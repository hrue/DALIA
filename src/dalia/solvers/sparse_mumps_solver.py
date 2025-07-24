# Copyright 2024-2025 DALIA authors. All rights reserved.

import warnings

from dalia import NDArray, sp, xp, backend_flags
from dalia.configs.dalia_config import SolverConfig
from dalia.core.solver import Solver


if backend_flags["mpi_avail"]:
    from mpi4py import MPI
    from mpi4py.MPI import Comm as mpi_comm
else:
    mpi_comm = None

# Try to import mumpspy with proper error handling
MUMPS_AVAILABLE = False
try:
    from mumpspy import MumpsSolver

    MUMPS_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"The mumpspy package is required to use the MumpsSolver: {e}")
    MumpsSolver = None


class SparseMumpsSolver(Solver):
    def __init__(
        self,
        config: SolverConfig,
        comm: mpi_comm,
        **kwargs,
    ) -> None:
        """Initializes the solver."""
        super().__init__(config)

        if not MUMPS_AVAILABLE:
            raise ImportError(
                "mumpspy is not available. Please install mumpspy and ensure "
                "the libdmumps.so library is properly linked."
            )
        
        self.comm: mpi_comm = comm
        self.rank: int = self.comm.Get_rank()
        self.comm_size: int = self.comm.size        

        # initialize MumpsSolver
        self.mumps_solver = MumpsSolver(verbose=False, mpi_rank=self.rank, comm=self.comm)

        self.nnz = None
        self.n = None 

    def cholesky(self, A: sp.sparse.spmatrix, **kwargs) -> None:
        """Compute Cholesky factor of input matrix."""

        A = sp.sparse.coo_matrix(A)
        self.mumps_solver.factorize(sparse_matrix=A, compute_determinant=True)
        self.nnz = self.mumps_solver.nnz
        self.n = A.shape[0]

    def solve(
        self,
        rhs: NDArray,
        **kwargs,
    ) -> NDArray:
        """Solve linear system using Cholesky factor."""

        if self.mumps_solver.is_factorized is False:
            raise ValueError("Cholesky factor not computed. Please call cholesky first.")
        
        rhs[:] = self.mumps_solver.solve(rhs)

        return rhs

    def logdet(
        self,
        **kwargs,
    ) -> float:
        """Compute logdet of input matrix using Cholesky factor."""

        if self.mumps_solver.is_factorized is False:
            raise ValueError("Cholesky factor not computed. Please call cholesky first.")

        return self.mumps_solver.logdet()

    def selected_inversion(self, **kwargs):

        if self.mumps_solver.is_factorized is False:
            raise ValueError("Cholesky factor not computed. Please call cholesky first.")

        self.a_inv_mat = self.mumps_solver.selected_inverse()

    def _structured_to_spmatrix(self,        
        sparsity_pattern: sp.sparse.spmatrix,
        **kwargs,
        ) -> None:
        """ Extract only those entries of a_inv that match A"""

        # set data entries in sparsity pattern to 1
        sparsity_pattern.data = xp.ones_like(sparsity_pattern.data, dtype=xp.float64)
        
        # Apply the sparsity pattern
        self.a_inv_mat = self.a_inv_mat.multiply(sparsity_pattern)

        # Return the sparse matrix
        return self.a_inv_mat

    def get_solver_memory(self) -> int:
        """Return the memory used by the solver in number of bytes"""

        if self.nnz is None or self.n is None:
            print("Matrix nnz not known yet. Please call cholesky first.")
            return 0
        
        memory_bytes = 8 * self.nnz + 4 * (self.nnz + self.n + 1)
        return memory_bytes


if __name__ == "__main__":
    import numpy as np
    from dalia.configs.dalia_config import SolverConfig

    # Initialize MPI
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()  
    mpi_size = comm.Get_size()

    # Example usage
    config = SolverConfig()
    solver = SparseMumpsSolver(config, comm=comm)
    print("SparseMumpsSolver initialized successfully.")

    n = 5
    a = np.array([4.0, 1.0, 4.0, 1.0, 4.0, 1.0, 4.0, 1.0, 4.0], dtype=np.float64)
    ia = np.array([1, 3, 5, 7, 9, 10], dtype=np.int32)  # Row pointers (1-based)
    ja = np.array(
        [1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int32
    )  # Column indices (1-based)

    print("-------- Reference calculation ----------")
    # construct matrix to check
    A_upper = sp.sparse.csr_matrix(
        (a, ja - 1, ia - 1), shape=(n, n)
    )  # Convert to 0-based indexing for scipy
    print("Test matrix A (lower triangular part):")
    print(A_upper.toarray())

    A = A_upper + A_upper.T - sp.sparse.diags(A_upper.diagonal())

    # Create a right-hand side vector
    rhs = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    # Reference solution
    L = sp.linalg.cholesky(A_upper.toarray(), lower=False)
    print("L:\n", L)
    logdet_ref = 2 * np.sum(np.log(L.diagonal()))
    solution_ref = sp.linalg.solve(A.toarray(), rhs)

    # Perform Cholesky decomposition
    solver.cholesky(A_upper, compute_determinant=True)

    print("nnz(L):", solver.nnz)
    print("Matrix size (n):", solver.n)
    
    # Test memory calculation
    memory_bytes = solver.get_solver_memory()
    print(f"Solver memory usage: {memory_bytes} bytes ({memory_bytes / 1024:.2f} KB)")

    # Solve the linear system
    solution = solver.solve(rhs.copy())  # Use copy to preserve original rhs
    print(f"Solution (Mumps):     {solution}")
    print(f"Solution (Reference):   {solution_ref}")
    print(f"Solution difference:    {np.abs(solution - solution_ref)}")

    # Compute log determinant
    logdet = solver.logdet()
    print(f"\nLog determinant (Mumps):   {logdet:.6f}")
    print(f"Log determinant (Reference): {logdet_ref:.6f}")
    print(f"Log determinant difference:  {abs(logdet - logdet_ref):.2e}")

    # Test selected inversion
    print("\n" + "-" * 60)
    print("TESTING SELECTED INVERSION")
    print("-" * 60)
    solver.selected_inversion()

    selected_inv_result = solver._structured_to_spmatrix(A)
    print(f"Selected inversion result:\n {selected_inv_result.toarray()}")

    print("\n" + "=" * 60)
    print("TESTING COMPLETED")
    print("=" * 60)
