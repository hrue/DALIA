# Copyright 2024-2025 DALIA authors. All rights reserved.

import warnings

from dalia import NDArray, sp, xp
from dalia.configs.dalia_config import SolverConfig
from dalia.core.solver import Solver

# Try to import pardisopy with proper error handling
PARDISO_AVAILABLE = False
try:
    from pardisopy import PardisoSolver

    PARDISO_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"The pardisopy package is required to use the PardisoSolver: {e}")
    PardisoSolver = None


class SparsePardisoSolver(Solver):
    def __init__(
        self,
        config: SolverConfig,
        **kwargs,
    ) -> None:
        """Initializes the solver."""
        super().__init__(config)

        if not PARDISO_AVAILABLE:
            raise ImportError(
                "pardisopy is not available. Please install pardisopy and ensure "
                "the libpardiso.dylib library is properly linked."
            )

        # initialize PardisoSolver and check license
        mtype = -2
        self.pardiso_solver = PardisoSolver(mtype=mtype, verbose=False)

        self.nnz = None
        self.n = None 

    def cholesky(self, A: sp.sparse.spmatrix, **kwargs) -> None:
        """Compute Cholesky factor of input matrix."""

        A = sp.sparse.csr_matrix(A)
        self.pardiso_solver.factorize(sparse_matrix=A, compute_determinant=True)
        self.nnz = self.pardiso_solver.nnz
        self.n = A.shape[0]

    def solve(
        self,
        rhs: NDArray,
        **kwargs,
    ) -> NDArray:
        """Solve linear system using Cholesky factor."""

        if self.pardiso_solver.is_factorized is False:
            raise ValueError("Cholesky factor not computed. Please call cholesky first.")
        
        rhs[:] = self.pardiso_solver.solve(rhs)

        return rhs

    def logdet(
        self,
        **kwargs,
    ) -> float:
        """Compute logdet of input matrix using Cholesky factor."""

        if self.pardiso_solver.is_factorized is False:
            raise ValueError("Cholesky factor not computed. Please call cholesky first.")

        return self.pardiso_solver.logdet()

    def selected_inversion(self, **kwargs):
        # Placeholder for the selected inversion method.

        if self.pardiso_solver.is_factorized is False:
            raise ValueError("Cholesky factor not computed. Please call cholesky first.")
        
        self.a_inv = self.pardiso_solver.selected_inverse()
        return self.a_inv

    def _structured_to_spmatrix(self,        
        sparsity_pattern: sp.sparse.spmatrix,
        **kwargs,
        ) -> None:
        """ Extract only those entries of a_inv that match A"""

        # set data entries in sparsity pattern to 1
        sparsity_pattern.data = xp.ones_like(sparsity_pattern.data, dtype=xp.float64)
        
        # construct sparse matrix using a_inv
        a_inv_mat = sp.sparse.csr_matrix(
            (self.a_inv, 
             self.pardiso_solver.ja - 1,
             self.pardiso_solver.ia - 1,  # Convert to 0-based indexing
             ),  # Convert to 0-based indexing
            shape=(self.n, self.n)
        )

        # Apply the sparsity pattern
        a_inv_mat = a_inv_mat.multiply(sparsity_pattern)

        # Return the sparse matrix
        return a_inv_mat



        print("In _structured_to_spmatrix. Nothing to be done here.")

    def get_solver_memory(self) -> int:
        """Return the memory used by the solver in number of bytes"""

        if self.nnz is None or self.n is None:
            print("Matrix dimensions not known yet. Please call cholesky first.")
            return 0
        
        memory_bytes = 8 * self.nnz + 4 * (self.nnz + self.n + 1)
        return memory_bytes


## write test


if __name__ == "__main__":
    import numpy as np

    from dalia.configs.dalia_config import SolverConfig

    # Example usage
    config = SolverConfig()
    solver = SparsePardisoSolver(config)
    print("SparsePardisoSolver initialized successfully.")

    n = 5
    a = np.array([4.0, 1.0, 4.0, 1.0, 4.0, 1.0, 4.0, 1.0, 4.5], dtype=np.float64)
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
    print(f"Solution (Pardiso):     {solution}")
    print(f"Solution (Reference):   {solution_ref}")
    print(f"Solution difference:    {np.abs(solution - solution_ref)}")

    # Compute log determinant
    logdet = solver.logdet()
    print(f"\nLog determinant (Pardiso):   {logdet:.6f}")
    print(f"Log determinant (Reference): {logdet_ref:.6f}")
    print(f"Log determinant difference:  {abs(logdet - logdet_ref):.2e}")

    # Test selected inversion
    print("\n" + "-" * 60)
    print("TESTING SELECTED INVERSION")
    print("-" * 60)
    selected_inv_result = solver.selected_inversion()
    print(f"Selected inversion result: {selected_inv_result}")

    
    print("\n" + "=" * 60)
    print("TESTING COMPLETED")
    print("=" * 60)
