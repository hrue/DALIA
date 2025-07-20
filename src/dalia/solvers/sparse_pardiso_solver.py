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
        self.pardiso_solver = PardisoSolver(mtype=mtype, verbose=True)

        self.L: sp.sparse.spmatrix = None

    def cholesky(self, A: sp.sparse.spmatrix, **kwargs) -> None:
        """Compute Cholesky factor of input matrix."""

        A = sp.sparse.csr_matrix(A)

        self.pardiso_solver.factorize(sparse_matrix=A, compute_determinant=True)
        # LU = sp.sparse.linalg.splu(A, diag_pivot_thresh=0, permc_spec="NATURAL")

    def solve(
        self,
        rhs: NDArray,
        **kwargs,
    ) -> NDArray:
        """Solve linear system using Cholesky factor."""

        if self.L is None:
            raise ValueError("Cholesky factor not computed")

        sp.sparse.linalg.spsolve_triangular(self.L, rhs, lower=True, overwrite_b=True)
        sp.sparse.linalg.spsolve_triangular(
            self.L.T, rhs, lower=False, overwrite_b=True
        )

        return rhs

    def logdet(
        self,
        **kwargs,
    ) -> float:
        """Compute logdet of input matrix using Cholesky factor."""

        if self.L is None:
            raise ValueError("Cholesky factor not computed")

        return 2 * xp.sum(xp.log(self.L.diagonal()))

    def selected_inversion(self, **kwargs):
        # Placeholder for the selected inversion method.
        return super().selected_inversion(**kwargs)

    def _structured_to_spmatrix(self, **kwargs) -> None:
        """Convert structured matrix to sparse matrix."""

        print("In _structured_to_spmatrix. Nothing to be done here.")

    def get_solver_memory(self) -> int:
        """Return the memory used by the solver in number of bytes"""
        if self.L is None:
            return 0

        return self.L.data.nbytes + self.L.indptr.nbytes + self.L.indices.nbytes


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
    logdet_ref = 2 * np.sum(np.log(L.diagonal()))
    solution_ref = sp.linalg.solve(A.toarray(), rhs)

    # Perform Cholesky decomposition
    solver.cholesky(A_upper, compute_determinant=True)

    # Solve the linear system
    solution = solver.solve(rhs)
    print("Solution to the linear system:", solution)
    print("Reference solution:", solution_ref)

    # Compute log determinant
    logdet = solver.logdet()
    print("Log determinant of the matrix:", logdet)
    print("Reference log determinant:", logdet_ref)
