from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix as SparseMatrix
from scipy.sparse.linalg import splu, cg, spilu, LinearOperator

# Try to import scikit-sparse for faster Cholesky (optional)
try:
    from sksparse.cholmod import cholesky as cholmod_cholesky

    HAS_CHOLMOD = True
except ImportError:
    cholmod_cholesky = None  # type: ignore
    HAS_CHOLMOD = False


class SolverMethod(Enum):
    DIRECT = auto()
    ITERATIVE = auto()


def compute_free_dofs(n_dof: int, fixed_dof_indices: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    Compute free DOF indices (complement of fixed DOFs).
    Cache this result and reuse across iterations.
    """
    all_dofs = np.arange(n_dof, dtype=np.int64)
    return np.setdiff1d(all_dofs, fixed_dof_indices)


class DirectSolver:
    """
    Direct solver using SuperLU factorization with symmetric ordering.
    For SPD matrices, uses Cholesky via CHOLMOD if available.
    """
    __slots__ = ('free_dofs', 'n_dof', '_u', '_use_cholmod')

    def __init__(self, n_dof: int, free_dofs: NDArray[np.int64]):
        self.free_dofs = free_dofs
        self.n_dof = n_dof
        self._u = np.zeros(n_dof, dtype=np.float64)
        self._use_cholmod = HAS_CHOLMOD

    def solve(self, K: SparseMatrix, f: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve Ku = f for free DOFs."""
        # Extract submatrix
        K_ff = K.tocsc()[self.free_dofs, :][:, self.free_dofs]
        f_f = f[self.free_dofs]

        if self._use_cholmod:
            # CHOLMOD Cholesky - fastest for SPD matrices
            factor = cholmod_cholesky(K_ff)
            u_f = factor(f_f)
        else:
            # SuperLU with symmetric ordering (better for symmetric matrices)
            lu = splu(K_ff, permc_spec='MMD_AT_PLUS_A')
            u_f = lu.solve(f_f)

        # Scatter back (reuse array)
        self._u.fill(0.0)
        self._u[self.free_dofs] = u_f
        return self._u


class IterativeSolver:
    """
    Iterative CG solver with ILU preconditioner and warm start.
    Much faster when solutions don't change drastically between iterations.
    """
    __slots__ = ('free_dofs', 'n_dof', '_u', '_u_prev', 'tol', 'maxiter')

    def __init__(self, n_dof: int, free_dofs: NDArray[np.int64], tol: float = 1e-8, maxiter: int = 1000):
        self.free_dofs = free_dofs
        self.n_dof = n_dof
        self._u = np.zeros(n_dof, dtype=np.float64)
        self._u_prev = None  # Previous solution for warm start
        self.tol = tol
        self.maxiter = maxiter

    def solve(self, K: SparseMatrix, f: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve Ku = f using preconditioned CG with warm start."""
        K_ff = K.tocsc()[self.free_dofs, :][:, self.free_dofs]
        f_f = f[self.free_dofs]

        # Build ILU preconditioner
        ilu = spilu(K_ff, drop_tol=1e-4)
        M = LinearOperator(K_ff.shape, matvec=ilu.solve)

        # Use previous solution as initial guess (warm start)
        x0 = self._u_prev if self._u_prev is not None else None

        u_f, info = cg(K_ff, f_f, x0=x0, rtol=self.tol, maxiter=self.maxiter, M=M)
        if info != 0:
            raise RuntimeError(f"CG solver did not converge (info: {info}).")

        # Cache solution for next iteration's warm start
        self._u_prev = u_f.copy()

        # Scatter back
        self._u.fill(0.0)
        self._u[self.free_dofs] = u_f
        return self._u


def create_solver(method: SolverMethod, n_dof: int, free_dofs: NDArray[np.int64]):
    """Factory function to create the appropriate solver."""
    if method == SolverMethod.DIRECT:
        return DirectSolver(n_dof, free_dofs)
    elif method == SolverMethod.ITERATIVE:
        return IterativeSolver(n_dof, free_dofs)
    else:
        raise ValueError(f"Unknown solver method: {method}")


def solve_displacements(
        stiffness_matrix: SparseMatrix,
        force_vector: NDArray[np.float64],
        free_dofs: NDArray[np.int64],
        method: SolverMethod
) -> NDArray[np.float64]:
    """
    Solve Ku = f with Dirichlet BCs (legacy interface).
    For better performance, use create_solver() and reuse the solver object.
    """
    if method == SolverMethod.DIRECT:
        K_ff = stiffness_matrix.tocsc()[free_dofs, :][:, free_dofs]
        f_f = force_vector[free_dofs]
        lu = splu(K_ff)
        u_f = lu.solve(f_f)
    else:
        K_ff = stiffness_matrix.tocsr()[free_dofs, :][:, free_dofs]
        f_f = force_vector[free_dofs]
        u_f, info = cg(K_ff, f_f, rtol=1e-8)
        if info != 0:
            raise RuntimeError(f"CG solver did not converge (info: {info}).")

    u = np.zeros(stiffness_matrix.shape[0], dtype=np.float64)
    u[free_dofs] = u_f
    return u


def compute_sensitivity(
        element_connectivity: NDArray[np.int64],
        element_stiffness: NDArray[np.float64],
        displacement_vector: NDArray[np.float64],
        penalization: float,
        young_modulus: float,
        young_modulus_min: float
) -> NDArray[np.float64]:
    """
    Compute compliance sensitivity (dc/dx) for each element.
    """
    sensitivity_factor = -penalization * (young_modulus - young_modulus_min)
    element_displacements = displacement_vector[element_connectivity]

    # Compute element strain energies: u_e^T * ke * u_e for each element
    element_strain_energies = np.einsum('ij,jk,ik->i', element_displacements, element_stiffness, element_displacements)
    return sensitivity_factor * element_strain_energies
