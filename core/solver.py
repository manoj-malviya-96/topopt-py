from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix as SparseMatrix
from scipy.sparse.linalg import splu, cg, spilu, LinearOperator

try:
    from sksparse.cholmod import analyze as cholmod_analyze

    HAS_CHOLMOD = True
except ImportError:
    cholmod_analyze = None  # type: ignore
    HAS_CHOLMOD = False


class SolverMethod(Enum):
    DIRECT = auto()
    ITERATIVE = auto()


def compute_free_dofs(n_dof: int, fixed_dofs: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return DOF indices not in fixed_dofs."""
    return np.setdiff1d(np.arange(n_dof, dtype=np.int64), fixed_dofs)


class DirectSolver:
    """LU factorization solver. Uses CHOLMOD if available.

    With CHOLMOD, the sparsity pattern of K_free is constant across SIMP
    iterations (only values change), so the symbolic analysis (fill-reducing
    ordering) is done once via analyze() and reused; only cholesky_inplace()
    redoes the numeric factorization each solve.
    """
    __slots__ = ('_free_dofs', '_n_dof', '_displacement', '_use_cholmod', '_factor')

    def __init__(self, n_dof: int, free_dofs: NDArray[np.int64]):
        self._free_dofs = free_dofs
        self._n_dof = n_dof
        self._displacement = np.zeros(n_dof, dtype=np.float64)
        self._use_cholmod = HAS_CHOLMOD
        self._factor = None

    def solve(self, K: SparseMatrix, f: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve Ku=f, returning full displacement vector (zeros at fixed DOFs)."""
        K_free = K.tocsc()[self._free_dofs, :][:, self._free_dofs]
        f_free = f[self._free_dofs]

        if self._use_cholmod:
            if self._factor is None:
                self._factor = cholmod_analyze(K_free)
            self._factor.cholesky_inplace(K_free)
            u_free = self._factor(f_free)
        else:
            u_free = splu(K_free, permc_spec='MMD_AT_PLUS_A').solve(f_free)

        self._displacement.fill(0.0)
        self._displacement[self._free_dofs] = u_free
        return self._displacement


class IterativeSolver:
    """Preconditioned CG solver with warm start."""
    __slots__ = ('_free_dofs', '_n_dof', '_displacement', '_prev_solution', '_tol', '_maxiter')

    def __init__(self, n_dof: int, free_dofs: NDArray[np.int64], tol: float = 1e-8, maxiter: int = 1000):
        self._free_dofs = free_dofs
        self._n_dof = n_dof
        self._displacement = np.zeros(n_dof, dtype=np.float64)
        self._prev_solution = None
        self._tol = tol
        self._maxiter = maxiter

    def solve(self, K: SparseMatrix, f: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve Ku=f using ILU-preconditioned CG."""
        K_free = K.tocsc()[self._free_dofs, :][:, self._free_dofs]
        f_free = f[self._free_dofs]

        ilu = spilu(K_free, drop_tol=1e-4)
        preconditioner = LinearOperator(K_free.shape, matvec=ilu.solve)

        u_free, info = cg(K_free, f_free, x0=self._prev_solution,
                          rtol=self._tol, maxiter=self._maxiter, M=preconditioner)
        if info != 0:
            raise RuntimeError(f"CG did not converge (info={info})")

        self._prev_solution = u_free.copy()
        self._displacement.fill(0.0)
        self._displacement[self._free_dofs] = u_free
        return self._displacement


def create_solver(method: SolverMethod, n_dof: int, free_dofs: NDArray[np.int64]):
    """Factory for solver instances."""
    if method == SolverMethod.DIRECT:
        return DirectSolver(n_dof, free_dofs)
    elif method == SolverMethod.ITERATIVE:
        return IterativeSolver(n_dof, free_dofs)
    raise ValueError(f"Unknown solver: {method}")


def compute_sensitivity(elem_dof_indices: NDArray[np.int64], ke: NDArray[np.float64],
                        displacement: NDArray[np.float64], density: NDArray[np.float64],
                        penal: float, E: float, E_min: float) -> NDArray[np.float64]:
    """Compute compliance sensitivity dc/dx for each element."""
    elem_u = displacement[elem_dof_indices]
    strain_energy = (elem_u @ ke * elem_u).sum(axis=1)
    return -penal * (E - E_min) * np.power(density, penal - 1) * strain_energy
