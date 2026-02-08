from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix as SparseMatrix
from scipy.sparse.linalg import spsolve, cg

from core.utils import time_benchmark


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


def _solve_reduced_system(
        stiffness_matrix: SparseMatrix,
        force_vector: NDArray[np.float64],
        free_dofs: NDArray[np.int64],
        use_cg: bool = False
) -> NDArray[np.float64]:
    """
    Solve reduced system K[free,free] * u[free] = f[free].
    """
    # Extract reduced system using fast sparse slicing (CSR for rows, then CSC for cols)
    K_ff = stiffness_matrix.tocsr()[free_dofs, :][:, free_dofs]
    f_f = force_vector[free_dofs]

    if use_cg:
        u_f, exit_code = cg(K_ff, f_f)
        if exit_code != 0:
            raise RuntimeError(f"CG solver did not converge (exit code: {exit_code}).")
    else:
        u_f = spsolve(K_ff.tocsc(), f_f)

    # Scatter back to full vector
    u = np.zeros(stiffness_matrix.shape[0], dtype=np.float64)
    u[free_dofs] = u_f
    return u


@time_benchmark
def solve_displacements(
        stiffness_matrix: SparseMatrix,
        force_vector: NDArray[np.float64],
        free_dofs: NDArray[np.int64],
        method: SolverMethod
) -> NDArray[np.float64]:
    """
    Solve Ku = f with Dirichlet BCs.
    Expects precomputed free_dofs from compute_free_dofs().
    Returns the full displacement vector (zeros at fixed DOFs).
    """
    match method:
        case SolverMethod.DIRECT:
            return _solve_reduced_system(stiffness_matrix, force_vector, free_dofs, use_cg=False)
        case SolverMethod.ITERATIVE:
            return _solve_reduced_system(stiffness_matrix, force_vector, free_dofs, use_cg=True)
        case _:
            raise ValueError(f"Unknown solver method: {method}")


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
