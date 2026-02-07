from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix as SparseMatrix
from scipy.sparse.linalg import spsolve, cg

from core.utils import time_benchmark


class SolverMethod(Enum):
    DIRECT = auto()
    ITERATIVE = auto()


@time_benchmark
def _modify_system_for_bcs(
        stiffness_matrix: SparseMatrix,
        force_vector: NDArray[np.float64],
        fixed_dof_indices: NDArray[np.int64]
) -> tuple[SparseMatrix, NDArray[np.float64]]:
    """
    Apply Dirichlet BCs by modifying the full system in-place (via a copy).
    Returns modified stiffness matrix and force vector.
    """
    stiffness_modified = stiffness_matrix.copy().tolil()  # LIL for efficient row/col assignment
    force_modified = force_vector.copy()

    # Enforce u[fixed_dof_indices] = 0 by zeroing corresponding rows/cols and setting diagonal to 1
    for fixed_dof in fixed_dof_indices:
        stiffness_modified[fixed_dof, :] = 0.0
        stiffness_modified[:, fixed_dof] = 0.0
        stiffness_modified[fixed_dof, fixed_dof] = 1.0
    force_modified[fixed_dof_indices] = 0.0

    return stiffness_modified.tocsc(), force_modified


def _solve_direct(
        stiffness_matrix: SparseMatrix,
        force_vector: NDArray[np.float64],
        fixed_dof_indices: NDArray[np.int64]
) -> NDArray[np.float64]:
    """
    Solve the modified full linear system with a direct solver.
    """
    stiffness_modified, force_modified = _modify_system_for_bcs(
        stiffness_matrix, force_vector, fixed_dof_indices
    )
    return spsolve(stiffness_modified, force_modified, use_umfpack=True)


def _solve_using_cg(
        stiffness_matrix: SparseMatrix,
        force_vector: NDArray[np.float64],
        fixed_dof_indices: NDArray[np.int64]
) -> NDArray[np.float64]:
    """
    Solve the modified full linear system with Conjugate Gradient.
    Raises RuntimeError if CG does not converge.
    """
    stiffness_modified, force_modified = _modify_system_for_bcs(
        stiffness_matrix, force_vector, fixed_dof_indices
    )
    solution_vector, exit_code = cg(stiffness_modified, force_modified)
    if exit_code != 0:
        raise RuntimeError(f"Conjugate Gradient solver did not converge (exit code: {exit_code}).")
    return solution_vector


def solve_displacements(
        stiffness_matrix: SparseMatrix,
        force_vector: NDArray[np.float64],
        fixed_dof_indices: NDArray[np.int64],
        method: SolverMethod
) -> NDArray[np.float64]:
    """
    Solve Ku = f with Dirichlet BCs on `fixed_dof_indices`.
    Returns the full displacement vector (zeros enforced on fixed DOFs).
    """

    match method:
        case SolverMethod.DIRECT:
            return _solve_direct(stiffness_matrix, force_vector, fixed_dof_indices)
        case SolverMethod.ITERATIVE:
            return _solve_using_cg(stiffness_matrix, force_vector, fixed_dof_indices)
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
