import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from core.filters import create_filter_kernel, apply_density_filter
from core.mesh import RectangularMesh, setup_problem
from core.solver import solve_displacements, SolverMethod, compute_free_dofs
from core.stiffness import build_element_stiffness, assemble_stiffness_matrix

logger = logging.getLogger(__name__)


@dataclass
class TopOptConfig:
    nelx: int = 30
    nely: int = 10
    problem_type: Literal["mbb", "cantilever"] = "cantilever"
    target_vol_frac: float = 0.5
    penalization: float = 3.0
    r_min: float = 1.5
    young_modulus: float = 1.0
    young_modulus_min: float = 1e-9
    move: float = 0.2
    max_iter: int = 100
    tol: float = 0.01
    solver_method: SolverMethod = SolverMethod.DIRECT
    use_filter: bool = True


def run_optimization(
        config: TopOptConfig,
) -> np.ndarray:
    """
    Run the topology optimization.
    """
    logger.info(f"Setting up {config.problem_type.upper()} problem: {config.nelx} x {config.nely} elements")

    mesh, fixed_dofs, force_vector = setup_problem(config.nelx, config.nely, config.problem_type)
    logger.info(f"Mesh: {mesh.n_elem} elements, {mesh.n_node} nodes, {mesh.n_dof} DOFs")

    # Run optimization
    logger.info("Starting optimization...")
    x_final, compliance = optimize_compliance(
        mesh=mesh,
        fixed_dofs=fixed_dofs,
        force_vector=force_vector,
        config=config,
        show_progress=True
    )
    logger.info(f"Final compliance: {compliance:.4f}")
    logger.info(f"Final volume fraction: {x_final.mean():.4f}")
    return x_final


def _oc_update(
        x: NDArray[np.float64],
        dc: NDArray[np.float64],
        dv: NDArray[np.float64],
        volfrac: float,
        move: float
) -> None:
    """
    Optimality Criteria update. Modifies x in-place.
    """
    lower_bound = np.maximum(x - move, 1e-3)
    upper_bound = np.minimum(x + move, 1.0)
    lambda_min, lambda_max = 0.0, 1e9

    # Run binary search
    for i in range(100):
        lambda_mid = 0.5 * (lambda_min + lambda_max)
        ratio = -dc / (dv * lambda_mid)
        be = np.where(ratio > 0, np.sqrt(ratio), 1.0)

        # Update and clip
        np.multiply(x, be, out=x)
        np.clip(x, lower_bound, upper_bound, out=x)

        # Bisection
        if x.mean() > volfrac:
            lambda_min = lambda_mid
        else:
            lambda_max = lambda_mid

        if abs(x.mean() - volfrac) < 1e-4:
            break


def optimize_compliance(
        mesh: RectangularMesh,
        fixed_dofs: NDArray[np.int64],
        force_vector: NDArray[np.float64],
        config: TopOptConfig,
        show_progress: bool = True
) -> tuple[NDArray[np.float64], float]:
    """
    Run SIMP topology optimization for compliance minimization.
    """
    stiffness_mat = build_element_stiffness(nu=0.3)
    filter_kernel = create_filter_kernel(config.r_min) if config.use_filter else None

    # Cache free DOFs once (fixed DOFs don't change during optimization)
    free_dofs = compute_free_dofs(mesh.n_dof, fixed_dofs)

    def apply_filter(input_array: NDArray[np.float64]) -> NDArray[np.float64]:
        if filter_kernel is None:
            return input_array
        x_2d = input_array.reshape((mesh.nelx, mesh.nely))
        return apply_density_filter(x_2d, filter_kernel, mode='constant').flatten()

    x = np.full(mesh.n_elem, config.target_vol_frac, dtype=np.float64)
    x_old = np.empty_like(x)

    pbar = tqdm(
        range(1, config.max_iter + 1),
        desc="Optimizing",
        disable=not show_progress,
        ncols=80
    )

    compliance = 0.0
    for iteration in pbar:
        # Apply density filter
        x_phys = apply_filter(x)

        # Assemble stiffness and solve
        global_stiffness_matrix = assemble_stiffness_matrix(
            mesh.elem_conn, stiffness_mat, x_phys,
            config.penalization, config.young_modulus, config.young_modulus_min
        )
        u = solve_displacements(global_stiffness_matrix, force_vector, free_dofs, config.solver_method)
        compliance = float(force_vector @ u)

        elem_u = u[mesh.elem_conn]
        strain_energy = np.einsum('ij,jk,ik->i', elem_u, stiffness_mat, elem_u)
        dc = -config.penalization * (config.young_modulus - config.young_modulus_min) * \
             (x_phys ** (config.penalization - 1)) * strain_energy

        # Filter sensitivities
        if filter_kernel is not None:
            dc_2d = dc.reshape((mesh.nelx, mesh.nely))
            dc = apply_density_filter(dc_2d, filter_kernel, mode='constant').flatten()

        dv = np.ones(mesh.n_elem) / mesh.n_elem

        # Store old design
        np.copyto(x_old, x)

        # OC update (in-place)
        _oc_update(x, dc, dv, config.target_vol_frac, config.move)

        # Compute change
        change = np.max(np.abs(x - x_old))
        volume = x.mean()

        # Update progress
        pbar.set_postfix({'C': f'{compliance:.2f}', 'V': f'{volume:.3f}', 'Δ': f'{change:.4f}'})
        # Check convergence
        if change < config.tol:
            pbar.set_description("Converged")
            break
    else:
        pbar.set_description("Max iter")

    pbar.close()

    # Return filtered density
    return apply_filter(x), compliance
