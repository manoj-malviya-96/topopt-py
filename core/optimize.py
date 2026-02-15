import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from core.filters import create_filter_kernel, apply_density_filter
from core.mesh import RectangularMesh, setup_problem
from core.solver import SolverMethod, compute_free_dofs, create_solver
from core.stiffness import build_element_stiffness, StiffnessAssembler

logger = logging.getLogger(__name__)


@dataclass
class TopOptConfig:
    """Configuration for topology optimization."""
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


def run_optimization(config: TopOptConfig) -> np.ndarray:
    """Run topology optimization and return final density field."""
    logger.info(f"Setting up {config.problem_type.upper()}: {config.nelx}x{config.nely}")

    mesh, fixed_dofs, force = setup_problem(config.nelx, config.nely, config.problem_type)
    logger.info(f"Mesh: {mesh.n_elem} elements, {mesh.n_node} nodes, {mesh.n_dof} DOFs")

    logger.info("Starting optimization...")
    density_final, compliance = optimize_compliance(mesh, fixed_dofs, force, config, show_progress=True)

    logger.info(f"Final compliance: {compliance:.4f}")
    logger.info(f"Final volume fraction: {density_final.mean():.4f}")
    return density_final


def _oc_update(density: NDArray[np.float64], dc: NDArray[np.float64],
               dv: NDArray[np.float64], volfrac: float, move: float) -> None:
    """Optimality Criteria update. Modifies density in-place via bisection."""
    n = density.size
    target_sum = volfrac * n

    lower = np.maximum(density - move, 1e-3)
    upper = np.minimum(density + move, 1.0)
    sensitivity_ratio = -dc / dv

    lam_min, lam_max = 0.0, 1e9
    density_new = np.empty_like(density)

    for _ in range(50):
        lam_mid = 0.5 * (lam_min + lam_max)
        ratio = sensitivity_ratio / lam_mid

        np.maximum(ratio, 0, out=density_new)
        np.sqrt(density_new, out=density_new)
        density_new[ratio <= 0] = 1.0

        np.multiply(density, density_new, out=density_new)
        np.clip(density_new, lower, upper, out=density_new)

        if density_new.sum() > target_sum:
            lam_min = lam_mid
        else:
            lam_max = lam_mid

        if abs(density_new.sum() - target_sum) < 1e-4 * n:
            break

    np.copyto(density, density_new)


def optimize_compliance(mesh: RectangularMesh, fixed_dofs: NDArray[np.int64],
                        force: NDArray[np.float64], config: TopOptConfig,
                        show_progress: bool = True) -> tuple[NDArray[np.float64], float]:
    """Run SIMP topology optimization for compliance minimization."""
    ke = build_element_stiffness(nu=0.3)
    kernel = create_filter_kernel(config.r_min) if config.use_filter else None

    free_dofs = compute_free_dofs(mesh.n_dof, fixed_dofs)
    solver = create_solver(config.solver_method, mesh.n_dof, free_dofs)
    assembler = StiffnessAssembler(mesh.elem_conn, ke)

    def filter_density(x: NDArray[np.float64]) -> NDArray[np.float64]:
        if kernel is None:
            return x
        return apply_density_filter(x.reshape(mesh.nelx, mesh.nely), kernel, 'constant').flatten()

    density = np.full(mesh.n_elem, config.target_vol_frac, dtype=np.float64)
    density_prev = np.empty_like(density)

    pbar = tqdm(range(1, config.max_iter + 1), desc="Optimizing",
                disable=not show_progress, ncols=80)

    compliance = 0.0
    for _ in pbar:
        density_phys = filter_density(density)

        K = assembler.assemble(density_phys, config.penalization,
                               config.young_modulus, config.young_modulus_min)
        u = solver.solve(K, force)
        compliance = float(force @ u)

        elem_u = u[mesh.elem_conn]
        strain_energy = np.einsum('ij,jk,ik->i', elem_u, ke, elem_u)
        dc = -config.penalization * (config.young_modulus - config.young_modulus_min) * \
             np.power(density_phys, config.penalization - 1) * strain_energy

        if kernel is not None:
            dc = apply_density_filter(dc.reshape(mesh.nelx, mesh.nely), kernel, 'constant').flatten()

        dv = np.ones(mesh.n_elem) / mesh.n_elem

        np.copyto(density_prev, density)
        _oc_update(density, dc, dv, config.target_vol_frac, config.move)

        change = np.max(np.abs(density - density_prev))
        pbar.set_postfix({'C': f'{compliance:.2f}', 'V': f'{density.mean():.3f}', 'Δ': f'{change:.4f}'})

        if change < config.tol:
            pbar.set_description("Converged")
            break
    else:
        pbar.set_description("Max iter")

    pbar.close()
    return filter_density(density), compliance
