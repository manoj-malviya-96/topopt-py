#!/usr/bin/env python3

import argparse
import logging
import sys
from typing import Literal

import numpy as np

from core.mesh import setup_problem, plot_density
from core.optimize import optimize_compliance, TopOptConfig
from core.solver import SolverMethod


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Topology Optimization using SIMP method',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--nelx', type=int, default=60,
                        help='Number of elements in x direction')
    parser.add_argument('--nely', type=int, default=20,
                        help='Number of elements in y direction')
    parser.add_argument('--problem', type=str, default='mbb',
                        choices=['mbb', 'cantilever'],
                        help='Problem type')
    parser.add_argument('--volfrac', type=float, default=0.5,
                        help='Volume fraction constraint')
    parser.add_argument('--penal', type=float, default=3.0,
                        help='Penalization factor for SIMP')
    parser.add_argument('--rmin', type=float, default=1.5,
                        help='Filter radius (in elements)')
    parser.add_argument('--maxiter', type=int, default=100,
                        help='Maximum number of iterations')
    parser.add_argument('--tol', type=float, default=0.01,
                        help='Convergence tolerance for design change')
    parser.add_argument('--solver', type=str, default='direct',
                        choices=['direct', 'iterative'],
                        help='Linear solver method')
    parser.add_argument('--no-filter', action='store_true',
                        help='Disable density filtering')
    parser.add_argument('--plot-every', type=int, default=0,
                        help='Plot intermediate results every N iterations (0 to disable)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save final result to file (e.g., result.png)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')

    return parser.parse_args()


logger = logging.getLogger(__name__)


def run_optimization(
        nelx: int,
        nely: int,
        problem_type: Literal["mbb", "cantilever"],
        volfrac: float,
        penalization: float,
        r_min: float,
        max_iter: int,
        tol: float,
        solver_method: SolverMethod,
        use_filter: bool,
        plot_every: int,
        save_path: str | None
) -> np.ndarray:
    """
    Run the topology optimization.
    """
    logger.info(f"Setting up {problem_type.upper()} problem: {nelx} x {nely} elements")

    mesh, fixed_dofs, force_vector = setup_problem(nelx, nely, problem_type)
    logger.info(f"Mesh: {mesh.n_elem} elements, {mesh.n_node} nodes, {mesh.n_dof} DOFs")

    config = TopOptConfig(
        target_vol_frac=volfrac,
        penalization=penalization,
        r_min=r_min,
        move=0.2,
        max_iter=max_iter,
        tol=tol,
        solver_method=solver_method,
        use_filter=use_filter
    )

    def plot_callback(iteration: int, x: np.ndarray, compliance: float,
                      volume: float, change: float) -> bool:
        if plot_every > 0 and iteration % plot_every == 0:
            plot_density(x, nelx, nely, iteration=iteration, show=True)
        return False

    # Run optimization
    logger.info("Starting optimization...")
    x_final, compliance = optimize_compliance(
        mesh=mesh,
        fixed_dofs=fixed_dofs,
        force_vector=force_vector,
        config=config,
        callback=plot_callback if plot_every > 0 else None,
        show_progress=True
    )

    logger.info("=" * 60)
    logger.info(f"Final compliance: {compliance:.4f}")
    logger.info(f"Final volume fraction: {x_final.mean():.4f}")

    # Plot final result
    plot_density(x_final, nelx, nely, show=True, save_path=save_path)

    if save_path:
        logger.info(f"Result saved to: {save_path}")

    return x_final


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    solver_method = SolverMethod.DIRECT if args.solver == 'direct' else SolverMethod.ITERATIVE

    try:
        run_optimization(
            nelx=args.nelx,
            nely=args.nely,
            problem_type=args.problem,
            volfrac=args.volfrac,
            penalization=args.penal,
            r_min=args.rmin,
            max_iter=args.maxiter,
            tol=args.tol,
            solver_method=solver_method,
            use_filter=not args.no_filter,
            plot_every=args.plot_every,
            save_path=args.save
        )
        return 0
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
