#!/usr/bin/env python3

import argparse
import logging
import sys

from core.mesh import plot_density
from core.optimize import run_optimization
from core.solver import SolverMethod
from core.utils import memory_benchmark, time_benchmark


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Topology Optimization using SIMP method',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--nelx', type=int, default=30,
                        help='Number of elements in x direction')
    parser.add_argument('--nely', type=int, default=10,
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
    parser.add_argument('--save', type=str, default=None,
                        help='Save final result to file (e.g., result.png)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')

    return parser.parse_args()


@time_benchmark
@memory_benchmark
def _run_optimization(*args):
    return run_optimization(*args)


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    solver_method = SolverMethod.DIRECT if args.solver == 'direct' else SolverMethod.ITERATIVE

    try:
        result = _run_optimization(
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
        )
        plot_density(result, args.nelx, args.nely, show=True, save_path=args.save)
        if args.save:
            logger.info(f"Result saved to: {args.save}")
        return 0
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
