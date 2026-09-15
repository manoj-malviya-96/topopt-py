#!/usr/bin/env python3
import argparse
import logging
import sys

from core.mesh import plot_density, create_optimization_gif, create_optimization_webm
from core.optimize import TopOptConfig, run_optimization
from core.solver import SolverMethod


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
    parser.add_argument('--nelx', type=int, default=200,
                        help='Number of elements in x direction')
    parser.add_argument('--nely', type=int, default=100,
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
    parser.add_argument('--gif', type=str, default=None,
                        help='Save optimization animation as GIF (e.g., results/optimization.gif)')
    parser.add_argument('--webm', type=str, default=None,
                        help='Save optimization animation as WebM video (e.g., results/optimization.webm)')
    parser.add_argument('--video-dpi', type=int, default=150,
                        help='Resolution (dpi) for the WebM export')
    parser.add_argument('--video-fps', type=int, default=10,
                        help='Frame rate for the WebM export')
    parser.add_argument('--transparent', action='store_true',
                        help='Render the WebM export with a transparent background')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    solver_method = SolverMethod.DIRECT if args.solver == 'direct' else SolverMethod.ITERATIVE

    try:
        config = TopOptConfig(
            nelx=args.nelx,
            nely=args.nely,
            problem_type=args.problem,
            target_vol_frac=args.volfrac,
            penalization=args.penal,
            r_min=args.rmin,
            max_iter=args.maxiter,
            tol=args.tol,
            solver_method=solver_method,
            use_filter=not args.no_filter,
        )
        if args.profile:
            import cProfile
            import pstats
            profiler = cProfile.Profile()
            profiler.enable()
            run_optimization(config)
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(30)
            return 0

        should_collect_history = args.gif is not None or args.webm is not None
        optimization_result = run_optimization(config, collect_history=should_collect_history)

        if should_collect_history:
            final_density, density_history = optimization_result
            if args.gif:
                create_optimization_gif(density_history, args.nelx, args.nely, args.gif)
                logger.info(f"GIF saved to: {args.gif}")
            if args.webm:
                create_optimization_webm(
                    density_history, args.nelx, args.nely, args.webm,
                    fps=args.video_fps, dpi=args.video_dpi, transparent=args.transparent,
                )
                logger.info(f"WebM saved to: {args.webm}")
        else:
            final_density = optimization_result

        plot_density(final_density, args.nelx, args.nely, show=True, save_path=args.save)
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
