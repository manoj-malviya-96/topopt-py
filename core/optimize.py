from abc import ABC, abstractmethod
from dataclasses import dataclass
from venv import logger

import numpy as np
from numpy.typing import NDArray


def std_optimize_update(
        x: NDArray[np.float64],
        x_target: float,
        derivative_objective: NDArray[np.float64],
        derivative_constraint: NDArray[np.float64],
        move: float,
        max_iter: int = 50,
        tol: float = 1e-4
) -> NDArray[np.float64]:
    """
    Optimality Criteria update with improved robustness, clarity, and efficiency.
    """
    # 1. Pre-calculate combined bounds once, outside the loop.
    lower_bound = np.maximum(x - move, 1e-3)
    upper_bound = np.minimum(x + move, 1.0)
    lambda_min, lambda_max = 0.0, 1e9
    x_new = x.copy()

    for _ in range(max_iter):
        lambda_mid = 0.5 * (lambda_min + lambda_max)

        # 2. Safely calculate the update factor 'B' to avoid sqrt of negative numbers.
        # We only update elements where the sensitivity condition (-dc/dv) is positive.
        ratio = -derivative_objective / (derivative_constraint * lambda_mid)

        # Default to B=1 (no change), then update only the valid elements.
        be = np.ones_like(x)
        update_mask = ratio > 0
        be[update_mask] = np.sqrt(ratio[update_mask])

        # 3. Apply the update and a single, combined clip.
        x_new = np.clip(x * be, lower_bound, upper_bound)

        # 4. Use standard bisection logic.
        x_mean = x_new.mean()
        if x_mean - x_target > 0:
            lambda_min = lambda_mid
        else:
            lambda_max = lambda_mid

        # Check for convergence
        if abs(x_mean - x_target) < tol:
            break

    return x_new


class OptimizerProblem(ABC):

    @abstractmethod
    def objective(self, x: NDArray[np.float64]) -> float:
        pass

    @abstractmethod
    def constraint(self, x: NDArray[np.float64]) -> float:
        pass

    @abstractmethod
    def derivative_objective(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def derivative_constraint(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        pass


@dataclass(frozen=True)
class OptimizerParams:
    x_target: float
    move: float = 0.2
    max_iteration: int = 50
    tol: float = 1e-4


def optimize(problem: OptimizerProblem, x_guess: NDArray[np.float64], params: OptimizerParams) -> NDArray[np.float64]:
    """
    Optimize the design variable x using the provided problem and parameters.
    """
    x = x_guess

    for i in range(params.max_iteration):
        # Calculate derivatives
        d_obj = problem.derivative_objective(x)
        d_con = problem.derivative_constraint(x)

        # Perform the update
        x_new = std_optimize_update(
            x, params.x_target, d_obj, d_con, params.move, params.max_iteration, params.tol
        )

        # Check for convergence
        if np.linalg.norm(x_new - x) < params.tol:
            break
        x = x_new
        logger.info(f"Iteration: {i}, Objective: {problem.objective(x_new)}, "
                    f"Constraint: {problem.constraint(x_new)}")

    return x
