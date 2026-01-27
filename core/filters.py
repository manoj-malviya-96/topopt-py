import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve as ndimage_convolve


def create_filter_kernel(r_min: float) -> NDArray[np.float64]:
    """
    Build a 2D convolution kernel for density filtering.
    """
    radius = int(np.ceil(r_min))
    # Using meshgrid and vectorized math is the correct, high-performance approach.
    dy, dx = np.meshgrid(
        np.arange(-radius, radius + 1),
        np.arange(-radius, radius + 1),
        indexing='ij'
    )
    h = np.maximum(0.0, r_min - np.sqrt(dx ** 2 + dy ** 2))
    return h / np.sum(h)


def apply_density_filter(
        x: NDArray[np.float64],
        kernel: NDArray[np.float64],
        mode: str
) -> NDArray[np.float64]:
    """
    Convolve design vector x with the kernel using scipy.ndimage.convolve.

    NOTE: scipy.ndimage.convolve is optimized for N-dimensional image filtering
    and is typically faster for this task. It also handles boundary conditions
    more directly.
    """
    hs = ndimage_convolve(np.ones_like(x), kernel, mode=mode)
    filtered = ndimage_convolve(x, kernel, mode=mode)
    hs[hs == 0] = 1.0
    return filtered / hs


def heaviside_projection(
        x: NDArray[np.float64],
        eta: float,
        beta: float
) -> NDArray[np.float64]:
    numerator = np.tanh(beta * eta) + np.tanh(beta * (x - eta))
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    if np.isclose(denominator, 0):
        raise ValueError("Denominator in Heaviside projection is zero, check eta and beta values.")
    return numerator / denominator


def d_heaviside_dx(
        x: NDArray[np.float64],
        eta: float,
        beta: float
) -> NDArray[np.float64]:
    """
    Derivative of the relaxed Heaviside projection wrt x.
    """
    t1 = np.tanh(beta * (x - eta))
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    if np.isclose(denominator, 0):
        raise ValueError("Denominator in Heaviside derivative is zero, check eta and beta values.")
    return beta * (1 - t1 ** 2) / denominator
