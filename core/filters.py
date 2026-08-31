from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve as ndimage_convolve

FilterMode = Literal["reflect", "constant", "nearest", "mirror", "wrap"]


def create_filter_kernel(r_min: float) -> NDArray[np.float64]:
    """Build normalized 2D cone kernel for density filtering."""
    radius = int(np.ceil(r_min))
    y, x = np.meshgrid(np.arange(-radius, radius + 1),
                       np.arange(-radius, radius + 1), indexing='ij')
    weights = np.maximum(0.0, r_min - np.sqrt(x ** 2 + y ** 2))
    return weights / weights.sum()


def compute_filter_normalization(shape: tuple[int, int], kernel: NDArray[np.float64],
                                 mode: FilterMode) -> NDArray[np.float64]:
    """Precompute boundary normalization for a shape/kernel/mode (invariant across SIMP iterations)."""
    normalization = ndimage_convolve(np.ones(shape, dtype=np.float64), kernel, mode=mode)
    normalization[normalization == 0] = 1.0
    return normalization


def apply_density_filter(density: NDArray[np.float64], kernel: NDArray[np.float64],
                         mode: FilterMode, normalization: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply density filter using precomputed boundary normalization."""
    return ndimage_convolve(density, kernel, mode=mode) / normalization


def heaviside_projection(x: NDArray[np.float64], eta: float, beta: float) -> NDArray[np.float64]:
    """Smooth Heaviside projection for sharp boundaries."""
    numer = np.tanh(beta * eta) + np.tanh(beta * (x - eta))
    denom = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    if np.isclose(denom, 0):
        raise ValueError("Heaviside denominator is zero")
    return numer / denom


def d_heaviside_dx(x: NDArray[np.float64], eta: float, beta: float) -> NDArray[np.float64]:
    """Derivative of Heaviside projection w.r.t. x."""
    t = np.tanh(beta * (x - eta))
    denom = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    if np.isclose(denom, 0):
        raise ValueError("Heaviside derivative denominator is zero")
    return beta * (1 - t ** 2) / denom
