from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray


@lru_cache(maxsize=None)
def build_element_stiffness(nu: float = 0.3):
    c1 = np.array([12, 3, -6, -3, -6, -3, 0, 3,
                   12, 3, 0, -3, -6, -3, -6, 12,
                   -3, 0, -3, -6, 3, 12, 3, -6,
                   3, -6, 12, 3, -6, -3, 12, 3,
                   0, 12, -3, 12])
    c2 = np.array([-4, 3, -2, 9, 2, -3, 4, -9,
                   -4, -9, 4, -3, 2, 9, -2, -4,
                   -3, 4, 9, 2, 3, -4, -9, -2,
                   3, 2, -4, 3, -2, 9, -4, -9,
                   4, -4, -3, -4])
    element_coefficients = (c1 + nu * c2) / (1 - nu ** 2) / 24.0
    element_matrix = np.zeros((8, 8), dtype=float)
    idx = 0
    for j in range(8):
        for i in range(j, 8):
            element_matrix[i, j] = element_coefficients[idx]
            element_matrix[j, i] = element_coefficients[idx]
            idx += 1
    return element_matrix


class StiffnessAssembler:
    """
    Pre-computes and caches the sparsity pattern for efficient assembly.
    The row/col indices only depend on mesh connectivity (constant).
    """
    __slots__ = ('rows', 'cols', 'n_dof', 'ke_flat', '_vals_buffer')

    def __init__(self, elem_conn: NDArray[np.int64], ke: NDArray[np.float64]):
        nodes_per_elem = ke.shape[0]
        self.n_dof = int(elem_conn.max()) + 1

        # Cache local index patterns
        i_loc = np.repeat(np.arange(nodes_per_elem, dtype=np.int64), nodes_per_elem)
        j_loc = np.tile(np.arange(nodes_per_elem, dtype=np.int64), nodes_per_elem)

        # Cache global row/col indices (constant for all iterations)
        self.rows = elem_conn[:, i_loc].ravel()
        self.cols = elem_conn[:, j_loc].ravel()

        # Cache flattened element stiffness
        self.ke_flat = ke.ravel()

        # Preallocate values buffer
        self._vals_buffer = np.empty(len(self.rows), dtype=np.float64)

    def assemble(
            self,
            x_phys: NDArray[np.float64],
            penal: float,
            young_modulus: float,
            young_modulus_min: float
    ) -> sp.csc_matrix:
        """Assemble stiffness matrix with given densities."""
        # SIMP material interpolation
        e_diff = young_modulus - young_modulus_min
        s_k = young_modulus_min + np.power(x_phys, penal) * e_diff

        # Compute values (reuse buffer to avoid allocation)
        np.multiply.outer(s_k, self.ke_flat, out=self._vals_buffer.reshape(len(s_k), -1))

        # Build COO and convert directly to CSC (solver needs CSC)
        result = sp.coo_matrix((self._vals_buffer, (self.rows, self.cols)), shape=(self.n_dof, self.n_dof))
        return result.tocsc()


# Legacy function for backwards compatibility
@lru_cache(maxsize=None)
def _get_local_indices(nodes_per_elem: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Cache local index patterns (constant for 8-node elements)."""
    i_loc = np.repeat(np.arange(nodes_per_elem, dtype=np.int64), nodes_per_elem)
    j_loc = np.tile(np.arange(nodes_per_elem, dtype=np.int64), nodes_per_elem)
    return i_loc, j_loc


def assemble_stiffness_matrix(
        elem_conn: NDArray[np.int64],
        ke: NDArray[np.float64],
        x_phys: NDArray[np.float64],
        penal: float,
        young_modulus: float,
        young_modulus_min: float
) -> sp.csr_matrix:
    """
    Assemble a global stiffness matrix using a fully vectorized approach.
    For better performance, use StiffnessAssembler class directly.
    """
    n_dof = int(elem_conn.max()) + 1
    nodes_per_elem = ke.shape[0]

    # SIMP material interpolation
    e_diff = young_modulus - young_modulus_min
    s_k = young_modulus_min + np.power(x_phys, penal) * e_diff

    # Get cached local indices
    i_loc, j_loc = _get_local_indices(nodes_per_elem)

    # Build global row/col indices using ravel for speed
    rows = elem_conn[:, i_loc].ravel()
    cols = elem_conn[:, j_loc].ravel()

    # Scale element stiffness by material factor
    ke_flat = ke.ravel()
    vals = (s_k[:, np.newaxis] * ke_flat).ravel()

    result = sp.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    return result.tocsr()
