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
    """
    n_dof = int(elem_conn.max()) + 1
    nodes_per_elem = ke.shape[0]

    s_k = young_modulus_min + (x_phys ** penal) * (young_modulus - young_modulus_min)

    i_loc = np.repeat(np.arange(nodes_per_elem), nodes_per_elem)
    j_loc = np.tile(np.arange(nodes_per_elem), nodes_per_elem)

    rows = elem_conn[:, i_loc].flatten()
    cols = elem_conn[:, j_loc].flatten()

    ke_flat = ke.flatten()
    vals = np.outer(s_k, ke_flat).flatten()

    result = sp.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    return result.tocsr()
