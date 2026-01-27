from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from utils import memory_benchmark, time_benchmark


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


@memory_benchmark
@time_benchmark
def assemble_stiffness_after(
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
    nodes_per_elem = ke.shape[0]  # Should be 8 in this case

    # 1. Calculate scaled stiffness for each element (already vectorized)
    s_k = young_modulus_min + x_phys ** penal * (young_modulus - young_modulus_min)
    i_loc = np.repeat(np.arange(nodes_per_elem), nodes_per_elem)
    j_loc = np.tile(np.arange(nodes_per_elem), nodes_per_elem)

    # Use the local indices to get the global DOF indices for all elements
    # and flatten them into 1D arrays
    rows = elem_conn[:, i_loc].flatten()
    cols = elem_conn[:, j_loc].flatten()

    # 3. Generate the values for all elements at once
    vals = np.kron(s_k, ke.flatten())

    # 4. Create the COO matrix from the vectorized data
    result = sp.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))

    # 5. Convert to CSR to sum duplicate entries and enforce symmetry
    result = result.tocsr()
    result = result + result.T - sp.diags(result.diagonal())
    return result
