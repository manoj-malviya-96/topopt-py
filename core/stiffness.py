from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray


@lru_cache(maxsize=None)
def build_element_stiffness(nu: float = 0.3) -> NDArray[np.float64]:
    """Build 8x8 element stiffness matrix for unit-square Q4 element."""
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
    coeffs = (c1 + nu * c2) / (1 - nu ** 2) / 24.0
    ke = np.zeros((8, 8), dtype=np.float64)
    idx = 0
    for j in range(8):
        for i in range(j, 8):
            ke[i, j] = ke[j, i] = coeffs[idx]
            idx += 1
    return ke


class StiffnessAssembler:
    """Caches sparsity pattern and COO->CSC scatter permutation for fast repeated assembly."""
    __slots__ = ('_n_dof', '_ke_flat', '_values', '_scatter_perm', '_nnz',
                 '_csc_indices', '_csc_indptr')

    def __init__(self, elem_dof_indices: NDArray[np.int64], ke: NDArray[np.float64]):
        dofs_per_elem = ke.shape[0]
        self._n_dof = int(elem_dof_indices.max()) + 1

        i_local = np.repeat(np.arange(dofs_per_elem, dtype=np.int64), dofs_per_elem)
        j_local = np.tile(np.arange(dofs_per_elem, dtype=np.int64), dofs_per_elem)

        rows = elem_dof_indices[:, i_local].ravel()
        cols = elem_dof_indices[:, j_local].ravel()
        self._ke_flat = ke.ravel()
        self._values = np.empty(len(rows), dtype=np.float64)

        # Precompute where each (row, col) entry lands in canonical CSC form
        # (sorted indices per column, duplicates summed), so later assembly
        # is an O(nnz) scatter-add instead of a full sort + dedup every call.
        order = np.lexsort((rows, cols))
        sorted_rows = rows[order]
        sorted_cols = cols[order]

        is_new = np.empty(len(order), dtype=bool)
        is_new[0] = True
        is_new[1:] = (sorted_rows[1:] != sorted_rows[:-1]) | (sorted_cols[1:] != sorted_cols[:-1])
        group_id = np.cumsum(is_new) - 1

        self._scatter_perm = np.empty(len(order), dtype=np.int64)
        self._scatter_perm[order] = group_id
        self._nnz = int(group_id[-1]) + 1
        self._csc_indices = sorted_rows[is_new]

        col_counts = np.bincount(sorted_cols[is_new], minlength=self._n_dof)
        self._csc_indptr = np.zeros(self._n_dof + 1, dtype=np.int64)
        np.cumsum(col_counts, out=self._csc_indptr[1:])

    def assemble(self, density: NDArray[np.float64], penal: float,
                 E: float, E_min: float) -> sp.csc_matrix:
        """Assemble global stiffness using SIMP interpolation."""
        material_factor = E_min + np.power(density, penal) * (E - E_min)
        np.multiply.outer(material_factor, self._ke_flat,
                          out=self._values.reshape(len(material_factor), -1))
        data = np.bincount(self._scatter_perm, weights=self._values, minlength=self._nnz)
        return sp.csc_matrix((data, self._csc_indices, self._csc_indptr),
                             shape=(self._n_dof, self._n_dof))

