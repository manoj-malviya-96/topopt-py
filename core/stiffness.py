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
    """Caches sparsity pattern for fast repeated assembly."""
    __slots__ = ('_rows', '_cols', '_n_dof', '_ke_flat', '_values')

    def __init__(self, elem_dof_indices: NDArray[np.int64], ke: NDArray[np.float64]):
        dofs_per_elem = ke.shape[0]
        self._n_dof = int(elem_dof_indices.max()) + 1

        i_local = np.repeat(np.arange(dofs_per_elem, dtype=np.int64), dofs_per_elem)
        j_local = np.tile(np.arange(dofs_per_elem, dtype=np.int64), dofs_per_elem)

        self._rows = elem_dof_indices[:, i_local].ravel()
        self._cols = elem_dof_indices[:, j_local].ravel()
        self._ke_flat = ke.ravel()
        self._values = np.empty(len(self._rows), dtype=np.float64)

    def assemble(self, density: NDArray[np.float64], penal: float,
                 E: float, E_min: float) -> sp.csc_matrix:
        """Assemble global stiffness using SIMP interpolation."""
        material_factor = E_min + np.power(density, penal) * (E - E_min)
        np.multiply.outer(material_factor, self._ke_flat,
                          out=self._values.reshape(len(material_factor), -1))
        coo = sp.coo_matrix((self._values, (self._rows, self._cols)),
                            shape=(self._n_dof, self._n_dof))
        return coo.tocsc()


@lru_cache(maxsize=None)
def _get_local_indices(dofs_per_elem: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Cached local DOF index patterns."""
    i_local = np.repeat(np.arange(dofs_per_elem, dtype=np.int64), dofs_per_elem)
    j_local = np.tile(np.arange(dofs_per_elem, dtype=np.int64), dofs_per_elem)
    return i_local, j_local


def assemble_stiffness_matrix(elem_dof_indices: NDArray[np.int64], ke: NDArray[np.float64],
                              density: NDArray[np.float64], penal: float,
                              E: float, E_min: float) -> sp.csr_matrix:
    """Vectorized global stiffness assembly. Prefer StiffnessAssembler for loops."""
    n_dof = int(elem_dof_indices.max()) + 1
    dofs_per_elem = ke.shape[0]

    material_factor = E_min + np.power(density, penal) * (E - E_min)
    i_local, j_local = _get_local_indices(dofs_per_elem)

    rows = elem_dof_indices[:, i_local].ravel()
    cols = elem_dof_indices[:, j_local].ravel()
    values = (material_factor[:, np.newaxis] * ke.ravel()).ravel()

    coo = sp.coo_matrix((values, (rows, cols)), shape=(n_dof, n_dof))
    return coo.tocsr()
