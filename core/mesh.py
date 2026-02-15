from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


@dataclass
class RectangularMesh:
    """
    Holds mesh data for a 2D rectangular structured mesh.
    """
    nelx: int
    nely: int
    n_elem: int
    n_node: int
    n_dof: int
    elem_conn: NDArray[np.int64]


def build_rectangular_mesh(nelx: int, nely: int) -> RectangularMesh:
    """
    Build a structured rectangular mesh of Q4 elements (bilinear quads).
    Each element has 4 nodes and 8 DOFs (2 per node).
    Node numbering: column-major (y varies fastest), starting bottom-left.
    Element numbering: also column-major.
    """
    n_elem = nelx * nely
    n_node = (nelx + 1) * (nely + 1)
    n_dof = 2 * n_node
    ex_flat, ey_flat = np.divmod(np.arange(n_elem, dtype=np.int64), nely)

    n1 = ex_flat * (nely + 1) + ey_flat
    n2 = n1 + (nely + 1)
    n3 = n2 + 1
    n4 = n1 + 1

    elem_conn = np.column_stack([
        2 * n1, 2 * n1 + 1,
        2 * n2, 2 * n2 + 1,
        2 * n3, 2 * n3 + 1,
        2 * n4, 2 * n4 + 1
    ])
    return RectangularMesh(
        nelx=nelx,
        nely=nely,
        n_elem=n_elem,
        n_node=n_node,
        n_dof=n_dof,
        elem_conn=elem_conn
    )


def get_mbb_boundary_conditions(mesh: RectangularMesh) -> tuple[
    NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """
    Set up boundary conditions for the MBB beam (half-symmetry).

    MBB beam: left edge is symmetric (u_x = 0), bottom-right corner is simply supported (u_y = 0).
    Load: unit downward force at top-left corner.
    """
    nelx, nely = mesh.nelx, mesh.nely
    left_nodes = np.arange(nely + 1)
    fixed_x_dofs = 2 * left_nodes
    bottom_right_node = nelx * (nely + 1)
    fixed_y_dof = 2 * bottom_right_node + 1

    fixed_dofs = np.concatenate([fixed_x_dofs, [fixed_y_dof]])
    top_left_node = nely
    load_dof = 2 * top_left_node + 1

    force_vector = np.zeros(mesh.n_dof, dtype=np.float64)
    force_vector[load_dof] = -1.0

    return fixed_dofs, np.array([load_dof], dtype=np.int64), force_vector


def get_cantilever_boundary_conditions(mesh: RectangularMesh) -> tuple[
    NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """
    Set up boundary conditions for a cantilever beam.

    Cantilever: left edge is fully fixed (u_x = u_y = 0).
    Load: unit downward force at mid-right edge.
    """
    nelx, nely = mesh.nelx, mesh.nely

    left_nodes = np.arange(nely + 1)
    fixed_dofs = np.concatenate([2 * left_nodes, 2 * left_nodes + 1])

    # Load at mid-right edge
    mid_right_node = nelx * (nely + 1) + nely // 2
    load_dof = 2 * mid_right_node + 1

    force_vector = np.zeros(mesh.n_dof, dtype=np.float64)
    force_vector[load_dof] = -1.0

    return fixed_dofs, np.array([load_dof], dtype=np.int64), force_vector


ProblemType = Literal["mbb", "cantilever"]


def setup_problem(nelx: int, nely: int, problem_type: ProblemType = "mbb") -> tuple[
    RectangularMesh, NDArray[np.int64], NDArray[np.float64]]:
    """
    Create mesh and boundary conditions for a standard topology optimization problem.
    """
    mesh = build_rectangular_mesh(nelx, nely)

    if problem_type == "mbb":
        fixed_dofs, _, force_vector = get_mbb_boundary_conditions(mesh)
    elif problem_type == "cantilever":
        fixed_dofs, _, force_vector = get_cantilever_boundary_conditions(mesh)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    return mesh, fixed_dofs, force_vector


def plot_density(
        x: NDArray[np.float64],
        nelx: int,
        nely: int,
        iteration: int | None = None,
        show: bool = True,
        save_path: str | None = None
) -> None:
    """
    Plot the density field as a grayscale image.
    """
    x_2d = x.reshape((nelx, nely)).T

    plt.figure(figsize=(max(6.0, nelx / 10), max(4.0, nely / 10)))
    plt.imshow(1 - x_2d, cmap='gray', origin='lower', vmin=0, vmax=1)
    plt.colorbar(label='Void (white) / Solid (black)')
    plt.axis('equal')
    plt.axis('off')

    title = "Topology Optimization Result"
    if iteration is not None:
        title += f" (Iteration {iteration})"
    plt.title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()
