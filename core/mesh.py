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
    nelx: int  # Number of elements in x direction
    nely: int  # Number of elements in y direction
    n_elem: int  # Total number of elements
    n_node: int  # Total number of nodes
    n_dof: int  # Total number of DOFs
    elem_conn: NDArray[np.int64]  # Element connectivity (n_elem, 8) for DOF indices


def build_rectangular_mesh(nelx: int, nely: int) -> RectangularMesh:
    """
    Build a structured rectangular mesh of Q4 elements (bilinear quads).
    Each element has 4 nodes and 8 DOFs (2 per node).

    Node numbering: column-major (y varies fastest), starting bottom-left.
    Element numbering: also column-major.

    Returns:
        RectangularMesh with element connectivity in terms of global DOF indices.
    """
    n_elem = nelx * nely
    n_node = (nelx + 1) * (nely + 1)
    n_dof = 2 * n_node

    # Build element connectivity: for each element, list its 8 DOFs
    # Nodes of element (ex, ey):
    #   n1 = ex*(nely+1) + ey        (bottom-left)
    #   n2 = (ex+1)*(nely+1) + ey    (bottom-right)
    #   n3 = (ex+1)*(nely+1) + ey+1  (top-right)
    #   n4 = ex*(nely+1) + ey+1      (top-left)
    # DOFs for node n: [2*n, 2*n+1]

    elem_conn = np.zeros((n_elem, 8), dtype=np.int64)

    for ex in range(nelx):
        for ey in range(nely):
            e = ex * nely + ey  # element index
            n1 = ex * (nely + 1) + ey
            n2 = (ex + 1) * (nely + 1) + ey
            n3 = (ex + 1) * (nely + 1) + ey + 1
            n4 = ex * (nely + 1) + ey + 1

            # DOFs: [n1_x, n1_y, n2_x, n2_y, n3_x, n3_y, n4_x, n4_y]
            elem_conn[e, :] = [
                2 * n1, 2 * n1 + 1,
                2 * n2, 2 * n2 + 1,
                2 * n3, 2 * n3 + 1,
                2 * n4, 2 * n4 + 1
            ]

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

    Returns:
        fixed_dofs: indices of DOFs with Dirichlet BCs (displacement = 0)
        load_dof: index of loaded DOF
        force_vector: full force vector
    """
    nelx, nely = mesh.nelx, mesh.nely

    # Fixed DOFs:
    # - Left edge: all nodes on x=0, fix u_x (horizontal displacement)
    #   Nodes: ey for ey in range(nely+1), DOF = 2*ey
    left_nodes = np.arange(nely + 1)
    fixed_x_dofs = 2 * left_nodes  # u_x DOFs

    # - Bottom-right corner: node at (nelx, 0), fix u_y
    bottom_right_node = nelx * (nely + 1)
    fixed_y_dof = 2 * bottom_right_node + 1

    fixed_dofs = np.concatenate([fixed_x_dofs, [fixed_y_dof]])

    # Force vector: unit downward load at top-left corner
    # Top-left node: (0, nely), node index = nely
    top_left_node = nely
    load_dof = 2 * top_left_node + 1  # y-direction

    force_vector = np.zeros(mesh.n_dof, dtype=np.float64)
    force_vector[load_dof] = -1.0  # downward

    return fixed_dofs, np.array([load_dof], dtype=np.int64), force_vector


def get_cantilever_boundary_conditions(mesh: RectangularMesh) -> tuple[
    NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """
    Set up boundary conditions for a cantilever beam.

    Cantilever: left edge is fully fixed (u_x = u_y = 0).
    Load: unit downward force at mid-right edge.

    Returns:
        fixed_dofs, load_dofs, force_vector
    """
    nelx, nely = mesh.nelx, mesh.nely

    # Fixed DOFs: entire left edge (both u_x and u_y)
    left_nodes = np.arange(nely + 1)
    fixed_dofs = np.concatenate([2 * left_nodes, 2 * left_nodes + 1])

    # Load at mid-right edge
    mid_right_node = nelx * (nely + 1) + nely // 2
    load_dof = 2 * mid_right_node + 1  # y-direction

    force_vector = np.zeros(mesh.n_dof, dtype=np.float64)
    force_vector[load_dof] = -1.0

    return fixed_dofs, np.array([load_dof], dtype=np.int64), force_vector


ProblemType = Literal["mbb", "cantilever"]


def setup_problem(nelx: int, nely: int, problem_type: ProblemType = "mbb") -> tuple[
    RectangularMesh, NDArray[np.int64], NDArray[np.float64]]:
    """
    Create mesh and boundary conditions for a standard topology optimization problem.

    Args:
        nelx: Number of elements in x direction
        nely: Number of elements in y direction
        problem_type: "mbb" or "cantilever"

    Returns:
        mesh, fixed_dofs, force_vector
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

    Args:
        x: Density vector (n_elem,)
        nelx: Number of elements in x
        nely: Number of elements in y
        iteration: Optional iteration number for title
        show: Whether to display the plot
        save_path: Optional path to save the figure
    """
    # Reshape to 2D grid (elements are column-major)
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
