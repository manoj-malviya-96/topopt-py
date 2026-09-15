import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Literal


@dataclass
class RectangularMesh:
    """Structured 2D mesh data container."""
    nelx: int
    nely: int
    n_elem: int
    n_node: int
    n_dof: int
    elem_conn: NDArray[np.int64]


def build_rectangular_mesh(nelx: int, nely: int) -> RectangularMesh:
    """Build Q4 mesh with column-major numbering (y varies fastest)."""
    n_elem = nelx * nely
    n_node = (nelx + 1) * (nely + 1)
    n_dof = 2 * n_node

    elem_x, elem_y = np.divmod(np.arange(n_elem, dtype=np.int64), nely)

    node_bl = elem_x * (nely + 1) + elem_y
    node_br = node_bl + (nely + 1)
    node_tr = node_br + 1
    node_tl = node_bl + 1

    elem_conn = np.column_stack([
        2 * node_bl, 2 * node_bl + 1,
        2 * node_br, 2 * node_br + 1,
        2 * node_tr, 2 * node_tr + 1,
        2 * node_tl, 2 * node_tl + 1
    ])

    return RectangularMesh(nelx, nely, n_elem, n_node, n_dof, elem_conn)


def get_mbb_boundary_conditions(mesh: RectangularMesh) -> tuple[
    NDArray[np.int64], NDArray[np.float64]]:
    """MBB beam: left edge symmetric (ux=0), bottom-right pinned (uy=0), load at top-left."""
    left_nodes = np.arange(mesh.nely + 1)
    fixed_ux = 2 * left_nodes

    bottom_right_node = mesh.nelx * (mesh.nely + 1)
    fixed_uy = 2 * bottom_right_node + 1

    fixed_dofs = np.concatenate([fixed_ux, [fixed_uy]])

    top_left_node = mesh.nely
    load_dof = 2 * top_left_node + 1

    force = np.zeros(mesh.n_dof, dtype=np.float64)
    force[load_dof] = -1.0

    return fixed_dofs, force


def get_cantilever_boundary_conditions(mesh: RectangularMesh) -> tuple[
    NDArray[np.int64], NDArray[np.float64]]:
    """Cantilever: left edge fully fixed, load at mid-right."""
    left_nodes = np.arange(mesh.nely + 1)
    fixed_dofs = np.concatenate([2 * left_nodes, 2 * left_nodes + 1])

    mid_right_node = mesh.nelx * (mesh.nely + 1) + mesh.nely // 2
    load_dof = 2 * mid_right_node + 1

    force = np.zeros(mesh.n_dof, dtype=np.float64)
    force[load_dof] = -1.0

    return fixed_dofs, force


ProblemType = Literal["mbb", "cantilever"]


def setup_problem(nelx: int, nely: int, problem_type: ProblemType = "mbb") -> tuple[
    RectangularMesh, NDArray[np.int64], NDArray[np.float64]]:
    """Create mesh and boundary conditions for standard problems."""
    mesh = build_rectangular_mesh(nelx, nely)

    if problem_type == "mbb":
        fixed_dofs, force = get_mbb_boundary_conditions(mesh)
    elif problem_type == "cantilever":
        fixed_dofs, force = get_cantilever_boundary_conditions(mesh)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    return mesh, fixed_dofs, force


def plot_density(density: NDArray[np.float64], nelx: int, nely: int,
                 show: bool = True, save_path: str | None = None) -> None:
    """Plot density field as grayscale image."""
    density_2d = density.reshape((nelx, nely)).T

    plt.figure(figsize=(max(6.0, nelx / 10), max(4.0, nely / 10)))
    plt.imshow(1 - density_2d, cmap='gray', origin='lower', vmin=0, vmax=1)
    plt.colorbar(label='Void (white) / Solid (black)')
    plt.axis('equal')
    plt.axis('off')
    plt.title("Topology Optimization Result")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def create_optimization_gif(
        density_history: list[NDArray[np.float64]],
        nelx: int,
        nely: int,
        output_path: str,
        frame_duration_ms: int = 100,
        loop_count: int = 0,
) -> None:
    from PIL import Image
    import io

    frames: list[Image.Image] = []
    figure_width = max(6.0, nelx / 10)
    figure_height = max(4.0, nely / 10)

    for iteration_index, density in enumerate(density_history):
        density_2d = density.reshape((nelx, nely)).T

        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
        ax.imshow(1 - density_2d, cmap='gray', origin='lower', vmin=0, vmax=1)
        ax.set_title(f"Iteration {iteration_index + 1}")
        ax.axis('equal')
        ax.axis('off')
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        frames.append(Image.open(buffer).copy())
        buffer.close()
        plt.close(fig)

    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=loop_count,
        )


def create_optimization_webm(
        density_history: list[NDArray[np.float64]],
        nelx: int,
        nely: int,
        output_path: str,
        fps: int = 10,
        dpi: int = 150,
        transparent: bool = False,
) -> None:
    from matplotlib.animation import FFMpegWriter

    figure_width = max(6.0, nelx / 10)
    figure_height = max(4.0, nely / 10)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
    ax.axis('equal')
    ax.axis('off')
    fig.tight_layout()

    first_density = 1 - density_history[0].reshape((nelx, nely)).T
    image = ax.imshow(first_density, cmap='gray', origin='lower', vmin=0, vmax=1)
    title = ax.set_title("Iteration 1")

    extra_args = ['-pix_fmt', 'yuva420p'] if transparent else None
    writer = FFMpegWriter(fps=fps, extra_args=extra_args)

    with writer.saving(fig, output_path, dpi=dpi):
        for iteration_index, density in enumerate(density_history):
            density_2d = density.reshape((nelx, nely)).T
            image.set_data(1 - density_2d)
            title.set_text(f"Iteration {iteration_index + 1}")
            writer.grab_frame(transparent=transparent)

    plt.close(fig)
