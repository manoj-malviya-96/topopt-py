import base64
import io
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException

from api.schemas import OptimizeRequest, OptimizeResponse
from core.mesh import setup_problem
from core.optimize import optimize_compliance, TopOptConfig
from core.solver import SolverMethod

logger = logging.getLogger(__name__)

app = FastAPI(title="TopOpt API", version="0.1.0")


def _render_density_png(density, nelx: int, nely: int) -> bytes:
    density_2d = density.reshape((nelx, nely)).T

    fig, ax = plt.subplots(figsize=(max(6.0, nelx / 10), max(4.0, nely / 10)))
    ax.imshow(1 - density_2d, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.axis("equal")
    ax.axis("off")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(request: OptimizeRequest) -> OptimizeResponse:
    solver_method = SolverMethod.DIRECT if request.solver == "direct" else SolverMethod.ITERATIVE
    config = TopOptConfig(
        nelx=request.nelx,
        nely=request.nely,
        problem_type=request.problem,
        target_vol_frac=request.volfrac,
        penalization=request.penal,
        r_min=request.rmin,
        max_iter=request.maxiter,
        tol=request.tol,
        solver_method=solver_method,
        use_filter=request.use_filter,
    )

    try:
        mesh, fixed_dofs, force = setup_problem(request.nelx, request.nely, request.problem)
        density, compliance, _ = optimize_compliance(
            mesh, fixed_dofs, force, config, show_progress=False, collect_history=False
        )
    except Exception as e:
        logger.exception("Optimization failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    png_bytes = _render_density_png(density, request.nelx, request.nely)
    return OptimizeResponse(
        compliance=compliance,
        volume_fraction=float(density.mean()),
        image_base64=base64.b64encode(png_bytes).decode("ascii"),
    )
