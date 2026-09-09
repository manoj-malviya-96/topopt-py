from typing import Literal

from pydantic import BaseModel, Field


class OptimizeRequest(BaseModel):
    nelx: int = Field(default=200, gt=0, le=500)
    nely: int = Field(default=100, gt=0, le=500)
    problem: Literal["mbb", "cantilever"] = "mbb"
    volfrac: float = Field(default=0.5, gt=0, lt=1)
    penal: float = Field(default=3.0, gt=0)
    rmin: float = Field(default=1.5, ge=0)
    maxiter: int = Field(default=100, gt=0, le=1000)
    tol: float = Field(default=0.01, gt=0)
    solver: Literal["direct", "iterative"] = "direct"
    use_filter: bool = True


class OptimizeResponse(BaseModel):
    compliance: float
    volume_fraction: float
    image_base64: str
