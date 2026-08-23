"""A single solution point on a continuation branch."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ContinuationPoint(BaseModel):
    """One computed equilibrium on a branch, with stability information.

    Eigenvalues are stored as ``(real, imaginary)`` pairs so the record stays
    serializable. ``kind`` marks whether the point is a detected bifurcation.
    """

    arclength: float = Field(description="Cumulative arclength along the branch")
    state: list[float] = Field(description="Equilibrium state vector")
    parameter: float = Field(description="Value of the continuation parameter")
    tangent: list[float] = Field(
        description="Unit tangent in (state, parameter) space",
    )
    eigenvalues: list[tuple[float, float]] = Field(
        description="Jacobian eigenvalues as (real, imaginary) pairs",
    )
    unstable_dimension: int = Field(
        description="Number of eigenvalues with positive real part",
    )
    stability: Literal["stable", "unstable", "saddle"] = Field(
        description="Stability classification of the equilibrium",
    )
    measure: float = Field(description="Scalar measure used on a bifurcation diagram")
    kind: str = Field(
        default="regular",
        description="Bifurcation label ('regular' for ordinary points)",
    )
    frequency: float | None = Field(
        default=None,
        description="Imaginary part of the critical pair at a Hopf point",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
