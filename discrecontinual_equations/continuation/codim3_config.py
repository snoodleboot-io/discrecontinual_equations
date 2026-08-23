"""Configuration for three-parameter continuation and codim-3 detection."""

from typing import Literal

from pydantic import Field

from discrecontinual_equations.continuation.codim2_config import Codim2Config


class Codim3Config(Codim2Config):
    """Codim-2 configuration plus the third active parameter and detectors."""

    third_parameter_index: int = Field(
        default=2,
        description="Index of the swept third active parameter",
    )
    codim3_curve: Literal["cusp", "generalized_hopf", "bogdanov_takens"] = Field(
        default="cusp",
        description="Which codim-2 curve to continue in three parameters",
    )
    codim3_detectors: list[str] = Field(
        default_factory=lambda: ["swallowtail"],
        description="Registry keys of the codim-3 test functions to scan",
    )
