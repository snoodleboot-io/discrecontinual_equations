"""Configuration for two-parameter continuation of a bifurcation curve."""

from typing import Literal

from pydantic import Field

from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)


class Codim2Config(ContinuationConfig):
    """Adds the second active parameter and codim-2 detection to the base config.

    ``continuation_parameter_index`` is the first active parameter; the curve is
    continued in ``second_parameter_index`` (which plays the role of the
    pseudo-arclength parameter). Defaults are tuned for the augmented curve system,
    whose Jacobian involves second derivatives and so tolerates less precision.
    """

    second_parameter_index: int = Field(
        default=1,
        description="Index of the second active parameter (the curve coordinate)",
    )
    curve: Literal["hopf", "fold"] = Field(
        default="hopf",
        description="Which codim-1 curve to continue",
    )
    codim2_detectors: list[str] = Field(
        default_factory=lambda: ["bogdanov_takens", "zero_hopf", "hopf_hopf"],
        description="Registry keys of the codim-2 test functions to scan",
    )
    curve_epsilon: float = Field(
        default=1e-5,
        description="Directional finite-difference step for the inner Jacobian",
    )
    derivative_provider: str = Field(
        default="automatic_differentiation",
        description="Registry key of the state-derivative provider",
    )

    finite_difference_epsilon: float = Field(
        default=1e-5,
        description="Outer finite-difference step for the augmented Jacobian",
    )
    newton_tolerance: float = Field(
        default=1e-8,
        description="Corrector tolerance (relaxed for the augmented system)",
    )
    maximum_newton_iterations: int = Field(
        default=25,
        description="Corrector iteration cap for the augmented system",
    )
    initial_step: float = Field(
        default=0.02,
        description="Initial arclength step along the curve",
    )
    maximum_step: float = Field(
        default=0.08,
        description="Largest arclength step along the curve",
    )
