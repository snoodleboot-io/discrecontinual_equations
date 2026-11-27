"""Configuration for two-parameter continuation of a bifurcation curve.

The fields select the curve (fold or Hopf), the two active parameters, the codim-2
test functions to scan along it, and corrector settings relaxed for the augmented
system. The detectors are those MatCont runs on the same curves: Bogdanov-Takens,
zero-Hopf and cusp along a fold curve; Bogdanov-Takens, zero-Hopf, double Hopf and
generalized Hopf along a Hopf curve.

References:
- Govaerts, W., Kuznetsov, Yu. A., Meijer, H. G. E. et al. "MATCONT: Continuation
  toolbox for ODEs in Matlab", manual for MatCont 7.1 (August 2019), sections 8.1.2 and
  8.2.2: the codim-2 points detected along fold and Hopf curves.
"""

import warnings
from pathlib import Path
from typing import Any, Literal

import pydantic
from pydantic import Field, model_validator

from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)

_PYDANTIC_DIRECTORY = str(Path(pydantic.__file__).parent)


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
    derivative_provider: str = Field(
        default="automatic_differentiation",
        description="Registry key of the state-derivative provider",
    )
    derivative_epsilon: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Relative step of every order of the finite_difference_derivatives "
            "provider; None (the default) steps an order-m difference by "
            "eps^(1/(m+2)), which balances its truncation against its rounding "
            "(DEQ-167). Refused with automatic_differentiation, which takes no "
            "step"
        ),
    )

    finite_difference_epsilon: float = Field(
        default=1e-5,
        description=(
            "Relative step of the central differences of the curve's augmented "
            "system: the finite-difference Jacobian's columns, and the columns "
            "the cusp and generalized-Hopf residuals difference themselves. An "
            "unknown x steps this times max(|x|, typical_x), the typical sizes "
            "set by typical and parameter_typical (one by default)"
        ),
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

    @model_validator(mode="before")
    @classmethod
    def _retire_curve_epsilon(cls, data: Any) -> Any:
        """Warn about, and drop, ``curve_epsilon`` (DEQ-166).

        It was never read: since the first commit the curves' columns have
        stepped ``finite_difference_epsilon``, and the derivative provider has
        its own ``derivative_epsilon``.
        """
        if isinstance(data, dict) and "curve_epsilon" in data:
            warnings.warn(
                "Codim2Config.curve_epsilon is deprecated and ignored: no curve "
                "ever read it. The augmented system's columns step "
                "finite_difference_epsilon, and the finite-difference derivative "
                "provider's step is derivative_epsilon (DEQ-166).",
                DeprecationWarning,
                # Attribute it to the caller, past pydantic's own frames: Python
                # shows a DeprecationWarning by default only from __main__.
                skip_file_prefixes=(_PYDANTIC_DIRECTORY, __file__),
            )
            data = {key: value for key, value in data.items() if key != "curve_epsilon"}
        return data

    @model_validator(mode="after")
    def _derivative_epsilon_needs_differences(self) -> "Codim2Config":
        """Refuse a derivative step for a provider that takes none."""
        if (
            self.derivative_epsilon is not None
            and self.derivative_provider == "automatic_differentiation"
        ):
            message = (
                "derivative_epsilon sets the step of the finite_difference_derivatives "
                "provider; automatic_differentiation takes no step"
            )
            raise ValueError(message)
        return self
