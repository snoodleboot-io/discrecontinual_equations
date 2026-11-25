"""Configuration for three-parameter continuation and codim-3 detection.

A codim-2 point continued in three parameters traces a curve. The fields pick
which codim-2 curve (``cusp``, ``generalized_hopf`` or ``bogdanov_takens``), the
third active parameter (the curve coordinate), and the codim-3 test functions
scanned along it (a swallowtail on a cusp curve, a degenerate Bautin on a
generalized-Hopf curve).

References:
- Govaerts, W., Kuznetsov, Yu. A. and Sijnave, B. "Continuation of codimension-2
  equilibrium bifurcations in CONTENT", in *Numerical Methods for Bifurcation
  Problems and Large-Scale Dynamical Systems*, IMA Vol. Math. Appl., Springer
  (2000) 163-184, sections 6 and 7: continuing cusp and generalized-Hopf curves in
  three parameters.
"""

from typing import Literal

from pydantic import Field

from discrecontinual_equations.continuation.codim2_config import Codim2Config


class Codim3Config(Codim2Config):
    """Codim-2 configuration plus the third active parameter and detectors.

    ``continuation_parameter_index`` and ``second_parameter_index`` are solved for
    along the curve; ``third_parameter_index`` is its coordinate, started at
    ``initial_parameter``.
    """

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
