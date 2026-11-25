"""Deprecated: ``DormandPrince89Config`` is :class:`PrinceDormand87Config`.

The solver it configures is Prince and Dormand's 8(7) pair, not an 8(9) one
(DEQ-139). The old name still works and warns.
"""

import warnings
from typing import Any

from discrecontinual_equations.solver.deterministic.prince_dormand87.prince_dormand87_config import (
    PrinceDormand87Config,
)


class DormandPrince89Config(PrinceDormand87Config):
    """Deprecated alias of :class:`PrinceDormand87Config`; warns when built."""

    def __init__(self, **data: Any) -> None:
        """Warn, then build a :class:`PrinceDormand87Config` from ``data``."""
        warnings.warn(
            "DormandPrince89Config is deprecated: use PrinceDormand87Config from "
            "discrecontinual_equations.solver.deterministic.prince_dormand87 "
            "(DEQ-139).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)
