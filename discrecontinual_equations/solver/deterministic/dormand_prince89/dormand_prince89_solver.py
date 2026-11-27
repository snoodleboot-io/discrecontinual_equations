"""Deprecated: ``DormandPrince89Solver`` is :class:`PrinceDormand87Solver`.

The pair was named "8(9)", but its embedded row is seventh order: it is Prince
and Dormand's (1981) 8(7) pair (DEQ-139). The old name still works and warns.
"""

import warnings

import numpy as np

from discrecontinual_equations.solver.deterministic.prince_dormand87.prince_dormand87_config import (
    PrinceDormand87Config,
)
from discrecontinual_equations.solver.deterministic.prince_dormand87.prince_dormand87_solver import (
    PrinceDormand87Solver,
)


class DormandPrince89Solver(PrinceDormand87Solver):
    """Deprecated alias of :class:`PrinceDormand87Solver`; warns when built.

    The old public attribute ``b9`` held the embedded weights of the error
    estimate, which are seventh order, not ninth. It is kept as a deprecated
    read-only property returning :attr:`PrinceDormand87Solver.b7` (DEQ-157).

    References:
        Prince and Dormand (1981); see ``docs/references.md``
        (PrinceDormand1981).
    """

    def __init__(self, solver_config: PrinceDormand87Config):
        """Warn, then build a :class:`PrinceDormand87Solver`.

        Args:
            solver_config: As for :class:`PrinceDormand87Solver`.
        """
        warnings.warn(
            "DormandPrince89Solver is deprecated: the pair is Prince and "
            "Dormand's 8(7), not 8(9). Use PrinceDormand87Solver from "
            "discrecontinual_equations.solver.deterministic.prince_dormand87 "
            "(DEQ-139).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(solver_config=solver_config)

    @property
    def b9(self) -> np.ndarray:
        """Deprecated: the seventh-order embedded weights, now ``b7``."""
        warnings.warn(
            "DormandPrince89Solver.b9 is deprecated: those weights are seventh "
            "order. Use PrinceDormand87Solver.b7 (DEQ-139, DEQ-157).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.b7
