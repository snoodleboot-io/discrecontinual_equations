"""Physical coupled-array sensor systems built on the continuation library."""

from discrecontinual_equations.systems.coupled_ring import (
    FerroelectricRing,
    FluxgateRing,
    UnidirectionalRing,
)

__all__ = [
    "FerroelectricRing",
    "FluxgateRing",
    "UnidirectionalRing",
]
