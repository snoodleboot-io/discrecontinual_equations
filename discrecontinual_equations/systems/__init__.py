"""Physical coupled-array sensor systems built on the continuation library."""

from discrecontinual_equations.systems.coupled_ring import (
    FerroelectricRing,
    FluxgateRing,
    UnidirectionalRing,
)
from discrecontinual_equations.systems.reflection_cubic import (
    ReflectionCubic,
    guckenheimer_holmes,
    reflection_cubic,
)

__all__ = [
    "FerroelectricRing",
    "FluxgateRing",
    "ReflectionCubic",
    "UnidirectionalRing",
    "guckenheimer_holmes",
    "reflection_cubic",
]
