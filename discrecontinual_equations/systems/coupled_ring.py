"""Coupled nonlinear sensor arrays as unidirectional bistable rings.

These are the physical systems behind the Aven-Palacios-Bulsara sensors: N
identical bistable cells connected in a unidirectional ring, cell ``i`` driven by
its downstream neighbour ``i + 1``. The devices differ only in the single-cell
nonlinearity - a soft ``tanh`` core for the coupled-core fluxgate magnetometer, a
quartic (Landau) double well for the ferroelectric electric-field sensor - so the
ring topology lives in :class:`UnidirectionalRing` and each device supplies its own
cell dynamics through :meth:`UnidirectionalRing._cell`.

Both devices take the same three parameters, in order: the ring coupling, the cell
gain, and the target-field bias. At zero bias each device's symmetric state breaks
symmetry as the coupling is varied; the ferroelectric ring additionally onsets
oscillations through a Hopf bifurcation, the mechanism its sensing exploits.
"""

import math
from abc import ABC, abstractmethod

from discrecontinual_equations.function.deterministic import DeterministicFunction


class UnidirectionalRing(DeterministicFunction, ABC):
    """N identical cells coupled forward in a ring; cell ``i`` sees cell ``i + 1``.

    The ring size is the number of state variables, so the same class serves any
    odd length: a three-cell prototype or a seven-cell array are the same dynamics.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        """Assemble the ring derivative from the per-cell dynamics."""
        count = len(point)
        return [self._cell(point[i], point[(i + 1) % count]) for i in range(count)]

    @abstractmethod
    def _cell(self, state: float, neighbour: float) -> float:
        """Time derivative of one cell given its state and downstream neighbour."""
        raise NotImplementedError


class FluxgateRing(UnidirectionalRing):
    """Coupled-core fluxgate magnetometer: soft (tanh) bistable cells.

    Parameters in order: coupling ``lambda``, gain ``c``, target-field bias
    ``eps``. At zero bias the trivial state breaks symmetry at ``lambda = 1 - c``.
    The cell uses a transcendental nonlinearity, so it is intended for numeric
    continuation rather than the jet-based automatic-differentiation backend.
    """

    def _cell(self, state: float, neighbour: float) -> float:
        coupling = self.parameters[0].value
        gain = self.parameters[1].value
        bias = self.parameters[2].value
        return -state + math.tanh(gain * state + coupling * neighbour + bias)


class FerroelectricRing(UnidirectionalRing):
    """Ferroelectric electric-field sensor: quartic (Landau) double-well cells.

    Parameters in order: coupling ``lambda``, Landau gain ``a``, target electric
    field ``eps``. At zero field the trivial state breaks symmetry at
    ``lambda = a`` and oscillations onset through a Hopf bifurcation at
    ``lambda = -2a``; a target field sweeps the double well, whose fold thresholds
    the coupling tunes. The cell is polynomial, so it is fully compatible with the
    automatic-differentiation backend.
    """

    def _cell(self, state: float, neighbour: float) -> float:
        coupling = self.parameters[0].value
        gain = self.parameters[1].value
        field = self.parameters[2].value
        return gain * state - state**3 - coupling * neighbour + field
