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
symmetry as the coupling is varied; the ferroelectric ring additionally carries a
rotating wave, the oscillation its sensing exploits. How that wave is born depends
on the sign of the gain, and :class:`FerroelectricRing` says how.
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
    field ``eps``; the cell is ``x_i' = a x_i - x_i^3 - lambda x_{i+1} + eps``.
    Rescaling ``x`` by ``sqrt|a|`` and time by ``1/|a|`` leaves only ``lambda / a``,
    so every threshold below is proportional to ``a``.

    At zero field the trivial state's Fourier modes have eigenvalues ``a - lambda``
    (``k = 0``) and ``a + lambda/2 +- i lambda sqrt(3)/2`` (``k = 1, 2``): the
    uniform mode crosses at ``lambda = a`` (symmetry breaking) and the wave pair at
    ``lambda = -2a``. Which of these is an *onset* depends on the sign of ``a``.

    * ``a < 0`` (monostable cells): the trivial state is stable for
      ``a < lambda < -2a`` and the crossing at ``-2a`` is a genuine Hopf onset of
      the rotating wave, born with the linear period ``4 pi / (lambda sqrt 3)``
      (measured 11.16 at ``a = -0.3``, ``lambda = 0.65``, against 11.16).
    * ``a > 0`` (double-well cells, the sensor's regime): the trivial state is
      never stable - its ``k = 0`` eigenvalue is positive for all ``lambda < a`` -
      so the Hopf at ``-2a`` creates a cycle nobody sees, and for
      ``lambda < -2a`` the attractors are the symmetric states. Those states,
      ``+-s(1, 1, 1)`` with ``s^2 = a - lambda``, are the attractors for weak
      coupling and lose stability by their own Hopf at ``lambda = 4a/7`` (found by
      continuation to four figures for ``a`` in ``[0.4, 1.4]``). The rotating wave
      the sensor reads appears at ``lambda ~ 0.34 a`` with an *infinite* period -
      ``1/T^2`` is linear in ``lambda``; ``T = 75`` at ``0.36 a``, ``44`` at
      ``0.40 a``, ``7.0`` at ``1.2 a`` - and coexists with the symmetric states up
      to ``4a/7`` (DEQ-14).

    A target field sweeps the double well, whose fold thresholds the coupling
    tunes. The cell is polynomial, so it is fully compatible with the
    automatic-differentiation backend.
    """

    def _cell(self, state: float, neighbour: float) -> float:
        coupling = self.parameters[0].value
        gain = self.parameters[1].value
        field = self.parameters[2].value
        return gain * state - state**3 - coupling * neighbour + field
