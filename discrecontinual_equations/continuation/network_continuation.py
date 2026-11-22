"""Following a heteroclinic network through a parameter, and finding where it breaks.

A robust connection persists while the parameter keeps its subspace invariant, its
source a saddle in the subspace and its target a sink there (Krupa and Melbourne
1995, (H1)). The first two survive any perturbation that keeps the symmetry that
makes the subspace invariant; the last two are open conditions on eigenvalues, and
one of them failing is how the connection ends. Breaking a *larger* symmetry - the
cyclic shift of the Guckenheimer-Holmes cycle, say, while keeping the reflections -
therefore moves the saddles and changes every eigenvalue ratio, and the network
survives until one of those margins reaches zero.

:class:`NetworkContinuation` follows the network along a list of parameter values,
refining every saddle from where it was, reporting each connection's margins and
each cycle's stability ratio, and optionally re-solving every connection seeded by
the previous one. :meth:`NetworkContinuation.locate_break` marches until a margin
changes sign and bisects on that margin.
"""

import itertools

import numpy as np

from discrecontinual_equations.continuation.heteroclinic_network import (
    ConnectionOrbit,
    HeteroclinicNetwork,
    NetworkConnection,
)

_SIDES = ("source", "target")


class NetworkPoint:
    """The network at one parameter value."""

    __slots__ = ["equilibria", "margins", "orbits", "parameter", "stability_ratios"]

    def __init__(
        self,
        parameter: float,
        equilibria: dict[str, np.ndarray],
        margins: dict[tuple[str, str], tuple[float, float]],
        stability_ratios: dict[tuple[str, ...], float],
        orbits: dict[tuple[str, str], ConnectionOrbit],
    ) -> None:
        self.parameter = parameter
        self.equilibria = equilibria
        self.margins = margins
        self.stability_ratios = stability_ratios
        self.orbits = orbits


class ConnectionBreak:
    """Where a connection stops existing, and which end gave way.

    ``side`` is ``"source"`` when the source stopped expanding into the subspace
    and ``"target"`` when the target stopped attracting within it.
    """

    __slots__ = ["connection", "parameter", "side"]

    def __init__(
        self,
        parameter: float,
        connection: NetworkConnection,
        side: str,
    ) -> None:
        self.parameter = parameter
        self.connection = connection
        self.side = side


class NetworkContinuation:
    """Continue every saddle, connection and cycle of a network in one parameter.

    The parameter is ``network.function.parameters[parameter_index]``. Both methods
    restore its value, and the saddles, when they finish.
    """

    __slots__ = ["_index", "_network"]

    def __init__(self, network: HeteroclinicNetwork, parameter_index: int) -> None:
        self._network = network
        self._index = parameter_index

    def trace(
        self,
        values: list[float],
        half_length: float | None = None,
        intervals: int | None = None,
    ) -> list[NetworkPoint]:
        """The network at each value; connections are solved when a mesh is given.

        Each connection is seeded by its own orbit at the previous value, so the
        solve is a correction rather than a fresh search.
        """
        network = self._network
        original = self._value()
        previous: dict[tuple[str, str], ConnectionOrbit] = {}
        points: list[NetworkPoint] = []
        try:
            for value in values:
                self._set(value)
                orbits: dict[tuple[str, str], ConnectionOrbit] = {}
                if half_length is not None and intervals is not None:
                    for connection in network.connections:
                        mesh = network.mesh(connection, half_length, intervals)
                        prior = previous.get(connection.key)
                        seed = None if prior is None else prior.solution.states
                        orbits[connection.key] = network.solve(connection, mesh, seed)
                    previous = orbits
                points.append(self._point(value, orbits))
        finally:
            self._set(original)
        return points

    def locate_break(
        self,
        start: float,
        stop: float,
        steps: int,
        tolerance: float,
    ) -> ConnectionBreak:
        """The first parameter between ``start`` and ``stop`` where a connection ends.

        Marches ``steps`` equal steps; at the first step where any margin is no
        longer positive, bisects every margin that changed sign there and returns
        the crossing nearest ``start``. Every margin must be positive at ``start``.
        """
        original = self._value()
        try:
            self._set(start)
            if self._weakest()[0] <= 0.0:
                message = f"a connection is already broken at the start value {start}"
                raise ValueError(message)
            values = np.linspace(start, stop, steps + 1)
            for low, high in itertools.pairwise(values):
                self._set(float(high))
                if self._weakest()[0] > 0.0:
                    continue
                return self._bisect_all(float(low), float(high), tolerance)
            message = f"every connection persists from {start} to {stop}"
            raise ValueError(message)
        finally:
            self._set(original)

    def _bisect_all(self, low: float, high: float, tolerance: float) -> ConnectionBreak:
        crossings = [
            (connection, side)
            for connection in self._network.connections
            for side in range(2)
            if self._network.connection_margins(connection)[side] <= 0.0
        ]
        located = [
            ConnectionBreak(
                self._bisect(connection, side, (low, high), tolerance),
                connection,
                _SIDES[side],
            )
            for connection, side in crossings
        ]
        return min(located, key=lambda found: abs(found.parameter - low))

    def _bisect(
        self,
        connection: NetworkConnection,
        side: int,
        bracket: tuple[float, float],
        tolerance: float,
    ) -> float:
        """Bisect one margin, which is positive at ``bracket[0]`` and not at ``[1]``."""
        low, high = bracket
        while abs(high - low) > tolerance:
            middle = 0.5 * (low + high)
            self._set(middle)
            if self._network.connection_margins(connection)[side] > 0.0:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    def _weakest(self) -> tuple[float, NetworkConnection]:
        margins = [
            (self._network.persistence_margin(connection), connection)
            for connection in self._network.connections
        ]
        return min(margins, key=lambda pair: pair[0])

    def _point(
        self,
        value: float,
        orbits: dict[tuple[str, str], ConnectionOrbit],
    ) -> NetworkPoint:
        network = self._network
        return NetworkPoint(
            value,
            {saddle.name: saddle.equilibrium.copy() for saddle in network.saddles},
            {c.key: network.connection_margins(c) for c in network.connections},
            {
                cycle: network.spectrum(cycle).stability_ratio
                for cycle in network.cycles()
            },
            orbits,
        )

    def _value(self) -> float:
        value = self._network.function.parameters[self._index].value
        if value is None:
            message = f"parameter {self._index} has no value to continue from"
            raise ValueError(message)
        return float(value)

    def _set(self, value: float) -> None:
        """Move the parameter and carry every saddle to the new value."""
        self._network.function.parameters[self._index].value = value
        self._network.refresh()
