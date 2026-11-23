"""The Guckenheimer-Holmes heteroclinic cycle and the growth of its passage times.

Split out of webplot_examples, which assembles the atlas from these. Both pages are
built from one :class:`HeteroclinicNetwork`: the connections are solved by the
projection boundary-value problem, and the residence times are measured on a
trajectory integrated in logarithmic coordinates and set against the ratio the
saddle eigenvalues predict.
"""

import math

import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.heteroclinic_network import (
    HeteroclinicNetwork,
    coordinate_plane,
)
from discrecontinual_equations.continuation.residence import Itinerary
from discrecontinual_equations.systems.reflection_cubic import guckenheimer_holmes
from discrecontinual_equations.webplot.scene import Scene
from discrecontinual_equations.webplot.scene_builder import curve_scene

# Krupa (1997) eq. 2.1 normalisation: e = 0.4, c = 0.5 at every saddle, so the
# cycle is attracting (2 a1 = -2 > a2 + a3 = -2.1) and each passage lasts
# c / e = 1.25 times the one before.
_GROWTH, _SELF, _FORWARD, _BACKWARD = 1.0, -1.0, -1.5, -0.6
_NAMES = ("xi1", "xi2", "xi3")
_HALF_LENGTH, _INTERVALS = 20.0, 400
_HORIZON, _SAMPLES = 1200.0, 240001
_RADIUS = 0.1
_START = (0.9, 0.3, 0.1)
_LOG_FLOOR = -300.0
_PLANE_TOLERANCE = 1.0e-10
# The trajectory is sampled finely for the residence times; one point in ten is
# plenty to draw it.
_THIN = 10
# Two revolutions are enough to see the trajectory close in on the cycle.
_DRAWN = 120.0
# Projection onto the plane normal to (1, 1, 1), so the three axes - and the
# three saddles - sit 120 degrees apart and every connection is seen face-on.
_ACROSS = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
_UP = np.array([-1.0, -1.0, 2.0]) / math.sqrt(6.0)


_Series = tuple[str, str, list[tuple[float, float]], str | None]


def _network() -> HeteroclinicNetwork:
    field = guckenheimer_holmes(_GROWTH, _SELF, _FORWARD, _BACKWARD)
    network = HeteroclinicNetwork(field)
    radius = math.sqrt(-_GROWTH / _SELF)
    for index, name in enumerate(_NAMES):
        network.add_saddle(name, radius * np.eye(3)[index])
    for index in range(3):
        following = (index + 1) % 3
        network.add_connection(
            _NAMES[index],
            _NAMES[following],
            coordinate_plane(3, (index, following)),
        )
    return network


def _project(states: np.ndarray) -> list[tuple[float, float]]:
    return list(zip(states @ _ACROSS, states @ _UP, strict=True))


def _trajectory(network: HeteroclinicNetwork) -> tuple[np.ndarray, np.ndarray]:
    """A trajectory near the cycle, integrated in ``y = ln x`` so nothing underflows."""
    field = network.function

    def rhs(_time: float, logs: np.ndarray) -> np.ndarray:
        state = np.exp(np.maximum(logs, _LOG_FLOOR))
        return np.array(field.eval(list(state))) / state

    solution = solve_ivp(
        rhs,
        (0.0, _HORIZON),
        np.log(_START),
        method="DOP853",
        rtol=1.0e-11,
        atol=1.0e-12,
        dense_output=True,
    )
    times = np.linspace(0.0, _HORIZON, _SAMPLES)
    return times, np.exp(solution.sol(times).T)


def _heteroclinic_network() -> Scene:
    network = _network()
    spectrum = network.spectrum(_NAMES)
    lines: list[_Series] = []
    defects = []
    for connection in network.connections:
        mesh = network.mesh(connection, _HALF_LENGTH, _INTERVALS)
        orbit = network.solve(connection, mesh)
        defects.append(orbit.full_space.subspace_defect)
        lines.append(
            (
                "connection",
                f"{connection.source} -> {connection.target}",
                _project(orbit.solution.states),
                None,
            ),
        )
    in_plane = "within 1e-10 of" if max(defects) < _PLANE_TOLERANCE else "off"
    times, states = _trajectory(network)
    early = times <= _DRAWN
    lines.append(
        ("trajectory", "nearby trajectory", _project(states[early][::_THIN]), None),
    )
    markers: list[_Series] = [
        (
            "saddle",
            "saddles",
            _project(np.array([saddle.equilibrium for saddle in network.saddles])),
            None,
        ),
    ]
    return curve_scene(
        lines,
        markers,
        ("(x1 - x2) / sqrt 2", "(2 x3 - x1 - x2) / sqrt 6"),
        "Guckenheimer-Holmes heteroclinic cycle",
        "Three saddle-to-saddle connections, each solved by the projection "
        "boundary-value problem in its invariant coordinate plane and checked by "
        f"the full-space solve, found {in_plane} that plane, viewed down the "
        "diagonal. Contracting over "
        "expanding "
        f"rate {spectrum.contracting[0]:.3f} / {spectrum.expanding[0]:.3f} at every "
        f"saddle gives rho = {spectrum.stability_ratio:.4f} > 1: the cycle "
        "attracts, as the trajectory spiralling onto it shows.",
    )


def _residence_time_growth() -> Scene:
    network = _network()
    predicted = float(network.spectrum(_NAMES).residence_ratios[0])
    times, states = _trajectory(network)
    itinerary = Itinerary.measure(
        times,
        states,
        [saddle.equilibrium for saddle in network.saddles],
        _RADIUS,
    )
    steps = np.diff(itinerary.durations)
    measured = [(float(k + 1), math.log(step)) for k, step in enumerate(steps)]
    anchor = measured[0][1]
    prediction = [
        (float(k + 1), anchor + k * math.log(predicted)) for k in range(len(steps))
    ]
    final = float(itinerary.growth_ratios()[-1])
    return curve_scene(
        [("prediction", f"slope ln(c/e) = ln {predicted:.4f}", prediction, None)],
        [("measured", "integrated", measured, None)],
        ("passage k", "ln(T(k+1) - T(k))"),
        "Residence-time growth near the Guckenheimer-Holmes cycle",
        "Time spent within 0.1 of each saddle on successive passages of a trajectory "
        "integrated by scipy DOP853 in log coordinates. Each passage "
        "lasts c/e times the last plus a constant, so successive differences tend "
        f"to growth by c/e: measured {final:.6f} on the last passages against "
        f"{predicted:.6f} from the saddle eigenvalues (Gaunersdorfer 1992, Lemma 1).",
    )


def _network_entries() -> list[tuple[str, Scene]]:
    return [
        ("heteroclinic_network.html", _heteroclinic_network()),
        ("residence_time_growth.html", _residence_time_growth()),
    ]
