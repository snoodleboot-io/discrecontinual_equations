"""Stability indices and cycle competition, checked against simulated trajectories.

Two predictions of :mod:`network_stability` are tested by integrating the flow,
in logarithmic coordinates ``y = ln x`` (the reflection cubic becomes
``y' = 1 + A e^{2y}``), with scipy or a numpy RK4 - never the library's
integrators:

* **which cycle wins.** Near ``delta = 0`` the losing cycle is barely unstable and
  trajectories follow it for several turns before switching; the prediction is
  that every trajectory eventually switches to the surviving cycle and stays;
* **how much of a neighbourhood a cycle attracts.** The index says the fraction
  ``Sigma_eps`` of an ``eps``-box about a point of a connection that converges to
  the cycle obeys ``1 - Sigma_eps ~ eps^sigma`` (``sigma > 0``). The box is in the
  two coordinates transverse to the connection; uniform in ``x`` there means
  ``ln x = ln eps - E`` with ``E`` unit-exponential, and the fraction is
  integrated over a grid in ``E`` rather than sampled, which reaches the small
  tails an exponent needs.
"""

import itertools
import math
from unittest import TestCase

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.heteroclinic_network import (
    HeteroclinicNetwork,
    coordinate_plane,
)
from discrecontinual_equations.continuation.network_stability import (
    NetworkCompetition,
)
from discrecontinual_equations.systems.reflection_cubic import reflection_cubic

_XI3, _XI4 = ("xi1", "xi2", "xi3"), ("xi1", "xi2", "xi4")
_BASE = {
    "e12": 1.0, "c21": 1.0, "e23": 1.0, "e24": 0.8, "e31": 1.0, "c32": 1.0,
    "e41": 1.0, "c42": 2.0, "c13": 1.5, "c34": 0.01, "c43": 0.01,
}  # fmt: skip


def _interaction(e: dict[str, float]) -> np.ndarray:
    mu = np.zeros((4, 4))
    mu[1, 0], mu[2, 0], mu[3, 0] = e["e12"], -e["c13"], -e["c14"]
    mu[0, 1], mu[2, 1], mu[3, 1] = -e["c21"], e["e23"], e["e24"]
    mu[0, 2], mu[1, 2], mu[3, 2] = e["e31"], -e["c32"], -e["c34"]
    mu[0, 3], mu[1, 3], mu[2, 3] = e["e41"], -e["c42"], -e["c43"]
    interaction = mu - 1.0
    np.fill_diagonal(interaction, -1.0)
    return interaction


def _competition(interaction: np.ndarray) -> NetworkCompetition:
    network = HeteroclinicNetwork(reflection_cubic(np.ones(4), interaction))
    for index in range(4):
        network.add_saddle(f"xi{index + 1}", np.eye(4)[index])
    for source, target in ((1, 2), (2, 3), (3, 1), (2, 4), (4, 1)):
        network.add_connection(
            f"xi{source}",
            f"xi{target}",
            coordinate_plane(4, (source - 1, target - 1)),
        )
    return NetworkCompetition(network)


def _choices_after_xi2(interaction: np.ndarray, start: np.ndarray, horizon: float):
    """The saddle each departure from xi2 heads for, as 2 (xi3) or 3 (xi4)."""

    def rhs(_time: float, logs: np.ndarray) -> np.ndarray:
        return 1.0 + interaction @ np.exp(2.0 * np.minimum(logs, 3.0))

    solution = solve_ivp(
        rhs,
        (0.0, horizon),
        np.log(start),
        method="DOP853",
        rtol=1.0e-10,
        atol=1.0e-10,
        dense_output=True,
    )
    assert solution.success
    times = np.linspace(0.0, horizon, int(horizon * 20))
    leader = np.argmax(solution.sol(times), axis=0)
    order = [int(leader[0])]
    order += [int(b) for a, b in itertools.pairwise(leader) if a != b]
    return [order[k + 1] for k in range(len(order) - 1) if order[k] == 1]


def _rk4_classify(
    interaction: np.ndarray,
    cycle: list[int],
    start: np.ndarray,
    transverse: np.ndarray,
) -> np.ndarray:
    """Which starts follow ``cycle`` for two full turns, staying near it.

    ``start`` is a point of the connection ``cycle[-1] -> cycle[0]`` (log
    coordinates); ``transverse`` holds the two off-plane log coordinates of each
    trajectory. A trajectory escapes if the saddle it is nearest to changes to one
    that does not follow in ``cycle``, or if a third coordinate exceeds 0.1 -
    the neighbourhood the local index is measured in.
    """
    size = len(cycle)
    off = [k for k in range(4) if k not in (cycle[-1], cycle[0])]
    logs = np.repeat(start[:, None], transverse.shape[1], axis=1)
    logs[off] = transverse
    following = {cycle[k]: cycle[(k + 1) % size] for k in range(size)}
    leader = np.full(transverse.shape[1], cycle[0])
    steps = np.zeros(transverse.shape[1], dtype=int)
    alive = np.ones(transverse.shape[1], dtype=bool)
    attracted = np.zeros(transverse.shape[1], dtype=bool)
    step = 0.01

    def rhs(values: np.ndarray) -> np.ndarray:
        return 1.0 + interaction @ np.exp(2.0 * np.minimum(values, 3.0))

    while alive.any():
        y = logs[:, alive]
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * step * k1)
        k3 = rhs(y + 0.5 * step * k2)
        k4 = rhs(y + step * k3)
        y = y + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        logs[:, alive] = y
        index = np.flatnonzero(alive)
        top = np.argmax(y, axis=0)
        escaped = (np.sort(y, axis=0)[-3] > math.log(0.1)) | (y.max(axis=0) > 1.0)
        for k in np.flatnonzero(top != leader[index]):
            if following.get(int(leader[index[k]])) == top[k]:
                leader[index[k]] = top[k]
                steps[index[k]] += 1
            else:
                escaped[k] = True
        done = steps[index] >= 2 * size
        attracted[index[done & ~escaped]] = True
        alive[index[done | escaped]] = False
    return attracted


def _connection_midpoint(interaction: np.ndarray, source: int, target: int):
    """Log coordinates where the in-plane connection crosses ``x_s = x_t``."""
    plane = [source, target]
    block = interaction[np.ix_(plane, plane)]

    def rhs(_time: float, logs: np.ndarray) -> np.ndarray:
        return 1.0 + block @ np.exp(2.0 * logs)

    def crossing(_time: float, logs: np.ndarray) -> float:
        return float(logs[1] - logs[0])

    crossing.terminal = True  # type: ignore[attr-defined]
    solution = solve_ivp(
        rhs,
        (0.0, 500.0),
        np.log([1.0 - 1.0e-9, 1.0e-6]),
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-12,
        events=crossing,
    )
    point = np.full(4, -np.inf)
    point[plane] = solution.y_events[0][0]
    return point


class TestCompetitionNearDeltaZero(TestCase):
    """Castro and Lohse (2014), Proposition 5.3, a few hundredths from ``delta = 0``.

    ``delta`` and ``delta~`` are both within 0.07 of zero. Trajectories start near
    xi1; each records where it goes each time it leaves xi2. The prediction is a
    run of the loser followed by the winner for good.
    """

    _CASES = (
        ({**_BASE, "c14": 0.8 * 1.5 - 0.02}, _XI4),
        ({**_BASE, "c14": 0.8 * 1.5 + 0.01}, _XI3),
    )

    @pytest.mark.slow
    def test_every_seed_ends_on_the_predicted_cycle(self):
        for eigenvalues, winner in self._CASES:
            interaction = _interaction(eigenvalues)
            competition = _competition(interaction)
            deltas = [competition.stabilities[c].delta for c in (_XI4, _XI3)]
            assert max(abs(delta) for delta in deltas) < 0.07
            assert min(deltas) < 0.0 < max(deltas)
            assert competition.winner == winner
            target = 2 if winner == _XI3 else 3
            for seed in range(3):
                rng = np.random.default_rng(seed)
                start = np.array([0.95, 0.2, 0.0, 0.0]) + rng.uniform(1e-3, 3e-2, 4)
                choices = _choices_after_xi2(interaction, start, 6000.0)
                first = choices.index(target)
                assert set(choices[first:]) == {target}
                assert len(choices) - first >= 3


class TestBasinFraction(TestCase):
    """The attracted fraction of a box shrinks its complement like ``eps^sigma``."""

    @pytest.mark.slow
    def test_complement_of_the_basin_scales_with_the_index(self):
        """Kirk-Silber, Proposition 5.3 case (i): the xi3-cycle's index along
        ``xi1 -> xi2`` is ``e23/e24 - 1 = 1``, so ``1 - Sigma`` falls a thousandfold
        from ``eps = 1e-2`` to ``1e-5``."""
        eigenvalues = {
            "e12": 1.0, "c13": 1.0, "c14": 1.0, "c21": 1.0, "e23": 1.0,
            "e24": 0.5, "e31": 1.0, "c32": 1.5, "c34": 0.5, "e41": 1.0,
            "c42": 1.5, "c43": 0.5,
        }  # fmt: skip
        interaction = _interaction(eigenvalues)
        stability = _competition(interaction).stabilities[_XI3]
        sigma = stability.index("xi1", "xi2")
        assert abs(sigma - 1.0) < 1.0e-12
        start = _connection_midpoint(interaction, 0, 1)
        spacing = 0.5
        grid = (np.arange(40) + 0.5) * spacing
        first, second = (e.ravel() for e in np.meshgrid(grid, grid, indexing="ij"))
        weight = np.exp(-first - second)
        complements = []
        sizes = (1.0e-2, 1.0e-5)
        for size in sizes:
            transverse = math.log(size) - np.vstack([first, second])
            attracted = _rk4_classify(interaction, [1, 2, 0], start, transverse)
            complements.append(float(weight[~attracted].sum() / weight.sum()))
        measured = math.log(complements[1] / complements[0]) / math.log(
            sizes[1] / sizes[0],
        )
        assert abs(measured - sigma) < 0.1
