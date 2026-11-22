"""Heteroclinic networks: robust cycles, their stability, and their passage times.

The oracles are closed forms from the literature on robust heteroclinic cycles:

* the Guckenheimer-Holmes system in Krupa's normalisation (Krupa 1997, eq. 2.1)
  ``x_i' = x_i (lam + a1 x_i^2 + a2 x_{i+1}^2 + a3 x_{i+2}^2)``, whose saddle
  spectra are explicit (eq. 2.2), whose cycle exists iff ``a1 < 0``,
  ``a2 < a1 < a3`` (2.3, Proposition 1), and which is stable iff
  ``2 a1 > a2 + a3`` (Theorem 1) - the same as ``prod c_j > prod e_j``
  (Krupa and Melbourne 1995, Theorem 2.7 and Remark 2.10(b));
* the residence times near the saddles of an attracting cycle, which grow so
  that ``t_{j+1} / t_j -> c_j / e_{j+1}`` (Gaunersdorfer 1992, Lemma 1), measured
  here by integrating with ``scipy.integrate.solve_ivp``, independent of the
  library's own integrators;
* the Kirk-Silber network of two cycles sharing a connection, whose competition
  is decided by the signs of ``delta`` and ``delta~`` (Castro and Lohse 2014,
  section 5, Proposition 5.3), checked by simulation.

Trajectories are integrated in logarithmic coordinates ``y = ln x``: the cycle
pushes coordinates towards zero geometrically, and ``y`` keeps their relative
precision where ``x`` would underflow after a few revolutions.
"""

import itertools
import math
from unittest import TestCase

import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.connecting_orbit import MeshSpec
from discrecontinual_equations.continuation.heteroclinic_network import (
    HeteroclinicNetwork,
    coordinate_plane,
)
from discrecontinual_equations.continuation.network_continuation import (
    NetworkContinuation,
)
from discrecontinual_equations.continuation.residence import Itinerary
from discrecontinual_equations.systems.reflection_cubic import (
    guckenheimer_holmes,
    reflection_cubic,
)

_LAMBDA, _A1, _A2, _A3 = 1.0, -1.0, -1.5, -0.6
_NAMES = ("xi1", "xi2", "xi3")


def _gh_network(field, guesses=None) -> HeteroclinicNetwork:
    """The three positive-axis saddles and their connections in coordinate planes."""
    radius = math.sqrt(-_LAMBDA / _A1)
    network = HeteroclinicNetwork(field)
    for index, name in enumerate(_NAMES):
        guess = guesses[index] if guesses is not None else radius * np.eye(3)[index]
        network.add_saddle(name, guess)
    for index in range(3):
        following = (index + 1) % 3
        network.add_connection(
            _NAMES[index],
            _NAMES[following],
            coordinate_plane(3, (index, following)),
        )
    return network


def _log_flow(field):
    def rhs(_time: float, logs: np.ndarray) -> np.ndarray:
        # Below e^-300 a coordinate contributes nothing to x^2 terms, and clamping
        # keeps f(x) / x a ratio of two nonzero numbers instead of 0 / 0.
        state = np.exp(np.maximum(logs, -300.0))
        return np.array(field.eval(list(state))) / state

    return rhs


def _simulate(field, start: np.ndarray, horizon: float, samples: int):
    """Integrate in log coordinates with scipy, return (times, states)."""
    solution = solve_ivp(
        _log_flow(field),
        (0.0, horizon),
        np.log(start),
        method="DOP853",
        rtol=1.0e-11,
        atol=1.0e-12,
        dense_output=True,
    )
    assert solution.success
    times = np.linspace(0.0, horizon, samples)
    return times, np.exp(solution.sol(times).T)


class TestGuckenheimerHolmesCycle(TestCase):
    """Oracle 1: the three connections in their planes, and the stability inequality."""

    def test_network_holds_one_cycle_through_the_three_saddles(self):
        network = _gh_network(guckenheimer_holmes(_LAMBDA, _A1, _A2, _A3))
        assert network.cycles() == [_NAMES]
        assert [c.source for c in network.connections] == list(_NAMES)

    def test_saddles_and_spectra_match_krupa_closed_form(self):
        """Krupa (1997) eq. 2.2: Df(xi1) = lam diag(-2, (a1-a3)/a1, (a1-a2)/a1)."""
        network = _gh_network(guckenheimer_holmes(_LAMBDA, _A1, _A2, _A3))
        spectrum = network.spectrum(_NAMES)
        expanding = _LAMBDA * (_A1 - _A3) / _A1
        contracting = _LAMBDA * (_A2 - _A1) / _A1
        for index, name in enumerate(_NAMES):
            saddle = network.saddle(name)
            exact = math.sqrt(-_LAMBDA / _A1) * np.eye(3)[index]
            assert np.max(np.abs(saddle.equilibrium - exact)) < 1.0e-12
        assert np.allclose(spectrum.radial, 2.0 * _LAMBDA, atol=1.0e-10)
        assert np.allclose(spectrum.expanding, expanding, atol=1.0e-10)
        assert np.allclose(spectrum.contracting, contracting, atol=1.0e-10)
        assert np.all(np.isneginf(spectrum.transverse))

    def test_each_connection_is_found_in_its_invariant_plane(self):
        """Seeded off the plane, the projection BVP must return to it.

        The seed carries a 1e-2 bump in the coordinate the plane excludes; the
        connection the solver returns must lie in the plane to rounding, start at
        the source and end at the target.
        """
        network = _gh_network(guckenheimer_holmes(_LAMBDA, _A1, _A2, _A3))
        for index, connection in enumerate(network.connections):
            mesh = network.mesh(connection, half_length=20.0, intervals=400)
            seed = network.seed(connection, mesh)
            times = np.linspace(-20.0, 20.0, 401)
            seed[:, (index + 2) % 3] += 1.0e-2 * np.exp(-(times**2))
            orbit = network.solve(connection, mesh, seed)
            source = network.saddle(connection.source).equilibrium
            target = network.saddle(connection.target).equilibrium
            assert orbit.subspace_defect < 1.0e-10
            assert np.linalg.norm(orbit.solution.states[0] - source) < 1.0e-3
            assert np.linalg.norm(orbit.solution.states[-1] - target) < 1.0e-3
            assert orbit.resolution.estimate < 1.0e-3

    def test_connection_matches_the_exact_quarter_circle_at_resonance(self):
        """At ``a2 + a3 = 2 a1`` the connection is the arc ``x1^2 + x2^2 = R^2``.

        With ``u = x1^2`` the plane flow is ``u' = 2u(lam + a1 u + a2 v)`` and
        ``u + v`` is conserved exactly when ``a2 + a3 = 2 a1``; on that circle
        ``u = R^2 / (1 + exp(k t))``, ``k = 2 (a1 - a2) R^2``, ``R^2 = -lam/a1``.
        The trapezoidal BVP is second order, so doubling the mesh must cut the
        error by close to four.
        """
        a2, a3 = -1.4, -0.6
        field = guckenheimer_holmes(_LAMBDA, _A1, a2, a3)
        network = _gh_network(field)
        connection = network.connections[0]
        radius_squared = -_LAMBDA / _A1
        rate = 2.0 * (_A1 - a2) * radius_squared
        errors = []
        for intervals in (200, 400):
            mesh = MeshSpec(
                half_length=16.0,
                intervals=intervals,
                phase_index=0,
                phase_value=math.sqrt(radius_squared / 2.0),
            )
            orbit = network.solve(connection, mesh)
            times = orbit.solution.times
            u = radius_squared / (1.0 + np.exp(rate * times))
            exact = np.column_stack(
                [np.sqrt(u), np.sqrt(radius_squared - u), np.zeros_like(u)],
            )
            errors.append(float(np.max(np.abs(orbit.solution.states - exact))))
        assert errors[0] < 1.5e-4
        assert 3.9 < errors[0] / errors[1] < 4.1

    def test_stability_inequality_matches_the_parameter_condition(self):
        """``prod c_j > prod e_j`` from computed spectra iff ``2 a1 > a2 + a3``."""
        verdicts = []
        for a2, a3 in itertools.product((-1.9, -1.5, -1.2), (-0.9, -0.6, -0.2)):
            network = _gh_network(guckenheimer_holmes(_LAMBDA, _A1, a2, a3))
            spectrum = network.spectrum(_NAMES)
            ratio = ((a2 - _A1) / (_A1 - a3)) ** 3
            assert abs(spectrum.stability_ratio - ratio) < 1.0e-10 * ratio
            closed_form = a2 + a3 < 2.0 * _A1
            assert spectrum.is_asymptotically_stable == closed_form
            verdicts.append(closed_form)
        assert any(verdicts)
        assert not all(verdicts)

    def test_no_cycle_when_the_existence_condition_fails(self):
        """Krupa (2.3): with ``a3 < a1`` xi1 is a sink in the x1x2 plane."""
        network = _gh_network(guckenheimer_holmes(_LAMBDA, _A1, _A2, -1.2))
        margins = [network.persistence_margin(c) for c in network.connections]
        assert all(margin < 0.0 for margin in margins)


class TestResidenceTimeGrowth(TestCase):
    """Oracle 2: passage times near an attracting cycle grow by c / e per saddle."""

    def test_symmetric_cycle_residence_grows_by_the_eigenvalue_ratio(self):
        field = guckenheimer_holmes(_LAMBDA, _A1, _A2, _A3)
        network = _gh_network(field)
        predicted = network.spectrum(_NAMES).residence_ratios
        assert np.allclose(predicted, 0.5 / 0.4, atol=1.0e-10)
        times, states = _simulate(field, np.array([0.9, 0.3, 0.1]), 2500.0, 500001)
        itinerary = Itinerary.measure(
            times,
            states,
            [network.saddle(name).equilibrium for name in _NAMES],
            radius=0.1,
        )
        order = itinerary.order
        assert len(order) > 12
        assert all(b == (a + 1) % 3 for a, b in itertools.pairwise(order))
        growth = itinerary.growth_ratios()
        assert abs(growth[-1] / predicted[0] - 1.0) < 1.0e-4

    def test_broken_cycle_residence_ratios_follow_each_saddle_pair(self):
        """With Z3 broken, each step grows by its own ``c_j / e_{j+1}``.

        Gaunersdorfer's Lemma 1 indexes the ratio by the saddle left and the
        saddle entered; the three are distinct here, so a swapped index fails.
        """
        field = guckenheimer_holmes(_LAMBDA, _A1, _A2, _A3)
        field.parameters[1].value = 0.85
        network = _gh_network(field)
        predicted = network.spectrum(_NAMES).residence_ratios
        assert np.ptp(predicted) > 0.2
        times, states = _simulate(field, np.array([0.9, 0.3, 0.1]), 2500.0, 500001)
        itinerary = Itinerary.measure(
            times,
            states,
            [network.saddle(name).equilibrium for name in _NAMES],
            radius=0.1,
        )
        measured = itinerary.step_ratios(3)
        assert np.max(np.abs(measured / predicted - 1.0)) < 1.0e-4

    def test_may_leonard_cycle_grows_by_the_same_ratio(self):
        """The squared GH system is May-Leonard with alpha = a2/a1, beta = a3/a1.

        Gaunersdorfer (1992, section 4a): eigenvalues ``1 - beta`` (expanding) and
        ``-(alpha - 1)`` (contracting) at each vertex, so the residence ratio is
        ``(alpha - 1) / (1 - beta)`` - integrated here as an independent system.
        """
        alpha, beta = _A2 / _A1, _A3 / _A1
        interaction = np.array(
            [[-1.0, -alpha, -beta], [-beta, -1.0, -alpha], [-alpha, -beta, -1.0]],
        )
        gh = _gh_network(guckenheimer_holmes(_LAMBDA, _A1, _A2, _A3))
        predicted = gh.spectrum(_NAMES).residence_ratios[0]
        assert abs(predicted - (alpha - 1.0) / (1.0 - beta)) < 1.0e-10

        def lotka_volterra(_time: float, logs: np.ndarray) -> np.ndarray:
            return 1.0 + interaction @ np.exp(logs)

        solution = solve_ivp(
            lotka_volterra,
            (0.0, 1500.0),
            np.log([0.8, 0.1, 0.05]),
            method="DOP853",
            rtol=1.0e-11,
            atol=1.0e-12,
            dense_output=True,
        )
        times = np.linspace(0.0, 1500.0, 300001)
        states = np.exp(solution.sol(times).T)
        itinerary = Itinerary.measure(times, states, list(np.eye(3)), radius=0.1)
        assert abs(itinerary.growth_ratios()[-1] / predicted - 1.0) < 1.0e-4


class TestSymmetryBrokenNetwork(TestCase):
    """Oracle 3: connections persist as Z3 is broken, until one breaks where located.

    Breaking ``lam_2 = 1 - eps`` keeps every coordinate plane invariant (Z2^3
    survives) but moves xi2 and changes every ratio. In closed form,
    ``c_2 = 1.5 lam_2 - 1`` vanishes at ``lam_2 = 2/3`` - xi2 stops being a sink in
    the x1x2 plane, so xi1 -> xi2 breaks there - before ``e_1 = lam_2 - 0.6``
    vanishes at 0.6, which would break the same connection from its source.
    """

    @staticmethod
    def _ratio(growth: float) -> float:
        contracting = (0.5, 1.5 * growth - 1.0, 1.5 - growth)
        expanding = (growth - 0.6, 1.0 - 0.6 * growth, 0.4)
        return math.prod(contracting) / math.prod(expanding)

    def test_connections_persist_and_ratio_follows_the_closed_form(self):
        field = guckenheimer_holmes(_LAMBDA, _A1, _A2, _A3)
        continuation = NetworkContinuation(_gh_network(field), parameter_index=1)
        values = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75]
        points = continuation.trace(values, half_length=24.0, intervals=480)
        assert [point.parameter for point in points] == values
        for point in points:
            exact = math.sqrt(point.parameter)
            assert abs(point.equilibria["xi2"][1] - exact) < 1.0e-12
            ratio = point.stability_ratios[_NAMES]
            assert abs(ratio - self._ratio(point.parameter)) < 1.0e-9 * ratio
            for orbit in point.orbits.values():
                assert orbit.subspace_defect < 1.0e-10
                assert orbit.resolution.estimate < 5.0e-3

    def test_cycle_loses_stability_before_its_connection_breaks(self):
        """``rho`` falls through one near ``lam_2 = 0.71``, ahead of the break at 2/3.

        Closed form: ``rho(0.72) = 1.144`` and ``rho(0.70) = 0.862``. No orbits are
        solved here - only the saddles and spectra are continued - because near the
        break the target's contraction ``c_2`` is small and the orbit's approach to
        it slow, which is a resolution question rather than a stability one.
        """
        field = guckenheimer_holmes(_LAMBDA, _A1, _A2, _A3)
        continuation = NetworkContinuation(_gh_network(field), parameter_index=1)
        values = [1.0, 0.9, 0.8, 0.72, 0.70, 0.68]
        points = continuation.trace(values)
        ratios = [point.stability_ratios[_NAMES] for point in points]
        for point, ratio in zip(points, ratios, strict=True):
            assert abs(ratio - self._ratio(point.parameter)) < 1.0e-9 * ratio
            assert min(min(pair) for pair in point.margins.values()) > 0.0
        assert [ratio > 1.0 for ratio in ratios] == [True] * 4 + [False] * 2

    def test_continuation_locates_the_first_break(self):
        field = guckenheimer_holmes(_LAMBDA, _A1, _A2, _A3)
        continuation = NetworkContinuation(_gh_network(field), parameter_index=1)
        located = continuation.locate_break(1.0, 0.5, steps=20, tolerance=1.0e-10)
        assert (located.connection.source, located.connection.target) == (
            "xi1",
            "xi2",
        )
        assert located.side == "target"
        assert abs(located.parameter - 2.0 / 3.0) < 1.0e-8


def _kirk_silber(eigenvalues: dict[str, float]):
    """Z2^4 cubic with unit saddles and the given eigenvalues at each.

    With ``lam = 1`` and ``A_jj = -1`` the saddles sit at unit distance and the
    eigenvalue at xi_j in direction k is ``1 + A_kj``, so each one is set directly.
    Names follow Castro and Lohse (2014), section 5.
    """
    mu = np.zeros((4, 4))
    e = eigenvalues
    mu[1, 0], mu[2, 0], mu[3, 0] = e["e12"], -e["c13"], -e["c14"]
    mu[0, 1], mu[2, 1], mu[3, 1] = -e["c21"], e["e23"], e["e24"]
    mu[0, 2], mu[1, 2], mu[3, 2] = e["e31"], -e["c32"], -e["c34"]
    mu[0, 3], mu[1, 3], mu[2, 3] = e["e41"], -e["c42"], -e["c43"]
    interaction = mu - 1.0
    np.fill_diagonal(interaction, -1.0)
    field = reflection_cubic(np.ones(4), interaction)
    network = HeteroclinicNetwork(field)
    for index in range(4):
        network.add_saddle(f"xi{index + 1}", np.eye(4)[index])
    for source, target in ((1, 2), (2, 3), (3, 1), (2, 4), (4, 1)):
        network.add_connection(
            f"xi{source}",
            f"xi{target}",
            coordinate_plane(4, (source - 1, target - 1)),
        )
    return field, network


class TestKirkSilberCompetition(TestCase):
    """Two cycles sharing xi1 -> xi2: which one the dynamics follows.

    Castro and Lohse (2014), Proposition 5.3, under Assumption 1 (both ratios
    above one, ``0 < e24/e23 < 1``): if ``delta < 0`` the xi4-cycle has every
    stability index ``-inf`` and the xi3-cycle attracts; if ``delta~ < 0`` the
    reverse. Both cases below make the winner the cycle with the *smaller*
    stability ratio in at least one of them, so the ratio alone cannot decide.
    """

    _CASES = (
        (
            {
                "e12": 1.0, "c13": 1.0, "c14": 1.0, "c21": 1.0, "e23": 1.0,
                "e24": 0.5, "e31": 1.0, "c32": 1.5, "c34": 0.5, "e41": 1.0,
                "c42": 1.5, "c43": 0.5,
            },
            "xi3",
        ),
        (
            {
                "e12": 1.0, "c13": 4.0, "c14": 0.5, "c21": 1.0, "e23": 1.0,
                "e24": 0.8, "e31": 1.0, "c32": 1.0, "c34": 0.2, "e41": 1.0,
                "c42": 2.0, "c43": 0.5,
            },
            "xi4",
        ),
    )  # fmt: skip

    def test_network_holds_two_cycles_sharing_a_connection(self):
        _, network = _kirk_silber(self._CASES[0][0])
        cycles = network.cycles()
        assert sorted(cycles) == [("xi1", "xi2", "xi3"), ("xi1", "xi2", "xi4")]

    def test_roles_follow_the_cycle_not_the_saddle(self):
        """At xi2 the x4 direction expands for one cycle and is transverse to one."""
        eigenvalues = self._CASES[0][0]
        _, network = _kirk_silber(eigenvalues)
        three = network.spectrum(("xi1", "xi2", "xi3"))
        four = network.spectrum(("xi1", "xi2", "xi4"))
        assert abs(three.transverse[1] - eigenvalues["e24"]) < 1.0e-10
        assert abs(four.transverse[1] - eigenvalues["e23"]) < 1.0e-10
        assert abs(three.transverse[0] + eigenvalues["c14"]) < 1.0e-10
        assert not three.is_asymptotically_stable
        assert not four.is_asymptotically_stable

    def test_competition_is_won_where_castro_lohse_predict(self):
        for eigenvalues, winner in self._CASES:
            field, network = _kirk_silber(eigenvalues)
            three = network.spectrum(("xi1", "xi2", "xi3"))
            four = network.spectrum(("xi1", "xi2", "xi4"))
            assert three.stability_ratio > 1.0
            assert four.stability_ratio > 1.0
            delta = self._delta(four)
            delta_tilde = self._delta(three)
            predicted = "xi3" if delta < 0.0 < delta_tilde else "xi4"
            assert delta_tilde < 0.0 < delta or delta < 0.0 < delta_tilde
            assert predicted == winner
            saddles = [network.saddle(f"xi{i}").equilibrium for i in range(1, 5)]
            for seed in range(3):
                rng = np.random.default_rng(seed)
                start = np.array([0.95, 0.2, 0.0, 0.0]) + rng.uniform(1e-3, 3e-2, 4)
                times, states = _simulate(field, start, 600.0, 120001)
                itinerary = Itinerary.measure(times, states, saddles, radius=0.2)
                after = [
                    itinerary.order[k + 1]
                    for k in range(len(itinerary.order) - 1)
                    if itinerary.order[k] == 1
                ]
                late = after[len(after) // 2 :]
                assert len(late) >= 3
                assert {f"xi{index + 1}" for index in late} == {winner}

    @staticmethod
    def _delta(cycle) -> float:
        """Castro and Lohse (2014) section 5 and Table 1: ``delta`` of one cycle.

        ``delta = b1 a2 a3 + b3 a2 + b2`` where, at each node, ``a = c / e`` and
        ``b = -t / e`` in that cycle's own roles: at xi1 ``a3 = c14/e12`` and
        ``b3 = c13/e12`` for the xi4-cycle; at xi2 ``b1 = -e23/e24``; at xi4
        ``a2 = c42/e41`` and ``b2 = c43/e41``. Every quantity is read from the
        library's role-classified spectrum, so a mislabelled role changes the sign.
        """
        a3 = cycle.contracting[0] / cycle.expanding[0]
        b3 = -cycle.transverse[0] / cycle.expanding[0]
        b1 = -cycle.transverse[1] / cycle.expanding[1]
        a2 = cycle.contracting[2] / cycle.expanding[2]
        b2 = -cycle.transverse[2] / cycle.expanding[2]
        return b1 * a2 * a3 + b3 * a2 + b2


class TestItinerary(TestCase):
    """The visit measurement on a trajectory whose dwell times are known."""

    def test_measures_known_dwell_times_and_order(self):
        saddles = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        times = np.linspace(0.0, 10.0, 10001)
        states = np.zeros((times.size, 2)) + 5.0
        visits = [(0, 1.0, 2.0), (1, 3.0, 5.5), (0, 6.0, 9.25)]
        for saddle, enter, leave in visits:
            inside = (times > enter) & (times < leave)
            states[inside] = saddles[saddle]
        itinerary = Itinerary.measure(times, states, saddles, radius=0.5)
        assert itinerary.order == [0, 1, 0]
        assert np.allclose(itinerary.durations, [1.0, 2.5, 3.25], atol=2.0e-3)
