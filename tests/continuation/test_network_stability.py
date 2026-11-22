"""Stability indices of heteroclinic cycles, and the competition between them.

The oracles are independent of the code under test:

* ``f^index`` against Podvigina and Ashwin's own two-variable table (2011, p. 905)
  and against the exact measure of ``{prod x_i^alpha_i < 1}`` in an ``eps``-box,
  which is a tail probability of a signed sum of exponentials in closed form;
* the Rock-Scissors-Paper cycle's indices, Lemma 4.1 of Garrido-da-Silva and
  Castro (2019), from the transition matrices they print;
* the Guckenheimer-Holmes cycle: ``+inf`` everywhere iff ``2 a1 > a2 + a3``
  (Krupa 1997, Theorem 1), else ``-inf``;
* each cycle of the Kirk-Silber network against Podvigina and Ashwin's closed
  forms for type B3- cycles (2011, section 4.2.1), with the eigenvalue ratios of
  Castro and Lohse (2014), Table 1, and their Propositions 5.2 and 5.3;
* the type C4- cycle's transition matrices against Podvigina and Ashwin's
  ``[[b, 1], [a, 0]]`` and the leading-eigenvector term against their formula;
* simulation: which cycle a trajectory follows, from several seeds, and the
  fraction of a small box that a cycle attracts, against ``eps^|sigma|``.

Trajectories are integrated in logarithmic coordinates with scipy or with a
numpy RK4, never with the library's integrators.
"""

import itertools
import math
from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.heteroclinic_network import (
    HeteroclinicNetwork,
    coordinate_plane,
)
from discrecontinual_equations.continuation.network_stability import (
    CycleStability,
    NetworkCompetition,
    f_index,
    stability_indices,
    transition_matrices,
    transverse_eigenvalues,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.systems.reflection_cubic import (
    Amplitude,
    guckenheimer_holmes,
    reflection_cubic,
)

_XI3, _XI4 = ("xi1", "xi2", "xi3"), ("xi1", "xi2", "xi4")
_KEYS = (
    "e12", "c13", "c14", "c21", "e23", "e24", "e31", "c32", "c34", "e41", "c42",
    "c43",
)  # fmt: skip


class _Sheared(DeterministicFunction):
    """``g(y) = S f(S^-1 y)``: the same flow in linearly sheared coordinates."""

    def setup(self, inner: DeterministicFunction, shear: np.ndarray) -> None:
        self.inner = inner
        self.shear = shear
        self.unshear = np.linalg.inv(shear)

    def eval(self, point: list, time: float | None = None) -> list:
        size = len(point)
        inner = [
            sum((float(self.unshear[i, k]) * point[k] for k in range(size)), 0.0)
            for i in range(size)
        ]
        value = self.inner.eval(inner, time)
        return [
            sum((float(self.shear[i, k]) * value[k] for k in range(size)), 0.0)
            for i in range(size)
        ]


_KS_EDGES = ((1, 2), (2, 3), (3, 1), (2, 4), (4, 1))
_C4_EDGES = ((1, 2), (2, 3), (3, 4), (4, 1))


def _sheared_network(
    interaction: np.ndarray,
    shear: np.ndarray,
    edges: tuple[tuple[int, int], ...],
) -> HeteroclinicNetwork:
    """The reflection cubic in coordinates ``y = S x``: saddles and planes mapped."""
    field = _Sheared(
        variables=[Amplitude() for _ in range(4)],
        parameters=[],
        results=[Amplitude() for _ in range(4)],
        time=None,
    )
    field.setup(reflection_cubic(np.ones(4), interaction), shear)
    network = HeteroclinicNetwork(field)
    for index in range(4):
        network.add_saddle(f"xi{index + 1}", shear @ np.eye(4)[index])
    for source, target in edges:
        plane = coordinate_plane(4, (source - 1, target - 1))
        network.add_connection(f"xi{source}", f"xi{target}", shear @ plane)
    return network


def _pa_f(alpha: float, beta: float) -> float:
    """Podvigina and Ashwin (2011), p. 905, transcribed case by case."""

    def plus(a: float, b: float) -> float:
        if a >= 0.0 and b >= 0.0:
            return math.inf
        if a <= 0.0 and b <= 0.0:
            return 0.0
        if a < 0.0 < b:
            return -b / a - 1.0 if b / a < -1.0 else 0.0
        return -a / b - 1.0 if a / b < -1.0 else 0.0

    return plus(alpha, beta) - plus(-alpha, -beta)


def _log_tail(alpha: np.ndarray, level: float) -> float:
    """``ln P(sum alpha_i E_i > level)`` for independent unit exponentials ``E_i``.

    For distinct non-zero ``alpha`` and ``level >= 0`` the tail is
    ``sum_{alpha_i > 0} C_i exp(-level / alpha_i)`` with
    ``C_i = prod_{k != i} alpha_i / (alpha_i - alpha_k)`` (partial fractions of the
    product of the factors' Laplace transforms). Summed in log space.
    """
    logs, signs = [], []
    for i, a in enumerate(alpha):
        if a <= 0.0:
            continue
        others = np.delete(alpha, i)
        coefficient = np.prod(a / (a - others))
        logs.append(math.log(abs(coefficient)) - level / a)
        signs.append(math.copysign(1.0, coefficient))
    top = max(logs)
    total = sum(s * math.exp(v - top) for s, v in zip(signs, logs, strict=True))
    return top + math.log(total)


def _measured_index(alpha: np.ndarray, depth: float) -> float:
    """Index of ``{alpha . ln x < 0}`` from its exact measure in a box of side e^R.

    With ``x_i`` uniform on ``(0, e^R)``, ``ln x_i = R - E_i``, so the set is
    ``sum alpha_i E_i > R sum alpha``; its measure fraction (or its complement's,
    by ``E -> -E``) decays like ``e^{R s}``, and ``s`` is read off between two
    depths so constants cancel (Podvigina and Ashwin 2011, Definition 5).
    """
    total = float(np.sum(alpha))
    if np.all(alpha > 0.0):
        return math.inf
    if np.all(alpha < 0.0):
        return -math.inf
    depths = (-depth, -2.0 * depth)
    if total > 0.0:  # the complement is the small part: sigma_+ > 0
        logs = [_log_tail(-alpha, -r * total) for r in depths]
        return (logs[1] - logs[0]) / (depths[1] - depths[0])
    logs = [_log_tail(alpha, r * total) for r in depths]
    return -(logs[1] - logs[0]) / (depths[1] - depths[0])


def _kirk_silber(e: dict[str, float]) -> tuple[np.ndarray, HeteroclinicNetwork]:
    """Z2^4 cubic with unit saddles and eigenvalue ``1 + A_kj`` at xi_j along x_k."""
    mu = np.zeros((4, 4))
    mu[1, 0], mu[2, 0], mu[3, 0] = e["e12"], -e["c13"], -e["c14"]
    mu[0, 1], mu[2, 1], mu[3, 1] = -e["c21"], e["e23"], e["e24"]
    mu[0, 2], mu[1, 2], mu[3, 2] = e["e31"], -e["c32"], -e["c34"]
    mu[0, 3], mu[1, 3], mu[2, 3] = e["e41"], -e["c42"], -e["c43"]
    interaction = mu - 1.0
    np.fill_diagonal(interaction, -1.0)
    network = HeteroclinicNetwork(reflection_cubic(np.ones(4), interaction))
    for index in range(4):
        network.add_saddle(f"xi{index + 1}", np.eye(4)[index])
    for source, target in ((1, 2), (2, 3), (3, 1), (2, 4), (4, 1)):
        network.add_connection(
            f"xi{source}",
            f"xi{target}",
            coordinate_plane(4, (source - 1, target - 1)),
        )
    return interaction, network


def _type_c4(roles: list[tuple[float, float, float]]) -> tuple:
    """Cycle xi1 -> xi2 -> xi3 -> xi4 -> xi1 in R^4: a type C4- cycle.

    ``roles[j] = (c, e, t)`` at the saddle on axis ``j``: contraction from axis
    ``j - 1``, expansion to ``j + 1``, transverse along ``j + 2``.
    """
    mu = np.zeros((4, 4))
    for j, (contracting, expanding, transverse) in enumerate(roles):
        mu[(j - 1) % 4, j] = -contracting
        mu[(j + 1) % 4, j] = expanding
        mu[(j + 2) % 4, j] = transverse
    interaction = mu - 1.0
    np.fill_diagonal(interaction, -1.0)
    network = HeteroclinicNetwork(reflection_cubic(np.ones(4), interaction))
    for index in range(4):
        network.add_saddle(f"xi{index + 1}", np.eye(4)[index])
    for j in range(4):
        network.add_connection(
            f"xi{j + 1}",
            f"xi{(j + 1) % 4 + 1}",
            coordinate_plane(4, (j, (j + 1) % 4)),
        )
    return interaction, network


def _ratios(e: dict[str, float], cycle: str) -> tuple[tuple, tuple]:
    """Castro and Lohse (2014), Table 1: ``(a1, a2, a3), (b1, b2, b3)``.

    Index 1 is xi2, 2 the cycle's own node (xi3 or xi4), 3 is xi1.
    """
    own, other = ("4", "3") if cycle == "xi4" else ("3", "4")
    a = (
        e["c21"] / e[f"e2{own}"],
        e[f"c{own}2"] / e[f"e{own}1"],
        e[f"c1{own}"] / e["e12"],
    )
    b = (
        -e[f"e2{other}"] / e[f"e2{own}"],
        e[f"c{own}{other}"] / e[f"e{own}1"],
        e[f"c1{other}"] / e["e12"],
    )
    return a, b


def _pa_b3(a: tuple, b: tuple) -> tuple[float, float, float] | None:
    """Podvigina and Ashwin (2011), section 4.2.1, B3-: (sigma1, sigma2, sigma3)."""
    a1, a2, a3 = a
    b1, b2, b3 = b
    lost = (-math.inf,) * 3
    if b1 < 0.0 < b2 and b3 > 0.0:
        if a1 * a2 * a3 < 1.0 or b1 * a2 * a3 + b3 * a2 + b2 < 0.0:
            return lost
        return (_pa_f(b1, 1.0), math.inf, _pa_f(b3 + b1 * a3, 1.0))
    if b1 < 0.0 and b2 < 0.0 < b3:
        if (
            a1 * a2 * a3 < 1.0
            or b2 * a1 * a3 + b1 * a3 + b3 < 0.0
            or b1 * a2 * a3 + a2 * b3 + b2 < 0.0
        ):
            return lost
        return (
            min(_pa_f(b1, 1.0), _pa_f(b1 + b2 * a1, 1.0)),
            _pa_f(b2, 1.0),
            _pa_f(b3 + b1 * a3, 1.0),
        )
    return None


def _same(first: float, second: float) -> bool:
    return first == second or abs(first - second) < 1.0e-9


class TestIndexFunction(TestCase):
    """``f^index``: the index of a half-space in logarithmic coordinates."""

    def test_two_variables_match_podvigina_ashwin_table(self):
        values = (-3.0, -1.5, -1.0, -0.4, 0.0, 0.3, 1.0, 2.5)
        for alpha, beta in itertools.product(values, values):
            if alpha == beta == 0.0:
                continue
            assert _same(f_index(np.array([alpha, beta])), _pa_f(alpha, beta))

    def test_matches_the_exact_measure_of_the_set(self):
        """Random ``alpha`` in 2-4 dimensions against the closed-form tail."""
        rng = np.random.default_rng(3)
        signs_seen = set()
        for _ in range(60):
            size = int(rng.integers(2, 5))
            alpha = rng.uniform(-2.0, 2.0, size)
            if np.min(np.abs(np.subtract.outer(alpha, alpha)) + np.eye(size)) < 0.05:
                continue
            if abs(np.sum(alpha)) < 0.05 or np.min(np.abs(alpha)) < 0.05:
                continue
            expected = _measured_index(alpha, depth=200.0)
            computed = f_index(alpha)
            signs_seen.add(np.sign(computed))
            if math.isinf(expected):
                assert computed == expected
            else:
                assert abs(computed - expected) < 1.0e-6
        assert {-1.0, 1.0} <= signs_seen

    def test_rejects_the_zero_vector(self):
        with pytest.raises(ValueError, match="undefined"):
            f_index(np.zeros(3))


class TestPublishedIndices(TestCase):
    """Indices that the literature prints, for the transition matrices it prints."""

    def test_rock_scissors_paper_cycle(self):
        """Garrido-da-Silva and Castro (2019), section 4 and Lemma 4.1."""
        grid = np.linspace(-0.85, 0.85, 9)
        finite = 0
        for ex, ey in itertools.product(grid, grid):
            if abs(ex + ey) < 1.0e-9:
                continue
            first = np.array([[(1 - ey) / 2, 1, 0], [-(1 + ex) / 2, 0, 1], [1, 0, 0]])
            second = np.array([[(1 - ex) / 2, 1, 0], [-(1 + ey) / 2, 0, 1], [1, 0, 0]])
            indices = stability_indices([first, second])
            if ex + ey < 0.0:
                expected = (
                    min((1 - ex) / (1 + ex), (1 - ey) ** 2 / (2 * (1 + ey))),
                    min((1 - ey) / (1 + ey), (1 - ex) ** 2 / (2 * (1 + ex))),
                )
                finite += 1
            else:
                expected = (-math.inf, -math.inf)
            assert all(_same(x, y) for x, y in zip(indices, expected, strict=True))
        assert finite > 20

    def test_guckenheimer_holmes_cycle_is_stable_exactly_when_krupa_says(self):
        """No transverse directions: ``+inf`` iff ``rho > 1`` iff ``2 a1 > a2 + a3``."""
        verdicts = []
        grid = list(itertools.product((-1.9, -1.5, -1.2), (-0.9, -0.6, -0.2)))
        grid += [(-1.41, -0.6), (-1.39, -0.6)]
        for a2, a3 in grid:
            network = HeteroclinicNetwork(guckenheimer_holmes(1.0, -1.0, a2, a3))
            for index in range(3):
                network.add_saddle(f"xi{index + 1}", np.eye(3)[index])
            for index in range(3):
                following = (index + 1) % 3
                network.add_connection(
                    f"xi{index + 1}",
                    f"xi{following + 1}",
                    coordinate_plane(3, (index, following)),
                )
            stability = CycleStability.of(network, _XI3)
            stable = a2 + a3 < -2.0
            assert stability.matrices[0].shape == (1, 1)
            assert np.all(stability.indices == (math.inf if stable else -math.inf))
            assert stability.is_asymptotically_stable == stable
            verdicts.append(stable)
        assert any(verdicts)
        assert not all(verdicts)

    def test_kirk_silber_cycles_match_the_type_b3_closed_forms(self):
        """Random eigenvalues, both cycles, every case of P&A section 4.2.1 B3-.

        ``sigma1, sigma2, sigma3`` are along the connections into xi2, into the
        cycle's own node, and into xi1. A positive transverse eigenvalue at the
        own node (``c34 < 0`` or ``c43 < 0``) puts the cycle in case (iv).
        """
        rng = np.random.default_rng(1)
        seen: dict[tuple[bool, bool], int] = {}
        for _ in range(250):
            e = {key: float(rng.uniform(0.1, 3.0)) for key in _KEYS}
            for key in ("c34", "c43"):
                if rng.uniform() < 0.4:
                    e[key] = -float(rng.uniform(0.05, 1.5))
            _, network = _kirk_silber(e)
            for cycle, own in ((_XI4, "xi4"), (_XI3, "xi3")):
                expected = _pa_b3(*_ratios(e, own))
                assert expected is not None
                stability = CycleStability.of(network, cycle)
                computed = (
                    stability.index("xi1", "xi2"),
                    stability.index("xi2", own),
                    stability.index(own, "xi1"),
                )
                assert all(_same(x, y) for x, y in zip(computed, expected, strict=True))
                case = (_ratios(e, own)[1][1] > 0.0, expected[0] > -math.inf)
                seen[case] = seen.get(case, 0) + 1
        assert len(seen) == 4
        assert min(seen.values()) >= 10

    def test_delta_is_castro_lohse_equation_three_and_decides_lemma_5_1(self):
        rng = np.random.default_rng(2)
        signs = set()
        for _ in range(60):
            e = {key: float(rng.uniform(0.2, 2.5)) for key in _KEYS}
            _, network = _kirk_silber(e)
            competition = NetworkCompetition(network)
            four, three = competition.stabilities[_XI4], competition.stabilities[_XI3]
            delta = (
                e["c43"] / e["e41"]
                + e["c13"] * e["c42"] / (e["e12"] * e["e41"])
                - e["e23"] * e["c14"] * e["c42"] / (e["e12"] * e["e41"] * e["e24"])
            )
            delta_tilde = (
                e["c34"] / e["e31"]
                + e["c14"] * e["c32"] / (e["e12"] * e["e31"])
                - e["e24"] * e["c13"] * e["c32"] / (e["e12"] * e["e31"] * e["e23"])
            )
            assert abs(four.delta - delta) < 1.0e-9 * max(1.0, abs(delta))
            assert abs(three.delta - delta_tilde) < 1.0e-9 * max(1.0, abs(delta))
            rho = e["c42"] * e["c14"] * e["c21"] / (e["e24"] * e["e41"] * e["e12"])
            if rho > 1.0:
                assert four.is_attractor == (delta > 0.0)
                signs.add(delta > 0.0)
        assert signs == {True, False}

    def test_castro_lohse_proposition_5_2_pattern(self):
        """``delta, delta~ > 0``: sigma~23 = sigma24 = +inf, 0 < sigma~12 < inf,
        -inf < sigma12 < 0, and exactly one of sigma~31, sigma41 is +inf."""
        e = dict(zip(_KEYS, (1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.5, 0.5, 1.0, 1.5,
                             2.0), strict=True))  # fmt: skip
        _, network = _kirk_silber(e)
        competition = NetworkCompetition(network)
        four, three = competition.stabilities[_XI4], competition.stabilities[_XI3]
        assert four.delta > 0.0
        assert three.delta > 0.0
        assert three.index("xi2", "xi3") == math.inf
        assert four.index("xi2", "xi4") == math.inf
        assert 0.0 < three.index("xi1", "xi2") < math.inf
        assert -math.inf < four.index("xi1", "xi2") < 0.0
        tails = (three.index("xi3", "xi1"), four.index("xi4", "xi1"))
        assert sum(value == math.inf for value in tails) == 1
        assert sorted(competition.survivors) == [_XI3, _XI4]
        assert competition.winner is None
        assert three.is_predominantly_stable != four.is_predominantly_stable

    def test_castro_lohse_proposition_5_3_cases(self):
        """(i) ``delta < 0``: the xi4-cycle is all -inf, the xi3-cycle p.a.s.;
        (ii) ``delta > 0 > delta~``: the xi3-cycle all -inf, sigma12 < 0, rest +inf."""
        case_one = dict(zip(_KEYS, (1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.5, 0.5,
                                    1.0, 1.5, 0.5), strict=True))  # fmt: skip
        competition = NetworkCompetition(_kirk_silber(case_one)[1])
        four, three = competition.stabilities[_XI4], competition.stabilities[_XI3]
        assert four.delta < 0.0 < three.delta
        assert np.all(four.indices == -math.inf)
        assert three.index("xi2", "xi3") == three.index("xi3", "xi1") == math.inf
        assert 0.0 < three.index("xi1", "xi2") < math.inf
        assert three.is_predominantly_stable
        assert competition.winner == _XI3

        case_two = dict(zip(_KEYS, (1.0, 4.0, 0.5, 1.0, 1.0, 0.8, 1.0, 1.0, 0.2,
                                    1.0, 2.0, 0.5), strict=True))  # fmt: skip
        competition = NetworkCompetition(_kirk_silber(case_two)[1])
        four, three = competition.stabilities[_XI4], competition.stabilities[_XI3]
        assert three.delta < 0.0 < four.delta
        assert np.all(three.indices == -math.inf)
        assert four.index("xi1", "xi2") < 0.0
        assert four.index("xi2", "xi4") == four.index("xi4", "xi1") == math.inf
        assert abs(four.index("xi1", "xi2") - (1.0 - 1.0 / 0.8)) < 1.0e-12
        assert competition.winner == _XI4

    def test_type_c_cycle_has_podvigina_ashwin_transition_matrices(self):
        """Type C4-: the contracting line arrives as the next node's transverse one,
        so ``M_j = [[b_j, 1], [a_j, 0]]`` (P&A 2011, section 4.2.2)."""
        roles = [(1.3, 0.7, 0.4), (0.9, 1.1, -0.5), (1.6, 0.8, -0.3), (0.6, 1.4, -0.9)]
        _, network = _type_c4(roles)
        cycle = ("xi1", "xi2", "xi3", "xi4")
        matrices = transition_matrices(network, cycle)
        for (contracting, expanding, transverse), matrix in zip(
            roles,
            matrices,
            strict=True,
        ):
            expected = [[-transverse / expanding, 1.0], [contracting / expanding, 0.0]]
            assert np.allclose(matrix, expected, atol=1.0e-12)
        values = transverse_eigenvalues(network, cycle)
        assert np.allclose(np.concatenate(values), [r[2] for r in roles], atol=1e-12)

    def test_type_c_leading_eigenvector_term_matches_podvigina_ashwin(self):
        """P&A (2011) section 4.2.2, C4- with ``b1 < 0``: ``sigma_1 =
        f^index(v22 / h, -v21 / h)`` from the eigenvectors of ``M4 M3 M2 M1``.

        Here that term is the smallest of Theorem 3.10's candidates, so it is the
        index; the row terms are larger.
        """
        a = (0.6, 2.3, 0.7, 2.0)
        b = (-0.5, 1.2, 0.2, 0.9)
        roles = [(a[j], 1.0, -b[j]) for j in range(4)]
        _, network = _type_c4(roles)
        stability = CycleStability.of(network, ("xi1", "xi2", "xi3", "xi4"))
        product = np.eye(2)
        for j in range(4):
            product = np.array([[b[j], 1.0], [a[j], 0.0]]) @ product
        values, vectors = np.linalg.eig(product)
        lead = int(np.argmax(np.abs(values)))
        first = vectors[:, lead] * np.sign(vectors[0, lead])
        second = vectors[:, 1 - lead]
        assert np.all(first > 0.0)
        h = first[0] * second[1] - first[1] * second[0]
        expected = _pa_f(second[1] / h, -second[0] / h)
        assert 0.5 < expected < 1.5
        assert abs(stability.index("xi4", "xi1") - expected) < 1.0e-9

    def test_complex_leading_eigenvalue_attracts_nothing(self):
        """C2- with ``(b1 b2 + a2 - a1)^2 + 4 a1 b1 b2 < 0``: all indices ``-inf``
        (P&A 2011, section 4.2.2), although ``|lambda| = 2`` and ``Re lambda > 1``."""
        for a1, a2, b1, b2 in ((2.0, 2.0, -0.5, 1.0), (2.4, 1.4, 0.6, -1.0)):
            assert (b1 * b2 + a2 - a1) ** 2 + 4 * a1 * b1 * b2 < 0.0
            first = np.array([[b1, 1.0], [a1, 0.0]])
            second = np.array([[b2, 1.0], [a2, 0.0]])
            assert np.all(stability_indices([first, second]) == -math.inf)

    def test_leading_eigenvalue_below_one_attracts_nothing(self):
        """C2- with ``max(b1 b2 + a1 + a2, 2(b1 b2 + a1 + a2 - a1 a2)) < 2``:
        all ``-inf`` (P&A 2011, section 4.2.2), though the eigenvector is positive."""
        a1, a2, b1, b2 = 0.4, 0.8, -0.2, 0.1
        total = b1 * b2 + a1 + a2
        assert max(total, 2.0 * (total - a1 * a2)) < 2.0
        first = np.array([[b1, 1.0], [a1, 0.0]])
        second = np.array([[b2, 1.0], [a2, 0.0]])
        assert np.all(stability_indices([first, second]) == -math.inf)

    def test_rounding_does_not_turn_an_infinite_index_finite(self):
        """A C4- cycle whose leading left eigenvector has an exact zero component,
        computed as about 1e-17: the indices into nodes 1 and 2 are ``+inf``."""
        a, b = (0.5, 1.5, 0.4, 1.9), (-1.0, 1.6, 0.5, 1.1)
        matrices = [np.array([[b[j], 1.0], [a[j], 0.0]]) for j in range(4)]
        indices = stability_indices(matrices)
        assert indices[1] == indices[2] == math.inf
        assert abs(indices[0]) < 1.0e-12
        assert abs(indices[3] + 0.25) < 1.0e-12


class TestCoordinateInvariance(TestCase):
    """The index is a property of the flow, not of the coordinates it is written in."""

    def test_kirk_silber_indices_survive_oblique_coordinates(self):
        """``g(y) = S f(S^-1 y)`` with each of several shears ``S``: the planes are
        no longer orthogonal, and every index must come out unchanged."""
        e = dict.fromkeys(_KEYS, 1.0) | {"e24": 0.5, "c43": 0.7, "c34": 0.4, "c13": 1.3}
        interaction, network = _kirk_silber(e)
        reference = {cycle: CycleStability.of(network, cycle) for cycle in (_XI3, _XI4)}
        assert reference[_XI3].index("xi1", "xi2") == 1.0
        for row, column in ((0, 2), (0, 3), (2, 3), (3, 0), (1, 3)):
            shear = np.eye(4)
            shear[row, column] = 0.3
            sheared = _sheared_network(interaction, shear, _KS_EDGES)
            for cycle, expected in reference.items():
                computed = CycleStability.of(sheared, cycle).indices
                assert all(
                    _same(x, y) for x, y in zip(computed, expected.indices, strict=True)
                )


class TestRepresentationErrors(TestCase):
    """Cycles the transition-matrix construction does not cover are refused."""

    def test_unknown_connection_and_index_lookup(self):
        _, network = _kirk_silber(dict.fromkeys(_KEYS, 1.0))
        with pytest.raises(KeyError):
            CycleStability.of(network, ("xi1", "xi3", "xi2"))
        stability = CycleStability.of(network, _XI3)
        with pytest.raises(KeyError):
            stability.index("xi2", "xi4")

    def test_delta_needs_one_unstable_transverse_node(self):
        _, network = _type_c4([(1.0, 1.0, -0.5)] * 4)
        stability = CycleStability.of(network, ("xi1", "xi2", "xi3", "xi4"))
        with pytest.raises(ValueError, match="exactly one node"):
            _ = stability.delta
        _, network = _type_c4([(1.0, 1.0, 0.5)] + [(1.0, 1.0, -0.5)] * 3)
        stability = CycleStability.of(network, ("xi1", "xi2", "xi3", "xi4"))
        with pytest.raises(ValueError, match="type B"):
            _ = stability.delta

    def test_delta_needs_one_transverse_direction(self):
        matrix = np.array([[2.0, 0.0, 0.0], [-0.5, 1.0, 0.0], [0.3, 0.0, 1.0]])
        stability = CycleStability(("a", "b"), [matrix, np.diag([2.0, 1.0, 1.0])])
        with pytest.raises(ValueError, match="one transverse direction"):
            _ = stability.delta

    def test_consecutive_connections_in_one_plane(self):
        """``xi1 -> xi2 -> xi1`` both in the x1x2 plane: no contracting direction."""
        _, network = _kirk_silber(dict.fromkeys(_KEYS, 1.0))
        loop = HeteroclinicNetwork(network.function)
        for saddle in network.saddles[:2]:
            loop.add_saddle(saddle.name, saddle.equilibrium)
        loop.add_connection("xi1", "xi2", coordinate_plane(4, (0, 1)))
        loop.add_connection("xi2", "xi1", coordinate_plane(4, (0, 1)))
        with pytest.raises(ValueError, match="exactly one direction"):
            CycleStability.of(loop, ("xi1", "xi2"))

    def test_type_c_directions_in_oblique_planes_are_refused(self):
        """A sheared C4- cycle: the contracting line must arrive as a transverse
        one, but measured orthogonally to oblique planes the two do not line up,
        and the construction refuses rather than guesses (see the module notes)."""
        a, b = (0.6, 2.3, 0.7, 2.0), (-0.5, 1.2, 0.2, 0.9)
        interaction, _ = _type_c4([(a[j], 1.0, -b[j]) for j in range(4)])
        shear = np.eye(4)
        shear[0, 3] = 0.3
        network = _sheared_network(interaction, shear, _C4_EDGES)
        with pytest.raises(ValueError, match="Assumption A"):
            CycleStability.of(network, ("xi1", "xi2", "xi3", "xi4"))

    def test_connections_in_subspaces_of_different_dimension(self):
        _, network = _kirk_silber(dict.fromkeys(_KEYS, 1.0))
        bent = HeteroclinicNetwork(network.function)
        for saddle in network.saddles:
            bent.add_saddle(saddle.name, saddle.equilibrium)
        bent.add_connection("xi1", "xi2", coordinate_plane(4, (0, 1)))
        bent.add_connection("xi2", "xi3", coordinate_plane(4, (1, 2)))
        bent.add_connection("xi3", "xi1", coordinate_plane(4, (0, 2, 3)))
        with pytest.raises(ValueError, match="different dimensions"):
            CycleStability.of(bent, _XI3)
