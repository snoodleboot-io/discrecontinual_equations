"""Independent checks of the stability indices, added in verification.

* A five-node cycle in R^5 with two transverse directions per node, where the
  permutation ``A_j`` of Garrido-da-Silva and Castro's (2019) eq. 7 is a 3-cycle
  rather than an involution, so applying it transposed or from the wrong side
  changes the answer. The oracle builds the matrices by hand from the local maps.
* Type C4- with ``b1 < 0`` and the other ``b_j > 0``: Lohse's corrected closed forms
  (Hamburger Beitraege zur Angewandten Mathematik 2015-08, Lemma B.5, correcting
  Podvigina and Ashwin 2011, pp. 908-909, as agreed with Podvigina). The original
  P&A branch keeps only the eigenvector term and is wrong in most cases.
* A type C cycle with every index finite and negative: it attracts a set of positive
  measure but is not predominantly stable (Lohse 2015: type C cycles can be
  predominantly unstable).
"""

import math
from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.heteroclinic_network import (
    HeteroclinicNetwork,
    coordinate_plane,
)
from discrecontinual_equations.continuation.network_stability import (
    CycleStability,
    stability_indices,
)
from discrecontinual_equations.systems.reflection_cubic import reflection_cubic


def _axis_cycle(columns: list[dict[int, float]]) -> tuple[CycleStability, tuple]:
    """Cycle through the unit saddles on the axes of R^n, in order.

    ``columns[j][s]`` is the eigenvalue at the saddle on axis ``j`` along axis
    ``j + s``; with unit growth the cubic's eigenvalue there is ``1 + A[j+s, j]``.
    """
    size = len(columns)
    interaction = -np.ones((size, size))
    for j, column in enumerate(columns):
        for shift, value in column.items():
            interaction[(j + shift) % size, j] = value - 1.0
    np.fill_diagonal(interaction, -1.0)
    network = HeteroclinicNetwork(reflection_cubic(np.ones(size), interaction))
    for index in range(size):
        network.add_saddle(f"x{index}", np.eye(size)[index])
    for j in range(size):
        network.add_connection(
            f"x{j}",
            f"x{(j + 1) % size}",
            coordinate_plane(size, (j, (j + 1) % size)),
        )
    nodes = tuple(f"x{index}" for index in range(size))
    return CycleStability.of(network, nodes), nodes


def _same(first: float, second: float) -> bool:
    return first == second or abs(first - second) < 1.0e-9 * max(1.0, abs(first))


def _f2(alpha: float, beta: float) -> float:
    """Podvigina and Ashwin's two-variable ``f^index`` (2011, p. 905)."""

    def plus(a: float, b: float) -> float:
        if a >= 0.0 and b >= 0.0:
            return math.inf
        if a + b <= 0.0:
            return 0.0
        return -(a + b) / min(a, b)

    return plus(alpha, beta) - plus(-alpha, -beta)


class TestNonInvolutivePermutation(TestCase):
    """nt = 2: leaving ``(x_{j-1}, x_{j+2}, x_{j+3})`` arrive as the transverse,
    expanding and transverse directions ``(x_{j+2}, x_{j+3}, x_{j-1})`` of the next
    node, so in ``(w, z1, z2)`` coordinates ``M_j = [[b1, 1, 0], [b2, 0, 1],
    [a, 0, 0]]`` with ``a = c/e``, ``b_s = -t_s/e``."""

    _CASES = (
        (
            (2.858, 1.778, 2.941, 0.702, 2.018),
            (1.065, 1.703, 0.762, 1.807, 1.316),
            (-1.814, 0.569, -0.918, -1.599, -1.97),
            (-0.802, -1.941, -1.865, -0.438, -1.257),
            (2.5435448072147, 1.9929701230228472, math.inf, math.inf, 4.0091201523667),
        ),
        (
            (0.833, 1.745, 1.734, 1.751, 2.896),
            (1.025, 0.836, 1.283, 1.462, 1.909),
            (-1.206, -0.609, -1.867, 0.303, -1.384),
            (-1.004, -0.512, -1.416, -1.564, 0.144),
            (math.inf, 15.148372365849625, 7.695165775346045, 3.825082508250825,
             12.256944444444446),
        ),
    )  # fmt: skip

    def test_indices_match_the_hand_built_transition_matrices(self):
        for c, e, t1, t2, printed in self._CASES:
            columns = [{-1: -c[j], 1: e[j], 2: t1[j], 3: t2[j]} for j in range(len(c))]
            stability, _ = _axis_cycle(columns)
            by_hand = [
                np.array(
                    [
                        [-t1[j] / e[j], 1.0, 0.0],
                        [-t2[j] / e[j], 0.0, 1.0],
                        [c[j] / e[j], 0.0, 0.0],
                    ],
                )
                for j in range(len(c))
            ]
            expected = stability_indices(by_hand)
            assert all(
                _same(x, y) for x, y in zip(stability.indices, expected, strict=True)
            )
            # values from an implementation written separately from the papers
            assert all(
                _same(x, y) for x, y in zip(stability.indices, printed, strict=True)
            )


class TestTypeC4CorrectedClosedForms(TestCase):
    """C4- with ``b1 < 0 < b2, b3, b4`` against Lohse's (2015) Lemma B.5.

    Lohse labels the indices in the opposite orientation; in this library's labels
    the row terms belong to the connections into xi1, xi4 and xi3, and each is paired
    with the eigenvector term of the same section. The term for the connection into
    xi2 is the eigenvector term alone. Podvigina and Ashwin's original branch, which
    keeps only the eigenvector terms, misses the row terms.
    """

    @staticmethod
    def _eigenvector_terms(a: tuple, b: tuple) -> list[float] | None:
        matrices = [np.array([[b[j], 1.0], [a[j], 0.0]]) for j in range(4)]
        terms = []
        for j in range(4):
            product = np.eye(2)
            for step in range(4):
                product = matrices[(j + step) % 4] @ product
            (p, q), (r, s) = product
            disc = (p - s) ** 2 + 4.0 * q * r
            if disc < 0.0:
                return None
            big = (p + s + math.sqrt(disc)) / 2.0
            small = (p + s - math.sqrt(disc)) / 2.0
            if big <= 1.0:
                return None
            first = np.array([q, big - p])
            second = np.array([q, small - p])
            if first[0] * first[1] <= 0.0:
                return None
            first = first * np.sign(first[0])
            h = first[0] * second[1] - first[1] * second[0]
            terms.append(_f2(second[1] / h, -second[0] / h))
        return terms

    def test_library_matches_lohse_and_not_the_original_branch(self):
        rng = np.random.default_rng(3)
        checked = differs = 0
        for _ in range(300):
            a = tuple(float(x) for x in rng.uniform(0.2, 2.5, 4))
            b = (
                float(rng.uniform(-1.5, -0.05)),
                *(float(x) for x in rng.uniform(0.05, 1.5, 3)),
            )
            terms = self._eigenvector_terms(a, b)
            if terms is None:
                continue
            _, _, a3, a4 = a
            b1, _, b3, b4 = b
            corrected = (
                min(terms[0], _f2(b1, 1.0)),
                terms[1],
                min(terms[2], _f2(b3 * (b1 * b4 + a4) + b1 * a3, b1 * b4 + a4)),
                min(terms[3], _f2(b1 * b4 + a4, b1)),
            )
            columns = [{-1: -a[j], 1: 1.0, 2: -b[j]} for j in range(4)]
            stability, _ = _axis_cycle(columns)
            assert all(
                _same(x, y) for x, y in zip(stability.indices, corrected, strict=True)
            )
            checked += 1
            differs += not all(
                _same(x, y) for x, y in zip(stability.indices, terms, strict=True)
            )
        assert checked > 50
        assert differs > checked // 2

    def test_simulated_case(self):
        """The case simulated in verification (Lotka-Volterra, log-log slopes over
        eps = 1e-14 .. 1e-30): ``1 - Sigma ~ eps^0.31`` into xi1 and ``Sigma ~
        eps^0.46`` into xi4, where the original branch predicts ``+inf`` for both."""
        a, b = (1.68, 0.86, 0.61, 1.0), (-0.76, 0.82, 1.02, 0.63)
        stability, _ = _axis_cycle([{-1: -a[j], 1: 1.0, 2: -b[j]} for j in range(4)])
        assert abs(stability.index("x3", "x0") - 0.24 / 0.76) < 1.0e-12
        assert stability.index("x0", "x1") == stability.index("x1", "x2") == math.inf
        assert abs(stability.index("x2", "x3") - _f2(1.0 - 0.76 * 0.63, -0.76)) < 1e-12
        assert -0.47 < stability.index("x2", "x3") < -0.45
        assert self._eigenvector_terms(a, b) == [math.inf] * 4


class TestPredominantlyUnstableCycle(TestCase):
    def test_every_index_negative_yet_an_attractor(self):
        a, b = (2.47, 1.16, 0.26, 2.93), (0.44, 0.14, -1.06, 0.06)
        stability, _ = _axis_cycle([{-1: -a[j], 1: 1.0, 2: -b[j]} for j in range(4)])
        assert np.all(stability.indices < 0.0)
        assert np.all(stability.indices > -math.inf)
        assert stability.is_attractor
        assert not stability.is_predominantly_stable
        assert not stability.is_asymptotically_stable

    def test_mixed_infinite_and_finite_indices_are_not_asymptotically_stable(self):
        a, b = (1.68, 0.86, 0.61, 1.0), (-0.76, 0.82, 1.02, 0.63)
        stability, _ = _axis_cycle([{-1: -a[j], 1: 1.0, 2: -b[j]} for j in range(4)])
        assert np.any(stability.indices == math.inf)
        assert not stability.is_asymptotically_stable
        assert not stability.is_predominantly_stable
        assert stability.is_attractor
