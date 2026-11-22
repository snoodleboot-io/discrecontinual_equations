"""Levy-area checks against references that do not go through the sampler's code.

The tests in ``test_levy_area.py`` measure the sampler's pathwise error against a
reference assembled by the sampler itself, and its law by moments that are
blind to two kinds of mistake. These close both, with references built here:

* **The sign and scale of the Fourier part, pathwise.** A path built explicitly
  from ``K`` Fourier coefficients (Mrongowius and Rossler, arXiv:2101.09542, (10)
  with ``a_k = sqrt(h) X_k / (sqrt(2) pi k)``, ``b_k = sqrt(h) Y_k / (sqrt(2) pi
  k)``, ``a_0 = -2 sum a_k``) is a smooth curve whose area is a plain line
  integral. :meth:`LevyAreaSampler.assemble` with zero remainders must return it.
  The law of ``(dW, A)`` does not change under ``A -> -A``, so no moment test can
  see a sign error in one of the series' two parts; this sees it.
* **The coupling between pairs that share a channel.** Given the increments,
  ``E[A_(1,2) A_(1,3) | dW] = h dW_2 dW_3 / 12`` and ``E[A_(1,2) A_(2,3) | dW] =
  -h dW_1 dW_3 / 12`` (from the series: the only surviving products are the
  ``dW``-linear parts that share ``X_1`` or ``X_2``). Unconditionally both are
  zero, which is all the existing test checks; a sampler that gives each pair its
  own independent remainder (as a diagonal-covariance shortcut would) has the
  right law for every single pair and fails here. The same regression on a
  fine-grid Riemann sum of a simulated Brownian path pins the reference value
  independently of the series.
* **Levy's conditional law itself**, through its characteristic function
  ``E[cos(u A) | dW] = (b / sinh b) exp(|dW|^2 (1 - b coth b) / (2 h))``, ``b = u
  h / 2`` (Levy 1950), at fixed increments, which a test of the unconditional
  distribution averages away.
"""

import numpy as np
import pytest

from discrecontinual_equations.solver.stochastic.wiener import LevyAreaSampler

_STEP = 0.5


def _fourier_path(
    delta_w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """``W(t) - W(0)`` on ``times`` for the path with these Fourier coefficients."""
    k = np.arange(1, x.shape[0] + 1)
    scale = np.sqrt(_STEP) / (np.sqrt(2.0) * np.pi * k)
    a, b = scale[:, np.newaxis] * x, scale[:, np.newaxis] * y
    phase = 2.0 * np.pi * np.outer(times, k) / _STEP
    return (
        np.outer(times / _STEP, delta_w)
        - np.sum(a, axis=0)
        + np.cos(phase) @ a
        + np.sin(phase) @ b
    )


def _polygon_area(path: np.ndarray, j: int, k: int) -> float:
    """``(1/2) int (W_j dW_k - W_k dW_j)`` along a finely sampled curve (midpoints)."""
    steps = np.diff(path, axis=0)
    middle = 0.5 * (path[:-1] + path[1:])
    return float(
        0.5 * np.sum(middle[:, j] * steps[:, k] - middle[:, k] * steps[:, j]),
    )


class TestFourierPartAgainstAnExplicitPath:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_assemble_is_the_area_of_the_curve_its_coefficients_describe(self, seed):
        generator = np.random.default_rng(seed)
        terms, dimension = 6, 3
        x = generator.standard_normal((terms, dimension))
        y = generator.standard_normal((terms, dimension))
        delta_w = np.sqrt(_STEP) * generator.standard_normal(dimension)
        times = np.linspace(0.0, _STEP, 100001)
        path = _fourier_path(delta_w, x, y, times)
        # The curve is a Brownian bridge plus the chord: it ends at dW.
        assert np.allclose(path[-1], delta_w, atol=1e-12)
        areas = LevyAreaSampler.assemble(
            delta_w,
            _STEP,
            x,
            y,
            np.zeros(dimension),
            np.zeros((dimension, dimension)),
        )
        for j, k in [(0, 1), (0, 2), (1, 2)]:
            # The midpoint rule errs by O((K / grid)^2) relative: about 1e-8 here,
            # against areas of order 0.1.
            assert areas[j, k] == pytest.approx(_polygon_area(path, j, k), abs=1e-7)


def _riemann_areas(paths: int, cells: int, seed: int):
    """Areas of three pairs and the increments, from Ito left sums on a fine grid."""
    generator = np.random.default_rng(seed)
    areas = np.empty((paths, 3))
    increments = np.empty((paths, 3))
    chunk = 1000
    for start in range(0, paths, chunk):
        steps = generator.standard_normal((chunk, cells, 3)) * np.sqrt(_STEP / cells)
        values = np.cumsum(steps, axis=1) - steps  # W at the left end of each cell
        for column, (j, k) in enumerate([(0, 1), (0, 2), (1, 2)]):
            areas[start : start + chunk, column] = 0.5 * np.sum(
                values[:, :, j] * steps[:, :, k] - values[:, :, k] * steps[:, :, j],
                axis=1,
            )
        increments[start : start + chunk] = np.sum(steps, axis=1)
    return areas, increments


def _pair_slopes(areas: np.ndarray, increments: np.ndarray) -> tuple[float, float]:
    """Slopes of ``A12 A13`` on ``dW2 dW3`` and of ``A12 A23`` on ``dW1 dW3``."""
    shared_first = np.polyfit(
        increments[:, 1] * increments[:, 2],
        areas[:, 0] * areas[:, 1],
        1,
    )[0]
    shared_second = np.polyfit(
        increments[:, 0] * increments[:, 2],
        areas[:, 0] * areas[:, 2],
        1,
    )[0]
    return float(shared_first), float(shared_second)


class TestCouplingBetweenPairs:
    def test_brownian_reference_has_the_conditional_covariance(self):
        # Pins h dW_2 dW_3 / 12 and -h dW_1 dW_3 / 12 without the series: Ito left
        # sums on 512 cells of a simulated path. Standard error of each slope
        # about 3 per cent of h / 12 at 20000 paths.
        areas, increments = _riemann_areas(20000, 512, seed=3)
        first, second = _pair_slopes(areas, increments)
        assert first == pytest.approx(_STEP / 12, rel=0.12)
        assert second == pytest.approx(-_STEP / 12, rel=0.12)

    @pytest.mark.parametrize("terms", [1, None])
    def test_sampler_has_the_conditional_covariance_at_any_truncation(self, terms):
        # Exact at any n: the dW-linear remainder is simulated exactly, with one
        # Psi_1 per channel shared by every pair the channel is in. At h = 1/2
        # and m = 3 the default truncation is n = 1.
        generator = np.random.default_rng(5)
        delta_w = np.sqrt(_STEP) * generator.standard_normal((400000, 3))
        sampler = LevyAreaSampler(np.random.default_rng(6))
        matrix = sampler.sample(delta_w, _STEP, terms=terms)
        areas = np.stack([matrix[:, 0, 1], matrix[:, 0, 2], matrix[:, 1, 2]], axis=1)
        first, second = _pair_slopes(areas, delta_w)
        assert first == pytest.approx(_STEP / 12, rel=0.04)
        assert second == pytest.approx(-_STEP / 12, rel=0.04)


def _levy_cosine(frequency: float, squared_increment: float) -> float:
    """``E[cos(u A) | |dW|^2]`` over a step ``_STEP``, Levy (1950)."""
    b = frequency * _STEP / 2.0
    return float(
        b
        / np.sinh(b)
        * np.exp(squared_increment / (2.0 * _STEP) * (1.0 - b / np.tanh(b))),
    )


class TestConditionalLaw:
    @pytest.mark.parametrize("squared_increment", [0.0, 2.0 * _STEP])
    def test_characteristic_function_at_fixed_increments_is_levys(
        self,
        squared_increment,
    ):
        # With 16 terms the law error is far below the Monte Carlo error (7e-4
        # at 4e5 draws); at n = 1 the same check is off by 0.01 at u = 12, which
        # is the approximation's law error, not a defect: M-R is exact in its
        # first two moments only.
        draws = 400000
        delta_w = np.zeros((draws, 2))
        delta_w[:, 0] = np.sqrt(squared_increment)
        areas = LevyAreaSampler(np.random.default_rng(9)).sample(
            delta_w,
            _STEP,
            terms=16,
        )[:, 0, 1]
        for frequency in [2.0, 6.0, 12.0]:
            cosines = np.cos(frequency * areas)
            error = 4.0 * np.std(cosines) / np.sqrt(draws)
            assert np.mean(cosines) == pytest.approx(
                _levy_cosine(frequency, squared_increment),
                abs=error,
            )
