"""The stochastic Benettin estimators at a reference state that never moves.

A D-bifurcation is the sign change of the top exponent of an invariant state: an
equilibrium whose noise vanishes there, so the Euler-Maruyama step leaves it exactly
where it was and only the tangent frame moves. Every coefficient the estimators need
is then the same on every step, and they are computed once (DEQ-76). The tests pin
the saving, that the exponents are unchanged bit for bit against the estimator with
reuse switched off, and that a state that does move is still re-evaluated each step.
"""

from unittest import TestCase, mock

import numpy as np

from discrecontinual_equations.continuation import lyapunov
from discrecontinual_equations.continuation.lyapunov import (
    MatrixNoiseSystem,
    StochasticLyapunovSettings,
    StochasticSystem,
    matrix_noise_lyapunov,
    stochastic_lyapunov,
)
from discrecontinual_equations.continuation.noise import NoiseMatrix
from discrecontinual_equations.continuation.stochastic import Ito, Stratonovich
from tests.continuation.test_lyapunov_costs import CountedLinearField, _random_matrix

_DIMENSION = 3
_SHORT, _LONG = 20, 220


def _settings(horizon: int, seed: int = 5) -> StochasticLyapunovSettings:
    return StochasticLyapunovSettings(
        dt=0.01,
        horizon=horizon,
        transient=10,
        seed=seed,
    )


def _calls_per_step(estimate, field: CountedLinearField) -> float:
    """Evaluations of ``field`` per horizon step, from two horizon lengths."""
    field.calls = 0
    estimate(_SHORT)
    short = field.calls
    field.calls = 0
    estimate(_LONG)
    return (field.calls - short) / (_LONG - _SHORT)


def _matrix_system(convention) -> tuple[MatrixNoiseSystem, list[CountedLinearField]]:
    drift = CountedLinearField(_random_matrix(1, 0.5) - np.eye(_DIMENSION))
    columns = [CountedLinearField(_random_matrix(10 + k, 0.3)) for k in range(2)]
    return MatrixNoiseSystem(drift, NoiseMatrix(columns), convention), [
        drift,
        *columns,
    ]


def _scalar_system(convention) -> tuple[StochasticSystem, list[CountedLinearField]]:
    drift = CountedLinearField(_random_matrix(1, 0.5) - np.eye(_DIMENSION))
    diffusion = CountedLinearField(_random_matrix(5, 0.3))
    return StochasticSystem(drift, diffusion, convention), [drift, diffusion]


_unchanged = lyapunov._unchanged  # noqa: SLF001 (the helper under test)


def _without_reuse():
    return mock.patch.object(lyapunov, "_unchanged", return_value=False)


class _FixedStateCases:
    """Shared by the scalar and the matrix estimator; ``estimator`` picks one."""

    estimator = staticmethod(matrix_noise_lyapunov)
    build = staticmethod(_matrix_system)

    def _run(self, system, start, horizon, seed=5):
        return self.estimator(system, start, settings=_settings(horizon, seed))

    def test_an_invariant_state_evaluates_its_fields_once(self):
        # Linear drift and linear noise leave the origin fixed: f(0) = G(0) = 0.
        for convention in (Ito(), Stratonovich()):
            system, fields = self.build(convention)
            for field in fields:
                per_step = _calls_per_step(
                    lambda horizon, s=system: self._run(
                        s, np.zeros(_DIMENSION), horizon
                    ),
                    field,
                )
                assert per_step == 0.0, (convention, per_step)

    def test_reuse_changes_no_bit_of_the_exponents(self):
        for convention in (Ito(), Stratonovich()):
            for seed in (1, 2, 3):
                system, _ = self.build(convention)
                reused = self._run(system, np.zeros(_DIMENSION), 400, seed)
                with _without_reuse():
                    fresh = self._run(system, np.zeros(_DIMENSION), 400, seed)
                np.testing.assert_array_equal(reused.exponents, fresh.exponents)

    def test_a_moving_state_is_evaluated_every_step(self):
        system, fields = self.build(Ito())
        start = np.ones(_DIMENSION)
        with _without_reuse():
            expected = [
                _calls_per_step(lambda h: self._run(system, start, h), field)
                for field in fields
            ]
        measured = [
            _calls_per_step(lambda h: self._run(system, start, h), field)
            for field in fields
        ]
        assert measured == expected
        assert min(measured) > 0.0
        reused = self._run(system, start, 400)
        with _without_reuse():
            fresh = self._run(system, start, 400)
        np.testing.assert_array_equal(reused.exponents, fresh.exponents)

    def test_the_origin_estimate_is_the_linear_tangent_flow(self):
        # Under Ito at the origin the tangent flow is dY = A Y dt + sum_k dW_k B_k Y
        # with A and the B_k the fields' own matrices; the reference integrates that
        # with the exact matrices, the estimator with its forward differences, which
        # for linear fields are exact to O(1e-9).
        system, fields = self.build(Ito())
        drift, noises = fields[0].matrix, [field.matrix for field in fields[1:]]
        dt, horizon, transient, seed = 0.01, 600, 10, 9
        generator = np.random.default_rng(seed)
        draws = len(noises) if self.estimator is matrix_noise_lyapunov else None
        for _ in range(transient):
            generator.standard_normal(draws)
        frame, totals = np.eye(_DIMENSION), np.zeros(_DIMENSION)
        for step in range(horizon):
            increments = np.atleast_1d(np.sqrt(dt) * generator.standard_normal(draws))
            variation = sum(dw * b for dw, b in zip(increments, noises, strict=True))
            frame = frame + drift @ frame * dt + variation @ frame
            if (step + 1) % 10 == 0:
                frame, upper = np.linalg.qr(frame)
                totals += np.log(np.abs(np.diag(upper)))
        reference = np.sort(totals / (horizon * dt))[::-1]
        estimated = self._run(system, np.zeros(_DIMENSION), horizon, seed)
        np.testing.assert_allclose(estimated.exponents, reference, rtol=0, atol=1e-6)


class TestMatrixFixedState(_FixedStateCases, TestCase):
    """:func:`matrix_noise_lyapunov` at an invariant state."""


class TestScalarFixedState(_FixedStateCases, TestCase):
    """:func:`stochastic_lyapunov` at an invariant state."""

    estimator = staticmethod(stochastic_lyapunov)
    build = staticmethod(_scalar_system)


class TestUnchanged(TestCase):
    """The reuse test compares bits, not values."""

    def test_signed_zeros_are_different_states(self):
        assert not _unchanged(np.array([0.0]), np.array([-0.0]))

    def test_no_anchor_is_never_unchanged(self):
        assert not _unchanged(np.zeros(2), None)

    def test_equal_bits_are_unchanged(self):
        state = np.array([0.25, -1.5])
        assert _unchanged(state, state.copy())
        assert not _unchanged(state, state + 1.0e-16 * np.array([0.0, 4.0]))
