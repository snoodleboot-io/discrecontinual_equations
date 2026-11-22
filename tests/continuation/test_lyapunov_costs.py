"""What the Benettin estimators spend, and that spending less changes no exponent.

Three kinds of waste are pinned here. The transient only has to carry the state onto
the attractor, so it must not evaluate a Jacobian it then throws away. Under Ito the
convention adds nothing to either drift, so the estimators must not form the
Stratonovich sums only to multiply them by zero. And the tangent noise term must be
contracted increments-first, which is a BLAS product, rather than as one nested loop
over four indices. Each test checks the saving and, against a reference loop written
out here, that the exponents did not move.
"""

from unittest import TestCase, mock

import numpy as np

from discrecontinual_equations.continuation import lyapunov
from discrecontinual_equations.continuation.lyapunov import (
    LyapunovSettings,
    MatrixNoiseSystem,
    StochasticLyapunovSettings,
    StochasticSystem,
    lyapunov_spectrum,
    matrix_noise_lyapunov,
    stochastic_lyapunov,
)
from discrecontinual_equations.continuation.noise import (
    NoiseMatrix,
    field_at,
    jacobian_at,
)
from discrecontinual_equations.continuation.stochastic import Ito, Stratonovich
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import LorenzField, State

_DIMENSION = 3
_RENORMALISE_EVERY = 10
_RK4_STAGES = 4


class CountedLinearField(DeterministicFunction):
    """``x' = A x`` that counts its own evaluations."""

    def __init__(self, matrix: np.ndarray) -> None:
        size = matrix.shape[0]
        super().__init__(
            variables=[State() for _ in range(size)],
            parameters=[],
            results=[State() for _ in range(size)],
            time=None,
        )
        self.matrix = matrix
        self.calls = 0

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        self.calls += 1
        return list(self.matrix @ np.asarray(point, dtype=float))


def _random_matrix(seed: int, scale: float, size: int = _DIMENSION) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return scale * generator.standard_normal((size, size))


def _drift() -> CountedLinearField:
    return CountedLinearField(_random_matrix(1, 0.5) - np.eye(_DIMENSION))


def _columns(drivers: int) -> list[CountedLinearField]:
    return [CountedLinearField(_random_matrix(10 + k, 0.3)) for k in range(drivers)]


def _lorenz() -> LorenzField:
    return LorenzField(
        variables=[State(), State(), State()],
        parameters=[],
        results=[State(), State(), State()],
        time=None,
    )


def _rk4(function, state: np.ndarray, dt: float) -> np.ndarray:
    """Classic RK4 of the state alone, written out independently of the module."""
    f1 = field_at(function, state)
    f2 = field_at(function, state + 0.5 * dt * f1)
    f3 = field_at(function, state + 0.5 * dt * f2)
    f4 = field_at(function, state + dt * f3)
    return state + dt / 6.0 * (f1 + 2.0 * f2 + 2.0 * f3 + f4)


def _accumulate(frame: np.ndarray, totals: np.ndarray, step: int) -> np.ndarray:
    if (step + 1) % _RENORMALISE_EVERY == 0:
        frame, upper = np.linalg.qr(frame)
        totals += np.log(np.abs(np.diag(upper)))
    return frame


def _transient_calls(estimate, function, short: int, long: int) -> int:
    """Evaluations of ``function`` per transient step, from two transient lengths."""
    function.calls = 0
    estimate(short)
    before = function.calls
    function.calls = 0
    estimate(long)
    return (function.calls - before) // (long - short)


class TestDeterministicTransient(TestCase):
    """The transient of :func:`lyapunov_spectrum` carries the state and nothing else."""

    def test_the_transient_costs_four_field_evaluations_a_step(self):
        field = _drift()

        def estimate(transient):
            lyapunov_spectrum(
                field,
                np.ones(_DIMENSION),
                settings=LyapunovSettings(horizon=20, transient=transient),
            )

        assert _transient_calls(estimate, field, 5, 25) == _RK4_STAGES

    def test_a_transient_is_the_same_as_starting_where_it_ends(self):
        # The frame is reset to the identity when the transient ends, so the spectrum
        # after T transient steps from x0 is, bit for bit, the spectrum with no
        # transient from the state RK4 reaches after T steps.
        dt, transient = 0.01, 300
        settings = LyapunovSettings(dt=dt, horizon=2000, transient=transient)
        start = np.array([1.0, 1.0, 1.0])
        settled = start.copy()
        for _ in range(transient):
            settled = _rk4(_lorenz(), settled, dt)
        with_transient = lyapunov_spectrum(_lorenz(), start, settings=settings)
        from_settled = lyapunov_spectrum(
            _lorenz(),
            settled,
            settings=LyapunovSettings(dt=dt, horizon=2000, transient=0),
        )
        np.testing.assert_array_equal(
            with_transient.exponents,
            from_settled.exponents,
        )


class TestStochasticCosts(TestCase):
    """:func:`stochastic_lyapunov` evaluates only what each phase and convention use."""

    def _estimate(self, system, transient, horizon=20):
        return stochastic_lyapunov(
            system,
            np.ones(_DIMENSION),
            settings=StochasticLyapunovSettings(
                dt=0.01,
                horizon=horizon,
                transient=transient,
                seed=3,
            ),
        )

    def test_the_ito_transient_takes_no_jacobian(self):
        drift, diffusion = _drift(), CountedLinearField(_random_matrix(5, 0.3))
        system = StochasticSystem(drift, diffusion, Ito())

        def estimate(transient):
            self._estimate(system, transient)

        assert _transient_calls(estimate, drift, 5, 25) == 1
        assert _transient_calls(estimate, diffusion, 5, 25) == 1

    def test_the_stratonovich_transient_takes_only_the_diffusion_jacobian(self):
        drift, diffusion = _drift(), CountedLinearField(_random_matrix(5, 0.3))
        system = StochasticSystem(drift, diffusion, Stratonovich())

        def estimate(transient):
            self._estimate(system, transient)

        assert _transient_calls(estimate, drift, 5, 25) == 1
        assert _transient_calls(estimate, diffusion, 5, 25) == 1 + _DIMENSION + 1

    def test_ito_is_the_uncorrected_euler_maruyama_tangent_flow(self):
        # Under Ito both drifts are used as given; the reference below never forms
        # the Stratonovich terms at all, and must agree exactly.
        drift, diffusion = _drift(), CountedLinearField(_random_matrix(5, 0.3))
        dt, horizon, transient, seed = 0.01, 400, 50, 3
        generator = np.random.default_rng(seed)
        state = np.ones(_DIMENSION)
        for _ in range(transient):
            increment = np.sqrt(dt) * generator.standard_normal()
            state = (
                state
                + field_at(drift, state) * dt
                + field_at(diffusion, state) * increment
            )
        frame, totals = np.eye(_DIMENSION), np.zeros(_DIMENSION)
        for i in range(horizon):
            increment = np.sqrt(dt) * generator.standard_normal()
            variation = jacobian_at(diffusion, state) @ frame
            frame = (
                frame + jacobian_at(drift, state) @ frame * dt + variation * increment
            )
            state = (
                state
                + field_at(drift, state) * dt
                + field_at(diffusion, state) * increment
            )
            frame = _accumulate(frame, totals, i)
        reference = np.sort(totals / (horizon * dt))[::-1]
        estimated = self._estimate(
            StochasticSystem(drift, diffusion, Ito()),
            transient,
            horizon,
        )
        np.testing.assert_allclose(estimated.exponents, reference, rtol=0, atol=1e-12)


class TestMatrixNoiseCosts(TestCase):
    """:func:`matrix_noise_lyapunov`: no Ito work, no transient Jacobians, BLAS noise."""

    def _estimate(self, system, transient, horizon=20, seed=7):
        return matrix_noise_lyapunov(
            system,
            np.ones(_DIMENSION),
            settings=StochasticLyapunovSettings(
                dt=0.01,
                horizon=horizon,
                transient=transient,
                seed=seed,
            ),
        )

    def test_the_ito_transient_takes_no_jacobian(self):
        drift, columns = _drift(), _columns(2)
        system = MatrixNoiseSystem(drift, NoiseMatrix(columns), Ito())

        def estimate(transient):
            self._estimate(system, transient)

        assert _transient_calls(estimate, drift, 5, 25) == 1
        assert _transient_calls(estimate, columns[0], 5, 25) == 1

    def test_the_stratonovich_transient_takes_only_the_column_jacobians(self):
        drift, columns = _drift(), _columns(2)
        system = MatrixNoiseSystem(drift, NoiseMatrix(columns), Stratonovich())

        def estimate(transient):
            self._estimate(system, transient)

        assert _transient_calls(estimate, drift, 5, 25) == 1
        assert _transient_calls(estimate, columns[1], 5, 25) == 1 + _DIMENSION + 1

    def test_ito_never_forms_the_stratonovich_shifts(self):
        system = MatrixNoiseSystem(_drift(), NoiseMatrix(_columns(2)), Ito())
        with (
            mock.patch.object(lyapunov, "state_drift_shift") as state_shift,
            mock.patch.object(lyapunov, "variational_drift_shift") as tangent_shift,
        ):
            self._estimate(system, 10)
        state_shift.assert_not_called()
        tangent_shift.assert_not_called()

    def test_stratonovich_still_forms_both_shifts(self):
        system = MatrixNoiseSystem(_drift(), NoiseMatrix(_columns(2)), Stratonovich())
        with (
            mock.patch.object(
                lyapunov,
                "state_drift_shift",
                wraps=lyapunov.state_drift_shift,
            ) as state_shift,
            mock.patch.object(
                lyapunov,
                "variational_drift_shift",
                wraps=lyapunov.variational_drift_shift,
            ) as tangent_shift,
        ):
            self._estimate(system, 10)
        assert state_shift.call_count == 10 + 20
        assert tangent_shift.call_count == 20

    def test_the_tangent_noise_matches_the_four_index_sum(self):
        # The reference is the textbook sum over drivers and the inner index,
        # sum_k dW_k sum_j (dG_k/dx)_ij Y_jl, evaluated by the nested-loop einsum the
        # estimator used to call. Contracting the increments first changes only the
        # order of the floating-point sum.
        for convention in (Ito(), Stratonovich()):
            drift, columns = _drift(), _columns(3)
            noise = NoiseMatrix(columns)
            factor = convention.drift_correction(1.0, 1.0)
            dt, horizon, seed = 0.01, 400, 11
            generator = np.random.default_rng(seed)
            state = np.ones(_DIMENSION)
            frame, totals = np.eye(_DIMENSION), np.zeros(_DIMENSION)
            for i in range(horizon):
                values = noise.columns_at(state)
                slopes = noise.jacobians_at(state)
                increments = np.sqrt(dt) * generator.standard_normal(len(columns))
                variational = jacobian_at(drift, state) + factor * np.einsum(
                    "kij,kjl->il",
                    slopes,
                    slopes,
                )
                variation = np.einsum("k,kij,jl->il", increments, slopes, frame)
                frame = frame + variational @ frame * dt + variation
                reference_drift = field_at(drift, state) + factor * np.einsum(
                    "kij,kj->i",
                    slopes,
                    values,
                )
                state = state + reference_drift * dt + increments @ values
                frame = _accumulate(frame, totals, i)
            reference = np.sort(totals / (horizon * dt))[::-1]
            estimated = self._estimate(
                MatrixNoiseSystem(drift, noise, convention),
                0,
                horizon,
                seed,
            )
            np.testing.assert_allclose(
                estimated.exponents,
                reference,
                rtol=0,
                atol=1e-12,
            )
