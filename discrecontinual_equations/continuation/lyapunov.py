"""Numerical Lyapunov exponents of a flow, and the dynamical (D) bifurcation.

The top Lyapunov exponent measures the mean exponential rate at which nearby
trajectories separate; its sign decides whether a reference motion is stable. In one
dimension it is available in closed form (see :mod:`.stochastic`), but in higher
dimensions it must be computed from the linearised flow. This module does so with
the Benettin QR algorithm: a set of tangent vectors is transported along the orbit
under the variational equation ``Y' = J(x) Y`` and periodically re-orthonormalised
by a QR factorisation; the exponents are the time-averaged logarithms of the
diagonal stretching factors.

A sign change of the top exponent as a parameter varies is a *dynamical (D)
bifurcation* - the loss of stability of the reference motion. For a linear system
``x' = A x`` the spectrum is exactly the real parts of the eigenvalues of ``A``,
which gives a clean check on the numerics.

Two stochastic estimators live here, differing only in how many Brownian motions
drive the flow. :func:`stochastic_lyapunov` takes one shared driver, which is all the
exponent needs and is the cheapest thing to integrate;
:func:`matrix_noise_lyapunov` takes an ``N x K`` amplitude, because a system whose
*density* can be asked for must have independent drivers (see :mod:`.noise`) and it
would be no use to have the two notions of stochastic bifurcation describe two
different systems. Locating the parameter at which the top exponent changes sign is
:mod:`.stochastic_threshold`.

All three take their field and Jacobian evaluations from :mod:`.noise` rather than
keeping their own. That is not only to avoid three copies of a finite difference: the
step there is scaled by the size of the state, and an estimator that used an absolute
step would silently report a zero Jacobian once the reference orbit grew past the point
where the perturbation rounds away - which a stochastic reference orbit above its own
dynamical threshold does. Sharing the helpers means the scalar and matrix estimators
cannot disagree about the same system for a reason as arbitrary as that.
"""

import numpy as np

from discrecontinual_equations.continuation.noise import (
    NoiseMatrix,
    field_at,
    jacobian_at,
    state_drift_shift,
    stratonovich_factor,
    variational_drift_shift,
)
from discrecontinual_equations.continuation.stochastic import NoiseConvention
from discrecontinual_equations.function.function import Function

_DEFAULT_DT = 1.0e-2
_DEFAULT_HORIZON = 20000
_DEFAULT_TRANSIENT = 2000
_RENORMALISE_EVERY = 10


class LyapunovSettings:
    """Integration controls for the Benettin estimate."""

    __slots__ = ["dt", "horizon", "transient"]

    def __init__(
        self,
        dt: float = _DEFAULT_DT,
        horizon: int = _DEFAULT_HORIZON,
        transient: int = _DEFAULT_TRANSIENT,
    ) -> None:
        self.dt = dt
        self.horizon = horizon
        self.transient = transient


class LyapunovSpectrum:
    """The Lyapunov exponents of a flow, largest first."""

    __slots__ = ["_exponents"]

    def __init__(self, exponents: np.ndarray) -> None:
        self._exponents = exponents

    @property
    def exponents(self) -> np.ndarray:
        """All computed exponents in descending order."""
        return self._exponents

    @property
    def top(self) -> float:
        """The largest Lyapunov exponent."""
        return float(self._exponents[0])


def _step(
    function: Function,
    state: np.ndarray,
    frame: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One RK4 step of the state and its co-evolving tangent frame."""
    f1 = field_at(function, state)
    y1 = jacobian_at(function, state) @ frame
    f2 = field_at(function, state + 0.5 * dt * f1)
    y2 = jacobian_at(function, state + 0.5 * dt * f1) @ (frame + 0.5 * dt * y1)
    f3 = field_at(function, state + 0.5 * dt * f2)
    y3 = jacobian_at(function, state + 0.5 * dt * f2) @ (frame + 0.5 * dt * y2)
    f4 = field_at(function, state + dt * f3)
    y4 = jacobian_at(function, state + dt * f3) @ (frame + dt * y3)
    new_state = state + dt / 6.0 * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
    new_frame = frame + dt / 6.0 * (y1 + 2.0 * y2 + 2.0 * y3 + y4)
    return new_state, new_frame


def lyapunov_spectrum(
    function: Function,
    start: np.ndarray,
    count: int | None = None,
    settings: LyapunovSettings | None = None,
) -> LyapunovSpectrum:
    """Estimate the Lyapunov spectrum of ``function`` along an orbit from ``start``.

    ``count`` tangent directions are tracked (all of them by default). The orbit is
    advanced through the transient first so it settles onto the attractor, then the
    exponents are accumulated over the horizon with periodic QR renewal. The
    reference orbit must stay bounded, as it does on any attractor.
    """
    control = settings if settings is not None else LyapunovSettings()
    dt = control.dt
    state = np.asarray(start, dtype=float).copy()
    dimension = state.size
    directions = dimension if count is None else count
    for _ in range(control.transient):
        state, _frame = _step(function, state, np.eye(dimension)[:, :directions], dt)
    frame = np.eye(dimension)[:, :directions]
    totals = np.zeros(directions)
    elapsed = 0.0
    for i in range(control.horizon):
        state, frame = _step(function, state, frame, dt)
        if (i + 1) % _RENORMALISE_EVERY == 0:
            frame, upper = np.linalg.qr(frame)
            totals += np.log(np.abs(np.diag(upper)))
        elapsed += dt
    exponents = np.sort(totals / elapsed)[::-1]
    return LyapunovSpectrum(exponents)


_DEFAULT_SEED = 0


class StochasticLyapunovSettings:
    """Integration controls for the stochastic Benettin estimate."""

    __slots__ = ["dt", "horizon", "seed", "transient"]

    def __init__(
        self,
        dt: float = _DEFAULT_DT,
        horizon: int = _DEFAULT_HORIZON,
        transient: int = _DEFAULT_TRANSIENT,
        seed: int = _DEFAULT_SEED,
    ) -> None:
        self.dt = dt
        self.horizon = horizon
        self.transient = transient
        self.seed = seed


class StochasticSystem:
    """A scalar-noise stochastic flow ``dx = f(x) dt + g(x) dW`` with a convention."""

    __slots__ = ["convention", "diffusion", "drift"]

    def __init__(
        self,
        drift: Function,
        diffusion: Function,
        convention: NoiseConvention,
    ) -> None:
        self.drift = drift
        self.diffusion = diffusion
        self.convention = convention


class _Coefficients:
    """Drift/diffusion values and Jacobians at a state, plus convention factor."""

    __slots__ = ["diffusion", "diffusion_jacobian", "drift", "drift_jacobian", "half"]

    def __init__(self, system: "StochasticSystem", state: np.ndarray) -> None:
        self.half = system.convention.drift_correction(1.0, 1.0)
        self.drift = field_at(system.drift, state)
        self.diffusion = field_at(system.diffusion, state)
        self.drift_jacobian = jacobian_at(system.drift, state)
        self.diffusion_jacobian = jacobian_at(system.diffusion, state)

    def reference_drift(self) -> np.ndarray:
        """Ito-equivalent drift of the reference (Stratonovich adds g_x g / 2)."""
        return self.drift + self.half * (self.diffusion_jacobian @ self.diffusion)

    def variational_drift(self) -> np.ndarray:
        """Ito-equivalent variational drift matrix (adds g_x^2 / 2 for Stratonovich)."""
        return self.drift_jacobian + self.half * (
            self.diffusion_jacobian @ self.diffusion_jacobian
        )


def stochastic_lyapunov(
    system: StochasticSystem,
    start: np.ndarray,
    count: int | None = None,
    settings: StochasticLyapunovSettings | None = None,
) -> LyapunovSpectrum:
    """Estimate the Lyapunov spectrum of a scalar-noise stochastic flow.

    The ``system`` carries ``dx = f(x) dt + g(x) dW`` and its noise convention.
    Reference and tangent frame are advanced together by Euler-Maruyama with a shared
    Brownian increment; the frame is re-orthonormalised by periodic QR. For the linear
    multiplicative system ``f = a x``, ``g = s x`` the top exponent is ``a - s**2 / 2``
    (Ito) or ``a`` (Stratonovich), so noise shifts the stability boundary by
    ``s**2 / 2`` - the stochastic (D) bifurcation.
    """
    control = settings if settings is not None else StochasticLyapunovSettings()
    dt = control.dt
    root_dt = np.sqrt(dt)
    generator = np.random.default_rng(control.seed)
    state = np.asarray(start, dtype=float).copy()
    dimension = state.size
    directions = dimension if count is None else count
    for _ in range(control.transient):
        coefficients = _Coefficients(system, state)
        increment = root_dt * generator.standard_normal()
        state = (
            state
            + coefficients.reference_drift() * dt
            + coefficients.diffusion * increment
        )
    frame = np.eye(dimension)[:, :directions]
    totals = np.zeros(directions)
    elapsed = 0.0
    for i in range(control.horizon):
        coefficients = _Coefficients(system, state)
        increment = root_dt * generator.standard_normal()
        variation = coefficients.diffusion_jacobian @ frame
        frame = (
            frame
            + coefficients.variational_drift() @ frame * dt
            + variation * increment
        )
        state = (
            state
            + coefficients.reference_drift() * dt
            + coefficients.diffusion * increment
        )
        if (i + 1) % _RENORMALISE_EVERY == 0:
            frame, upper = np.linalg.qr(frame)
            totals += np.log(np.abs(np.diag(upper)))
        elapsed += dt
    exponents = np.sort(totals / elapsed)[::-1]
    return LyapunovSpectrum(exponents)


class MatrixNoiseSystem:
    """A flow ``dx = f(x) dt + G(x) dW`` with ``K`` independent Brownian drivers."""

    __slots__ = ["convention", "drift", "noise"]

    def __init__(
        self,
        drift: Function,
        noise: NoiseMatrix,
        convention: NoiseConvention,
    ) -> None:
        self.drift = drift
        self.noise = noise
        self.convention = convention


class _MatrixCoefficients:
    """The Ito-equivalent drifts at one state, with each field evaluated once."""

    __slots__ = ["columns", "jacobians", "reference", "variational"]

    def __init__(
        self,
        system: "MatrixNoiseSystem",
        state: np.ndarray,
        factor: float,
    ) -> None:
        self.columns = system.noise.columns_at(state)
        self.jacobians = system.noise.jacobians_at(state)
        self.reference = field_at(system.drift, state) + state_drift_shift(
            self.jacobians,
            self.columns,
            factor,
        )
        self.variational = jacobian_at(system.drift, state) + variational_drift_shift(
            self.jacobians,
            factor,
        )


def matrix_noise_lyapunov(
    system: MatrixNoiseSystem,
    start: np.ndarray,
    count: int | None = None,
    settings: StochasticLyapunovSettings | None = None,
) -> LyapunovSpectrum:
    """Estimate the Lyapunov spectrum of a flow driven by ``K`` independent noises.

    This is :func:`stochastic_lyapunov` with the single Brownian increment replaced by
    one per driver, and it exists because the scalar-noise estimator and the planar
    stationary density cannot be asked about the same system: the noise that makes a
    density possible has independent drivers, and the noise the scalar estimator
    accepts has exactly one (see :mod:`.noise`). A one-column
    :class:`~.noise.NoiseMatrix` reproduces :func:`stochastic_lyapunov` increment for
    increment, so nothing is forked - the scalar case is recovered, not reimplemented.

    Two sanity points the tests lean on. With additive noise the amplitude has no
    Jacobian, so the tangent equation carries no noise at all and the spectrum of a
    linear system is exactly the real parts of its eigenvalues, with no Monte-Carlo
    error. With noise ``G = diag(s_i x_i)`` the components decouple into geometric
    Brownian motions and the spectrum is ``{a_i - s_i^2 / 2}`` (Ito) exactly.
    """
    control = settings if settings is not None else StochasticLyapunovSettings()
    dt = control.dt
    root_dt = np.sqrt(dt)
    factor = stratonovich_factor(system.convention)
    generator = np.random.default_rng(control.seed)
    drivers = system.noise.drivers
    state = np.asarray(start, dtype=float).copy()
    dimension = state.size
    directions = dimension if count is None else count
    for _ in range(control.transient):
        coefficients = _MatrixCoefficients(system, state, factor)
        increments = root_dt * generator.standard_normal(drivers)
        state = state + coefficients.reference * dt + increments @ coefficients.columns
    frame = np.eye(dimension)[:, :directions]
    totals = np.zeros(directions)
    elapsed = 0.0
    for i in range(control.horizon):
        coefficients = _MatrixCoefficients(system, state, factor)
        increments = root_dt * generator.standard_normal(drivers)
        variation = np.einsum(
            "k,kij,jl->il",
            increments,
            coefficients.jacobians,
            frame,
        )
        frame = frame + coefficients.variational @ frame * dt + variation
        state = state + coefficients.reference * dt + increments @ coefficients.columns
        if (i + 1) % _RENORMALISE_EVERY == 0:
            frame, upper = np.linalg.qr(frame)
            totals += np.log(np.abs(np.diag(upper)))
        elapsed += dt
    exponents = np.sort(totals / elapsed)[::-1]
    return LyapunovSpectrum(exponents)
