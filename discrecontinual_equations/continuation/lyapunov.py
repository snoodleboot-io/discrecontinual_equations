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
"""

import numpy as np

from discrecontinual_equations.continuation.stochastic import NoiseConvention
from discrecontinual_equations.function.function import Function

_STEP = 1.0e-7
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


def _field(function: Function, state: np.ndarray) -> np.ndarray:
    return np.array(function.eval(point=list(state), time=None), dtype=float)


def _jacobian(function: Function, state: np.ndarray) -> np.ndarray:
    base = _field(function, state)
    dimension = state.size
    columns = np.empty((dimension, dimension))
    for j in range(dimension):
        shifted = state.copy()
        shifted[j] += _STEP
        columns[:, j] = (_field(function, shifted) - base) / _STEP
    return columns


def _step(
    function: Function,
    state: np.ndarray,
    frame: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One RK4 step of the state and its co-evolving tangent frame."""
    f1 = _field(function, state)
    y1 = _jacobian(function, state) @ frame
    f2 = _field(function, state + 0.5 * dt * f1)
    y2 = _jacobian(function, state + 0.5 * dt * f1) @ (frame + 0.5 * dt * y1)
    f3 = _field(function, state + 0.5 * dt * f2)
    y3 = _jacobian(function, state + 0.5 * dt * f2) @ (frame + 0.5 * dt * y2)
    f4 = _field(function, state + dt * f3)
    y4 = _jacobian(function, state + dt * f3) @ (frame + dt * y3)
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
        self.drift = _field(system.drift, state)
        self.diffusion = _field(system.diffusion, state)
        self.drift_jacobian = _jacobian(system.drift, state)
        self.diffusion_jacobian = _jacobian(system.diffusion, state)

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
