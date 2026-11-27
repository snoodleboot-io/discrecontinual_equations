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
    """One RK4 step of the state and its co-evolving tangent frame.

    Each stage point is formed once and shared by the field and its Jacobian; the
    arithmetic is otherwise the textbook step's, so the result is the same to the
    bit.
    """
    half = 0.5 * dt
    f1 = field_at(function, state)
    y1 = jacobian_at(function, state) @ frame
    second = state + half * f1
    f2 = field_at(function, second)
    y2 = jacobian_at(function, second) @ (frame + half * y1)
    third = state + half * f2
    f3 = field_at(function, third)
    y3 = jacobian_at(function, third) @ (frame + half * y2)
    fourth = state + dt * f3
    f4 = field_at(function, fourth)
    y4 = jacobian_at(function, fourth) @ (frame + dt * y3)
    new_state = state + dt / 6.0 * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
    new_frame = frame + dt / 6.0 * (y1 + 2.0 * y2 + 2.0 * y3 + y4)
    return new_state, new_frame


def _advance(function: Function, state: np.ndarray, dt: float) -> np.ndarray:
    """The state half of :func:`_step`, operation for operation, without the frame.

    The transient only has to carry the state onto the attractor; the frame starts
    afresh where it ends. Leaving out the four Jacobians makes a transient step cost
    four field evaluations instead of ``4 (n + 2)``, and because the state never
    depended on the frame it lands on the same state bit for bit.
    """
    f1 = field_at(function, state)
    f2 = field_at(function, state + 0.5 * dt * f1)
    f3 = field_at(function, state + 0.5 * dt * f2)
    f4 = field_at(function, state + dt * f3)
    return state + dt / 6.0 * (f1 + 2.0 * f2 + 2.0 * f3 + f4)


def lyapunov_spectrum(
    function: Function,
    start: np.ndarray,
    count: int | None = None,
    settings: LyapunovSettings | None = None,
) -> LyapunovSpectrum:
    """Estimate the Lyapunov spectrum of ``function`` along an orbit from ``start``.

    Complexity: ``O(T C_f + H (n C_f + n^2 k))`` for ``T`` transient and ``H``
    horizon steps and ``k`` directions: a transient step advances the state alone
    (four field evaluations), a horizon step takes ``4(n + 2)``; plus a QR every
    tenth step; ``O(n^2)`` memory.

    ``count`` tangent directions are tracked (all of them by default). The orbit is
    advanced through the transient first so it settles onto the attractor, then the
    exponents are accumulated over the horizon with periodic QR renewal. The
    reference orbit must stay bounded, as it does on any attractor.

    The tangent frame starts at the identity when the transient ends, so the
    transient advances the state alone. That gives the same limit: the exponents are
    properties of the orbit, the same for almost every initial frame (Oseledets; Skokos,
    Lect. Notes Phys. 790, 2010, Theorem 2), and where the frame starts moves only
    the ``O(1/t)`` term of the finite-time estimate (ibid., eq. 63).
    """
    control = settings if settings is not None else LyapunovSettings()
    dt = control.dt
    state = np.asarray(start, dtype=float).copy()
    dimension = state.size
    directions = dimension if count is None else count
    for _ in range(control.transient):
        state = _advance(function, state, dt)
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


def _scalar_state_drift(
    system: "StochasticSystem",
    state: np.ndarray,
    diffusion: np.ndarray,
    half: float,
    slope: np.ndarray | None = None,
) -> np.ndarray:
    """Ito-equivalent drift of the reference (Stratonovich adds g_x g / 2).

    Under Ito the shift is skipped rather than formed and multiplied by zero, and the
    diffusion Jacobian is taken only when the shift needs it and no ``slope`` is given.
    """
    drift = field_at(system.drift, state)
    if half == 0.0:
        return drift
    if slope is None:
        slope = jacobian_at(system.diffusion, state)
    return drift + half * (slope @ diffusion)


def _unchanged(state: np.ndarray, anchor: np.ndarray | None) -> bool:
    """Whether ``state`` is ``anchor`` bit for bit, so its coefficients still hold.

    The fields are evaluated with no time argument and parameters that do not change
    during an estimate, so every coefficient is a function of the state's bits alone.
    A reference state that never moves - an equilibrium whose noise vanishes there,
    the case every D-bifurcation estimate is about - then needs them only once.
    Comparing bytes rather than values keeps ``-0.0`` and ``0.0`` apart, so reuse
    cannot change a single bit of the estimate.
    """
    return anchor is not None and state.tobytes() == anchor.tobytes()


class _Coefficients:
    """Everything one horizon step needs at a state, each field evaluated once.

    A transient step needs only :func:`_scalar_state_drift`: the drift Jacobian
    feeds nothing but the tangent frame, which does not exist yet.
    """

    __slots__ = ["diffusion", "diffusion_jacobian", "reference", "variational"]

    def __init__(self, system: "StochasticSystem", state: np.ndarray) -> None:
        half = system.convention.drift_correction(1.0, 1.0)
        self.diffusion = field_at(system.diffusion, state)
        self.diffusion_jacobian = jacobian_at(system.diffusion, state)
        self.reference = _scalar_state_drift(
            system,
            state,
            self.diffusion,
            half,
            self.diffusion_jacobian,
        )
        # The variational drift, with g_x^2 / 2 added for Stratonovich only.
        self.variational = jacobian_at(system.drift, state)
        if half != 0.0:
            self.variational = self.variational + half * (
                self.diffusion_jacobian @ self.diffusion_jacobian
            )


def stochastic_lyapunov(
    system: StochasticSystem,
    start: np.ndarray,
    count: int | None = None,
    settings: StochasticLyapunovSettings | None = None,
) -> LyapunovSpectrum:
    """Estimate the Lyapunov spectrum of a scalar-noise stochastic flow.

    Complexity: per horizon step ``O(n (C_f + C_g) + n^2 k)`` plus, under
    Stratonovich only, the ``O(n^3)`` product ``g_x g_x``; a transient step
    advances the state alone, ``O(C_f + C_g)`` under Ito and ``O(C_f + n C_g)``
    under Stratonovich. ``O(n^2)`` memory.

    The ``system`` carries ``dx = f(x) dt + g(x) dW`` and its noise convention.
    Reference and tangent frame are advanced together by Euler-Maruyama with a shared
    Brownian increment; the frame is re-orthonormalised by periodic QR. For the linear
    multiplicative system ``f = a x``, ``g = s x`` the top exponent is ``a - s**2 / 2``
    (Ito) or ``a`` (Stratonovich), so noise shifts the stability boundary by
    ``s**2 / 2`` - the stochastic (D) bifurcation.

    The estimate is of the Euler-Maruyama map, not of the flow: for that linear
    system it converges to ``E log|1 + a dt + s sqrt(dt) Z| / dt`` (``Z`` standard
    normal), which is ``a - s**2 / 2`` only up to ``O(dt)``; without noise it is
    ``log(1 + a dt) / dt``, below ``a`` by about ``a**2 dt / 2``.

    The coefficients - both fields and both Jacobians - are a function of the state
    alone, so while the state has not moved they are reused rather than evaluated
    again (DEQ-76). At an equilibrium whose noise vanishes there, the reference
    state of every D-bifurcation estimate, the step leaves the state exactly where it
    was and the fields are evaluated once per estimate instead of once per step; the
    exponents are the same bit for bit.

    References:
        [HigMaoYuan2007] D. J. Higham, X. Mao and C. Yuan, "Almost sure and moment
        exponential stability in the numerical simulation of stochastic
        differential equations", SIAM J. Numer. Anal. 45(2), 592-609 (2007),
        doi:10.1137/060658138: Theorem 4.1 (the sample exponent ``a - s**2 / 2``
        of ``dx = a x dt + s x dB``), eqs. (4.3)-(4.4) (the Euler-Maruyama
        recurrence and its almost-sure growth rate).
    """
    control = settings if settings is not None else StochasticLyapunovSettings()
    dt = control.dt
    root_dt = np.sqrt(dt)
    generator = np.random.default_rng(control.seed)
    state = np.asarray(start, dtype=float).copy()
    dimension = state.size
    directions = dimension if count is None else count
    half = system.convention.drift_correction(1.0, 1.0)
    for _ in range(control.transient):
        diffusion = field_at(system.diffusion, state)
        reference = _scalar_state_drift(system, state, diffusion, half)
        increment = root_dt * generator.standard_normal()
        state = state + reference * dt + diffusion * increment
    frame = np.eye(dimension)[:, :directions]
    totals = np.zeros(directions)
    elapsed = 0.0
    coefficients: _Coefficients | None = None
    anchor: np.ndarray | None = None
    for i in range(control.horizon):
        if coefficients is None or not _unchanged(state, anchor):
            coefficients = _Coefficients(system, state)
            anchor = state
        increment = root_dt * generator.standard_normal()
        variation = coefficients.diffusion_jacobian @ frame
        frame = frame + coefficients.variational @ frame * dt + variation * increment
        state = state + coefficients.reference * dt + coefficients.diffusion * increment
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


def _matrix_state_drift(
    system: "MatrixNoiseSystem",
    state: np.ndarray,
    columns: np.ndarray,
    factor: float,
    jacobians: np.ndarray | None = None,
) -> np.ndarray:
    """Ito-equivalent drift of the reference, the matrix twin of the scalar one.

    Under Ito the ``O(K n^2)`` shift is skipped, and the column Jacobians are taken
    only when the shift needs them and none are given.
    """
    drift = field_at(system.drift, state)
    if factor == 0.0:
        return drift
    if jacobians is None:
        jacobians = system.noise.jacobians_at(state)
    return drift + state_drift_shift(jacobians, columns, factor)


class _MatrixCoefficients:
    """The Ito-equivalent drifts at one state, with each field evaluated once.

    A transient step needs only :func:`_matrix_state_drift`. Under Ito neither
    Stratonovich shift is formed, the variational one being ``O(K n^3)``.
    """

    __slots__ = ["columns", "jacobians", "reference", "variational"]

    def __init__(
        self,
        system: "MatrixNoiseSystem",
        state: np.ndarray,
        factor: float,
    ) -> None:
        self.columns = system.noise.columns_at(state)
        self.jacobians = system.noise.jacobians_at(state)
        self.reference = _matrix_state_drift(
            system,
            state,
            self.columns,
            factor,
            self.jacobians,
        )
        self.variational = jacobian_at(system.drift, state)
        if factor != 0.0:
            self.variational = self.variational + variational_drift_shift(
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

    Complexity: per horizon step ``O(K n C_g + n C_f + K n^2 + n^2 k)`` (the
    tangent noise contracts the increments first), plus the ``O(K n^3)``
    variational shift under Stratonovich only; a transient step advances the
    state alone. ``O(K n^2)`` memory for the column Jacobians.

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

    As in :func:`stochastic_lyapunov`, the coefficients are reused while the state's
    bytes are unchanged, so an estimate at an invariant state - the noisy Hopf
    film's origin under conformal noise - evaluates its fields once (DEQ-76).

    References:
        [HigMaoYuan2007] D. J. Higham, X. Mao and C. Yuan, SIAM J. Numer. Anal.
        45(2), 592-609 (2007), doi:10.1137/060658138: Theorem 4.1, the scalar
        exponent each decoupled component above attains.
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
        columns = system.noise.columns_at(state)
        reference = _matrix_state_drift(system, state, columns, factor)
        increments = root_dt * generator.standard_normal(drivers)
        state = state + reference * dt + increments @ columns
    frame = np.eye(dimension)[:, :directions]
    totals = np.zeros(directions)
    elapsed = 0.0
    coefficients: _MatrixCoefficients | None = None
    anchor: np.ndarray | None = None
    for i in range(control.horizon):
        if coefficients is None or not _unchanged(state, anchor):
            coefficients = _MatrixCoefficients(system, state, factor)
            anchor = state
        increments = root_dt * generator.standard_normal(drivers)
        # sum_k dW_k (dG_k/dx) Y, increments first: an O(K n^2) sum and one BLAS
        # product. A three-operand einsum without a contraction path runs as one
        # nested loop over all four indices, O(K n^2 k) with no BLAS.
        variation = np.tensordot(increments, coefficients.jacobians, axes=1) @ frame
        frame = frame + coefficients.variational @ frame * dt + variation
        state = state + coefficients.reference * dt + increments @ coefficients.columns
        if (i + 1) % _RENORMALISE_EVERY == 0:
            frame, upper = np.linalg.qr(frame)
            totals += np.log(np.abs(np.diag(upper)))
        elapsed += dt
    exponents = np.sort(totals / elapsed)[::-1]
    return LyapunovSpectrum(exponents)
