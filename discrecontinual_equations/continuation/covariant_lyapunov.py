"""Covariant Lyapunov vectors, the angles between them, and finite-time exponents.

:mod:`.lyapunov` returns the Lyapunov *rates* by Benettin's QR method. The
orthonormal frame that method carries is not a set of directions of the dynamics:
the Gram-Schmidt vectors are orthogonal by construction, they depend on the
order and the inner product the orthogonalisation used, and the vector at one
point is not mapped by the tangent flow onto the vector at the image point. The
intrinsic directions - the Oseledets splitting - are the *covariant* Lyapunov
vectors: ``v_j(t)`` is carried by the tangent flow onto a multiple of
``v_j(t + tau)``, grows on average at the rate ``lambda_j``, and does not depend
on any norm. Only they can answer whether the stable and unstable directions
ever become tangent, which is what hyperbolicity is.

They are computed here by the dynamical algorithm of Ginelli, Poggi, Turchi,
Chate, Livi and Politi (Phys. Rev. Lett. 99, 130601, 2007; in the numerical
form of Ginelli, Chate, Livi and Politi, J. Phys. A 46, 254005, 2013, Sec. 4).
A forward Benettin pass is run exactly as for the exponents, but the
upper-triangular factors ``R_n`` of each QR step are kept, with the
Gram-Schmidt bases ``Q_n``. Since ``J_n Q_n = Q_{n+1} R_n``, a covariant vector
written in the Gram-Schmidt basis, ``v_n = Q_n c_n``, satisfies
``c_n = R_n^{-1} c_{n+1} gamma_n``, where ``gamma_n`` is its growth factor. Run
backward from any non-singular upper-triangular ``C``, the columns converge to
the covariant coefficients, because the backward map contracts every direction
of the ``j``-th Gram-Schmidt subspace onto its least-contracted one - the
``j``-th covariant vector (Ginelli et al. 2013, Eqs. 32-40). The backward
solve is done by triangular substitution rather than by forming ``R^{-1}``, as
Kuptsov and Parlitz recommend (J. Nonlinear Sci. 22, 2012, Eq. 66): column
``j`` then depends only on the leading ``j x j`` block, so a badly conditioned
contracting tail cannot spoil the leading vectors. The starting ``C`` is the
identity: the ``j``-th Gram-Schmidt vector is orthogonal to the first ``j - 1``
covariant vectors' span, so its component along the ``j``-th is never zero, and
a deterministic start keeps the answer reproducible.

The growth factors are themselves the second product. Their logarithms over a
window are the *finite-time* exponents (Ginelli et al. 2013, Eq. 29), and over
the whole record they average to the exponents - the "growth rate of the
``i``-th vector" of the 2007 letter, which must agree with Benettin's. On the
same window the two differ by a boundary term only: the diagonal of the
backward recursion telescopes to ``sum log gamma_j - sum log |R_jj| =
log|c_jj(0)| - log|c_jj(T)|``. The finite-time exponents are kept as a
distribution, not a limit, because at finite time that distribution is the
observable: under noise the limit can be negative while a fraction of windows
still expands, and its median is what crosses zero at the dynamical threshold.

Hyperbolicity is read off the angles. Between two vectors it is
``arccos |v_j . v_k|`` in ``[0, pi/2]`` (Ginelli et al. 2007, Eq. 4); between
the span of the first ``k`` vectors and the span of the rest it is the smallest
principal angle (Ginelli et al. 2013, Sec. 7). A distribution of the latter that
reaches zero signals tangencies; one bounded away from zero is the numerical
signature of a dominated, hyperbolic-like splitting.

The stochastic finite-time exponents call
:func:`~.lyapunov.matrix_noise_lyapunov` over one window per seed rather than
integrating a noisy tangent flow a second time here, so they inherit every
correction that estimator carries and cannot drift away from it.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.linalg import solve_triangular

from discrecontinual_equations.continuation.lyapunov import (
    _RENORMALISE_EVERY,
    MatrixNoiseSystem,
    StochasticLyapunovSettings,
    _step,
    matrix_noise_lyapunov,
)
from discrecontinual_equations.continuation.noise import field_at, jacobian_at
from discrecontinual_equations.function.function import Function

_DEFAULT_DT = 1.0e-2
_DEFAULT_TRANSIENT = 200
_DEFAULT_SAMPLES = 2000
_DEFAULT_BACKWARD_TRANSIENT = 200
_MAP_INTERVAL = 1.0
_HALF_PI = 0.5 * np.pi


class TangentDynamics(ABC):
    """One sampling interval of a reference orbit and the tangent frame along it."""

    @property
    @abstractmethod
    def interval(self) -> float:
        """Time advanced by one call to :meth:`advance`."""
        raise NotImplementedError

    @abstractmethod
    def advance(
        self,
        state: np.ndarray,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Move the state one interval on, and the frame by the tangent map."""
        raise NotImplementedError


class FlowTangent(TangentDynamics):
    """A flow ``x' = f(x)``, advanced by ``steps`` RK4 steps of ``dt`` per interval.

    The steps are those of :func:`~.lyapunov.lyapunov_spectrum`, and the default
    interval is its renormalisation interval, so the forward pass here is the
    Benettin pass that estimator runs.
    """

    __slots__ = ["_dt", "_function", "_steps"]

    def __init__(
        self,
        function: Function,
        dt: float = _DEFAULT_DT,
        steps: int = _RENORMALISE_EVERY,
    ) -> None:
        if steps < 1:
            message = "a sampling interval needs at least one step"
            raise ValueError(message)
        self._function = function
        self._dt = dt
        self._steps = steps

    @property
    def interval(self) -> float:
        return self._dt * self._steps

    def advance(
        self,
        state: np.ndarray,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(self._steps):
            state, frame = _step(self._function, state, frame, self._dt)
        return state, frame


class MapTangent(TangentDynamics):
    """An iterated map ``x -> F(x)``, one iterate per interval of unit time."""

    __slots__ = ["_function"]

    def __init__(self, function: Function) -> None:
        self._function = function

    @property
    def interval(self) -> float:
        return _MAP_INTERVAL

    def advance(
        self,
        state: np.ndarray,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        jacobian = jacobian_at(self._function, state)
        return field_at(self._function, state), jacobian @ frame


class CovariantLyapunovSettings:
    """Lengths of the four stages of the dynamical algorithm, in sampling intervals.

    ``transient`` settles the orbit on the attractor and the Gram-Schmidt frame
    on its asymptotic basis; ``samples`` is how many points the vectors are
    returned at; ``backward_transient`` is the stretch past the last sample that
    is recorded only so the backward iteration can converge before it reaches
    the samples (Ginelli et al. 2013, Sec. 4.2, stages i-iv).

    Both transients matter for the vectors, not only the backward one. The
    ``j``-th vector is found inside the span of the first ``j`` Gram-Schmidt
    vectors, so if the forward frame has not converged at a sample, the vector
    there carries the frame's error; each stage converges like
    ``exp(-gap * time)``, with ``gap`` the smallest difference between
    neighbouring exponents, and needs ``gap * time`` of twenty or more for
    vectors good to ``1e-9``.
    """

    __slots__ = ["backward_transient", "samples", "transient"]

    def __init__(
        self,
        transient: int = _DEFAULT_TRANSIENT,
        samples: int = _DEFAULT_SAMPLES,
        backward_transient: int = _DEFAULT_BACKWARD_TRANSIENT,
    ) -> None:
        if samples < 1:
            message = "at least one sample is needed"
            raise ValueError(message)
        if transient < 0 or backward_transient < 0:
            message = "transients cannot be negative"
            raise ValueError(message)
        self.transient = transient
        self.samples = samples
        self.backward_transient = backward_transient


class AngleDistribution:
    """Angles in ``[0, pi/2]`` sampled along an orbit, one per sample point."""

    __slots__ = ["_values"]

    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    @property
    def values(self) -> np.ndarray:
        """Every sampled angle, in orbit order."""
        return self._values

    @property
    def minimum(self) -> float:
        """The closest approach to tangency seen."""
        return float(np.min(self._values))

    @property
    def median(self) -> float:
        """The median angle."""
        return float(np.median(self._values))

    def quantile(self, level: float) -> float:
        """The angle below which a fraction ``level`` of samples lie."""
        return float(np.quantile(self._values, level))

    def fraction_below(self, angle: float) -> float:
        """Fraction of samples within ``angle`` of tangency."""
        return float(np.mean(self._values < angle))


class FiniteTimeExponents:
    """A distribution of exponents over windows of a fixed length.

    ``values`` has one row per window and one column per exponent, in the order
    the exponents are reported (largest first); ``window`` is the window's
    length in time.
    """

    __slots__ = ["_values", "_window"]

    def __init__(self, values: np.ndarray, window: float) -> None:
        self._values = values
        self._window = window

    @property
    def values(self) -> np.ndarray:
        """One row of exponents per window."""
        return self._values

    @property
    def window(self) -> float:
        """Length of each window in time."""
        return self._window

    @property
    def mean(self) -> np.ndarray:
        """Mean over windows; for tiling windows, the exponent over their union."""
        return np.mean(self._values, axis=0)

    @property
    def median(self) -> np.ndarray:
        """Median over windows, per exponent."""
        return np.median(self._values, axis=0)

    @property
    def spread(self) -> np.ndarray:
        """Standard deviation over windows, per exponent."""
        return np.std(self._values, axis=0)

    def quantile(self, level: float) -> np.ndarray:
        """Quantile over windows, per exponent."""
        return np.quantile(self._values, level, axis=0)

    def fraction_positive(self) -> np.ndarray:
        """Fraction of windows in which each exponent is positive."""
        return np.mean(self._values > 0.0, axis=0)


class CovariantLyapunov:
    """Covariant Lyapunov vectors along an orbit, with the spectrum and growth.

    ``vectors[n]`` holds the unit covariant vectors at ``states[n]`` as columns,
    ordered as the exponents are. ``exponents`` is Benettin's spectrum over the
    sampled stretch; ``growth`` holds the logarithm of each vector's growth
    factor over each sampling interval, from which the covariant growth rates
    and the finite-time exponents are formed.
    """

    __slots__ = ["_exponents", "_growth", "_interval", "_states", "_vectors"]

    def __init__(
        self,
        states: np.ndarray,
        vectors: np.ndarray,
        exponents: np.ndarray,
        growth: np.ndarray,
        interval: float,
    ) -> None:
        self._states = states
        self._vectors = vectors
        self._exponents = exponents
        self._growth = growth
        self._interval = interval

    @property
    def states(self) -> np.ndarray:
        """Reference states, one row per sample."""
        return self._states

    @property
    def vectors(self) -> np.ndarray:
        """Unit covariant vectors, ``(samples, dimension, count)``."""
        return self._vectors

    @property
    def interval(self) -> float:
        """Time between consecutive samples."""
        return self._interval

    @property
    def times(self) -> np.ndarray:
        """Sample times, measured from the first sample."""
        return self._interval * np.arange(self._states.shape[0])

    @property
    def exponents(self) -> np.ndarray:
        """Benettin's exponents over the sampled stretch, largest first."""
        return self._exponents

    @property
    def growth(self) -> np.ndarray:
        """``log`` of each vector's growth factor per interval, ``(samples, count)``."""
        return self._growth

    @property
    def covariant_exponents(self) -> np.ndarray:
        """The mean growth rate of each covariant vector over the sampled stretch."""
        return np.mean(self._growth, axis=0) / self._interval

    def finite_time_exponents(self, window: int) -> FiniteTimeExponents:
        """Covariant growth rates over consecutive windows of ``window`` intervals.

        The windows tile the sampled stretch from its start and any remainder is
        dropped, so their mean is the covariant exponent over the stretch they
        cover.
        """
        count = self._growth.shape[0] // window if window > 0 else 0
        if count < 1:
            message = (
                f"a window of {window} intervals does not fit in "
                f"{self._growth.shape[0]} samples"
            )
            raise ValueError(message)
        blocks = self._growth[: count * window].reshape(count, window, -1)
        return FiniteTimeExponents(
            blocks.sum(axis=1) / (window * self._interval),
            window * self._interval,
        )

    def angles(self, first: int, second: int) -> AngleDistribution:
        """``arccos |v_first . v_second|`` at every sample (Ginelli et al., Eq. 4)."""
        cosines = np.abs(
            np.einsum("ni,ni->n", self._vectors[:, :, first], self._vectors[:, :, second]),
        )
        return AngleDistribution(np.arccos(np.clip(cosines, 0.0, 1.0)))

    def splitting_angles(self, leading: int) -> AngleDistribution:
        """Smallest principal angle between the first ``leading`` vectors' span and
        the rest's.

        With ``leading`` the number of non-negative exponents this is the angle
        between the centre-unstable and the stable subspace, the quantity whose
        distribution decides hyperbolicity.
        """
        count = self._vectors.shape[2]
        if not 0 < leading < count:
            message = f"the split must leave vectors on both sides, got {leading}"
            raise ValueError(message)
        angles = np.empty(self._vectors.shape[0])
        for n, basis in enumerate(self._vectors):
            head, _ = np.linalg.qr(basis[:, :leading])
            tail, _ = np.linalg.qr(basis[:, leading:])
            largest = np.linalg.svd(head.T @ tail, compute_uv=False)[0]
            angles[n] = np.arccos(min(largest, 1.0))
        return AngleDistribution(angles)


class _ForwardRecord:
    """What the forward pass leaves for the backward one."""

    __slots__ = ["bases", "states", "uppers"]

    def __init__(self, length: int, dimension: int, count: int) -> None:
        self.states = np.empty((length, dimension))
        self.bases = np.empty((length, dimension, count))
        self.uppers = np.empty((length, count, count))


def _forward(
    dynamics: TangentDynamics,
    start: np.ndarray,
    count: int,
    control: CovariantLyapunovSettings,
) -> _ForwardRecord:
    """Benettin's pass, keeping each Gram-Schmidt basis and triangular factor.

    Record ``n`` holds the state and basis at sample ``n`` and the factor with
    ``J_n Q_n = Q_{n+1} R_n`` that carries them to sample ``n + 1``.
    """
    state = np.asarray(start, dtype=float).copy()
    dimension = state.size
    frame = np.eye(dimension)[:, :count]
    for _ in range(control.transient):
        state, frame = dynamics.advance(state, frame)
        frame, _upper = np.linalg.qr(frame)
    record = _ForwardRecord(
        control.samples + control.backward_transient,
        dimension,
        count,
    )
    for n in range(record.states.shape[0]):
        record.states[n] = state
        record.bases[n] = frame
        state, frame = dynamics.advance(state, frame)
        frame, record.uppers[n] = np.linalg.qr(frame)
    return record


def covariant_lyapunov_vectors(
    dynamics: TangentDynamics,
    start: np.ndarray,
    count: int | None = None,
    settings: CovariantLyapunovSettings | None = None,
) -> CovariantLyapunov:
    """The first ``count`` covariant Lyapunov vectors along an orbit from ``start``.

    The forward pass records the triangular factors over the samples and the
    backward transient past them; the backward pass starts from the identity at
    the far end, solves ``R_n c_n = c_{n+1}`` column by column, normalises, and
    keeps the vectors once it reaches the samples. ``count`` defaults to the
    full dimension; fewer vectors need only the leading block of every factor,
    so they cost proportionally less (Ginelli et al. 2013, Sec. 4.2).
    """
    control = settings if settings is not None else CovariantLyapunovSettings()
    dimension = np.asarray(start).size
    vectors_wanted = dimension if count is None else count
    if not 0 < vectors_wanted <= dimension:
        message = f"count must lie in [1, {dimension}], got {vectors_wanted}"
        raise ValueError(message)
    record = _forward(dynamics, start, vectors_wanted, control)
    samples = control.samples
    coefficients = np.eye(vectors_wanted)
    vectors = np.empty((samples, dimension, vectors_wanted))
    growth = np.empty((samples, vectors_wanted))
    for n in range(record.uppers.shape[0] - 1, -1, -1):
        coefficients = solve_triangular(record.uppers[n], coefficients)
        norms = np.linalg.norm(coefficients, axis=0)
        coefficients = coefficients / norms
        if n < samples:
            vectors[n] = record.bases[n] @ coefficients
            growth[n] = -np.log(norms)
    stretches = np.log(np.abs(np.diagonal(record.uppers[:samples], axis1=1, axis2=2)))
    exponents = np.mean(stretches, axis=0) / dynamics.interval
    return CovariantLyapunov(
        record.states[:samples],
        vectors,
        exponents,
        growth,
        dynamics.interval,
    )


def stochastic_finite_time_exponents(
    system: MatrixNoiseSystem,
    start: np.ndarray,
    window: int,
    windows: int,
    settings: StochasticLyapunovSettings | None = None,
) -> FiniteTimeExponents:
    """Finite-time exponents of a noisy flow over ``windows`` windows of ``window``
    steps.

    Each window is one call of :func:`~.lyapunov.matrix_noise_lyapunov` with the
    horizon set to the window and the seed advanced by one per window, so the
    windows are independent draws after the same transient, and the same
    ``settings`` give the same distribution at every parameter value - the
    common random numbers that make its median a usable test function in the
    parameter (see :mod:`.stochastic_threshold`). ``window`` must be a whole
    number of the estimator's renormalisation intervals, or the last partial
    interval would be timed but not measured.
    """
    control = settings if settings is not None else StochasticLyapunovSettings()
    if window < _RENORMALISE_EVERY or window % _RENORMALISE_EVERY != 0:
        message = (
            f"the window must be a positive multiple of {_RENORMALISE_EVERY} steps, "
            f"got {window}"
        )
        raise ValueError(message)
    if windows < 1:
        message = "at least one window is needed"
        raise ValueError(message)
    rows = []
    for index in range(windows):
        rows.append(
            matrix_noise_lyapunov(
                system,
                start,
                settings=StochasticLyapunovSettings(
                    dt=control.dt,
                    horizon=window,
                    transient=control.transient,
                    seed=control.seed + index,
                ),
            ).exponents,
        )
    return FiniteTimeExponents(np.array(rows), window * control.dt)
