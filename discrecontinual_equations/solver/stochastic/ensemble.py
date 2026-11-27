"""Many sample paths of one equation, stepped together.

The per-path solvers in this package step one state at a time and append each
step to a :class:`~discrecontinual_equations.curve.Curve`; one path of 50,000
Euler-Maruyama steps costs about four seconds, almost all of it Python overhead
per step. Every ensemble statistic - a stationary covariance, an escape rate, a
residence-time density - wants thousands of paths, which that cost puts out of
reach. This module steps a ``(paths, n)`` array at once with ``(paths, m)``
Wiener increments, so the per-step cost is a few array operations however many
paths there are, provided the equation evaluates its drift and diffusion as
arrays too (``eval_batch`` and ``diffusion_batch`` on
:class:`~discrecontinual_equations.function.stochastic.StochasticFunction`; the
defaults loop, which still works, only slowly).

Two schemes are offered, the same two that take matrix noise: Euler-Maruyama,
and Milstein, which for noise that does not commute adds the per-path solver's
Levy-area term ``sum_{j,k} (dG_k/dx) G_j A_(j,k)`` with the areas of every path
drawn at once (DEQ-65; before, such noise was refused).
The step formulas are the per-path solvers' with one leading axis, and the
algebra is shared in spirit rather than in code so that each solver file stays
readable against its derivation; the test suite pins the two to each other
instead. With a single path and a shared :class:`~.wiener.WienerSource`, the
Euler-Maruyama ensemble reproduces the per-path solver's path hex for hex.

What this is not: a replacement for the per-path solvers, which keep their
``Curve`` output and their place in the plotting code; nor a Wiener source for
convergence studies, since an ensemble's increments come from its own generator
unless a source is handed in, in which case the source is asked once per path
and step and the ensemble is only as fast as that source.
"""

import warnings
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.finite_difference import CENTRAL_STEP, step_scales
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.solver.solver_config import StochasticConfig
from discrecontinual_equations.solver.stochastic.coefficients import (
    coupled,
    warn_wrong_declaration,
)
from discrecontinual_equations.solver.stochastic.wiener import (
    LevyAreaSampler,
    WienerSource,
)

_MATRIX_BATCH = 3
# Commutativity is judged by coefficients.coupled, with the per-path solver's
# tolerance, relative to the largest product over the ensemble.


@dataclass(frozen=True)
class EnsembleResult:
    """What an ensemble run returns."""

    final: np.ndarray
    """The states at the end time, ``(paths, n)``."""

    times: np.ndarray
    """The times at which ``samples`` were kept (empty when none were)."""

    samples: np.ndarray
    """The kept states, ``(len(times), paths, n)``; empty when none were kept."""


class BatchedCoefficients:
    """The Ito drift and diffusion of an equation at many points at once.

    The batched twin of :class:`~.coefficients.ItoCoefficients`: the same
    conventions (the Stratonovich correction is added, the derivative is a
    central difference with the same step), applied to a ``(paths, n)`` array
    of points so that the ensemble's per-step cost does not grow with the paths
    when the equation's batch methods are arrays. The step is relative, as
    there: ``slope_step * max(|x_j|, typical_j)`` for each path and component,
    each quotient divided by the width actually stepped, so that a single path
    differences exactly as the per-path solver does (DEQ-118).

    References:
        Dennis and Schnabel (1983), Sec. 5.4 and Algorithm A5.4.1; SciPy
        1.14.1, ``scipy/optimize/_numdiff.py``. See
        :class:`~.coefficients.ItoCoefficients` and ``docs/references.md``.
    """

    __slots__ = [
        "_checked",
        "_declared",
        "_equation",
        "_slope_step",
        "_stratonovich",
        "_typical",
    ]

    def __init__(
        self,
        equation: DifferentialEquation,
        calculus: str,
        slope_step: float = CENTRAL_STEP,
        typical: float | np.ndarray | None = None,
    ) -> None:
        self._equation = equation
        self._stratonovich = calculus == "stratonovich"
        self._slope_step = slope_step
        self._typical = typical
        derivative = equation.derivative
        self._declared = (
            isinstance(derivative, StochasticFunction) and derivative.diagonal_noise
        )
        self._checked = False

    @property
    def declared_diagonal(self) -> bool:
        """Whether the equation promises ``b_i`` depends on ``x_i`` alone.

        False once :meth:`confirm_declaration` has found the promise broken.
        """
        return self._declared

    def confirm_declaration(self, points: np.ndarray, time: float) -> None:
        """Check a ``diagonal_noise`` declaration once, at the first points used.

        :meth:`~.coefficients.ItoCoefficients.confirm_declaration` over every
        path at once: ``1 + 2 n`` batched evaluations once per solve, one
        direction held at a time. A broken promise is withdrawn with a warning
        and the solve goes on as for undeclared noise, bit for bit; a coupling
        that vanishes at every starting point is not caught (DEQ-48).
        """
        if not self._declared or self._checked:
            return
        self._checked = True
        if self.is_matrix(self.diffusion(points, time)):
            return
        scale = 0.0
        cross = 0.0
        for j, slope in self.diagonal_directions(points, time):
            magnitude = np.abs(slope)
            scale = float(np.max([scale, np.max(magnitude)]))
            magnitude[:, j] = 0.0
            cross = float(np.max([cross, np.max(magnitude)]))
        if coupled(cross, scale):
            self._declared = False
            warn_wrong_declaration(self._equation.derivative, cross, scale)

    def diffusion(self, points: np.ndarray, time: float) -> np.ndarray:
        """``b`` at every point, ``(paths, n)``, or ``G``, ``(paths, n, m)``."""
        return np.asarray(
            self._equation.derivative.diffusion_batch(points, time),
            dtype=float,
        )

    @staticmethod
    def is_matrix(diffusion: np.ndarray) -> bool:
        """Whether a value of :meth:`diffusion` is the matrix form."""
        return diffusion.ndim == _MATRIX_BATCH

    def drivers(self, points: np.ndarray, time: float) -> int:
        """How many independent Wiener processes drive the equation: ``m``."""
        diffusion = self.diffusion(points[:1], time)
        return diffusion.shape[2] if self.is_matrix(diffusion) else diffusion.shape[1]

    @staticmethod
    def spread(diffusion: np.ndarray, increments: np.ndarray) -> np.ndarray:
        """What the increments do to every path: ``b * dW`` or ``G @ dW``."""
        if BatchedCoefficients.is_matrix(diffusion):
            # An einsum, not a batched matmul: three times faster on the small
            # blocks an SDE has, at the price of a last-bit difference from the
            # per-path solver's ``G @ dW`` (its sum may round once differently).
            # The vector form below is the per-path arithmetic exactly.
            return np.einsum("pim,pm->pi", diffusion, increments)
        return diffusion * increments

    def written_drift(self, points: np.ndarray, time: float) -> np.ndarray:
        """``a`` at every point as the equation wrote it, ``(paths, n)``."""
        return np.asarray(
            self._equation.derivative.eval_batch(points, time),
            dtype=float,
        )

    def drift(self, points: np.ndarray, time: float) -> np.ndarray:
        """``a`` at every point in the Ito sense, ``(paths, n)``."""
        drift = self.written_drift(points, time)
        if not self._stratonovich:
            return drift
        diffusion = self.diffusion(points, time)
        if self.is_matrix(diffusion):
            jacobians = self.column_jacobians(points, time)
            return drift + 0.5 * np.einsum("pkij,pjk->pi", jacobians, diffusion)
        # Diagonal noise: (1/2) b_i db_i/dx_i, which is the (i, i, i) entries of the
        # column Jacobians and needs only them, (paths, n) not (paths, n, n, n).
        return drift + 0.5 * diffusion * self.diagonal_slopes(points, time)

    def diagonal_slopes(self, points: np.ndarray, time: float) -> np.ndarray:
        """``d b_i / d x_i`` of a vector diffusion at every point, ``(paths, n)``.

        The same central differences as :meth:`column_jacobians`, value for
        value, keeping only the entries the Stratonovich correction of diagonal
        noise reads: ``O(paths n)`` memory where the Jacobians take ``O(paths
        n^3)`` (DEQ-48). Still ``2 n`` batched evaluations, since ``b_i`` may
        depend on every component and only ``x_i`` may be moved to read
        ``d b_i / d x_i``; two when the equation declares diagonal noise, which
        moves every component at once. The declaration is checked at the first
        points it is used.
        """
        self.confirm_declaration(points, time)
        if self._declared:
            steps = self.steps(points)
            forward = points + steps
            backward = points - steps
            return (self.diffusion(forward, time) - self.diffusion(backward, time)) / (
                forward - backward
            )
        slopes = np.empty(points.shape)
        for j in range(points.shape[1]):
            slopes[:, j] = self._difference(points, time, j)[:, j]
        return slopes

    def diagonal_directions(
        self,
        points: np.ndarray,
        time: float,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """``(j, d b / d x_j)`` for every direction of a vector diffusion.

        One ``(paths, n)`` difference at a time, the values of
        :meth:`column_jacobians` without the array that holds them all.
        """
        for j in range(points.shape[1]):
            yield j, self._difference(points, time, j)

    def steps(self, points: np.ndarray) -> np.ndarray:
        """The step of every path and component, ``(paths, n)``."""
        return self._slope_step * step_scales(points, self._typical)

    def _difference(self, points: np.ndarray, time: float, j: int) -> np.ndarray:
        """The central difference of the diffusion in direction ``j``.

        Divided, path by path, by the width the machine actually stepped.
        """
        step = self.steps(points)[:, j]
        forward = points.copy()
        backward = points.copy()
        forward[:, j] += step
        backward[:, j] -= step
        width = forward[:, j] - backward[:, j]
        difference = self.diffusion(forward, time) - self.diffusion(backward, time)
        return difference / width.reshape((-1,) + (1,) * (difference.ndim - 1))

    def column_jacobians(self, points: np.ndarray, time: float) -> np.ndarray:
        """``dG_k / dx`` for every path and driver, ``(paths, m, n, n)``.

        Central differences in every state direction, ``2 n`` batched diffusion
        evaluations whatever the number of paths. For the vector form column
        ``k`` is ``b_k e_k``, so its Jacobian has one nonzero row.
        """
        paths, dimension = points.shape
        diffusion = self.diffusion(points, time)
        matrix = self.is_matrix(diffusion)
        drivers = diffusion.shape[2] if matrix else diffusion.shape[1]
        jacobians = np.zeros((paths, drivers, dimension, dimension))
        for j in range(dimension):
            difference = self._difference(points, time, j)
            if matrix:
                jacobians[:, :, :, j] = np.transpose(difference, (0, 2, 1))
            else:
                index = np.arange(drivers)
                jacobians[:, index, index, j] = difference
        return jacobians


class StochasticEnsemble:
    """Step many paths of one stochastic equation at once.

    ``scheme`` is ``"euler_maruyama"`` or ``"milstein"``. The increments come
    from a generator this ensemble owns, seeded from the configuration, unless a
    :class:`~.wiener.WienerSource` is given, which is then asked once per path
    and step - the route by which one member of an ensemble is driven by the
    same Brownian path as a per-path solver.

    Milstein with noise that does not commute needs the Levy areas ``A_(j,k)``
    between the drivers of every path (DEQ-65). Without a source they are drawn
    for all paths in one call of :class:`~.wiener.LevyAreaSampler`, given the
    ``(paths, m)`` increments, on a stream spawned from ``random_seed`` as the
    per-path solver spawns its own, so that the increments a seeded ensemble
    draws do not change; they take ``O(paths m^2)`` memory, besides the
    sampler's ``2 paths m n`` Fourier normals. With a source, each path asks it
    for its areas with its increments, and the step decides beforehand whether
    to ask, as the per-path solver does.
    """

    __slots__ = ["_areas", "_config", "_generator", "_scheme", "_warned", "_wiener"]

    def __init__(
        self,
        config: StochasticConfig,
        scheme: str = "euler_maruyama",
        wiener: WienerSource | None = None,
    ) -> None:
        if scheme not in ("euler_maruyama", "milstein"):
            message = f"unknown ensemble scheme {scheme!r}"
            raise ValueError(message)
        self._config = config
        self._scheme = scheme
        self._wiener = wiener
        self._generator = np.random.default_rng(config.random_seed)
        # The Levy areas on a stream of their own, spawned from the seed as
        # MilsteinSolver spawns its, so drawing them leaves the increments alone.
        sequence = np.random.SeedSequence(config.random_seed)
        self._areas = LevyAreaSampler(np.random.default_rng(sequence.spawn(1)[0]))
        self._warned = False

    def solve(
        self,
        equation: DifferentialEquation,
        initial: np.ndarray,
        paths: int | None = None,
        keep_every: int | None = None,
    ) -> EnsembleResult:
        """Integrate from ``initial`` (one state, or one row per path).

        ``paths`` replicates a single initial state; ``keep_every`` stores the
        ensemble every that many steps (the initial state included) so a
        trajectory statistic can be read without storing every step.
        """
        states = np.array(initial, dtype=float)
        if states.ndim == 1:
            states = np.tile(states, (paths or 1, 1))
        elif paths is not None and paths != states.shape[0]:
            message = f"initial has {states.shape[0]} rows but paths = {paths}"
            raise ValueError(message)
        coefficients = BatchedCoefficients(
            equation,
            self._config.calculus,
            typical=self._config.typical_sizes,
        )
        count = states.shape[0]
        time = float(self._config.start_time)
        drivers = coefficients.drivers(states, time)
        dt = float(self._config.dt)
        times = self._config.times
        kept_times, kept = [], []
        if keep_every:
            kept_times.append(time)
            kept.append(states.copy())
        self._warned = False
        for i in range(self._config.n_steps):
            time = float(times[i])
            if self._scheme == "euler_maruyama":
                increments, _ = self._increments(time, dt, count, drivers)
                states = self._euler_maruyama(
                    coefficients,
                    states,
                    time,
                    dt,
                    increments,
                )
            else:
                states = self._milstein(coefficients, states, time, dt, drivers)
            if keep_every and (i + 1) % keep_every == 0:
                kept_times.append(float(times[i + 1]))
                kept.append(states.copy())
        return EnsembleResult(
            final=states,
            times=np.array(kept_times, dtype=float),
            samples=np.array(kept) if kept else np.empty((0, count, states.shape[1])),
        )

    def _increments(
        self,
        time: float,
        dt: float,
        paths: int,
        drivers: int,
        *,
        levy_area: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """``(paths, m)`` increments, and ``(paths, m, m)`` areas if asked for."""
        if self._wiener is None:
            # standard_normal scaled afterwards: the same law as normal(0, sqrt(dt))
            # at about half the cost, which is most of a step at this size.
            increments = self._generator.standard_normal((paths, drivers)) * np.sqrt(
                dt,
            )
            if not levy_area:
                return increments, None
            return increments, self._areas.sample(increments, dt)
        draws = [
            self._wiener.increments(time, dt, drivers, levy_area=levy_area)
            for _ in range(paths)
        ]
        increments = np.array([draw.delta_w for draw in draws], dtype=float)
        if not levy_area:
            return increments, None
        return increments, np.array([draw.levy_area for draw in draws], dtype=float)

    @staticmethod
    def _euler_maruyama(
        coefficients: BatchedCoefficients,
        states: np.ndarray,
        time: float,
        dt: float,
        increments: np.ndarray,
    ) -> np.ndarray:
        # The per-path step, y + a dt + b * dW, with a leading axis; with one path
        # the arithmetic is the same operations in the same order, hence hex-equal.
        drift = coefficients.drift(states, time)
        diffusion = coefficients.diffusion(states, time)
        return states + drift * dt + coefficients.spread(diffusion, increments)

    def _milstein(
        self,
        coefficients: BatchedCoefficients,
        states: np.ndarray,
        time: float,
        dt: float,
        drivers: int,
    ) -> np.ndarray:
        # The per-path Milstein step, with the drift as written (the dW dW form
        # absorbs the Stratonovich correction), the delta_jk dt term in the Ito
        # form only, and the Levy-area term when the noise does not commute; see
        # MilsteinSolver for the derivation.
        drift = coefficients.written_drift(states, time)
        diffusion = coefficients.diffusion(states, time)
        count = states.shape[0]
        increments = None
        # Vector noise tries the O(paths n) form first, which needs the
        # increments as it goes. A source is asked for areas with the increments
        # or not at all, so with one the verdict is taken first; the generator's
        # areas can be drawn afterwards, given the increments.
        if not BatchedCoefficients.is_matrix(diffusion) and (
            self._wiener is None
            or self._vector_commutes(coefficients, states, time, diffusion)
        ):
            increments, _ = self._increments(time, dt, count, drivers)
            correction = self._diagonal_correction(
                coefficients,
                states,
                time,
                dt,
                diffusion,
                increments,
            )
            if correction is not None:
                return (
                    states
                    + drift * dt
                    + coefficients.spread(diffusion, increments)
                    + correction
                )
        jacobians = coefficients.column_jacobians(states, time)
        if BatchedCoefficients.is_matrix(diffusion):
            columns = np.transpose(diffusion, (0, 2, 1))
        else:
            columns = np.einsum("pk,kl->pkl", diffusion, np.eye(diffusion.shape[1]))
        products = np.einsum("pkil,pjl->pjki", jacobians, columns)
        scale = float(np.max(np.abs(products)))
        asymmetry = float(
            np.max(np.abs(products - np.transpose(products, (0, 2, 1, 3)))),
        )
        # NaN is non-commuting, as in the per-path check: a product that cannot
        # be judged is not taken as commuting, and one path gone to NaN or
        # infinity would otherwise strip every other path of its areas (DEQ-48).
        commuting = not coupled(asymmetry, scale)
        if increments is None:
            increments, areas = self._increments(
                time,
                dt,
                count,
                drivers,
                levy_area=not commuting,
            )
        else:
            # Drawn already, for vector noise the O(paths n) form found not to
            # commute - the verdict this one repeats - and so without a source.
            areas = self._areas.sample(increments, dt)
        correction = 0.5 * np.einsum("pjki,pj,pk->pi", products, increments, increments)
        if self._config.calculus == "ito":
            correction -= 0.5 * dt * np.einsum("pkki->pi", products)
        if areas is not None:
            # sum_{j,k} (dG_k/dx) G_j A_(j,k), the per-path solver's indices.
            self._warn_cost(drivers)
            correction += np.einsum("pjki,pjk->pi", products, areas)
        return (
            states
            + drift * dt
            + coefficients.spread(diffusion, increments)
            + correction
        )

    @staticmethod
    def _vector_commutes(
        coefficients: BatchedCoefficients,
        states: np.ndarray,
        time: float,
        diffusion: np.ndarray,
    ) -> bool:
        """The verdict :meth:`_diagonal_correction` reaches, before any increment.

        The same products, one direction at a time, and the same two numbers
        compared; asked only when a source must be told whether to give areas.
        """
        coefficients.confirm_declaration(states, time)
        if coefficients.declared_diagonal:
            return True
        scale = 0.0
        asymmetry = 0.0
        for j, slope in coefficients.diagonal_directions(states, time):
            magnitude = np.abs(diffusion[:, j, None] * slope)
            scale = float(np.max([scale, np.max(magnitude)]))
            magnitude[:, j] = 0.0
            asymmetry = float(np.max([asymmetry, np.max(magnitude)]))
        return not coupled(asymmetry, scale)

    def _warn_cost(self, drivers: int) -> None:
        if self._warned:
            return
        self._warned = True
        message = (
            "StochasticEnsemble: the noise does not commute ((dG_k/dx) G_j differs "
            "from (dG_j/dx) G_k), so every such step draws the Levy areas between "
            f"its {drivers} drivers for every path: 2 m n + m (m + 1) / 2 extra "
            "normals a path and step with n = ceil(0.092 sqrt(m / dt)) Fourier "
            "terms (DEQ-65). The euler_maruyama scheme needs none, at strong "
            "order 0.5."
        )
        warnings.warn(message, UserWarning, stacklevel=4)

    def _diagonal_correction(  # noqa: PLR0913 (the step's own arguments)
        self,
        coefficients: BatchedCoefficients,
        states: np.ndarray,
        time: float,
        dt: float,
        diffusion: np.ndarray,
        increments: np.ndarray,
    ) -> np.ndarray | None:
        """The Milstein term of vector noise in ``O(paths n)`` memory, or None.

        ``MilsteinSolver._diagonal_correction`` with a leading axis, the products
        ``P[j, k] = b_j d b_k / d x_j`` formed one direction ``j`` at a time and
        folded into the sum as they come, so neither the ``(paths, n, n, n)``
        Jacobians nor the products are ever held (DEQ-48). The commutativity
        verdict is the general one's: the largest off-diagonal ``|P[j, k]|``
        against the largest ``|P[j, k]|`` over the ensemble. None when the noise
        does not commute, so that the general step adds the Levy areas. Declared
        diagonal noise commutes by its declaration and needs only the slopes:
        ``P`` is diagonal, two batched evaluations make it, and the sum over
        ``j`` has one term, once the declaration has been checked.
        """
        coefficients.confirm_declaration(states, time)
        if coefficients.declared_diagonal:
            own = diffusion * coefficients.diagonal_slopes(states, time)
            correction = 0.5 * increments * (own * increments)
            if self._config.calculus == "ito":
                correction -= 0.5 * dt * own
            return correction
        scale = 0.0
        asymmetry = 0.0
        pushed = np.zeros(states.shape)
        own = np.empty(states.shape)
        for j, slope in coefficients.diagonal_directions(states, time):
            products = diffusion[:, j, None] * slope
            magnitude = np.abs(products)
            # np.max, not max: max(0.0, nan) is 0.0, which let a NaN path hide the
            # coupling of every other path in its direction.
            scale = float(np.max([scale, np.max(magnitude)]))
            own[:, j] = products[:, j]
            magnitude[:, j] = 0.0
            asymmetry = float(np.max([asymmetry, np.max(magnitude)]))
            pushed += products * increments[:, j, None]
        if coupled(asymmetry, scale):
            return None
        correction = 0.5 * increments * pushed
        if self._config.calculus == "ito":
            correction -= 0.5 * dt * own
        return correction
