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
and Milstein for commuting noise with the same refusal as the per-path solver.
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

from dataclasses import dataclass

import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver_config import StochasticConfig
from discrecontinual_equations.solver.stochastic.wiener import WienerSource

_MATRIX_BATCH = 3
# The same tolerance as the per-path Milstein solver: how far (dG_k/dx) G_j may
# differ from (dG_j/dx) G_k, relative to the largest product over the ensemble.
_COMMUTATIVITY_TOLERANCE = 1.0e-6


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
    when the equation's batch methods are arrays.
    """

    __slots__ = ["_equation", "_slope_step", "_stratonovich"]

    def __init__(
        self,
        equation: DifferentialEquation,
        calculus: str,
        slope_step: float = 1.0e-6,
    ) -> None:
        self._equation = equation
        self._stratonovich = calculus == "stratonovich"
        self._slope_step = slope_step

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
        jacobians = self.column_jacobians(points, time)
        if self.is_matrix(diffusion):
            return drift + 0.5 * np.einsum("pkij,pjk->pi", jacobians, diffusion)
        # Diagonal noise: (1/2) b_i db_i/dx_i, the (i, i, i) entries of the Jacobians.
        index = np.arange(points.shape[1])
        return drift + 0.5 * diffusion * jacobians[:, index, index, index]

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
            forward = points.copy()
            backward = points.copy()
            forward[:, j] += self._slope_step
            backward[:, j] -= self._slope_step
            difference = (
                self.diffusion(forward, time) - self.diffusion(backward, time)
            ) / (2.0 * self._slope_step)
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
    """

    __slots__ = ["_config", "_generator", "_scheme", "_wiener"]

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
        coefficients = BatchedCoefficients(equation, self._config.calculus)
        count = states.shape[0]
        time = float(self._config.start_time)
        drivers = coefficients.drivers(states, time)
        dt = float(self._config.dt)
        times = self._config.times
        kept_times, kept = [], []
        if keep_every:
            kept_times.append(time)
            kept.append(states.copy())
        for i in range(self._config.n_steps):
            time = float(times[i])
            increments = self._increments(time, dt, count, drivers)
            if self._scheme == "euler_maruyama":
                states = self._euler_maruyama(
                    coefficients,
                    states,
                    time,
                    dt,
                    increments,
                )
            else:
                states = self._milstein(coefficients, states, time, dt, increments)
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
    ) -> np.ndarray:
        if self._wiener is None:
            # standard_normal scaled afterwards: the same law as normal(0, sqrt(dt))
            # at about half the cost, which is most of a step at this size.
            return self._generator.standard_normal((paths, drivers)) * np.sqrt(dt)
        return np.array(
            [self._wiener.increments(time, dt, drivers).delta_w for _ in range(paths)],
            dtype=float,
        )

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
        increments: np.ndarray,
    ) -> np.ndarray:
        # The per-path Milstein step for commuting noise, with the drift as written
        # (the dW dW form absorbs the Stratonovich correction) and the delta_jk dt
        # term in the Ito form only; see MilsteinSolver for the derivation.
        drift = coefficients.written_drift(states, time)
        diffusion = coefficients.diffusion(states, time)
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
        if scale > 0.0 and asymmetry > _COMMUTATIVITY_TOLERANCE * scale:
            message = (
                "the Milstein ensemble needs commuting noise: (dG_k/dx) G_j must "
                f"equal (dG_j/dx) G_k for every driver pair; they differ by "
                f"{asymmetry:.3g} against a scale of {scale:.3g}. Use the "
                "euler_maruyama scheme (DEQ-28)."
            )
            raise ValueError(message)
        correction = 0.5 * np.einsum("pjki,pj,pk->pi", products, increments, increments)
        if self._config.calculus == "ito":
            correction -= 0.5 * dt * np.einsum("pkki->pi", products)
        return (
            states
            + drift * dt
            + coefficients.spread(diffusion, increments)
            + correction
        )
