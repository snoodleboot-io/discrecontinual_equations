"""Coloured noise, by making the noise a state.

A white-noise equation ``dx = f(x) dt + G(x) dW`` has drivers with no memory. Real
noise has a correlation time, and the standard way to give it one without leaving
the Markovian world is to *embed* it (Hänggi and Jung, "Colored noise in dynamical
systems", Adv. Chem. Phys. 89, 1995, Sec. VI.A): each Wiener driver is replaced by an
Ornstein-Uhlenbeck process ``eta`` that becomes one more component of the state,

    ``dx = f(x) dt + G(x) eta dt``,  ``d eta = -(eta / tau) dt + (1 / tau) dW``.

The pair ``(x, eta)`` is Markovian, so every solver and every estimator in the
library applies to it unchanged; the noise on the augmented state is a constant
``(n + m) x m`` matrix acting on ``eta`` only, which makes Ito and Stratonovich the
same thing there.

The normalisation is the one that keeps the *intensity* fixed as ``tau`` varies, not
the variance. ``eta`` has stationary variance ``1 / (2 tau)`` and correlation
``exp(-|t - s| / tau) / (2 tau)``, which integrates to one for every ``tau`` - Hänggi
and Jung's ``(D / tau) exp(-|t - s| / tau)`` with ``D = 1/2`` - and tends to
``delta(t - s)`` as ``tau -> 0``. So ``tau -> 0`` is the white-noise equation, and in
the Stratonovich sense, because ``eta`` is a smooth path (Wong and Zakai): the white
limit of ``G(x) eta`` is ``G(x) o dW``, whatever convention the white equation was
written in. The unit-variance normalisation ``d eta = -(eta / tau) dt +
sqrt(2 / tau) dW`` looks more natural and is wrong for this purpose: its intensity is
``tau``, so as ``tau -> 0`` the noise does not become white, it disappears.

Starting ``eta`` at zero is a preparation, not the stationary noise: for ``t``
shorter than a few ``tau`` the drivers are quieter than they will be. Use
:meth:`ColouredNoise.stationary_drivers` for initial states that are already in
the stationary law of the noise (Hänggi and Jung, eq. 6.9).
"""

import numpy as np

from discrecontinual_equations.function.function import Function
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.variable import Variable

_MATRIX = 2


class ColouredNoise(StochasticFunction):
    """``white`` with each of its ``drivers`` Wiener processes replaced by OU noise.

    The augmented state is ``[x_1 .. x_n, eta_1 .. eta_m]``. ``drivers`` is ``m``:
    for a ``white`` whose diffusion is the vector form (one driver per component)
    it is ``n``, the default; for the matrix form it is the number of columns.
    """

    __slots__ = ["_correlation_time", "_drivers", "_white"]

    def __init__(
        self,
        white: StochasticFunction,
        correlation_time: float,
        drivers: int | None = None,
    ) -> None:
        if correlation_time <= 0.0:
            message = (
                "a correlation time must be positive; tau = 0 is the white "
                "equation itself, in the Stratonovich sense"
            )
            raise ValueError(message)
        count = len(white.variables) if drivers is None else drivers
        if count < 1:
            message = "coloured noise needs at least one driver"
            raise ValueError(message)
        noise = [Variable(name=f"eta{k}", discretization=[]) for k in range(count)]
        super().__init__(
            [*white.variables, *noise],
            white.parameters,
            [Variable(name=f"d{v.name}/dt", discretization=[]) for v in white.variables]
            + [Variable(name=f"d{v.name}/dt", discretization=[]) for v in noise],
            white.time,
        )
        self._white = white
        self._correlation_time = float(correlation_time)
        self._drivers = count

    @property
    def white(self) -> StochasticFunction:
        """The white-noise equation this one colours."""
        return self._white

    @property
    def correlation_time(self) -> float:
        """``tau``."""
        return self._correlation_time

    @property
    def drivers(self) -> int:
        """Number of coloured drivers, ``m``."""
        return self._drivers

    @property
    def dimension(self) -> int:
        """Number of state variables of the white equation, ``n``."""
        return len(self._white.variables)

    @property
    def driver_variance(self) -> float:
        """Stationary variance of each ``eta``: ``1 / (2 tau)``."""
        return 0.5 / self._correlation_time

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        """``[f(x) + G(x) eta, -eta / tau]``."""
        state = np.asarray(point, dtype=float)
        x, eta = state[: self.dimension], state[self.dimension :]
        drift = np.asarray(self._white.eval(point=list(x), time=time), dtype=float)
        amplitude = np.asarray(
            self._white.diffusion(point=list(x), time=time),
            dtype=float,
        )
        forced = drift + self._spread(amplitude, eta)
        return [*forced.tolist(), *(-eta / self._correlation_time).tolist()]

    def diffusion(
        self,
        point: list[float],  # noqa: ARG002 (constant noise)
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[list[float]]:
        """The constant ``(n + m) x m`` matrix: ``0`` over ``I / tau``."""
        return self._noise().tolist()

    def eval_batch(self, points: np.ndarray, time: float | None = None) -> np.ndarray:
        """The augmented drift at every row, through the white batch methods."""
        x, eta = points[:, : self.dimension], points[:, self.dimension :]
        drift = np.asarray(self._white.eval_batch(x, time), dtype=float)
        amplitude = np.asarray(self._white.diffusion_batch(x, time), dtype=float)
        if amplitude.ndim == _MATRIX + 1:
            forced = drift + np.einsum("pim,pm->pi", amplitude, eta)
        else:
            self._check_vector(amplitude.shape[1])
            forced = drift + amplitude * eta
        return np.concatenate([forced, -eta / self._correlation_time], axis=1)

    def diffusion_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> np.ndarray:
        """The constant matrix, once per row."""
        noise = self._noise()
        return np.broadcast_to(noise, (points.shape[0], *noise.shape))

    def stationary_drivers(
        self,
        states: np.ndarray,
        generator: np.random.Generator,
    ) -> np.ndarray:
        """Append drivers drawn from their stationary law to ``(paths, n)`` states."""
        rows = np.atleast_2d(np.asarray(states, dtype=float))
        eta = generator.standard_normal((rows.shape[0], self._drivers)) * np.sqrt(
            self.driver_variance,
        )
        return np.concatenate([rows, eta], axis=1)

    def driver_columns(self) -> list[Function]:
        """The columns of the augmented noise as ``Function`` objects, one per driver.

        This is what :class:`~discrecontinual_equations.continuation.noise.NoiseMatrix`
        takes, so the continuation tools - the stochastic Lyapunov estimator, the grid
        Fokker-Planck solver - can be handed the augmented system directly.
        """
        return [
            _DriverColumn(self.variables, self._noise()[:, k])
            for k in range(self._drivers)
        ]

    def _noise(self) -> np.ndarray:
        noise = np.zeros((self.dimension + self._drivers, self._drivers))
        noise[self.dimension :, :] = np.eye(self._drivers) / self._correlation_time
        return noise

    def _spread(self, amplitude: np.ndarray, eta: np.ndarray) -> np.ndarray:
        if amplitude.ndim == _MATRIX:
            return amplitude @ eta
        self._check_vector(amplitude.shape[0])
        return amplitude * eta

    def _check_vector(self, components: int) -> None:
        if components != self._drivers:
            message = (
                f"the white diffusion is the vector form with {components} "
                f"components, which means {components} drivers, not {self._drivers}"
            )
            raise ValueError(message)


class _DriverColumn(Function):
    """A constant column of the augmented noise."""

    __slots__ = ["_column"]

    def __init__(self, variables: list[Variable], column: np.ndarray) -> None:
        super().__init__(variables, [], variables, None)
        self._column = column

    def eval(
        self,
        point: list[float],  # noqa: ARG002 (constant field)
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return self._column.tolist()
