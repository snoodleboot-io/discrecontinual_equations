"""One stochastic system, written once, for both the analysis and the solvers.

The analysis side (Fokker-Planck, matrix-noise Lyapunov) takes a drift
``Function`` and a :class:`~.noise.NoiseMatrix`; the solvers take a
``StochasticFunction`` whose ``diffusion()`` is the ``n x m`` matrix ``G``.
:class:`~.noise.NoiseMatrixFunction` and :func:`~.noise.noise_matrix_from` carry
one to the other (DEQ-44).

The oracle is the noisy-Hopf conformal system of the noisy-Hopf film:
``dz = (mu + i - |z|^2) z dt + s z (dB_1 + i dB_2)``. Its planar density is
``C r^(2 mu / s^2 - 2) exp(-r^2 / s^2)`` (:mod:`.test_noisy_hopf_thresholds`), so
``u = r^2`` is Gamma-distributed with shape ``mu / s^2`` and scale ``s^2`` and
``E[r^2] = mu`` exactly; by isotropy the stationary covariance is ``(mu / 2) I``.
The conformal noise has no Stratonovich correction, so this holds in either
calculus.
"""

from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    StationaryFokkerPlanck,
)
from discrecontinual_equations.continuation.noise import (
    NoiseMatrix,
    NoiseMatrixFunction,
    noise_matrix_from,
)
from discrecontinual_equations.continuation.stochastic import Ito
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.solver.stochastic.ensemble import StochasticEnsemble
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config import (  # noqa: E501
    EulerMaruyamaConfig,
)
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_solver import (  # noqa: E501
    EulerMaruyamaSolver,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_config import (
    MilsteinConfig,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_solver import (
    MilsteinSolver,
)
from discrecontinual_equations.variable import Variable
from tests.continuation.fields import (
    Alpha,
    ConformalColumn,
    HopfField,
    Sigma,
    State,
    Time,
)

_MU = 1.0
_SIGMA = 0.7
_PLANE = 2


class HandWrittenNoisyHopf(StochasticFunction):
    """The same system written directly for the solvers, as it was before the bridge.

    The arithmetic copies :class:`HopfField` and :class:`ConformalColumn` operation
    for operation, so the bridge has to reproduce it to the last bit.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        radius = x * x + y * y
        return [mu * x - y - x * radius, x + mu * y - y * radius]

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[list[float]]:
        sigma = self.parameters[1].value
        x, y = point[0], point[1]
        return [[sigma * x, -sigma * y], [sigma * y, sigma * x]]


class DiagonalNoise(StochasticFunction):
    """Vector-form diffusion ``(s x, s (1 + y^2))``: the diagonal case."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return [-point[0], -point[1]]

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        sigma = self.parameters[1].value
        return [sigma * point[0], sigma * (1.0 + point[1] * point[1])]


class OneDriver(StochasticFunction):
    """One Brownian motion pushing both components: ``G`` is ``2 x 1``."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return [-point[0], -point[1]]

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[list[float]]:
        sigma = self.parameters[1].value
        return [[sigma * point[1]], [sigma]]


class ForcedHopf(HopfField):
    """The Hopf drift plus ``(cos t, 0)``: a drift that reads the time it is given."""

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        base = super().eval(point, time)
        return [base[0] + np.cos(time or 0.0), base[1]]


def _parameters(mu: float = _MU, sigma: float = _SIGMA):
    return [Alpha(value=mu), Sigma(value=sigma)]


def _drift(parameters, time=None):
    return HopfField(
        variables=[State(), State()],
        parameters=parameters,
        results=[State(), State()],
        time=time,
    )


def _noise(parameters):
    return NoiseMatrix(
        [
            ConformalColumn(index, [State(), State()], parameters, [State(), State()])
            for index in range(_PLANE)
        ],
    )


def _hand_written(parameters, function_type=HandWrittenNoisyHopf, time=None):
    return function_type([State(), State()], parameters, [State(), State()], time)


def _clock() -> Time:
    # The per-path solvers write a timed curve, so their functions carry a time.
    return Time()


def _equation(function: StochasticFunction) -> DifferentialEquation:
    return DifferentialEquation(
        variables=function.variables,
        time=Variable(name="t", discretization=[]),
        parameters=function.parameters,
        derivative=function,
    )


def _points(count: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-3.0, 3.0, size=(count, _PLANE))


def _path(solver_type, config_type, function, seed: int) -> np.ndarray:
    config = config_type(
        start_time=0.0,
        end_time=2.0,
        step_size=0.01,
        random_seed=seed,
    )
    solver = solver_type(config)
    solver.solve(_equation(function), [0.8, -0.3])
    return np.array([result.discretization for result in solver.solution.results])


class TestNoiseMatrixFunction(TestCase):
    """The bridge from the analysis side to the solvers."""

    def test_diffusion_is_the_noise_matrix_exactly(self):
        parameters = _parameters()
        noise = _noise(parameters)
        bridged = NoiseMatrixFunction(_drift(parameters), noise)
        for point in _points(200, seed=1):
            matrix = np.array(bridged.diffusion(list(point)), dtype=float)
            assert matrix.shape == (_PLANE, noise.drivers)
            assert np.array_equal(matrix, noise.columns_at(point).T)

    def test_drift_is_the_drift_exactly(self):
        parameters = _parameters()
        drift = _drift(parameters)
        bridged = NoiseMatrixFunction(drift, _noise(parameters))
        for point in _points(200, seed=2):
            assert bridged.eval(list(point)) == drift.eval(list(point))

    def test_it_matches_the_hand_written_function_at_random_points(self):
        parameters = _parameters()
        bridged = NoiseMatrixFunction(_drift(parameters), _noise(parameters))
        written = _hand_written(parameters)
        for point in _points(200, seed=3):
            assert bridged.eval(list(point)) == written.eval(list(point))
            assert np.array_equal(
                np.array(bridged.diffusion(list(point))),
                np.array(written.diffusion(list(point))),
            )

    def test_the_drift_is_given_the_solver_s_time(self):
        parameters = _parameters()
        drift = ForcedHopf([State(), State()], parameters, [State(), State()], None)
        bridged = NoiseMatrixFunction(drift, _noise(parameters))
        assert bridged.eval([0.1, 0.2], 2.0) == drift.eval([0.1, 0.2], 2.0)
        assert bridged.eval([0.1, 0.2], 2.0) != drift.eval([0.1, 0.2], None)

    def test_it_reads_the_drift_s_variables_and_parameters(self):
        parameters = _parameters()
        drift = _drift(parameters)
        bridged = NoiseMatrixFunction(drift, _noise(parameters))
        assert bridged.variables is drift.variables
        assert bridged.parameters is drift.parameters
        assert bridged.noise.drivers == _PLANE

    def test_euler_maruyama_path_is_the_hand_written_path(self):
        parameters = _parameters()
        bridged = NoiseMatrixFunction(_drift(parameters, _clock()), _noise(parameters))
        for seed in (0, 7, 42):
            through_bridge = _path(
                EulerMaruyamaSolver, EulerMaruyamaConfig, bridged, seed
            )
            by_hand = _path(
                EulerMaruyamaSolver,
                EulerMaruyamaConfig,
                _hand_written(parameters, time=_clock()),
                seed,
            )
            assert through_bridge.shape == (_PLANE, 201)
            assert np.array_equal(through_bridge, by_hand)

    def test_milstein_path_is_the_hand_written_path(self):
        parameters = _parameters()
        bridged = NoiseMatrixFunction(_drift(parameters, _clock()), _noise(parameters))
        through_bridge = _path(MilsteinSolver, MilsteinConfig, bridged, 5)
        by_hand = _path(
            MilsteinSolver,
            MilsteinConfig,
            _hand_written(parameters, time=_clock()),
            5,
        )
        assert np.array_equal(through_bridge, by_hand)

    def test_ensemble_is_the_hand_written_ensemble(self):
        parameters = _parameters()
        bridged = NoiseMatrixFunction(_drift(parameters), _noise(parameters))
        config = EulerMaruyamaConfig(
            start_time=0.0,
            end_time=1.0,
            step_size=0.01,
            random_seed=11,
        )
        through_bridge = StochasticEnsemble(config).solve(
            _equation(bridged),
            [0.8, -0.3],
            paths=50,
        )
        by_hand = StochasticEnsemble(config).solve(
            _equation(_hand_written(parameters)),
            [0.8, -0.3],
            paths=50,
        )
        assert np.array_equal(through_bridge.final, by_hand.final)


class TestNoiseMatrixFrom(TestCase):
    """The reverse bridge: a solver's function, read as a noise matrix."""

    def test_columns_are_the_matrix_columns_exactly(self):
        written = _hand_written(_parameters())
        noise = noise_matrix_from(written, drivers=_PLANE)
        assert noise.drivers == _PLANE
        for point in _points(200, seed=4):
            expected = np.array(written.diffusion(list(point))).T
            assert np.array_equal(noise.columns_at(point), expected)

    def test_a_vector_diffusion_gives_the_diagonal_columns(self):
        written = _hand_written(_parameters(), DiagonalNoise)
        noise = noise_matrix_from(written)
        assert noise.drivers == _PLANE
        for point in _points(50, seed=5):
            expected = np.diag(written.diffusion(list(point)))
            assert np.array_equal(noise.columns_at(point), expected)

    def test_a_single_driver_gives_one_column(self):
        written = _hand_written(_parameters(), OneDriver)
        noise = noise_matrix_from(written, drivers=1)
        assert noise.drivers == 1
        for point in _points(20, seed=9):
            expected = np.array(written.diffusion(list(point))).T
            assert np.array_equal(noise.columns_at(point), expected)

    def test_the_driver_count_defaults_to_the_state_dimension(self):
        written = _hand_written(_parameters())
        assert noise_matrix_from(written).drivers == _PLANE

    def test_a_matrix_of_the_wrong_width_is_refused(self):
        written = _hand_written(_parameters())
        noise = noise_matrix_from(written, drivers=3)
        with pytest.raises(ValueError, match="3 drivers"):
            noise.columns_at(np.array([0.5, 0.5]))

    def test_a_driver_count_below_one_is_refused(self):
        with pytest.raises(ValueError, match="at least one driver"):
            noise_matrix_from(_hand_written(_parameters()), drivers=0)

    def test_round_trip_from_the_noise_matrix(self):
        parameters = _parameters()
        noise = _noise(parameters)
        back = noise_matrix_from(NoiseMatrixFunction(_drift(parameters), noise))
        for point in _points(100, seed=6):
            assert np.array_equal(back.columns_at(point), noise.columns_at(point))
            assert np.array_equal(back.jacobians_at(point), noise.jacobians_at(point))

    def test_round_trip_from_the_stochastic_function(self):
        written = _hand_written(_parameters())
        again = NoiseMatrixFunction(written, noise_matrix_from(written))
        for point in _points(100, seed=8):
            assert again.eval(list(point)) == written.eval(list(point))
            assert np.array_equal(
                np.array(again.diffusion(list(point))),
                np.array(written.diffusion(list(point))),
            )


class TestConformalCovariance(TestCase):
    """The ticket's oracle: one system, the same covariance from both sides."""

    def test_fokker_planck_of_the_reverse_bridge_is_the_original(self):
        # The hand-written solver function, read back for the analysis side, gives
        # the very density the NoiseMatrix gives: the same numbers, not close ones.
        parameters = _parameters()
        grid = DensityGrid.box([-3.0, -3.0], [3.0, 3.0], [40, 40])
        original = StationaryFokkerPlanck(
            _drift(parameters),
            _noise(parameters),
            Ito(),
            grid,
        ).solve()
        written = _hand_written(parameters)
        bridged = StationaryFokkerPlanck(
            written,
            noise_matrix_from(written),
            Ito(),
            grid,
        ).solve()
        assert np.array_equal(original.values, bridged.values)

    def test_grid_and_ensemble_give_the_closed_form_covariance(self):
        parameters = _parameters()
        drift, noise = _drift(parameters), _noise(parameters)
        exact = 0.5 * _MU * np.eye(_PLANE)

        # Even counts keep the origin, where D = s^2 r^2 I vanishes, off the grid.
        grid = DensityGrid.box([-3.0, -3.0], [3.0, 3.0], [60, 60])
        density = StationaryFokkerPlanck(drift, noise, Ito(), grid).solve()
        from_grid = np.array(density.covariance())

        # The same two objects, through the bridge, into the ensemble solver: 200
        # paths from the unit circle, sampled every 0.1 after a burn-in of eight
        # relaxation times (the log-radius relaxes at rate 2 mu). Measured: 0.507,
        # 0.490 on the diagonal and -0.011 off it; seed 1 gives 0.514, 0.500, -0.002.
        config = EulerMaruyamaConfig(
            start_time=0.0,
            end_time=12.0,
            step_size=0.02,
            random_seed=2026,
        )
        result = StochasticEnsemble(config).solve(
            _equation(NoiseMatrixFunction(drift, noise)),
            [1.0, 0.0],
            paths=200,
            keep_every=5,
        )
        settled = result.samples[result.times >= 4.0].reshape(-1, _PLANE)
        from_ensemble = settled.T @ settled / settled.shape[0]

        assert np.max(np.abs(from_grid - exact)) < 0.005
        assert np.max(np.abs(from_ensemble - exact)) < 0.03
        assert np.max(np.abs(from_grid - from_ensemble)) < 0.03
