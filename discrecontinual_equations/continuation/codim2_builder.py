"""Drive two-parameter continuation of a bifurcation curve.

Given a codim-1 seed (a fold or Hopf point found by ordinary continuation), the
driver builds the augmented curve residual, continues it with the ordinary engine
(via :meth:`ContinuerBuilder.assemble`), and scans the result for codim-2 points.
Strategies (residual, test functions, helpers) are resolved from the registry;
the driver imports abstractions and orchestration only.
"""

import numpy as np
from sweet_tea.abstract_factory import AbstractFactory
from sweet_tea.factory import Factory

from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.codim2 import (
    Codim2Point,
    Codim2TestFunction,
)
from discrecontinual_equations.continuation.codim2_config import Codim2Config
from discrecontinual_equations.continuation.derivative_provider import (
    DerivativeProvider,
)
from discrecontinual_equations.continuation.residual import ResidualFunction
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)
from discrecontinual_equations.differential_equation import DifferentialEquation

_HOPF_RESIDUAL = "hopf_curve_residual"
_FOLD_RESIDUAL = "fold_curve_residual"


class CurveSeed:
    """A codim-1 point used to seed a two-parameter curve continuation."""

    __slots__ = ["frequency", "parameter_a", "state"]

    def __init__(
        self,
        state: list[float],
        parameter_a: float,
        frequency: float | None = None,
    ) -> None:
        self.state = state
        self.parameter_a = parameter_a
        self.frequency = frequency


class Codim2Result:
    """The continued curve and the codim-2 points found along it."""

    __slots__ = ["_curve", "_parameters", "_points"]

    def __init__(
        self,
        curve: Branch,
        points: list[Codim2Point],
        parameters: list[tuple[float, float, float | None]],
    ) -> None:
        self._curve = curve
        self._points = points
        self._parameters = parameters

    @property
    def curve(self) -> Branch:
        """The continued codim-1 curve as an augmented branch."""
        return self._curve

    @property
    def points(self) -> list[Codim2Point]:
        """Detected codim-2 bifurcations."""
        return self._points

    @property
    def curve_parameters(self) -> list[tuple[float, float, float | None]]:
        """Decoded ``(p_a, p_b, omega)`` along the curve, for plotting."""
        return self._parameters


class Codim2Driver:
    """Continue a bifurcation curve in two parameters and detect codim-2 points."""

    @staticmethod
    def run(
        config: Codim2Config,
        equation: DifferentialEquation,
        seed: CurveSeed,
    ) -> Codim2Result:
        """Continue the configured curve from ``seed`` and analyze it."""
        problem = Factory.create(
            "two_parameter_problem",
            configuration={
                "equation": equation,
                "first_index": config.continuation_parameter_index,
                "second_index": config.second_parameter_index,
                "time": config.evaluation_time,
                "provider": AbstractFactory[DerivativeProvider].create(
                    config.derivative_provider,
                ),
            },
        )
        dimension = len(equation.variables)
        seed_state, normalization, residual_key = Codim2Driver._seed(
            config,
            problem,
            seed,
        )
        residual = AbstractFactory[ResidualFunction].create(
            residual_key,
            configuration={
                "problem": problem,
                "dimension": dimension,
                "normalization": normalization,
            },
        )
        curve_config = config.model_copy(update={"detectors": []})
        continuer = ContinuerBuilder.assemble(curve_config, residual)
        curve = continuer.solve(equation, seed_state.tolist())

        analyzer = Factory.create(
            "codim2_analyzer",
            configuration={
                "test_functions": [
                    AbstractFactory[Codim2TestFunction].create(key)
                    for key in config.codim2_detectors
                ],
            },
        )
        points = analyzer.analyze(curve, residual, problem)
        parameters = [
            (decoded.parameter_a, decoded.parameter_b, decoded.omega)
            for decoded in (
                residual.decode(np.asarray(point.state, dtype=float), point.parameter)
                for point in curve.points
            )
        ]
        return Codim2Result(curve, points, parameters)

    @staticmethod
    def _seed(
        config: Codim2Config,
        problem: TwoParameterProblem,
        seed: CurveSeed,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        equilibrium = np.asarray(seed.state, dtype=float)
        parameter_a = float(seed.parameter_a)
        parameter_b = float(config.initial_parameter)
        jacobian = problem.state_jacobian(equilibrium, parameter_a, parameter_b)
        if config.curve == "fold":
            null_vector = _null_vector(jacobian)
            state = np.concatenate([equilibrium, null_vector, [parameter_a]])
            return state, null_vector, _FOLD_RESIDUAL
        real_vector, imag_vector, omega = _hopf_vectors(jacobian, seed.frequency or 0.0)
        state = np.concatenate(
            [equilibrium, real_vector, imag_vector, [omega], [parameter_a]],
        )
        return state, real_vector.copy(), _HOPF_RESIDUAL


def _null_vector(jacobian: np.ndarray) -> np.ndarray:
    _left, _singular, right = np.linalg.svd(jacobian)
    vector = right[-1]
    return vector / np.linalg.norm(vector)


def _hopf_vectors(
    jacobian: np.ndarray,
    frequency: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    values, vectors = np.linalg.eig(jacobian)
    target = 1j * frequency
    index = min(
        range(len(values)),
        key=lambda position: abs(values[position] - target),
    )
    eigenvector = vectors[:, index]
    real_vector = eigenvector.real
    imag_vector = eigenvector.imag
    scale = 1.0 / np.linalg.norm(real_vector)
    real_vector = real_vector * scale
    imag_vector = imag_vector * scale
    return real_vector, imag_vector, abs(float(values[index].imag))
