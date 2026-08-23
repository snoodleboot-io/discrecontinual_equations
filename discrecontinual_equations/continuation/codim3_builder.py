"""Drive three-parameter continuation of a cusp curve and detect codim-3 points.

Given a cusp seed (an equilibrium with two parameters placing it on the cusp
curve), the driver builds the augmented cusp-curve residual, continues it with the
ordinary engine via :meth:`ContinuerBuilder.assemble`, and scans the result for
swallowtail points. Strategies are resolved from the registry; the driver imports
abstractions and orchestration only.
"""

import numpy as np
from sweet_tea.abstract_factory import AbstractFactory
from sweet_tea.factory import Factory

from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.codim3 import (
    Codim3Point,
    Codim3TestFunction,
)
from discrecontinual_equations.continuation.codim3_config import Codim3Config
from discrecontinual_equations.continuation.derivative_provider import (
    DerivativeProvider,
)
from discrecontinual_equations.continuation.residual import ResidualFunction
from discrecontinual_equations.differential_equation import DifferentialEquation


class Codim3Seed:
    """A codim-2 point used to seed a three-parameter curve continuation."""

    __slots__ = ["frequency", "parameter_a", "parameter_b", "state"]

    def __init__(
        self,
        state: list[float],
        parameter_a: float,
        parameter_b: float,
        frequency: float | None = None,
    ) -> None:
        self.state = state
        self.parameter_a = parameter_a
        self.parameter_b = parameter_b
        self.frequency = frequency


class Codim3Result:
    """The continued cusp curve and the codim-3 points found along it."""

    __slots__ = ["_curve", "_parameters", "_points"]

    def __init__(
        self,
        curve: Branch,
        points: list[Codim3Point],
        parameters: list[tuple[float, float, float]],
    ) -> None:
        self._curve = curve
        self._points = points
        self._parameters = parameters

    @property
    def curve(self) -> Branch:
        """The continued cusp curve as an augmented branch."""
        return self._curve

    @property
    def points(self) -> list[Codim3Point]:
        """Detected codim-3 bifurcations."""
        return self._points

    @property
    def curve_parameters(self) -> list[tuple[float, float, float]]:
        """Decoded ``(p_a, p_b, p_c)`` along the curve, for plotting."""
        return self._parameters


class Codim3Driver:
    """Continue a cusp curve in three parameters and detect codim-3 points."""

    @staticmethod
    def run(
        config: Codim3Config,
        equation: DifferentialEquation,
        seed: Codim3Seed,
    ) -> Codim3Result:
        """Continue the cusp curve from ``seed`` and analyze it."""
        third = equation.derivative.parameters[config.third_parameter_index]
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
        third.value = float(config.initial_parameter)
        dimension = len(equation.variables)
        equilibrium = np.asarray(seed.state, dtype=float)
        jacobian = problem.state_jacobian(
            equilibrium,
            float(seed.parameter_a),
            float(seed.parameter_b),
        )
        seed_state, normalization, residual_key = _build_seed(
            config,
            seed,
            equilibrium,
            jacobian,
        )
        residual = AbstractFactory[ResidualFunction].create(
            residual_key,
            configuration={
                "problem": problem,
                "dimension": dimension,
                "normalization": normalization,
                "third": third,
            },
        )
        curve_config = config.model_copy(update={"detectors": []})
        continuer = ContinuerBuilder.assemble(curve_config, residual)
        curve = continuer.solve(equation, seed_state.tolist())

        analyzer = Factory.create(
            "codim3_analyzer",
            configuration={
                "test_functions": [
                    AbstractFactory[Codim3TestFunction].create(key)
                    for key in config.codim3_detectors
                ],
                "third": third,
            },
        )
        points = analyzer.analyze(curve, residual, problem)
        parameters = [
            (decoded.parameter_a, decoded.parameter_b, decoded.parameter_c)
            for decoded in (
                residual.decode(np.asarray(point.state, dtype=float), point.parameter)
                for point in curve.points
            )
        ]
        return Codim3Result(curve, points, parameters)


def _build_seed(
    config: Codim3Config,
    seed: Codim3Seed,
    equilibrium: np.ndarray,
    jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    if config.codim3_curve == "generalized_hopf":
        real_vector, imag_vector, omega = _hopf_vectors(
            jacobian,
            seed.frequency or 0.0,
        )
        state = np.concatenate(
            [
                equilibrium,
                real_vector,
                imag_vector,
                [omega, seed.parameter_a, seed.parameter_b],
            ],
        )
        return state, real_vector.copy(), "generalized_hopf_curve_residual"
    null_vector = _null_vector(jacobian)
    if config.codim3_curve == "bogdanov_takens":
        chain = _jordan_chain(jacobian, null_vector)
        state = np.concatenate(
            [
                equilibrium,
                null_vector,
                chain,
                [seed.parameter_a, seed.parameter_b],
            ],
        )
        return state, null_vector, "bogdanov_takens_curve_residual"
    state = np.concatenate(
        [equilibrium, null_vector, [seed.parameter_a], [seed.parameter_b]],
    )
    return state, null_vector, "cusp_curve_residual"


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
    return real_vector * scale, imag_vector * scale, abs(float(values[index].imag))


def _null_vector(jacobian: np.ndarray) -> np.ndarray:
    _left, _singular, right = np.linalg.svd(jacobian)
    vector = right[-1]
    return vector / np.linalg.norm(vector)


def _jordan_chain(jacobian: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    chain, _residual, _rank, _singular = np.linalg.lstsq(jacobian, kernel, rcond=None)
    projection = float(kernel @ chain) / float(kernel @ kernel)
    return chain - projection * kernel
