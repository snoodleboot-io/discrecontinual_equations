import itertools
import math
from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.continuation.detection import (
    BifurcationDetector,
    DetectionContext,
)
from discrecontinual_equations.continuation.diagram import BifurcationDiagram
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

LOCATION_TOLERANCE = 2.0e-2
STATE_TOLERANCE = 5.0e-2
FREQUENCY_TOLERANCE = 1.0e-1
MARKER_PARAMETER = 0.5


class Mu(Parameter, name="Continuation parameter", abbreviation="mu"):
    pass


class X(Variable, name="State x", abbreviation="x"):
    pass


class Y(Variable, name="State y", abbreviation="y"):
    pass


class Z(Variable, name="State z", abbreviation="z"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


class SaddleNode(DeterministicFunction):
    """x' = mu - x^2, a fold at (x, mu) = (0, 0)."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        return [mu - point[0] * point[0]]


class Transcritical(DeterministicFunction):
    """x' = mu x - x^2, a branch point at the origin."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x = point[0]
        return [mu * x - x * x]


class Pitchfork(DeterministicFunction):
    """x' = mu x - x^3, a supercritical pitchfork at the origin."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x = point[0]
        return [mu * x - x * x * x]


class Cusp(DeterministicFunction):
    """x' = mu + 3x - x^3, an S-curve with two folds at (x, mu) = (+-1, -+2)."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x = point[0]
        return [mu + 3.0 * x - x * x * x]


class Hopf(DeterministicFunction):
    """Planar Andronov-Hopf normal form with a Hopf point at mu = 0."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        radius_squared = x * x + y * y
        return [
            mu * x - y - x * radius_squared,
            x + mu * y - y * radius_squared,
        ]


class Hopf3D(DeterministicFunction):
    """Three-dimensional system with a Hopf point at mu = 0 and a stable z axis."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x, y, z = point[0], point[1], point[2]
        return [mu * x - y, x + mu * y, -z]


class Linear(DeterministicFunction):
    """x' = mu - x, equilibrium x = mu with no fold (safe for natural parameter)."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return [self.parameters[0].value - point[0]]


class ParameterMarkerDetector(BifurcationDetector):
    """A custom detector fired at a chosen parameter value.

    Exists only to demonstrate that a new detector can be injected without any
    change to the driver (open/closed) and that the driver honours injection.
    """

    @property
    def kind(self) -> str:
        return "marker"

    def test_value(self, context: DetectionContext) -> float:
        return context.continuation_state.parameter - MARKER_PARAMETER


def build_equation(function_type, variables):
    mu = Mu(value=0.0)
    results = [X() for _ in variables]
    derivative = function_type(
        variables=variables,
        parameters=[mu],
        results=results,
        time=None,
    )
    return DifferentialEquation(
        variables=variables,
        time=Time(),
        parameters=[mu],
        derivative=derivative,
    )


class TestContinuer(TestCase):
    def test_saddle_node_detects_fold(self):
        equation = build_equation(SaddleNode, [X()])
        config = ContinuationConfig(
            detectors=["fold"],
            initial_parameter=3.0,
            direction=-1,
            measure="component",
            initial_step=0.04,
            maximum_step=0.12,
            parameter_lower_bound=-0.6,
            parameter_upper_bound=3.2,
        )
        continuer = ContinuerBuilder.build(config, equation)
        branch = continuer.solve(equation, initial_values=[math.sqrt(3.0)])

        folds = [point for point in branch.special_points if point.kind == "fold"]
        assert folds
        assert any(abs(point.parameter) < LOCATION_TOLERANCE for point in folds)
        stabilities = {point.stability for point in branch.points}
        assert "stable" in stabilities
        assert "unstable" in stabilities

    def test_transcritical_detects_branch_point(self):
        equation = build_equation(Transcritical, [X()])
        config = ContinuationConfig(
            detectors=["branch_point"],
            initial_parameter=-2.0,
            measure="component",
            initial_step=0.04,
            maximum_step=0.12,
            parameter_lower_bound=-2.0,
            parameter_upper_bound=2.0,
        )
        continuer = ContinuerBuilder.build(config, equation)
        branch = continuer.solve(equation, initial_values=[0.0])

        branch_points = [
            point for point in branch.special_points if point.kind == "transcritical"
        ]
        assert branch_points
        assert any(
            abs(point.parameter) < LOCATION_TOLERANCE
            and abs(point.state[0]) < STATE_TOLERANCE
            for point in branch_points
        )
        assert all(point.kind != "branch_point" for point in branch.special_points)

    def test_pitchfork_detects_branch_point_and_switches(self):
        equation = build_equation(Pitchfork, [X()])
        config = ContinuationConfig(
            detectors=["branch_point"],
            initial_parameter=-2.0,
            measure="component",
            initial_step=0.04,
            parameter_lower_bound=-2.0,
            parameter_upper_bound=2.0,
        )
        continuer = ContinuerBuilder.build(config, equation)
        branch = continuer.solve(equation, initial_values=[0.0])

        branch_points = [
            point for point in branch.special_points if point.kind == "pitchfork"
        ]
        assert branch_points
        assert any(abs(point.parameter) < LOCATION_TOLERANCE for point in branch_points)

        seed_state, seed_parameter = continuer.seed_secondary_branch(branch_points[0])
        assert abs(seed_state[0]) + abs(seed_parameter) > 0.0

    def test_branch_point_classification_separates_pitchfork_and_transcritical(self):
        def kinds_for(function_type: type) -> set[str]:
            equation = build_equation(function_type, [X()])
            config = ContinuationConfig(
                detectors=["branch_point"],
                initial_parameter=-2.0,
                measure="component",
                initial_step=0.04,
                maximum_step=0.12,
                parameter_lower_bound=-2.0,
                parameter_upper_bound=2.0,
            )
            continuer = ContinuerBuilder.build(config, equation)
            branch = continuer.solve(equation, initial_values=[0.0])
            return {point.kind for point in branch.special_points}

        assert "pitchfork" in kinds_for(Pitchfork)
        assert "transcritical" not in kinds_for(Pitchfork)
        assert "transcritical" in kinds_for(Transcritical)
        assert "pitchfork" not in kinds_for(Transcritical)

    def test_cusp_detects_two_folds(self):
        equation = build_equation(Cusp, [X()])
        config = ContinuationConfig(
            detectors=["fold"],
            initial_parameter=-2.0,
            direction=1,
            measure="component",
            initial_step=0.03,
            maximum_step=0.1,
            parameter_lower_bound=-3.0,
            parameter_upper_bound=3.0,
        )
        continuer = ContinuerBuilder.build(config, equation)
        branch = continuer.solve(equation, initial_values=[-2.0])

        folds = [point for point in branch.special_points if point.kind == "fold"]
        assert len(folds) >= 2
        parameters = sorted(point.parameter for point in folds)
        assert any(abs(value - (-2.0)) < LOCATION_TOLERANCE for value in parameters)
        assert any(abs(value - 2.0) < LOCATION_TOLERANCE for value in parameters)

    def test_hopf_detects_hopf_point_with_frequency(self):
        equation = build_equation(Hopf, [X(), Y()])
        config = ContinuationConfig(
            detectors=["hopf"],
            initial_parameter=-1.0,
            measure="norm",
            initial_step=0.05,
            maximum_step=0.15,
            parameter_lower_bound=-1.0,
            parameter_upper_bound=1.0,
        )
        continuer = ContinuerBuilder.build(config, equation)
        branch = continuer.solve(equation, initial_values=[0.0, 0.0])

        hopf_points = [point for point in branch.special_points if point.kind == "hopf"]
        assert hopf_points
        assert any(abs(point.parameter) < LOCATION_TOLERANCE for point in hopf_points)
        assert any(
            point.frequency is not None
            and abs(point.frequency - 1.0) < FREQUENCY_TOLERANCE
            for point in hopf_points
        )

    def test_three_dimensional_hopf(self):
        equation = build_equation(Hopf3D, [X(), Y(), Z()])
        config = ContinuationConfig(
            detectors=["hopf"],
            initial_parameter=-1.0,
            measure="norm",
            initial_step=0.05,
            maximum_step=0.15,
            parameter_lower_bound=-1.0,
            parameter_upper_bound=1.0,
        )
        continuer = ContinuerBuilder.build(config, equation)
        branch = continuer.solve(equation, initial_values=[0.0, 0.0, 0.0])

        hopf_points = [point for point in branch.special_points if point.kind == "hopf"]
        assert hopf_points
        assert any(abs(point.parameter) < LOCATION_TOLERANCE for point in hopf_points)

    def test_injected_custom_detector_is_used(self):
        equation = build_equation(SaddleNode, [X()])
        config = ContinuationConfig(
            detectors=[],
            initial_parameter=0.2,
            direction=1,
            measure="component",
            initial_step=0.05,
            parameter_lower_bound=-0.6,
            parameter_upper_bound=1.5,
        )
        base = ContinuerBuilder.build(config, equation)
        injected = base._components.model_copy(  # noqa: SLF001 (test wiring)
            update={"detectors": [ParameterMarkerDetector()]},
        )
        continuer = type(base)(solver_config=config, components=injected)
        branch = continuer.solve(equation, initial_values=[math.sqrt(0.2)])

        markers = [point for point in branch.special_points if point.kind == "marker"]
        assert markers
        assert any(
            abs(point.parameter - MARKER_PARAMETER) < LOCATION_TOLERANCE
            for point in markers
        )

    def test_natural_parameterization_traces_branch(self):
        # parameterization="natural" selects a different constraint strategy,
        # injected by the factory into the generic Newton corrector.
        equation = build_equation(Linear, [X()])
        config = ContinuationConfig(
            detectors=[],
            parameterization="natural",
            initial_parameter=-1.0,
            direction=1,
            measure="component",
            parameter_lower_bound=-1.0,
            parameter_upper_bound=1.0,
        )
        continuer = ContinuerBuilder.build(config, equation)
        branch = continuer.solve(equation, initial_values=[-1.0])

        assert len(branch) > 5
        for point in branch.points:
            assert math.isclose(point.state[0], point.parameter, abs_tol=1e-6)
        assert max(point.parameter for point in branch.points) > 0.5

    def test_secant_root_finder_localizes_fold(self):
        equation = build_equation(SaddleNode, [X()])
        config = ContinuationConfig(
            detectors=["fold"],
            root_finder="secant",
            initial_parameter=3.0,
            direction=-1,
            measure="component",
            parameter_lower_bound=-0.6,
            parameter_upper_bound=3.2,
        )
        continuer = ContinuerBuilder.build(config, equation)
        branch = continuer.solve(equation, initial_values=[math.sqrt(3.0)])

        folds = [point for point in branch.special_points if point.kind == "fold"]
        assert folds
        assert any(abs(point.parameter) < LOCATION_TOLERANCE for point in folds)

    def test_measure_component_matches_state(self):
        equation = build_equation(SaddleNode, [X()])
        config = ContinuationConfig(
            detectors=["fold"],
            initial_parameter=3.0,
            direction=-1,
            measure="component",
            parameter_upper_bound=3.2,
            parameter_lower_bound=0.0,
        )
        continuer = ContinuerBuilder.build(config, equation)
        branch = continuer.solve(equation, initial_values=[math.sqrt(3.0)])

        sample = branch.points[0]
        assert math.isclose(sample.measure, sample.state[0], abs_tol=1e-9)
        assert np.isfinite(sample.parameter)

    def test_diagram_segments_split_on_stability_change(self):
        equation = build_equation(SaddleNode, [X()])
        config = ContinuationConfig(
            detectors=["fold"],
            initial_parameter=3.0,
            direction=-1,
            measure="component",
            parameter_lower_bound=-0.6,
            parameter_upper_bound=3.2,
        )
        continuer = ContinuerBuilder.build(config, equation)
        branch = continuer.solve(equation, initial_values=[math.sqrt(3.0)])

        diagram = BifurcationDiagram(branch)
        segments = diagram.segments()
        labels = {segment.stability for segment in segments}
        # The saddle-node branch has both a stable and an unstable arc.
        assert "stable" in labels
        assert "unstable" in labels
        # Every segment is internally consistent in length.
        for segment in segments:
            assert len(segment.parameters) == len(segment.measures)
        # Adjacent segments connect: the last point of one equals the first
        # of the next, so a renderer draws an unbroken curve.
        for earlier, later in itertools.pairwise(segments):
            assert earlier.parameters[-1] == later.parameters[0]
            assert earlier.measures[-1] == later.measures[0]

    def test_diagram_passes_through_special_points(self):
        equation = build_equation(SaddleNode, [X()])
        config = ContinuationConfig(
            detectors=["fold"],
            initial_parameter=3.0,
            direction=-1,
            measure="component",
            parameter_lower_bound=-0.6,
            parameter_upper_bound=3.2,
        )
        continuer = ContinuerBuilder.build(config, equation)
        branch = continuer.solve(equation, initial_values=[math.sqrt(3.0)])

        diagram = BifurcationDiagram(branch)
        assert diagram.special_points() == branch.special_points
        assert any(point.kind == "fold" for point in diagram.special_points())
