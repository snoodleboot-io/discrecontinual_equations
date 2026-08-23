"""Automatic recursive exploration: the explorer escalates degeneracies on its own.

Two organizing centres are used as end-to-end oracles. The swallowtail normal form
drives the fold to cusp to swallowtail cascade, and an extended Bautin normal form
drives the Hopf to generalized-Hopf to degenerate-Bautin cascade. In each case the
explorer is given only an equilibrium seed and the parameter ranges, and is asked
to reach the codimension-3 point at the origin without being told the intermediate
bifurcations.
"""

from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.exploration import (
    BifurcationExplorer,
    ExplorationConfig,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable
from discrecontinual_equations.webplot.scene_builder import (
    exploration_diagram,
    exploration_skeleton,
    exploration_tree,
)

_ORIGIN_TOLERANCE = 5.0e-2
_MIN_CURVE_POINTS = 10


class First(Parameter, name="First", abbreviation="p1"):
    pass


class Second(Parameter, name="Second", abbreviation="p2"):
    pass


class Third(Parameter, name="Third", abbreviation="p3"):
    pass


class State(Variable, name="State", abbreviation="s"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


class Swallowtail(DeterministicFunction):
    """x' = b1 + b2 x + b3 x^2 - x^4; swallowtail at the origin."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        first = self.parameters[0].value
        second = self.parameters[1].value
        third = self.parameters[2].value
        x = point[0]
        return [first + second * x + third * x * x - x**4]


class ExtendedBautin(DeterministicFunction):
    """Planar quintic normal form; degenerate Bautin at the origin."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        first = self.parameters[0].value
        second = self.parameters[1].value
        third = self.parameters[2].value
        x, y = point[0], point[1]
        radius = x * x + y * y
        growth = second * radius + third * radius * radius
        return [first * x - y + x * growth, x + first * y + y * growth]


class BogdanovTakensUnfolding(DeterministicFunction):
    """x' = y, y' = b1 + b2 x + b3 y + x^2 + x y.

    The Bogdanov-Takens curve is analytic: b1 = b3^2, b2 = 2 b3.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        first = self.parameters[0].value
        second = self.parameters[1].value
        third = self.parameters[2].value
        x, y = point[0], point[1]
        return [y, first + second * x + third * y + x * x + x * y]


class DegenerateBogdanovTakens(DeterministicFunction):
    """x' = y, y' = b1 + b2 x + b3 y + b3 x^2 + x y.

    BT curve b1 = b3^3, b2 = 2 b3^2; the quadratic coefficient a = b3, so the
    degenerate Bogdanov-Takens (a = 0) sits at the origin.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        first = self.parameters[0].value
        second = self.parameters[1].value
        third = self.parameters[2].value
        x, y = point[0], point[1]
        return [y, first + second * x + third * y + third * x * x + x * y]


class BogdanovTakensTypeChange(DeterministicFunction):
    """x' = y, y' = b1 + b2 x + b3 y + x^2 + b3 x y.

    BT curve b1 = 1, b2 = 2 (x* = -1); the quadratic coefficient a = 1 while the
    cross coefficient b = b3, so the b = 0 degenerate Bogdanov-Takens sits at
    (b1, b2, b3) = (1, 2, 0).
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        first = self.parameters[0].value
        second = self.parameters[1].value
        third = self.parameters[2].value
        x, y = point[0], point[1]
        return [y, first + second * x + third * y + x * x + third * x * y]


def _equation(function_type: type, count: int, values: list[float]):
    parameters = [
        First(value=values[0]),
        Second(value=values[1]),
        Third(value=values[2]),
    ]
    variables = [State() for _ in range(count)]
    derivative = function_type(
        variables=variables,
        parameters=parameters,
        results=[State() for _ in range(count)],
        time=None,
    )
    return DifferentialEquation(
        variables=variables,
        time=Time(),
        parameters=parameters,
        derivative=derivative,
    )


class TestBifurcationExplorer(TestCase):
    def test_fold_cusp_swallowtail_cascade(self):
        equation = _equation(Swallowtail, 1, [0.1875, -1.0, 1.0])
        config = ExplorationConfig(
            parameters=[0, 1, 2],
            ranges=[(-1.4, 1.4), (-1.4, 1.4), (-1.4, 1.4)],
            initial_parameter=0.1875,
            initial_step=0.01,
            maximum_step=0.02,
            maximum_points=800,
        )
        root = BifurcationExplorer.explore(config, equation, [0.5])
        kinds = {point.kind for point in root.flatten()}
        assert "fold" in kinds
        assert "cusp" in kinds
        assert "zero_hopf" not in kinds
        swallowtails = [p for p in root.flatten() if p.kind == "swallowtail"]
        assert swallowtails
        assert all(
            all(abs(value) < _ORIGIN_TOLERANCE for value in point.parameters)
            for point in swallowtails
        )

    def test_hopf_generalized_hopf_degenerate_bautin_cascade(self):
        equation = _equation(ExtendedBautin, 2, [-0.3, 0.2, 0.4])
        config = ExplorationConfig(
            parameters=[0, 1, 2],
            ranges=[(-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6)],
            initial_parameter=-0.3,
            initial_step=0.01,
            maximum_step=0.02,
            maximum_points=900,
        )
        root = BifurcationExplorer.explore(config, equation, [0.01, 0.0])
        kinds = {point.kind for point in root.flatten()}
        assert "hopf" in kinds
        assert "generalized_hopf" in kinds
        degenerate = [p for p in root.flatten() if p.kind == "degenerate_bautin"]
        assert degenerate
        assert all(
            all(abs(value) < _ORIGIN_TOLERANCE for value in point.parameters)
            for point in degenerate
        )

    def test_fold_bogdanov_takens_curve_cascade(self):
        equation = _equation(BogdanovTakensUnfolding, 2, [-0.5, 0.6, 0.3])
        config = ExplorationConfig(
            parameters=[0, 1, 2],
            ranges=[(-1.0, 0.25), (-0.4, 1.4), (-0.6, 0.9)],
            initial_parameter=-0.5,
            initial_step=0.01,
            maximum_step=0.03,
            maximum_points=600,
        )
        seed = (-0.6 - (0.6**2 - 4 * (-0.5)) ** 0.5) / 2
        root = BifurcationExplorer.explore(config, equation, [seed, 0.0])
        curves = [
            child
            for node in root.children
            for child in node.children
            if child.description == "bogdanov_takens curve"
        ]
        assert curves
        segment = max(
            (seg for curve in curves for seg in curve.segments),
            key=len,
            default=[],
        )
        assert len(segment) >= _MIN_CURVE_POINTS
        for parameter_a, parameter_b, parameter_c in segment:
            assert abs(parameter_a - parameter_c**2) < 1.0e-3
            assert abs(parameter_b - 2.0 * parameter_c) < 1.0e-3

    def test_degenerate_bogdanov_takens_cascade(self):
        equation = _equation(DegenerateBogdanovTakens, 2, [-0.4, 0.32, 0.4])
        config = ExplorationConfig(
            parameters=[0, 1, 2],
            ranges=[(-0.8, 0.3), (-0.2, 0.9), (-0.6, 0.7)],
            initial_parameter=-0.4,
            initial_step=0.01,
            maximum_step=0.03,
            maximum_points=700,
        )
        roots = [v.real for v in np.roots([0.4, 0.32, -0.4]) if abs(v.imag) < 1.0e-9]
        root = BifurcationExplorer.explore(config, equation, [roots[0], 0.0])
        degenerate = [
            point
            for point in root.flatten()
            if point.kind == "degenerate_bogdanov_takens"
        ]
        assert degenerate
        assert all(
            all(abs(value) < _ORIGIN_TOLERANCE for value in point.parameters)
            for point in degenerate
        )

    def test_degenerate_bogdanov_takens_b_cascade(self):
        equation = _equation(BogdanovTakensTypeChange, 2, [0.6, 2.0, 0.4])
        config = ExplorationConfig(
            parameters=[0, 1, 2],
            ranges=[(0.2, 1.6), (1.2, 2.8), (-0.7, 0.7)],
            initial_parameter=0.6,
            initial_step=0.01,
            maximum_step=0.03,
            maximum_points=800,
        )
        roots = [v.real for v in np.roots([1.0, 2.0, 0.6]) if abs(v.imag) < 1.0e-9]
        root = BifurcationExplorer.explore(config, equation, [roots[0], 0.0])
        flat = root.flatten()
        target = (1.0, 2.0, 0.0)
        b_points = [p for p in flat if p.kind == "degenerate_bogdanov_takens_b"]
        assert b_points
        assert any(
            all(
                abs(value - expected) < _ORIGIN_TOLERANCE
                for value, expected in zip(point.parameters, target, strict=False)
            )
            for point in b_points
        )
        # a != 0 along this curve, so the a = 0 degeneracy must not fire
        assert not [p for p in flat if p.kind == "degenerate_bogdanov_takens"]


class TestExplorationDiagrams(TestCase):
    def _swallowtail_root(self):
        equation = _equation(Swallowtail, 1, [0.1875, -1.0, 1.0])
        config = ExplorationConfig(
            parameters=[0, 1, 2],
            ranges=[(-1.4, 1.4), (-1.4, 1.4), (-1.4, 1.4)],
            initial_parameter=0.1875,
            initial_step=0.01,
            maximum_step=0.02,
            maximum_points=800,
        )
        return BifurcationExplorer.explore(config, equation, [0.5])

    def test_two_parameter_diagram_has_curve_and_cusps(self):
        scene = exploration_diagram(self._swallowtail_root(), ("b1", "b2"), ("d", "s"))
        lines = [s for s in scene.series if s.role == "line"]
        markers = {s.kind for s in scene.series if s.role == "marker"}
        assert lines
        assert all(len(s.points) >= 2 for s in lines)
        assert "cusp" in markers

    def test_skeleton_lifts_curves_into_three_parameters(self):
        scene = exploration_skeleton(
            self._swallowtail_root(),
            ("b1", "b2", "b3"),
            ("d", "s"),
        )
        surface = scene.surface
        assert surface is not None
        assert surface.grid == []
        assert surface.loci
        assert all(len(point) == 3 for locus in surface.loci for point in locus.points)

    def test_skeleton_marks_organizing_centres(self):
        scene = exploration_skeleton(
            self._swallowtail_root(),
            ("b1", "b2", "b3"),
            ("d", "s"),
        )
        markers = scene.surface.markers
        assert markers
        assert all(marker.kind == "swallowtail" for marker in markers)
        assert all(len(marker.points[0]) == 3 for marker in markers)

    def test_tree_has_root_and_escalated_children(self):
        scene = exploration_tree(self._swallowtail_root(), ("t", "s"))
        tree = scene.tree
        assert tree is not None
        labels = [node.label for node in tree.nodes]
        assert "equilibrium branch" in labels
        assert "fold curve" in labels
        assert "cusp curve" in labels
        assert len(tree.edges) == len(tree.nodes) - 1
        root_nodes = [node for node in tree.nodes if node.level == 0]
        assert len(root_nodes) == 1
