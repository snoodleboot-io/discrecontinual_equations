"""Coupled-array sensor systems, as first-class library systems.

The ring topology is checked directly against a hand-written expansion, and each
device is driven through the continuation library to confirm its known onset: the
fluxgate breaks symmetry at ``lambda = 1 - c`` and the ferroelectric ring onsets
oscillations at ``lambda = -2a``.
"""

import math
from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.continuation.exploration import (
    BifurcationExplorer,
    ExplorationConfig,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.systems import FerroelectricRing, FluxgateRing
from discrecontinual_equations.variable import Variable

_TOLERANCE = 2.0e-2
_MIN_SEGMENT = 5


class Coupling(Parameter, name="Coupling", abbreviation="lam"):
    pass


class Gain(Parameter, name="Gain", abbreviation="c"):
    pass


class Field(Parameter, name="Field", abbreviation="eps"):
    pass


class State(Variable, name="State", abbreviation="s"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


def _equation(
    function_type: type,
    coupling: float,
    gain: float,
    field: float,
    count: int = 3,
):
    parameters = [
        Coupling(value=coupling),
        Gain(value=gain),
        Field(value=field),
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


class TestHighDimensionalRing(TestCase):
    def test_seven_cell_ring_hopf_spectrum(self):
        count = 7
        equation = _equation(FerroelectricRing, 0.0, 1.0, 0.0, count)
        detected: dict[str, set] = {}
        for direction in (1, -1):
            config = ContinuationConfig(
                continuation_parameter_index=0,
                initial_parameter=0.0,
                direction=direction,
                measure="norm",
                parameter_lower_bound=-5.0,
                parameter_upper_bound=3.5,
                initial_step=0.02,
                maximum_step=0.05,
                maximum_points=3000,
                detectors=["branch_point", "hopf"],
            )
            branch = ContinuerBuilder.build(config, equation).solve(
                equation,
                [0.0] * count,
            )
            for point in branch.special_points:
                detected.setdefault(point.kind, set()).add(round(point.parameter, 2))
        assert any(abs(v - 1.0) < _TOLERANCE for v in detected.get("pitchfork", set()))
        for mode in (1, 3):
            oracle = 1.0 / math.cos(2.0 * math.pi * mode / count)
            assert any(
                abs(v - oracle) < _TOLERANCE for v in detected.get("hopf", set())
            ), oracle

    def test_seven_cell_ring_exploration_continues_hopf_curves(self):
        count = 7
        equation = _equation(FerroelectricRing, 0.0, 1.0, 0.0, count)
        config = ExplorationConfig(
            parameters=[0, 1, 2],
            ranges=[(-3.0, 3.0), (0.4, 1.8), (-1.0, 1.0)],
            initial_parameter=0.0,
            initial_step=0.03,
            maximum_step=0.06,
            maximum_points=150,
        )
        root = BifurcationExplorer.explore(config, equation, [0.0] * count)
        oracles = [1.0 / math.cos(2.0 * math.pi * mode / count) for mode in (1, 3)]
        slopes = []
        for node in root.children:
            for segment in node.segments:
                usable = [lam / a for lam, a in segment if abs(a) > _TOLERANCE]
                if len(usable) >= _MIN_SEGMENT:
                    slopes.append(sum(usable) / len(usable))
        assert slopes
        for slope in slopes:
            assert any(abs(slope - oracle) < _TOLERANCE for oracle in oracles), slope


class TestCoupledRings(TestCase):
    def test_fluxgate_cell_and_ring(self):
        equation = _equation(FluxgateRing, 0.2, 3.0, 0.1)
        point = [0.5, -0.3, 0.1]
        expected = [
            -point[i] + math.tanh(3.0 * point[i] + 0.2 * point[(i + 1) % 3] + 0.1)
            for i in range(3)
        ]
        assert equation.derivative.eval(point=point, time=None) == expected

    def test_ferroelectric_cell_and_ring(self):
        equation = _equation(FerroelectricRing, 0.2, 1.0, 0.0)
        point = [0.5, -0.3, 0.1]
        expected = [
            1.0 * point[i] - point[i] ** 3 - 0.2 * point[(i + 1) % 3] for i in range(3)
        ]
        assert equation.derivative.eval(point=point, time=None) == expected

    def test_fluxgate_symmetry_breaks_at_one_minus_gain(self):
        equation = _equation(FluxgateRing, 0.0, 3.0, 0.0)
        branch = self._continue(equation, "branch_point")
        assert any(
            abs(p.parameter - (1.0 - 3.0)) < _TOLERANCE
            for p in branch.special_points
            if p.kind == "pitchfork"
        )

    def test_ferroelectric_oscillation_onset(self):
        equation = _equation(FerroelectricRing, 0.0, 1.0, 0.0)
        branch = self._continue(equation, "hopf")
        assert any(
            abs(p.parameter - (-2.0 * 1.0)) < _TOLERANCE
            for p in branch.special_points
            if p.kind == "hopf"
        )

    def _continue(self, equation: DifferentialEquation, detector: str):
        config = ContinuationConfig(
            continuation_parameter_index=0,
            initial_parameter=1.6,
            direction=-1,
            measure="norm",
            parameter_lower_bound=-3.2,
            parameter_upper_bound=1.7,
            initial_step=0.01,
            maximum_step=0.03,
            maximum_points=2500,
            detectors=[detector],
        )
        continuer = ContinuerBuilder.build(config, equation)
        return continuer.solve(equation, list(np.zeros(3)))
