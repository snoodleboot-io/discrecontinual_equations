"""Evolving a bifurcation diagram over a second parameter.

Two checks with analytic oracles: for the cusp normal form the diagram in one
parameter has no fold below the cusp and a fold pair above it, so sweeping the
unfolding parameter must switch the fold count from zero to two; and for the
coupled-core fluxgate ring the trivial state breaks symmetry exactly at
``lambda = 1 - c``, so the detected branch point must track the gain that way.
"""

import math
from unittest import TestCase

from discrecontinual_equations.continuation.family_builder import FamilyDriver
from discrecontinual_equations.continuation.family_config import FamilyConfig
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable
from discrecontinual_equations.webplot.scene_builder import surface_scene

_LOCATION_TOLERANCE = 2.0e-2
_EXPECTED_FOLDS = 2


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


class Cusp(DeterministicFunction):
    """x' = p1 + p2 x - x^3; a fold pair exists only for p2 > 0."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        first = self.parameters[0].value
        second = self.parameters[1].value
        x = point[0]
        return [first + second * x - x**3]


class Fluxgate(DeterministicFunction):
    """Coupled-core fluxgate ring; symmetry breaks at coupling 1 - gain."""

    size = 3

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        coupling = self.parameters[0].value
        gain = self.parameters[1].value
        count = type(self).size
        return [
            -point[i] + math.tanh(gain * point[i] + coupling * point[(i + 1) % count])
            for i in range(count)
        ]


class FerroelectricRing(DeterministicFunction):
    """Electric-field sensor ring: a x - x^3 - lambda x_next + eps.

    Parameters are coupling (First), Landau gain (Second), target field (Third).
    """

    size = 3

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        coupling = self.parameters[0].value
        gain = self.parameters[1].value
        field = self.parameters[2].value
        count = type(self).size
        return [
            gain * point[i] - point[i] ** 3 - coupling * point[(i + 1) % count] + field
            for i in range(count)
        ]


def _equation(
    function_type: type[DeterministicFunction],
    count: int,
    second: float,
) -> DifferentialEquation:
    parameters = [First(value=0.0), Second(value=second)]
    derivative = function_type(
        variables=[State() for _ in range(count)],
        parameters=parameters,
        results=[State() for _ in range(count)],
        time=None,
    )
    return DifferentialEquation(
        variables=[State() for _ in range(count)],
        time=Time(),
        parameters=parameters,
        derivative=derivative,
    )


class TestFamily(TestCase):
    def test_cusp_family_switches_fold_count(self):
        config = FamilyConfig(
            continuation_parameter_index=0,
            family_parameter_index=1,
            family_values=[-0.5, 0.0, 0.5, 1.0],
            initial_parameter=-3.0,
            direction=1,
            measure="component",
            parameter_lower_bound=-3.2,
            parameter_upper_bound=3.2,
            initial_step=0.01,
            maximum_step=0.05,
            maximum_points=1200,
            detectors=["fold"],
        )
        family = FamilyDriver.run(config, _equation(Cusp, 1, 0.0), [-(3.0 ** (1 / 3))])
        counts = {
            item.value: len([p for p in item.branch.special_points if p.kind == "fold"])
            for item in family.slices
        }
        assert counts[-0.5] == 0
        assert counts[0.0] == 0
        assert counts[0.5] == _EXPECTED_FOLDS
        assert counts[1.0] == _EXPECTED_FOLDS

    def test_fluxgate_symmetry_break_tracks_gain(self):
        config = FamilyConfig(
            continuation_parameter_index=0,
            family_parameter_index=1,
            family_values=[2.0, 3.0, 4.0],
            initial_parameter=0.0,
            direction=-1,
            measure="norm",
            parameter_lower_bound=-5.0,
            parameter_upper_bound=0.2,
            initial_step=0.01,
            maximum_step=0.03,
            maximum_points=2000,
            detectors=["branch_point"],
        )
        family = FamilyDriver.run(config, _equation(Fluxgate, 3, 3.0), [0.0, 0.0, 0.0])
        for item in family.slices:
            points = [p for p in item.branch.special_points if p.kind == "pitchfork"]
            assert len(points) == 1
            assert abs(points[0].parameter - (1.0 - item.value)) < _LOCATION_TOLERANCE


class TestElectricFieldSensor(TestCase):
    def _equation(self, coupling: float, gain: float, field: float):
        parameters = [
            First(value=coupling),
            Second(value=gain),
            Third(value=field),
        ]
        derivative = FerroelectricRing(
            variables=[State(), State(), State()],
            parameters=parameters,
            results=[State(), State(), State()],
            time=None,
        )
        return DifferentialEquation(
            variables=[State(), State(), State()],
            time=Time(),
            parameters=parameters,
            derivative=derivative,
        )

    def test_onset_branch_point_and_hopf(self):
        config = FamilyConfig(
            continuation_parameter_index=0,
            family_parameter_index=1,
            family_values=[0.5, 1.0, 1.25],
            initial_parameter=1.6,
            direction=-1,
            measure="norm",
            parameter_lower_bound=-3.2,
            parameter_upper_bound=1.7,
            initial_step=0.01,
            maximum_step=0.03,
            maximum_points=2500,
            detectors=["branch_point", "hopf"],
        )
        family = FamilyDriver.run(
            config,
            self._equation(0.0, 1.0, 0.0),
            [0.0, 0.0, 0.0],
        )
        for item in family.slices:
            gain = item.value
            branch = [
                p.parameter for p in item.branch.special_points if p.kind == "pitchfork"
            ]
            hopf = [p.parameter for p in item.branch.special_points if p.kind == "hopf"]
            assert any(abs(b - gain) < _LOCATION_TOLERANCE for b in branch)
            assert any(abs(h + 2.0 * gain) < _LOCATION_TOLERANCE for h in hopf)

    def test_detection_fold_locus_tracks_coupling(self):
        config = FamilyConfig(
            continuation_parameter_index=2,
            family_parameter_index=0,
            family_values=[-0.5, 0.0, 0.3, 0.6],
            initial_parameter=-1.3,
            direction=1,
            measure="component",
            parameter_lower_bound=-1.4,
            parameter_upper_bound=1.4,
            initial_step=0.005,
            maximum_step=0.02,
            maximum_points=3000,
            detectors=["fold"],
        )
        seed = _newton(1.0 - (-0.5), -1.3)
        family = FamilyDriver.run(config, self._equation(-0.5, 1.0, 0.0), [seed] * 3)
        coefficient = 2.0 / (3.0 * math.sqrt(3.0))
        for item in family.slices:
            effective = 1.0 - item.value
            oracle = coefficient * effective**1.5
            folds = [
                p.parameter for p in item.branch.special_points if p.kind == "fold"
            ]
            assert len(folds) == _EXPECTED_FOLDS
            for fold in folds:
                assert min(abs(fold - oracle), abs(fold + oracle)) < 1.0e-2


def _newton(effective_gain: float, field: float) -> float:
    x = field
    for _ in range(80):
        residual = effective_gain * x - x**3 + field
        derivative = effective_gain - 3.0 * x * x
        x -= residual / derivative
    return x


class TestSurface(TestCase):
    def test_fold_locus_is_the_semicubical_cusp(self):
        config = FamilyConfig(
            continuation_parameter_index=0,
            family_parameter_index=1,
            family_values=[round(0.1 + 0.1 * i, 3) for i in range(14)],
            initial_parameter=-3.0,
            direction=1,
            measure="component",
            parameter_lower_bound=-3.2,
            parameter_upper_bound=3.2,
            initial_step=0.01,
            maximum_step=0.05,
            maximum_points=1500,
            detectors=["fold"],
        )
        family = FamilyDriver.run(config, _equation(Cusp, 1, 0.0), [-(3.0 ** (1 / 3))])
        scene = surface_scene(family, ("p1", "p2", "x"), ("Cusp", "surface"))
        surface = scene.surface
        assert surface is not None
        assert len(surface.grid) == len(family.slices)
        fold = next(locus for locus in surface.loci if locus.kind == "fold")
        coefficient = 2.0 / (3.0 * math.sqrt(3.0))
        for parameter, family_value, _measure in fold.points:
            oracle = coefficient * family_value**1.5
            assert min(abs(parameter - oracle), abs(parameter + oracle)) < 1.0e-6
