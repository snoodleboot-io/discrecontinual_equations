"""The stage: a continuation re-shaped by frame, built and rendered without a browser.

The system under test is the linear saddle ``x' = x - p, y' = -y`` with its
equilibrium at ``(p, 0)``: the field, the manifolds and the eigenvalues are all
known in closed form, so every number the builder produces can be checked.
"""

import json
from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.continuation_point import ContinuationPoint
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable
from discrecontinual_equations.webplot.report import AtlasEntry, _atlas_html
from discrecontinual_equations.webplot.stage import (
    Cycle,
    CycleBranch,
    Equilibrium,
    Frame,
    Lattice,
    StageScene,
    StageSystem,
    Term,
    Timeline,
    View,
    evaluate_terms,
    stage_payload,
)
from discrecontinual_equations.webplot.stage_builder import (
    Continued,
    Film,
    cycles_at,
    frame_equilibria,
    saddle_manifolds,
    sample_field,
    stage_scene,
)
from discrecontinual_equations.webplot.stage_renderer import StageRenderer

_UNIT = ((-1.0, 1.0), (-1.0, 1.0))


class Shift(Parameter, name="Shift", abbreviation="p"):
    pass


class State(Variable, name="State", abbreviation="v"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


class ShiftedSaddle(DeterministicFunction):
    """x' = x - p, y' = -y: a saddle at (p, 0) with eigenvalues +1 and -1."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        p = self.parameters[0].value
        return [point[0] - p, -point[1]]


def _equation() -> DifferentialEquation:
    parameters = [Shift(value=0.0)]
    return DifferentialEquation(
        variables=[State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=ShiftedSaddle(
            variables=[State(), State()],
            parameters=parameters,
            results=[State(), State()],
            time=None,
        ),
    )


def _point(parameter: float, x: float, stability: str = "saddle") -> ContinuationPoint:
    return ContinuationPoint(
        arclength=0.0,
        state=[x, 0.0],
        parameter=parameter,
        tangent=[1.0, 0.0, 0.0],
        eigenvalues=[(1.0, 0.0), (-1.0, 0.0)],
        unstable_dimension=1,
        stability=stability,
        measure=x,
    )


def _saddle_branch() -> Branch:
    branch = Branch()
    for p in (0.0, 0.5, 1.0):
        branch.append_point(_point(p, p))
    return branch


_SYSTEM = StageSystem("Saddle", "", "p")


def _film(frames, box=_UNIT, grid=3, describe=None) -> Film:
    return Film(frames, Lattice(box, grid), _SYSTEM, describe)


class TestFrameEquilibria(TestCase):
    def test_interpolates_the_state_between_branch_points(self):
        found = frame_equilibria(_saddle_branch(), 0.25)
        assert len(found) == 1
        assert abs(found[0].x - 0.25) < 1.0e-12
        assert found[0].stability == "saddle"

    def test_takes_every_crossing_of_a_folded_branch(self):
        """A branch that rounds a fold crosses a parameter value twice."""
        branch = Branch()
        for p, x, stability in (
            (0.0, -1.0, "stable"),
            (0.5, -0.5, "stable"),
            (1.0, 0.0, "stable"),  # the fold
            (0.5, 0.5, "saddle"),
            (0.0, 1.0, "saddle"),
        ):
            branch.append_point(_point(p, x, stability))
        found = frame_equilibria(branch, 0.25)
        assert sorted(item.x for item in found) == [-0.75, 0.75]
        assert {item.stability for item in found} == {"stable", "saddle"}

    def test_a_point_exactly_on_the_value_appears_once(self):
        found = frame_equilibria(_saddle_branch(), 0.5)
        assert len(found) == 1
        assert found[0].x == 0.5

    def test_nothing_outside_the_branch(self):
        assert frame_equilibria(_saddle_branch(), 2.0) == []


class TestCyclesAt(TestCase):
    """A frame takes every cycle at its parameter, not just the nearest."""

    @staticmethod
    def _cycle(parameter: float, amplitude: float) -> Cycle:
        return Cycle(parameter, 6.0, amplitude, [(1.0, 0.0)], [(0.0, 0.0)])

    def test_takes_the_cycle_within_half_a_frame(self):
        cycles = [self._cycle(p, 0.5) for p in (0.0, 0.1)]
        assert cycles_at(cycles, 0.04, 0.1) == [0]
        assert cycles_at(cycles, 0.16, 0.1) == []
        assert cycles_at([], 0.0, 0.1) == []

    def test_takes_both_cycles_of_a_folded_branch(self):
        """Two cycles at one parameter differ in amplitude, so both are kept."""
        cycles = [self._cycle(0.0, 0.9), self._cycle(0.0, 0.3)]
        assert sorted(cycles_at(cycles, 0.0, 0.1)) == [0, 1]

    def test_leaves_out_the_same_orbit_sampled_twice(self):
        """Traced points crowd in the parameter where the branch turns.

        The two orbits in range differ by 0.002 against a branch that spans
        0.9, so they are one cycle sampled twice. The far-off points are what
        give the branch its range - the tolerance is a fraction of that, not of
        the pair being compared.
        """
        cycles = [
            self._cycle(0.0, 0.9),
            self._cycle(0.001, 0.902),
            self._cycle(5.0, 0.3),
            self._cycle(5.0, 1.2),
        ]
        assert cycles_at(cycles, 0.0, 0.1) == [0]

    def test_the_tolerance_does_not_tighten_as_the_orbits_shrink(self):
        """Near a Hopf the vanishing cycle is sampled many times over.

        The three small orbits are one cycle; a tolerance relative to the
        amplitudes being compared would have called them three.
        """
        cycles = [
            self._cycle(0.0, 0.093),
            self._cycle(0.001, 0.087),
            self._cycle(-0.001, 0.1),
            self._cycle(0.0, 1.009),
        ]
        assert cycles_at(cycles, 0.0, 0.05) == [0, 3]

    def test_reports_the_smallest_orbit_first(self):
        """So the order does not flip with which point sat nearest."""
        cycles = [self._cycle(0.001, 0.9), self._cycle(0.0, 0.3)]
        assert cycles_at(cycles, 0.0, 0.1) == [1, 0]


class TestStageScene(TestCase):
    def test_samples_the_field_row_major(self):
        equation = _equation()
        equation.derivative.parameters[0].value = 0.0
        field = sample_field(equation.derivative, Lattice(_UNIT, 3))
        assert len(field) == 2 * 3 * 3
        # node (ix=2, iy=0) is (x=1, y=-1): u = x - p = 1, v = -y = 1
        assert field[2 * 2 : 2 * 2 + 2] == [1.0, 1.0]
        # node (ix=0, iy=2) is (x=-1, y=1): u = -1, v = -1
        assert field[2 * (2 * 3 + 0) : 2 * (2 * 3 + 0) + 2] == [-1.0, -1.0]

    def test_stages_a_saddle_with_its_four_manifold_branches(self):
        scene = stage_scene(
            Continued(_equation(), 0, _saddle_branch()),
            _film([0.0, 0.5], box=((-2.0, 3.0), (-2.0, 2.0)), grid=5),
        )
        assert len(scene.frames) == 2
        for frame in scene.frames:
            assert len(frame.field) == 2 * 5 * 5
            assert len(frame.equilibria) == 1
            assert frame.cycles == []
            kinds = sorted(item.kind for item in frame.manifolds)
            assert kinds == ["stable", "stable", "unstable", "unstable"]
        # The unstable manifold of x' = x - p, y' = -y is the line y = 0 and the
        # stable manifold the line x = p; at p = 0.5 both must hold exactly.
        second = scene.frames[1]
        for manifold in second.manifolds:
            assert len(manifold.points) > 100
            if manifold.kind == "unstable":
                assert all(abs(y) < 1.0e-6 for _, y in manifold.points)
            else:
                assert all(abs(x - 0.5) < 1.0e-6 for x, _ in manifold.points)

    def test_describe_labels_each_frame(self):
        scene = stage_scene(
            Continued(_equation(), 0, _saddle_branch()),
            _film([0.0], describe=lambda frame: f"{len(frame.equilibria)} equilibrium"),
        )
        assert scene.frames[0].label == "1 equilibrium"

    def test_leaves_the_parameter_at_the_last_frame(self):
        equation = _equation()
        stage_scene(Continued(equation, 0, _saddle_branch()), _film([0.0, 0.5]))
        assert equation.derivative.parameters[0].value == 0.5

    def test_carries_the_branch_and_the_terminus_through(self):
        scene = stage_scene(
            Continued(_equation(), 0, _saddle_branch(), terminus="the edge"),
            _film([0.0]),
        )
        assert [point.parameter for point in scene.timeline.arcs[0]] == [0.0, 0.5, 1.0]
        assert scene.cycles.terminus == "the edge"
        assert scene.cycles.cycles == []


class Ring3(DeterministicFunction):
    """x_i' = a x_i - x_i^3 - p x_{i+1}: a three-cell ring with a = 1."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        p = self.parameters[0].value
        return [point[i] - point[i] ** 3 - p * point[(i + 1) % 3] for i in range(3)]


def _ring() -> DifferentialEquation:
    parameters = [Shift(value=0.0)]
    return DifferentialEquation(
        variables=[State(), State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=Ring3(
            variables=[State(), State(), State()],
            parameters=parameters,
            results=[State(), State(), State()],
            time=None,
        ),
    )


def _ring_terms() -> list[list[Term]]:
    terms = []
    for i in range(3):
        own = [0, 0, 0]
        own[i] = 1
        cube = [0, 0, 0]
        cube[i] = 3
        neighbour = [0, 0, 0]
        neighbour[(i + 1) % 3] = 1
        terms.append([Term(1.0, own), Term(-1.0, cube), Term(-1.0, neighbour, 1)])
    return terms


def _ring_view(terms=None) -> View:
    return View(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [(-1.6, 1.6)] * 3,
        _ring_terms() if terms is None else terms,
    )


def _ring_point(parameter: float, s: float, stability: str) -> ContinuationPoint:
    return ContinuationPoint(
        arclength=0.0,
        state=[s, s, s],
        parameter=parameter,
        tangent=[0.0, 0.0, 0.0, 1.0],
        eigenvalues=[(-2.0, 0.0), (-0.5, 1.0), (-0.5, -1.0)],
        unstable_dimension=0,
        stability=stability,
        measure=s,
    )


class TestTranscendentalField(TestCase):
    """A field the jets reject can still be staged, via finite differences."""

    def test_saddle_manifolds_of_a_field_with_a_radius(self):
        class Radial(DeterministicFunction):
            """A saddle written with a square root, which Jets refuse."""

            def eval(self, point, time=None):  # noqa: ARG002 (base signature)
                x, y = point[0], point[1]
                radius = (x * x + y * y + 1.0) ** 0.5
                return [x / radius, -y / radius]

        function = Radial(
            variables=[State(), State()],
            parameters=[Shift(value=0.0)],
            results=[State(), State()],
            time=None,
        )
        manifolds = saddle_manifolds(function, np.zeros(2), ((-2.0, 2.0), (-2.0, 2.0)))
        assert sorted(m.kind for m in manifolds) == [
            "stable",
            "stable",
            "unstable",
            "unstable",
        ]
        for manifold in manifolds:
            assert len(manifold.points) > 100


class TestSpecialPoints(TestCase):
    def test_a_fold_at_a_pitchfork_is_the_pitchfork(self):
        branch = Branch()
        branch.append_point(_point(0.0, -1.0, "stable"))
        branch.append_point(_point(1.0, 0.0, "stable"))
        branch.append_point(_point(0.0, 1.0, "saddle"))
        fold = _point(1.0, 0.0)
        fold.kind = "fold"
        pitchfork = _point(1.0, 0.0)
        pitchfork.kind = "pitchfork"
        for special in (fold, pitchfork, pitchfork):
            branch.append_special_point(special)
        hopf = _point(0.5, -0.5)
        hopf.kind = "hopf"
        branch.append_special_point(hopf)
        scene = stage_scene(Continued(_equation(), 0, [branch, branch]), _film([0.0]))
        assert sorted(s.kind for s in scene.timeline.special) == ["hopf", "pitchfork"]


class TestView(TestCase):
    def test_terms_reproduce_the_ring(self):
        equation = _ring()
        equation.derivative.parameters[0].value = 0.3
        for state in ([0.2, -0.7, 1.1], [1.0, 1.0, 1.0], [0.0, 0.5, -0.5]):
            expected = equation.derivative.eval(point=state, time=None)
            claimed = evaluate_terms(_ring_terms(), state, 0.3)
            assert all(
                abs(a - b) < 1.0e-12 for a, b in zip(claimed, expected, strict=True)
            )

    def test_a_wrong_field_is_refused(self):
        terms = _ring_terms()
        terms[1][0] = Term(2.0, [0, 1, 0])  # a mistyped gain on cell 2
        branch = Branch()
        branch.append_point(_ring_point(0.0, 1.0, "stable"))
        with pytest.raises(ValueError, match="disagrees"):
            stage_scene(
                Continued(_ring(), 0, branch),
                Film([0.0], Lattice(_UNIT, 3, view=_ring_view(terms)), _SYSTEM),
            )

    def test_projects_equilibria_and_cycles_and_skips_field_and_manifolds(self):
        branch = Branch()
        for p, s in ((-0.5, 1.2), (0.0, 1.0), (0.5, 0.7)):
            branch.append_point(_ring_point(p, s, "saddle"))
        scene = stage_scene(
            Continued(_ring(), 0, [branch, branch]),
            Film([0.0, 0.25], Lattice(_UNIT, 3, view=_ring_view()), _SYSTEM),
        )
        first, second = scene.frames
        assert first.field == []
        assert second.field == []
        assert first.manifolds == []  # a 3-D saddle's manifolds are not curves
        assert first.cycles == []
        assert [(e.x, e.y) for e in first.equilibria] == [(1.0, 1.0)]  # deduplicated
        assert abs(second.equilibria[0].x - 0.85) < 1.0e-12
        assert len(scene.timeline.arcs) == 2
        assert stage_payload(scene)["view"]["field"][0][2] == {
            "c": -1.0,
            "e": [0, 1, 0],
            "q": 1,
        }


class TestPayloadAndRenderer(TestCase):
    @staticmethod
    def _scene() -> StageScene:
        frame = Frame(
            0.0,
            [0.0] * 8,
            [Equilibrium(0.0, 0.0, "stable", [(-1.0, 2.0), (-1.0, -2.0)])],
            [],
            [0],
        )
        frame.label = "one stable focus"
        cycle = Cycle(
            0.0,
            6.28,
            0.5,
            [(1.0, 0.0), (0.5, 0.0)],
            [(0.5, 0.0), (0.0, 0.5)],
        )
        cycle.error = 0.03
        system = StageSystem(
            "Focus & cycle",
            "A <test> subtitle",
            "μ",
            equation=r"\dot x = -x",
            note="Measured on nothing.",
        )
        return StageScene(
            system,
            Lattice(_UNIT, 2, "u", "w"),
            Timeline([], []),
            CycleBranch([cycle], "homoclinic"),
            [frame],
        )

    def test_payload_is_plain_json_in_the_page_shape(self):
        payload = stage_payload(self._scene())
        json.dumps(payload)
        assert payload["grid"] == {"nx": 2, "ny": 2}
        assert payload["system"]["x_label"] == "u"
        assert payload["system"]["cycle_terminus"] == "homoclinic"
        assert payload["frames"][0]["cycles"] == [0]
        assert payload["frames"][0]["label"] == "one stable focus"
        eig = payload["frames"][0]["equilibria"][0]["eig"]
        assert eig == [[-1.0, 2.0], [-1.0, -2.0]]
        assert payload["cycles"][0]["multipliers"] == [[1.0, 0.0], [0.5, 0.0]]
        assert payload["cycles"][0]["error"] == 0.03

    def test_renderer_embeds_everything_and_leaves_no_placeholder(self):
        html = StageRenderer(library="/*d3*/").render(self._scene())
        assert "__" not in html.replace("__STAGE__", "")
        assert "/*d3*/" in html
        assert "window.__STAGE__ = {" in html
        assert "A &lt;test&gt; subtitle" in html
        assert "Measured on nothing." in html
        assert "<svg" in html  # the typeset equation
        assert '"cycle_terminus": "homoclinic"' in html

    def test_atlas_marks_a_stage_card(self):
        html = _atlas_html(
            [
                AtlasEntry("a.html", "A plot", "still"),
                AtlasEntry("b.html", "A stage", "moving", kind="stage"),
            ],
        )
        assert html.count('class="card"') == 1
        assert html.count('class="card stage"') == 1
        assert "play" in html
