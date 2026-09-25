"""The stage: a continuation re-shaped by frame, built and rendered without a browser.

The system under test is the linear saddle ``x' = x - p, y' = -y`` with its
equilibrium at ``(p, 0)``: the field, the manifolds and the eigenvalues are all
known in closed form, so every number the builder produces can be checked.
"""

import json
from unittest import TestCase

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
    Timeline,
    stage_payload,
)
from discrecontinual_equations.webplot.stage_builder import (
    Continued,
    Film,
    frame_equilibria,
    nearest_cycle,
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


def _film(frames, box=_UNIT, grid=3, describe=None) -> Film:
    return Film(frames, Lattice(box, grid), StageSystem("Saddle", "", "p"), describe)


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


class TestNearestCycle(TestCase):
    def test_picks_the_cycle_within_half_a_frame(self):
        cycles = [Cycle(p, 6.0, 0.1, [(1.0, 0.0)], [(0.0, 0.0)]) for p in (0.0, 0.1)]
        assert nearest_cycle(cycles, 0.04, 0.1) == 0
        assert nearest_cycle(cycles, 0.16, 0.1) is None
        assert nearest_cycle([], 0.0, 0.1) is None


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
            assert frame.cycle is None
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
        assert [point.parameter for point in scene.timeline.points] == [0.0, 0.5, 1.0]
        assert scene.cycles.terminus == "the edge"
        assert scene.cycles.cycles == []


class TestPayloadAndRenderer(TestCase):
    @staticmethod
    def _scene() -> StageScene:
        frame = Frame(
            0.0,
            [0.0] * 8,
            [Equilibrium(0.0, 0.0, "stable", [(-1.0, 2.0), (-1.0, -2.0)])],
            [],
            0,
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
        assert payload["frames"][0]["cycle"] == 0
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
