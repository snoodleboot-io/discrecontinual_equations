"""Scene shaping and D3 rendering for the webplot subsystem.

The scene builders are tested against known branch/result structure, and the D3
renderer is checked for a self-contained document carrying the scene data - all
without a browser, the same way the plotly renderer's data shaping is tested.
"""

import json
import re
from unittest import TestCase

from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.continuation_point import ContinuationPoint
from discrecontinual_equations.webplot.renderer import D3Renderer
from discrecontinual_equations.webplot.scene import Axes, Scene, Series
from discrecontinual_equations.webplot.scene_builder import (
    bifurcation_scene,
    with_equation,
)


def _point(parameter: float, measure: float, stability: str, kind: str):
    return ContinuationPoint(
        arclength=0.0,
        state=[measure],
        parameter=parameter,
        tangent=[0.0, 1.0],
        eigenvalues=[],
        unstable_dimension=0,
        stability=stability,
        measure=measure,
        kind=kind,
    )


class TestSceneBuilder(TestCase):
    def test_bifurcation_scene_splits_stability_and_marks_folds(self):
        points = [
            _point(1.0, 1.0, "stable", "none"),
            _point(0.5, 0.7, "stable", "none"),
            _point(0.0, 0.0, "stable", "fold"),
            _point(0.5, -0.7, "unstable", "none"),
        ]
        branch = Branch()
        for point in points:
            branch.append_point(point)
        branch.append_special_point(points[2])
        scene = bifurcation_scene(branch, "Fold", "a fold")
        lines = [s for s in scene.series if s.role == "line"]
        markers = [s for s in scene.series if s.role == "marker"]
        assert {line.kind for line in lines} == {"stable", "unstable"}
        assert len(markers) == 1
        assert markers[0].kind == "fold"
        assert markers[0].points == [(0.0, 0.0)]


class TestD3Renderer(TestCase):
    def test_render_is_self_contained_and_carries_scene(self):
        scene = Scene(
            "Title",
            "subtitle",
            Axes("p_b", "p_a"),
            [
                Series("line", "curve", "curve", [(0.0, 0.0), (1.0, 1.0)]),
                Series("marker", "swallowtail", "swallowtail", [(0.5, 0.5)]),
            ],
        )
        html = D3Renderer().render(scene)
        # The D3 library is inlined, so no tag loads anything externally.
        assert "<script src" not in html
        assert "<link" not in html
        assert 'href="index.html"' in html
        match = re.search(r"window\.__SCENE__ = (\{.*?\});</script>", html, re.DOTALL)
        assert match is not None
        payload = json.loads(match.group(1))
        kinds = {series["kind"] for series in payload["series"]}
        assert "swallowtail" in kinds
        assert payload["xLabel"] == "p_b"

    def test_render_reuses_injected_library(self):
        scene = Scene("t", "s", Axes("x", "y"), [])
        html = D3Renderer(library="/* stub d3 */").render(scene)
        assert "/* stub d3 */" in html


class TestPublicationFeatures(TestCase):
    def test_equation_renders_as_inline_svg(self):
        scene = with_equation(
            Scene("t", "s", Axes("x", "y"), []),
            r"\dot{x} = \lambda - x^2",
        )
        html = D3Renderer().render(scene)
        assert '<svg fill="currentColor"' in html
        assert '<div class="eqn" id="equation"><svg' in html

    def test_absent_equation_leaves_slot_empty(self):
        html = D3Renderer().render(Scene("t", "s", Axes("x", "y"), []))
        assert '<div class="eqn" id="equation"></div>' in html

    def test_three_themes_and_line_type_cycle_present(self):
        html = D3Renderer().render(Scene("t", "s", Axes("x", "y"), []))
        for name in ("dark", "light", "print"):
            assert f'data-theme="{name}"' in html
        assert "DASH_CYCLE" in html
        assert "mono:true" in html

    def test_surface_carries_slice_projection_controls(self):
        html = D3Renderer().render(Scene("t", "s", Axes("x", "y"), []))
        assert "drawSurfaceSlice" in html
        assert "mountSurfaceTools" in html
        assert 'id="tools"' in html
