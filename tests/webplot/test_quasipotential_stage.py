"""The optional quasipotential panel, built and rendered without a browser.

The system is the stochastic stage's own: the Hopf drift with conformal noise
``s z dB``, whose quasipotential about the cycle ``r = sqrt(mu)`` is radial in
closed form, ``U = (r^2 - mu - mu log(r^2 / mu)) / s^2``. The grid is coarse
enough for the fast loop; ``tests/continuation/test_quasipotential.py`` does
the fine-grid check of the same closed form.
"""

import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.quasipotential import (
    CycleAttractor,
    OrderedLineIntegral,
)
from discrecontinual_equations.webplot.quasipotential_stage import (
    QuasipotentialSettings,
    with_quasipotential,
)
from discrecontinual_equations.webplot.stage import stage_payload
from discrecontinual_equations.webplot.stochastic_stage_builder import density_window
from discrecontinual_equations.webplot.stochastic_stage_renderer import (
    StochasticStageRenderer,
)
from examples.continuation import stage_noisy_hopf as film
from tests.continuation.fields import ConformalColumn
from tests.webplot.test_stochastic_stage import (
    _SIGMA,
    _build,
    _columns,
    _deterministic,
    _even_grid,
    _hopf,
    _parameters,
)

_STARTS = ((0.3, 0.0), (1.35, 0.85))
_CAP = 4.0


def _family(mu, grid):
    if mu <= 0.0:
        return None
    parameters = _parameters(mu)
    angle = np.linspace(0.0, 2.0 * math.pi, 200, endpoint=False)
    cycle = math.sqrt(mu) * np.column_stack([np.cos(angle), np.sin(angle)])
    solver = OrderedLineIntegral(
        _hopf(parameters),
        _columns(ConformalColumn, parameters),
        grid,
    )
    return solver, CycleAttractor(cycle)


def _settings():
    return QuasipotentialSettings(_family, _even_grid(48, 2.2), _STARTS, _CAP)


def _with_panel():
    return with_quasipotential(_build(), _settings())


class TestQuasipotentialPanel(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = _with_panel()

    def test_frames_without_an_attractor_have_no_field(self):
        panel = self.scene.stochastic.quasipotential
        assert len(panel.frames) == len(self.scene.frames)
        assert panel.frames[0] is None  # mu < 0: the attractor is the origin
        assert all(frame is not None for frame in panel.frames[1:])

    def test_the_field_is_the_closed_form_on_the_drawn_window(self):
        panel = self.scene.stochastic.quasipotential
        nx, ny = panel.shape
        frame = panel.frames[2]  # mu = 1
        values = np.array(frame.values).reshape(ny, nx)
        (x0, x1), (y0, y1) = panel.box
        cell = (x1 - x0) / nx
        xs = np.linspace(x0 + cell / 2, x1 - cell / 2, nx)
        ys = np.linspace(y0 + cell / 2, y1 - cell / 2, ny)
        grid_x, grid_y = np.meshgrid(xs, ys)
        squared = grid_x**2 + grid_y**2
        exact = (squared - 1.0 - np.log(squared)) / _SIGMA**2
        away = (np.sqrt(squared) > 4 * cell) & (exact < _CAP)
        assert np.max(values) <= _CAP
        assert np.max(np.abs(values[away] - exact[away])) < 0.15

    def test_paths_run_from_their_start_onto_the_cycle(self):
        frame = self.scene.stochastic.quasipotential.frames[2]
        assert len(frame.paths) == len(_STARTS)
        for start, path in zip(_STARTS, frame.paths, strict=True):
            assert np.allclose(path[0], start)
            # Arrived within two cells of the cycle: on this coarse grid the
            # interpolated U stops falling about a cell and a half out.
            radius = math.hypot(*path[-1])
            assert abs(radius - 1.0) < 2.0 * _settings().grid.spacings[0]

    def test_payload_carries_the_panel_in_the_page_shape(self):
        payload = stage_payload(self.scene)
        json.dumps(payload)
        block = payload["stochastic"]["quasipotential"]
        assert block["frames"][0] is None
        assert block["nx"] * block["ny"] == len(block["frames"][2]["values"])
        assert block["frames"][2]["cap"] == _CAP
        assert len(block["frames"][2]["paths"]) == len(_STARTS)

    def test_a_stage_without_the_panel_has_no_key_and_no_markup(self):
        scene = _build()
        assert "quasipotential" not in stage_payload(scene)["stochastic"]
        html = StochasticStageRenderer(library="/*d3*/").render(scene)
        assert 'id="qp-panel"' not in html
        assert 'id="l-quasipotential"' not in html

    def test_a_deterministic_stage_is_refused(self):
        _continued, scene = _deterministic()
        with pytest.raises(ValueError, match="stochastic"):
            with_quasipotential(scene, _settings())


class TestQuasipotentialRenderer(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = StochasticStageRenderer(library="/*d3*/").render(_with_panel())

    def test_the_page_has_the_panel_and_no_placeholder_left(self):
        html = self.html
        assert "__" not in html.replace("__STAGE__", "")
        assert 'id="qp-panel"' in html
        assert 'id="qp-heat"' in html
        assert 'id="l-quasipotential"' in html
        assert '"quasipotential": {' in html

    @pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
    def test_every_script_block_parses(self):
        blocks = re.findall(r"<script>(.*?)</script>", self.html, re.DOTALL)
        assert len(blocks) == 3
        node = shutil.which("node")
        for index, block in enumerate(blocks):
            path = Path(f"/tmp/quasipotential_stage_block_{index}.js")  # noqa: S108
            path.write_text(block)
            subprocess.run([node, "--check", str(path)], check=True)  # noqa: S603


class TestNoisyHopfFilmQuasipotential(TestCase):
    """The film's own marches, on its own grid, against its own closed form.

    This is the only test that imports an example. It does so because what it
    checks is the film itself - its grid, box, cap and closed form, as the page
    quotes them - and a copy of those here would let the page drift from them.

    Measured on the built page, within ``U <= 4`` on the drawn box and beyond four
    cells of the origin: 0.03 to 0.08 for every frame with ``mu >= 0.1``, and 0.19
    at ``mu = 0.025``, where the cycle is three cells from the singular origin.
    """

    def test_the_marched_u_follows_the_closed_form_the_page_quotes(self):
        grid = film.quasipotential_grid()
        xs, ys, _extent = density_window(grid, film.BOX)
        points = grid.points().reshape(*grid.shape, 2)[xs, ys].reshape(-1, 2)
        radius = np.linalg.norm(points, axis=1)
        for mu, bar in ((0.025, 0.25), (1.0, 0.08)):
            problem = film.quasipotential_at(mu, grid)
            assert problem is not None
            solver, attractor = problem
            values = solver.solve(attractor).values[xs, ys].ravel()
            exact = film.exact_quasipotential(mu, points)
            kept = (exact <= film.QUASIPOTENTIAL_CAP) & (radius > 4 * grid.spacings[0])
            assert np.max(np.abs(values[kept] - exact[kept])) < bar
        assert film.quasipotential_at(0.0, grid) is None
