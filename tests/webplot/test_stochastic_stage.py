"""The stochastic stage, built and rendered without a browser, against closed forms.

Two oracles carry the threshold-locating pipeline. The one-dimensional pitchfork
``dx = (a x - x^3) dt + s x o dW`` (Stratonovich) has its phenomenological
threshold at ``a = s^2 / 2`` and its dynamical one at ``a = 0``, both in closed
form, and is what the pipeline is pointed at first. The planar Hopf drift with
conformal noise ``s z dB`` has ``D = s^2 r^2 I``, so its density is radial and
craters at ``mu = s^2`` exactly, while the origin's exponent is ``mu`` itself.
The grids here are coarse enough for the fast loop, so the planar check is
against what the closed-form density gives at the probe's own cells rather than
against ``s^2`` outright; ``tests/continuation`` does the fine-grid version.
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

from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.continuation_point import ContinuationPoint
from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    GridDensity,
    StationaryFokkerPlanck,
)
from discrecontinual_equations.continuation.noise import NoiseMatrix
from discrecontinual_equations.continuation.root_finder import (
    RootFinderSettings,
    Secant,
)
from discrecontinual_equations.continuation.stochastic import (
    Ito,
    LyapunovExponent,
    Stratonovich,
)
from discrecontinual_equations.continuation.stochastic_threshold import RadialCrater
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.webplot.stage import (
    Lattice,
    Projection,
    StageSystem,
    Term,
    View,
    stage_payload,
)
from discrecontinual_equations.webplot.stage_builder import (
    Continued,
    Film,
    stage_scene,
)
from discrecontinual_equations.webplot.stage_renderer import StageRenderer
from discrecontinual_equations.webplot.stochastic_stage_builder import (
    D_BIFURCATION,
    P_BIFURCATION,
    ClosedForms,
    Diagnostics,
    Measurements,
    NoiseModel,
    Thresholds,
    check_noise,
    density_window,
    frame_density,
    locate_thresholds,
    stochastic_stage,
)
from discrecontinual_equations.webplot.stochastic_stage_renderer import (
    StochasticStageRenderer,
)
from tests.continuation.fields import (
    Alpha,
    ConformalColumn,
    DiagonalMultiplicativeColumn,
    Diffusion,
    Drift,
    HopfField,
    Sigma,
    State,
    Time,
    _scalar,
)

_SIGMA = 0.7
_BOX = ((-1.5, 1.5), (-1.5, 1.5))
_IDENTITY = [[1.0, 0.0], [0.0, 1.0]]
_SYSTEM = StageSystem("Noisy Hopf", "", "μ")


def _parameters(mu):
    return [Alpha(value=mu), Sigma(value=_SIGMA)]


def _hopf(parameters):
    return HopfField(
        variables=[State(), State()],
        parameters=parameters,
        results=[State(), State()],
        time=None,
    )


def _columns(column_type, parameters):
    return NoiseMatrix(
        [
            column_type(index, [State(), State()], parameters, [State(), State()])
            for index in range(2)
        ],
    )


def _hopf_terms():
    return [
        [
            Term(1.0, [1, 0], 1),
            Term(-1.0, [0, 1]),
            Term(-1.0, [3, 0]),
            Term(-1.0, [1, 2]),
        ],
        [
            Term(1.0, [1, 0]),
            Term(1.0, [0, 1], 1),
            Term(-1.0, [2, 1]),
            Term(-1.0, [0, 3]),
        ],
    ]


def _conformal_terms(s=_SIGMA):
    return [
        [[Term(s, [1, 0])], [Term(s, [0, 1])]],
        [[Term(-s, [0, 1])], [Term(s, [1, 0])]],
    ]


def _even_grid(cells=40, half=2.5):
    # An even count keeps every cell centre off the origin, where conformal
    # noise vanishes and the diffusion is singular.
    return DensityGrid.box([-half, -half], [half, half], [cells, cells])


def _hopf_density_at(mu, grid):
    if mu <= 0.0:
        return None
    parameters = _parameters(mu)
    return StationaryFokkerPlanck(
        _hopf(parameters),
        _columns(ConformalColumn, parameters),
        Ito(),
        grid,
    )


def _secant():
    return Secant(
        RootFinderSettings(
            value_tolerance=1.0e-8,
            fraction_tolerance=1.0e-5,
            max_iterations=8,
        ),
    )


def _thresholds(probe, phenomenological=(0.15, 1.0), dynamical=(-0.5, 0.5)):
    return Thresholds(probe, phenomenological, dynamical, _secant())


def _origin_branch(parameters):
    branch = Branch()
    for p in parameters:
        stable = p < 0.0
        branch.append_point(
            ContinuationPoint(
                arclength=0.0,
                state=[0.0, 0.0],
                parameter=p,
                tangent=[0.0, 0.0, 1.0],
                eigenvalues=[(p, 1.0), (p, -1.0)],
                unstable_dimension=0 if stable else 2,
                stability="stable" if stable else "unstable",
                measure=0.0,
            ),
        )
    return branch


def _equation(parameters):
    return DifferentialEquation(
        variables=[State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=_hopf(parameters),
    )


def _deterministic():
    continued = Continued(
        _equation(_parameters(-0.5)),
        0,
        _origin_branch([-0.5, 0.0, 1.5]),
    )
    return continued, stage_scene(
        continued,
        Film([-0.25, 1.0], Lattice(_BOX, 3), _SYSTEM),
    )


def _viewed(frames):
    continued = Continued(
        _equation(_parameters(frames[0])),
        0,
        _origin_branch([-0.5, 0.0, 1.5]),
    )
    view = View([Projection("x - y", _IDENTITY, ("x", "y"))], list(_BOX), _hopf_terms())
    scene = stage_scene(continued, Film(frames, Lattice(_BOX, 3, view=view), _SYSTEM))
    return continued, scene


def _probe_prediction(grid, offset):
    """What the closed-form density says at the probe's two cells.

    The probe reads ``log p`` at two radii; with ``p = r^a exp(-r^2 / s^2)`` its
    zero is at ``a = (r2^2 - r1^2) / (s^2 log(r2 / r1))`` rather than at
    ``a = 0``, and ``mu = s^2 (1 + a / 2)``. That is the probe's own bias, which
    a fine grid makes small and a coarse one does not.
    """
    index = list(grid.nearest([0.0, 0.0]))
    index[0] += offset
    x1, y1 = grid.axis(0)[index[0]], grid.axis(1)[index[1]]
    x2 = grid.axis(0)[index[0] + 1]
    r1, r2 = math.hypot(x1, y1), math.hypot(x2, y1)
    a = ((r2 * r2 - r1 * r1) / _SIGMA**2) / math.log(r2 / r1)
    return _SIGMA**2 * (1.0 + 0.5 * a)


class TestDeterministicUnchanged(TestCase):
    """A deterministic film must not notice that stochastic ones exist."""

    def test_payload_has_no_stochastic_keys(self):
        _continued, scene = _deterministic()
        payload = stage_payload(scene)
        assert "stochastic" not in payload
        assert all("density" not in frame for frame in payload["frames"])

    def test_the_stochastic_renderer_draws_it_exactly_as_the_stage_renderer(self):
        _continued, scene = _deterministic()
        before = StageRenderer(library="/*d3*/").render(scene)
        after = StochasticStageRenderer(library="/*d3*/").render(scene)
        assert before == after
        assert 'id="heat"' not in after


def _synthetic(grid, profile):
    first, second = np.meshgrid(grid.axis(0), grid.axis(1), indexing="ij")
    raw = profile(np.hypot(first, second))
    return GridDensity(grid, raw / (raw.sum() * grid.cell_volume))


class TestFrameDensity(TestCase):
    def test_a_ring_reports_its_crest_and_a_window_of_values(self):
        grid = _even_grid(40, 2.0)
        density = _synthetic(grid, lambda r: np.exp(-((r - 0.8) ** 2) / 0.01))
        frame = frame_density(density, ((-1.0, 1.0), (-1.0, 1.0)), (0.0, 0.0))
        cell = float(grid.spacings[0])
        assert frame.crest is not None
        assert abs(frame.crest - 0.8) < cell
        assert len(frame.maxima) > 8
        assert len(frame.values) == 20 * 20
        assert max(frame.values) == 1.0
        assert frame.peak > 0.0

    def test_a_centred_peak_has_no_crest(self):
        grid = _even_grid(40, 2.0)
        density = _synthetic(grid, lambda r: np.exp(-(r**2) / 0.3))
        frame = frame_density(density, ((-1.0, 1.0), (-1.0, 1.0)), (0.0, 0.0))
        assert frame.crest is None
        assert math.hypot(*frame.mode) < float(grid.spacings[0])

    def test_a_ring_counts_even_while_the_centre_is_higher(self):
        # The grid overstates the centre cells of a singular diffusion; the
        # ring is a ring as soon as it is one, whoever is taller.
        grid = _even_grid(40, 2.0)
        density = _synthetic(
            grid,
            lambda r: 3.0 * np.exp(-(r**2) / 0.02) + np.exp(-((r - 0.8) ** 2) / 0.01),
        )
        frame = frame_density(density, ((-1.0, 1.0), (-1.0, 1.0)), (0.0, 0.0))
        assert math.hypot(*frame.mode) < float(grid.spacings[0])
        assert frame.crest is not None
        assert abs(frame.crest - 0.8) < float(grid.spacings[0])

    def test_values_are_rows_of_constant_y(self):
        grid = _even_grid(40, 2.0)
        density = _synthetic(grid, lambda r: np.exp(-(r**2) / 0.3))
        frame = frame_density(density, ((-1.0, 1.0), (-1.0, 1.0)), (0.0, 0.0))
        xs, ys, _extent = density_window(grid, ((-1.0, 1.0), (-1.0, 1.0)))
        block = density.values[xs, ys]
        nx = block.shape[0]
        # value at (ix = 3, iy = 7) sits at index iy * nx + ix
        expected = block[3, 7] / block.max()
        assert abs(frame.values[7 * nx + 3] - expected) < 1.0e-12

    def test_the_window_is_the_cells_inside_it_and_their_edges(self):
        grid = _even_grid(40, 2.0)
        xs, ys, extent = density_window(grid, ((-1.0, 1.0), (-1.0, 1.0)))
        assert xs.stop - xs.start == 20
        assert ys.stop - ys.start == 20
        half = 0.5 * float(grid.spacings[0])
        assert abs(extent[0][0] - (float(grid.axis(0)[xs.start]) - half)) < 1.0e-12
        assert abs(extent[0][1] - (float(grid.axis(0)[xs.stop - 1]) + half)) < 1.0e-12
        with pytest.raises(ValueError, match="holds no cell"):
            density_window(grid, ((5.0, 6.0), (-1.0, 1.0)))


class TestLocateThresholds(TestCase):
    """The pipeline on systems whose two thresholds are known in closed form."""

    def test_the_pitchfork_puts_p_at_half_the_variance_and_d_at_zero(self):
        sigma = 0.6
        grid = DensityGrid([np.linspace(0.05, 3.0, 601)])

        def density_at(a, on):
            return StationaryFokkerPlanck(
                _scalar(Drift, a, sigma),
                NoiseMatrix([_scalar(Diffusion, a, sigma)]),
                Stratonovich(),
                on,
            )

        def exponent_at(a):
            exponent = LyapunovExponent(
                _scalar(Drift, a, sigma),
                _scalar(Diffusion, a, sigma),
                Stratonovich(),
            )
            return exponent.at(0.0)

        located = locate_thresholds(
            Diagnostics(
                Measurements(density_at, exponent_at, (-0.3, 0.0, 0.3)),
                _thresholds(RadialCrater([0.05]), (0.05, 0.5), (-0.3, 0.3)),
                grid,
                centre=(0.0,),
            ),
        )
        assert abs(located.phenomenological - 0.5 * sigma * sigma) < 1.0e-2
        assert abs(located.dynamical) < 1.0e-6
        sampled = [p for p, _v in located.exponents]
        assert all(value in sampled for value in (-0.3, 0.0, 0.3))

    def test_the_conformal_hopf_craters_where_the_closed_form_says(self):
        # Four cells out on a 60-cell grid the finite-volume error is under a
        # percent of the density; what is left is the probe's own bias, which
        # the closed form predicts at these cells. A grid fine enough to bring
        # that under two percent of s^2 belongs to the slow suite.
        grid = _even_grid(60)
        located = locate_thresholds(
            Diagnostics(
                Measurements(_hopf_density_at, lambda mu: mu, (-0.5, 1.5)),
                _thresholds(RadialCrater([0.0, 0.0], axis=0, offset=4)),
                grid,
            ),
        )
        assert abs(located.phenomenological - _probe_prediction(grid, 4)) < 0.03
        assert located.phenomenological > _SIGMA**2
        assert abs(located.dynamical) < 1.0e-9

    def test_a_finer_locating_grid_is_used_when_given(self):
        coarse, fine = _even_grid(20), _even_grid(60)
        located = locate_thresholds(
            Diagnostics(
                Measurements(_hopf_density_at, lambda mu: mu, ()),
                Thresholds(
                    RadialCrater([0.0, 0.0], axis=0, offset=4),
                    (0.15, 1.0),
                    (-0.5, 0.5),
                    _secant(),
                    grid=fine,
                ),
                coarse,
            ),
        )
        assert abs(located.phenomenological - _probe_prediction(fine, 4)) < 0.03

    def test_a_bracket_without_a_sign_change_is_refused(self):
        with pytest.raises(ValueError, match="does not change sign"):
            locate_thresholds(
                Diagnostics(
                    Measurements(_hopf_density_at, lambda mu: mu, ()),
                    _thresholds(
                        RadialCrater([0.0, 0.0], offset=1),
                        dynamical=(0.1, 0.5),
                    ),
                    _even_grid(20),
                ),
            )

    def test_a_bracket_reaching_where_no_density_exists_is_refused(self):
        with pytest.raises(ValueError, match="no stationary density"):
            locate_thresholds(
                Diagnostics(
                    Measurements(_hopf_density_at, lambda mu: mu, ()),
                    _thresholds(
                        RadialCrater([0.0, 0.0], offset=1),
                        phenomenological=(-0.2, 1.0),
                    ),
                    _even_grid(20),
                ),
            )


class TestCheckNoise(TestCase):
    def test_conformal_terms_pass_and_their_drift_is_already_ito(self):
        parameters = _parameters(0.3)
        view = View(_IDENTITY, list(_BOX), _hopf_terms())
        matrix = _columns(ConformalColumn, parameters)
        check_noise(
            NoiseModel(_conformal_terms(), matrix, Ito()),
            view,
            _hopf(parameters),
            parameters[0],
            0.3,
        )
        # Under Stratonovich the conformal correction vanishes identically, so
        # the same plain drift is still the Ito drift and still passes.
        stratonovich = NoiseModel(_conformal_terms(), matrix, Stratonovich())
        check_noise(stratonovich, view, _hopf(parameters), parameters[0], 0.3)
        assert stratonovich.name == "Stratonovich"
        assert stratonovich.paths().convention == "Stratonovich"

    def test_a_wrong_noise_term_is_refused(self):
        parameters = _parameters(0.3)
        terms = _conformal_terms()
        terms[1][0] = [Term(_SIGMA, [0, 1])]  # the sign of -s y mistyped
        view = View(_IDENTITY, list(_BOX), _hopf_terms())
        noise = NoiseModel(terms, _columns(ConformalColumn, parameters), Ito())
        with pytest.raises(ValueError, match="disagrees with its matrix"):
            check_noise(noise, view, _hopf(parameters), parameters[0], 0.3)

    def test_a_stratonovich_drift_left_uncorrected_is_refused(self):
        # diag(s x, s y) shifts the Ito drift by s^2 (x, y) / 2, so the plain
        # Hopf terms are not what Euler-Maruyama should integrate.
        parameters = [Sigma(value=_SIGMA), Sigma(value=_SIGMA)]
        terms = [
            [[Term(_SIGMA, [1, 0])], []],
            [[], [Term(_SIGMA, [0, 1])]],
        ]
        view = View(_IDENTITY, list(_BOX), _hopf_terms())
        matrix = _columns(DiagonalMultiplicativeColumn, parameters)
        drift = _hopf(_parameters(0.3))
        with pytest.raises(ValueError, match="Ito drift"):
            check_noise(
                NoiseModel(terms, matrix, Stratonovich()),
                view,
                drift,
                drift.parameters[0],
                0.3,
            )
        check_noise(
            NoiseModel(terms, matrix, Ito()),
            view,
            drift,
            drift.parameters[0],
            0.3,
        )


def _build(frames=(-0.25, 0.25, 1.0)):
    continued, scene = _viewed(list(frames))
    parameters = continued.equation.derivative.parameters
    noise = NoiseModel(
        _conformal_terms(),
        _columns(ConformalColumn, parameters),
        Ito(),
        seed=11,
        step=0.01,
    )
    diagnostics = Diagnostics(
        Measurements(
            _hopf_density_at,
            lambda mu: mu,
            (-0.25, 1.0),
            ClosedForms(
                exponent=lambda mu: mu,
                crest=lambda mu: math.sqrt(mu - _SIGMA**2) if mu > _SIGMA**2 else None,
            ),
        ),
        _thresholds(RadialCrater([0.0, 0.0], axis=0, offset=2)),
        _even_grid(40),
    )
    return stochastic_stage(
        scene,
        continued,
        noise,
        diagnostics,
        lambda frame: "spike" if frame.density.crest is None else "crater",
    )


class TestStochasticStage(TestCase):
    def test_frames_carry_densities_that_follow_the_closed_form(self):
        scene = _build()
        below, between, above = scene.frames
        assert below.density is not None
        assert below.density.values == []  # no density on the plane at mu < 0
        assert between.density.crest is None  # past D, before P: still a spike
        assert above.density.crest is not None
        cell = 5.0 / 39.0
        assert abs(above.density.crest - math.sqrt(1.0 - _SIGMA**2)) < cell
        assert [frame.label for frame in scene.frames] == ["spike", "spike", "crater"]

    def test_both_thresholds_land_on_the_timeline_at_their_own_values(self):
        scene = _build()
        kinds = {point.kind: point for point in scene.timeline.special}
        assert P_BIFURCATION in kinds
        assert D_BIFURCATION in kinds
        assert abs(kinds[D_BIFURCATION].parameter) < 1.0e-9
        assert (
            abs(kinds[P_BIFURCATION].parameter - _probe_prediction(_even_grid(40), 2))
            < 0.05
        )
        assert "μ" in kinds[P_BIFURCATION].label
        assert kinds[P_BIFURCATION].parameter > kinds[D_BIFURCATION].parameter

    def test_payload_carries_the_stochastic_block_in_the_page_shape(self):
        scene = _build()
        payload = stage_payload(scene)
        json.dumps(payload)
        block = payload["stochastic"]
        assert block["convention"] == "Ito"
        assert block["seed"] == 11
        assert block["step"] == 0.01
        assert block["centre"] == [0.0, 0.0]
        assert len(block["noise"]) == 2
        assert all(len(driver) == 2 for driver in block["noise"])
        assert block["noise"][1][0] == [{"c": -_SIGMA, "e": [0, 1], "q": 0}]
        assert block["density"]["nx"] == block["density"]["ny"]
        assert block["density"]["nx"] * block["density"]["ny"] == len(
            payload["frames"][2]["density"]["values"],
        )
        assert payload["frames"][0]["density"]["values"] == []
        assert block["exponent"][0][0] == -0.5  # the bracket's end, memoised
        assert block["exact_exponent"] == [[-0.25, -0.25], [0.25, 0.25], [1.0, 1.0]]
        assert [p for p, _r in block["exact_crest"]] == [1.0]

    def test_a_scene_without_a_view_is_refused(self):
        continued, scene = _deterministic()
        with pytest.raises(ValueError, match="View"):
            stochastic_stage(
                scene,
                continued,
                NoiseModel(
                    _conformal_terms(),
                    _columns(ConformalColumn, continued.equation.derivative.parameters),
                    Ito(),
                ),
                Diagnostics(
                    Measurements(_hopf_density_at, lambda mu: mu, ()),
                    _thresholds(RadialCrater([0.0, 0.0], offset=1)),
                    _even_grid(20),
                ),
            )


class TestStochasticRenderer(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = StochasticStageRenderer(library="/*d3*/").render(_build())

    def test_the_page_has_its_panels_and_no_placeholder_left(self):
        html = self.html
        assert "__" not in html.replace("__STAGE__", "")
        assert 'id="heat"' in html
        assert 'id="exp"' in html
        assert 'id="clock"' not in html
        assert '"stochastic": {' in html
        assert "Ito" in html
        assert 'id="l-cycle"' in html  # the shared skeleton reads these two
        assert 'id="l-manifolds"' in html

    @pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
    def test_every_script_block_parses(self):
        # A JavaScript typo renders a blank page that no payload test catches.
        blocks = re.findall(r"<script>(.*?)</script>", self.html, re.DOTALL)
        assert len(blocks) == 3
        node = shutil.which("node")
        for index, block in enumerate(blocks):
            path = Path(f"/tmp/stochastic_stage_block_{index}.js")  # noqa: S108
            path.write_text(block)
            subprocess.run([node, "--check", str(path)], check=True)  # noqa: S603
