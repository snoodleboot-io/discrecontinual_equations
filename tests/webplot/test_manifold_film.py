"""The manifold film: a grown surface, scrubbed by geodesic distance."""

import json
import re
from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.geodesic_level_sets import (
    GeodesicLevelSets,
    GeodesicSettings,
)
from discrecontinual_equations.continuation.manifold import StableManifold
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable
from discrecontinual_equations.webplot.manifold_film import (
    FilmCurve,
    FilmMarker,
    ManifoldFilm,
    film_payload,
)
from discrecontinual_equations.webplot.manifold_film_renderer import (
    ManifoldFilmRenderer,
)


class Rate(Parameter, name="Rate", abbreviation="k"):
    """Unused; the field needs one parameter."""


class Coordinate(Variable, name="Coordinate", abbreviation="q"):
    """One state coordinate."""


class Saddle3(DeterministicFunction):
    """x' = -x, y' = -2y, z' = z: the stable manifold is the plane z = 0."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        return [-point[0], -2.0 * point[1], point[2]]


def _film() -> ManifoldFilm:
    function = Saddle3(
        variables=[Coordinate() for _ in range(3)],
        parameters=[Rate(value=0.0)],
        results=[Coordinate() for _ in range(3)],
        time=None,
    )
    jacobian = AutomaticDifferentiation().jacobian(function, np.zeros(3), 0.0)
    mesh = GeodesicLevelSets(GeodesicSettings(0.1, 0.1, (0.05, 0.15))).compute(
        function,
        np.zeros(3),
        jacobian,
        StableManifold(),
        distance=0.6,
    )
    film = ManifoldFilm(
        "A plane",
        "The stable plane of a linear saddle.",
        mesh,
        equation=r"\dot x = -x",
        note="Grown as geodesic level sets.",
    )
    film.curves.append(FilmCurve("unstable", "unstable", [[0, 0, 0], [0, 0, 0.5]]))
    film.markers.append(FilmMarker("saddle", "saddle", [0.0, 0.0, 0.0]))
    return film


def _payload(html: str) -> dict:
    match = re.search(r"window.__FILM__ = (.*?);</script>", html, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


class TestManifoldFilm(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.film = _film()
        cls.html = ManifoldFilmRenderer().render(cls.film)

    def test_the_payload_is_the_mesh(self):
        payload = _payload(self.html)
        mesh = self.film.mesh
        assert len(payload["vertices"]) == 3 * len(mesh.vertices)
        assert payload["triangles"] == mesh.triangles.ravel().tolist()
        assert len(payload["rings"]) == len(mesh.rings)
        assert payload["ringDistances"][-1] == round(mesh.ring_distances[-1], 6)
        assert np.allclose(
            np.reshape(payload["vertices"], (-1, 3)),
            mesh.vertices,
            atol=1.0e-4,
        )

    def test_the_crochet_pattern_travels_with_it(self):
        payload = _payload(self.html)
        rows = self.film.mesh.crochet_pattern()
        assert [row[2] for row in payload["pattern"]] == [r.stitches for r in rows]
        assert "Save as CSV" in self.html

    def test_curves_and_markers_are_projected(self):
        payload = _payload(self.html)
        assert payload["curves"][0]["points"][-1] == [0.0, 0.0, 0.5]
        assert payload["markers"][0]["point"] == [0.0, 0.0, 0.0]

    def test_the_scrub_bar_is_geodesic_distance(self):
        assert 'id="r" type="range"' in self.html
        assert 'aria-label="geodesic distance"' in self.html

    def test_space_on_a_focused_control_does_not_play(self):
        # Space presses a checkbox or a button; only elsewhere does it play.
        assert 'ev.target.tagName === "BUTTON"' in self.html
        assert 'ev.key === "ArrowRight"' in self.html

    def test_the_page_is_one_file(self):
        assert "<script src" not in self.html
        assert "A plane" in self.html
        assert "<svg" in self.html  # the typeset equation

    def test_a_projection_picks_the_drawn_coordinates(self):
        film = _film()
        film.projection = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], float)
        swapped = np.reshape(film_payload(film)["vertices"], (-1, 3))
        assert np.allclose(swapped[:, 0], film.mesh.vertices[:, 1], atol=1.0e-4)


class TestHowTheGrowthEnded(TestCase):
    """The page says whether the growth reached its distance, and if not why."""

    def test_a_complete_growth_says_so(self):
        payload = film_payload(_film())
        assert payload["stopped"] is None
        assert payload["lost"] == 0
        assert payload["warnings"] == []

    def test_a_stopped_growth_carries_its_reason_and_losses(self):
        film = _film()
        film.mesh.stopped = "3 of 40 landings failed at the smallest step 0.01"
        film.mesh.lost = 2
        film.mesh.warnings = ["ring 4 at distance 0.5 accepted at the smallest step"]
        html = ManifoldFilmRenderer().render(film)
        payload = _payload(html)
        assert payload["stopped"] == film.mesh.stopped
        assert payload["lost"] == 2
        assert payload["warnings"] == film.mesh.warnings
        # Shown in its own line beneath the film, as text, not markup.
        assert 'id="growth"' in html
        assert 'getElementById("growth").textContent' in html
