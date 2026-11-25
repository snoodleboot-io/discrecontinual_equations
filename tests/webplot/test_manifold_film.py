"""The manifold film: a grown surface, scrubbed by geodesic distance."""

import json
import re
from itertools import pairwise
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


def _unpack(payload: dict) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """What the page's ``unpack`` does, in Python, step for step."""
    steps = np.reshape(payload["vertexSteps"], (-1, 3))
    vertices = np.cumsum(steps, axis=0) / 10.0 ** payload["vertexDigits"]
    if "triangles" in payload:
        rings = [np.array(ring) for ring in payload["rings"]]
        return vertices, np.reshape(payload["triangles"], (-1, 3)), rings
    sizes = payload["ringSizes"]
    starts = np.cumsum([1, *sizes[:-1]])
    rings = [
        np.arange(start, start + size)
        for start, size in zip(starts, sizes, strict=True)
    ]
    seed = sizes[0]
    triangles = [[0, 1 + (i + 1) % seed, 1 + i] for i in range(seed)]
    bits = iter(payload["bands"])
    for inner, outer in pairwise(rings):
        i = j = 0
        for _ in range(len(inner) + len(outer)):
            a, b = inner[i % len(inner)], outer[j % len(outer)]
            if next(bits) == "0":
                triangles.append([a, inner[(i + 1) % len(inner)], b])
                i += 1
            else:
                triangles.append([a, outer[(j + 1) % len(outer)], b])
                j += 1
    return vertices, np.array(triangles), rings


class TestManifoldFilm(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.film = _film()
        cls.html = ManifoldFilmRenderer().render(cls.film)

    def test_the_payload_is_the_mesh(self):
        payload = _payload(self.html)
        mesh = self.film.mesh
        vertices, triangles, rings = _unpack(payload)
        assert triangles.tolist() == mesh.triangles.tolist()
        assert [ring.tolist() for ring in rings] == [r.tolist() for r in mesh.rings]
        assert payload["ringDistances"][-1] == round(mesh.ring_distances[-1], 6)
        # Exactly the coordinates rounded as before the packing, bit for bit.
        digits = payload["vertexDigits"]
        assert np.array_equal(vertices, np.round(mesh.vertices, digits))
        assert np.allclose(vertices, mesh.vertices, atol=1.0e-4)

    def test_the_packed_mesh_is_far_smaller_than_the_plain_one(self):
        # 2.9 times on this small plane; 3.4 on the Lorenz film, whose rings
        # are longer (746 kB to 222 kB).
        payload = film_payload(self.film)
        mesh = self.film.mesh
        digits = payload["vertexDigits"]
        plain = len(
            json.dumps(
                {
                    "vertices": np.round(mesh.vertices, digits).ravel().tolist(),
                    "triangles": mesh.triangles.ravel().tolist(),
                    "distances": np.round(mesh.distances, 6).tolist(),
                    "rings": [ring.tolist() for ring in mesh.rings],
                },
            ),
        )
        keys = ("vertexDigits", "vertexSteps", "ringSizes", "bands")
        packed = len(json.dumps({key: payload[key] for key in keys}))
        assert "triangles" not in payload
        assert packed < plain / 2.5

    def test_a_mesh_not_zipped_ring_by_ring_is_sent_as_it_is(self):
        film = _film()
        film.mesh.triangles = film.mesh.triangles[:, [0, 2, 1]]
        payload = film_payload(film)
        assert payload["triangles"] == film.mesh.triangles.ravel().tolist()
        assert len(payload["rings"]) == len(film.mesh.rings)
        _, triangles, _ = _unpack(payload)
        assert triangles.tolist() == film.mesh.triangles.tolist()

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
        swapped, _, _ = _unpack(film_payload(film))
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
