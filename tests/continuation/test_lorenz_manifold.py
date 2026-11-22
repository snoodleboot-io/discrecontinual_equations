"""The Lorenz manifold to geodesic distance 100, against what is known of it.

Kept apart from ``test_geodesic_level_sets.py`` the way the connecting-orbit
tests are kept apart: it grows the film's whole surface. That is under 2 s
since the BVPs are traced to a first guess, but it is the film's own cost and
grows with the distance, so it lives where it can be left out.

What is checked, and where each number comes from:

* the z-axis is in the manifold and is a geodesic of it, so every level set
  crosses it at ``z = +-r`` exactly - measured to 1.1e-11 relative, which is
  the BVP tolerance, since a chord along a straight line is the arc;
* the manifold winds around the positive z-axis like a helix: the level
  sets' tangent where they cross the axis turns steadily with height -
  measured 7.06 rad (just over one full turn) between r = 1 and r = 100;
* it reaches around both secondary equilibria: rays from ``C+`` in the plane
  ``z = rho - 1`` meet it within 25 in 6 of 8 directions - measured;
* its size: 10,417 vertices, 20,402 triangles, 68 rings at these settings,
  no BVP lost - measured.

What is *not* checked. Krauskopf and Osinga's figures show the two scrolls
rolled up around ``C+-`` in several layers, and the crocheted model's size;
from memory, and unverified because the papers cannot be fetched here, the
crochet pattern has 25,511 stitches and the computed manifold runs to a
geodesic distance of about 170. Neither is asserted. At distance 100 the
scrolls have only begun (one layer around ``C+`` in each direction above).
Beyond about 105 the flow runs along the level sets in places and some BVPs
have no nearby solution; with ``loss=0.03`` the growth reaches 135.6 at
80,422 vertices with 1,063 points dropped, and rays from ``C+`` then cross
three layers in four of the eight directions - measured, but with holes in
the mesh, so not asserted either.
"""

import math
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
from tests.continuation.fields import Alpha, LorenzField, State

_DISTANCE = 100.0
_RHO, _BETA = 28.0, 8.0 / 3.0


def _crossings(
    vertices: np.ndarray,
    triangles: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    length: float,
) -> int:
    """How many triangles the segment from ``origin`` along ``direction`` meets."""
    corners = vertices[triangles]
    first = corners[:, 1] - corners[:, 0]
    second = corners[:, 2] - corners[:, 0]
    normal = np.cross(direction, second)
    det = np.einsum("ij,ij->i", first, normal)
    usable = np.abs(det) > 1.0e-12
    inverse = 1.0 / np.where(usable, det, 1.0)
    offset = origin - corners[:, 0]
    u = np.einsum("ij,ij->i", offset, normal) * inverse
    cross = np.cross(offset, first)
    v = (cross @ direction) * inverse
    t = np.einsum("ij,ij->i", second, cross) * inverse
    hit = usable & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0) & (t < length)
    return int(np.count_nonzero(hit))


class TestLorenzManifold(TestCase):
    @classmethod
    def setUpClass(cls):
        function = LorenzField(
            variables=[State(), State(), State()],
            parameters=[Alpha(value=0.0)],
            results=[State(), State(), State()],
            time=None,
        )
        jacobian = AutomaticDifferentiation().jacobian(function, np.zeros(3), 0.0)
        cls.mesh = GeodesicLevelSets(
            GeodesicSettings(initial_radius=1.0, step=1.5, spacing=(1.5, 4.5)),
        ).compute(
            function,
            np.zeros(3),
            jacobian,
            StableManifold(),
            distance=_DISTANCE,
        )

    def test_it_reaches_the_distance_without_losing_a_point(self):
        assert self.mesh.ring_distances[-1] == _DISTANCE
        assert self.mesh.lost == 0
        assert self.mesh.seed == "taylor"
        assert self.mesh.residual < 1.0e-9

    def test_the_z_axis_is_a_geodesic_of_it(self):
        for ring, distance in zip(
            self.mesh.rings[1:],
            self.mesh.ring_distances[1:],
            strict=True,
        ):
            points = self.mesh.vertices[ring]
            for sign in (1.0, -1.0):
                side = points[sign * points[:, 2] > 0.0]
                nearest = side[np.argmin(np.hypot(side[:, 0], side[:, 1]))]
                assert abs(abs(nearest[2]) - distance) < 1.0e-8 * distance

    def test_it_winds_around_the_z_axis_as_a_helix(self):
        angles = []
        for ring in self.mesh.rings:
            points = self.mesh.vertices[ring]
            above = np.flatnonzero(points[:, 2] > 0.0)
            index = above[np.argmin(np.hypot(points[above, 0], points[above, 1]))]
            tangent = points[(index + 1) % len(points)] - points[index - 1]
            angles.append(math.atan2(tangent[1], tangent[0]))
        # A tangent is a line, so its angle is defined modulo pi.
        turned = np.unwrap(2.0 * np.array(angles)) / 2.0
        turned *= np.sign(turned[-1] - turned[0])
        assert float(np.min(np.diff(turned))) > -0.05
        assert turned[-1] - turned[0] > 2.0 * math.pi

    def test_it_reaches_around_the_secondary_equilibria(self):
        offset = math.sqrt(_BETA * (_RHO - 1.0))
        for sign in (1.0, -1.0):
            centre = np.array([sign * offset, sign * offset, _RHO - 1.0])
            met = sum(
                _crossings(
                    self.mesh.vertices,
                    self.mesh.triangles,
                    centre,
                    np.array([math.cos(angle), math.sin(angle), 0.0]),
                    25.0,
                )
                > 0
                for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
            )
            assert met >= 6

    def test_the_mesh_size_is_the_measured_one(self):
        # A regression guard on what this growth measured, not a literature
        # value: 10,417 vertices and 68 rings at these settings.
        assert abs(len(self.mesh.vertices) - 10_417) < 500
        assert abs(len(self.mesh.rings) - 68) <= 3
