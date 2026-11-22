"""The Lorenz manifold, grown as far as it can be trusted, against what is known.

Kept apart from ``test_geodesic_level_sets.py`` the way the connecting-orbit
tests are kept apart: it grows the film's whole surface and then two larger
ones - 5 s to 35 s in all, measured, depending on what else the machine is
doing - which is the cost of the thing under test rather than a defect.

What is checked, and where each number comes from - all measured here:

* the z-axis is in the manifold and is a geodesic of it, so every level set
  crosses it at ``z = +-r`` exactly - measured to 3.9e-12 relative, which is
  the BVP tolerance, since a chord along a straight line is the arc;
* the manifold winds around the positive z-axis like a helix: the level
  sets' tangent where they cross the axis turns steadily with height -
  measured 6.90 rad (just over one full turn) between r = 1 and r = 100;
* it reaches around both secondary equilibria by r = 100: rays from ``C+``
  in the plane ``z = rho - 1`` meet it within 25 in 6 of 8 directions;
* its size at the film's settings: 10,289 vertices, 20,154 triangles, 67
  rings, no BVP lost;
* grown strictly towards 170 it stops at 142.87, with no point lost, and
  says why: 10 of 891 landings fail at the smallest step. There the flow
  runs along the level set (``f . t / |f| = 1.0`` at the failing points) and
  those BVPs have no nearby solution. By then the scrolls have formed: rays
  from ``C+`` cross three layers of the surface in four of the eight
  directions;
* allowed to drop 3% of a ring (``loss=0.03``) it reaches 150 with 62 points
  lost, and the crochet pattern still adds up round by round.

What is *not* checked. Krauskopf and Osinga's figures show the two scrolls
rolled up around ``C+-`` in more layers, and the crocheted model's size;
from memory, and unverified because the papers cannot be fetched here, the
crochet pattern has 25,511 stitches and the computed manifold runs to a
geodesic distance of about 170. Neither is asserted, and this
implementation's strict growth stops short of 170.
"""

import math
from itertools import pairwise
from unittest import TestCase

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

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
        # value: 10,289 vertices and 67 rings at these settings.
        assert abs(len(self.mesh.vertices) - 10_289) < 500
        assert abs(len(self.mesh.rings) - 67) <= 3


def _lorenz(distance: float, loss: float = 0.0):
    function = LorenzField(
        variables=[State(), State(), State()],
        parameters=[Alpha(value=0.0)],
        results=[State(), State(), State()],
        time=None,
    )
    jacobian = AutomaticDifferentiation().jacobian(function, np.zeros(3), 0.0)
    return GeodesicLevelSets(
        GeodesicSettings(initial_radius=1.0, step=1.5, spacing=(1.5, 4.5), loss=loss),
    ).compute(function, np.zeros(3), jacobian, StableManifold(), distance=distance)


class TestLorenzManifoldAsFarAsItGoes(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = _lorenz(170.0)

    def test_the_strict_growth_stops_and_says_why(self):
        assert self.mesh.stopped is not None
        assert "landings failed" in self.mesh.stopped
        assert self.mesh.lost == 0
        # Measured: 142.87, short of the 170 asked for.
        assert 130.0 < self.mesh.ring_distances[-1] < 170.0

    def test_the_scrolls_have_formed_around_the_secondary_equilibria(self):
        offset = math.sqrt(_BETA * (_RHO - 1.0))
        for sign in (1.0, -1.0):
            centre = np.array([sign * offset, sign * offset, _RHO - 1.0])
            layered = sum(
                _crossings(
                    self.mesh.vertices,
                    self.mesh.triangles,
                    centre,
                    sign * np.array([math.cos(angle), math.sin(angle), 0.0]),
                    25.0,
                )
                >= 3
                for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
            )
            assert layered >= 4


class TestLossyLorenzManifold(TestCase):
    def test_the_crochet_pattern_adds_up_when_points_are_lost(self):
        mesh = _lorenz(150.0, loss=0.03)
        assert mesh.stopped is None
        assert mesh.lost > 0
        pattern = mesh.crochet_pattern()
        for previous, current in pairwise(pattern):
            assert current.stitches == previous.stitches + (
                current.increases - current.decreases
            )


_SIGMA = 10.0


def _lorenz_rhs(_t, state):
    x, y, z = state
    return [_SIGMA * (y - x), x * (_RHO - z) - y, x * y - _BETA * z]


def _off_manifold(point: np.ndarray, normal: np.ndarray) -> float:
    """How far ``point`` is from W^s(0) along ``normal``, found independently.

    Forward in time W^s(0) runs into the origin and everything beside it
    leaves along the unstable direction, one side each way. So flow a point
    ``point + e normal`` to the time ``T`` at which the orbit is closest to the
    origin and read its unstable coordinate there: it changes sign where the
    line crosses W^s(0). Root-find that (DOP853, rtol 1e-12), then take ``T``
    again from the root's own orbit, which comes closer, until it settles.
    """
    jacobian = np.array([[-_SIGMA, _SIGMA, 0.0], [_RHO, -1.0, 0.0], [0.0, 0.0, -_BETA]])
    values, left = np.linalg.eig(jacobian.T)
    unstable = np.real(left[:, int(np.argmax(values.real))])

    def closest(start):
        path = solve_ivp(
            _lorenz_rhs,
            (0.0, 12.0),
            start,
            method="DOP853",
            rtol=1.0e-12,
            atol=1.0e-15,
            dense_output=True,
        )
        times = np.linspace(0.0, path.t[-1], 40001)
        return float(times[np.argmin(np.linalg.norm(path.sol(times), axis=0))])

    def coordinate(start, horizon):
        path = solve_ivp(
            _lorenz_rhs,
            (0.0, horizon),
            start,
            method="DOP853",
            rtol=1.0e-12,
            atol=1.0e-15,
        )
        return float(unstable @ path.y[:, -1])

    offset, horizon = 0.0, closest(point)
    for _ in range(4):
        base = point + offset * normal

        def side(e, base=base, horizon=horizon):
            return coordinate(base + e * normal, horizon)

        here, width, bracket = side(0.0), 1.0e-9, None
        while bracket is None and width < 1.0:
            for sign in (1.0, -1.0):
                if bracket is None and np.sign(side(sign * width)) != np.sign(here):
                    bracket = sorted((0.0, sign * width))
            width *= 3.0
        assert bracket is not None
        root = brentq(side, *bracket, xtol=1.0e-15, rtol=1.0e-14)
        offset += root
        later = closest(point + offset * normal)
        if abs(root) < 1.0e-12 or abs(later - horizon) < 1.0e-3:
            break
        horizon = later
    return abs(offset)


def _normals(mesh) -> np.ndarray:
    corners = mesh.vertices[mesh.triangles]
    faces = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = np.zeros_like(mesh.vertices)
    for k in range(3):
        np.add.at(normals, mesh.triangles[:, k], faces)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def _outer_ring_error(step: float) -> float:
    function = LorenzField(
        variables=[State(), State(), State()],
        parameters=[Alpha(value=0.0)],
        results=[State(), State(), State()],
        time=None,
    )
    jacobian = AutomaticDifferentiation().jacobian(function, np.zeros(3), 0.0)
    mesh = GeodesicLevelSets(
        GeodesicSettings(initial_radius=1.0, step=step, spacing=(step, 3.0 * step)),
    ).compute(function, np.zeros(3), jacobian, StableManifold(), distance=20.0)
    normals = _normals(mesh)
    # Half the ring: the other half is its image under (x, y) -> (-x, -y).
    ring = mesh.rings[-1]
    chosen = ring[np.linspace(0, len(ring) // 2, 4, endpoint=False).astype(int)]
    return max(_off_manifold(mesh.vertices[i], normals[i]) for i in chosen)


class TestLorenzManifoldIsOnTheManifold(TestCase):
    """Vertices against W^s(0) itself, located without the grower.

    The z-axis tests above are exact but see only the axis. Here four points
    of the ring at distance 20 are each held to the stable manifold through
    them, found by flowing the line along the mesh normal into the origin
    (see :func:`_off_manifold`). Measured: within 7.6e-3 at step 1.5 and
    spacing (1.5, 4.5), within 9.2e-4 at half of both - the error of the
    ring's interpolation the next orbits start from, falling faster than
    the square of the spacing. On the film's own mesh (step 1.5, out to 100)
    the same probe measured a median 1e-3 and a worst 0.23 (at r = 74.5),
    so the film is a picture of the manifold to a few tenths, not 1e-10:
    the BVP's tolerance is how well each orbit is solved, not how far the
    ring it starts from is off the manifold.
    """

    def test_the_ring_at_distance_twenty_converges_onto_the_manifold(self):
        coarse = _outer_ring_error(1.5)
        fine = _outer_ring_error(0.75)
        assert coarse < 2.0e-2
        assert fine < 2.5e-3
        assert coarse / fine > 4.0
