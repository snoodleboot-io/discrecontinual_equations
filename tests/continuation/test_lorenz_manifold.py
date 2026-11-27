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
* grown strictly towards 170 it stops at 165.25, with no point lost, and
  says why: 1 of 1,248 landings fails at the smallest step, 0.094; seven
  rings from 157.09 on were accepted there with a warning, turning by up to
  0.95. (It stopped at 157.38 while the continuation in the flight time
  gave a landing up after 200 steps: at a symmetric pair of points at
  (+-60.09, -+25.9, 76.85), where the flow runs along the level set
  (``|f . t| / |f| = 0.9993``), the landings go 18.7 inside the ring
  before they come round to the outward one at tau = 0.263, some 400 steps
  of Delta / 4 - measured independently, DOP853. A row lost at 200 steps is
  now continued again with 4,000, once per front. The growth to 157.38
  costs what it did, the same mesh; the 28 rings beyond it take the rest
  of 824 s, against 64 s for the 200-step growth stopping at 157.38, run
  side by side. The classes below grow without the escalation, to keep
  the suite fast: it stops at 157.38. It stopped at 157.16 before
  failed landings were retried alone, and at 142.87 before the back-trace
  stopped counting false crossings far from the ring.)
  By then the scrolls have formed: rays from ``C+`` cross three layers of
  the surface in four of the eight directions;
* allowed to drop 3% of a ring (``loss=0.03``) it reaches 170 with 1 point
  lost (57 without the escalation; to 160 it now loses none, 30 without),
  and the crochet pattern still adds up round by round.

What the papers report, read from them: Krauskopf and Osinga (SIAM J. Appl.
Dyn. Syst. 2(4) 2003, Sec. 4.5, preprint of 7 May 2003) grew it from 20
points at distance 1 with alpha in (0.3, 0.4), Delta.alpha in (0.1, 1.0),
Delta >= 0.01 and spacing 0.25 to 1.0, to 151.75 in 72 circles, the last of
2,300 points; the IJBC 2005 survey (Sec. 2.3) to 154.75 in 75 bands with
spacing 0.67 to 2.0. The crocheted model (Math. Intelligencer 26(4), 2004)
runs to 110.75 (25,511 stitches, by the authors' crochet page). With the
paper's step control this grower reproduces their band distances exactly
and their last circle to within 4% in number and 1.3% in points
(:class:`TestLorenzManifoldAtThePublishedSettings`).
"""

import math
from contextlib import nullcontext
from itertools import pairwise
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pytest
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


_ESCALATED = (
    "discrecontinual_equations.continuation.geodesic_level_sets._ESCALATED_STEPS"
)


def _lorenz(distance: float, loss: float = 0.0, *, escalate: bool = True):
    function = LorenzField(
        variables=[State(), State(), State()],
        parameters=[Alpha(value=0.0)],
        results=[State(), State(), State()],
        time=None,
    )
    jacobian = AutomaticDifferentiation().jacobian(function, np.zeros(3), 0.0)
    grower = GeodesicLevelSets(
        GeodesicSettings(initial_radius=1.0, step=1.5, spacing=(1.5, 4.5), loss=loss),
    )
    with nullcontext() if escalate else patch(_ESCALATED, ()):
        return grower.compute(
            function,
            np.zeros(3),
            jacobian,
            StableManifold(),
            distance=distance,
        )


class TestLorenzManifoldAsFarAsItGoes(TestCase):
    """Grown without the escalated continuation budget, which stops it at
    157.38 in about a minute; with it the growth reaches 165.25 but takes
    many times longer, too long for every run of the suite."""

    @classmethod
    def setUpClass(cls):
        cls.mesh = _lorenz(170.0, escalate=False)

    def test_the_strict_growth_stops_and_says_why(self):
        assert self.mesh.stopped is not None
        assert "landings failed" in self.mesh.stopped
        assert self.mesh.lost == 0
        # Krauskopf and Osinga grew it to 151.75 (SIAM J. Appl. Dyn. Syst.
        # 2003, Sec. 4.5) and 154.75 (the IJBC 2005 survey, Sec. 2.3).
        assert 151.75 < self.mesh.ring_distances[-1] < 170.0

    @pytest.mark.slow
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
    @pytest.mark.slow
    def test_the_crochet_pattern_adds_up_when_points_are_lost(self):
        # Without the escalated budget, as above: it reaches 160 with 30
        # points lost. With it, it loses none to 160 and 1 to 170, slowly.
        mesh = _lorenz(160.0, loss=0.03, escalate=False)
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
    (see :func:`_off_manifold`). Measured: within 6.6e-3 at step 1.5 and
    spacing (1.5, 4.5), within 8.4e-4 at half of both - the error of the
    ring's interpolation the next orbits start from, falling faster than
    the square of the spacing. On the film's own mesh (step 1.5, out to 100)
    the same probe measured a median 1e-3 and a worst 0.23 (at r = 74.5),
    so the film is a picture of the manifold to a few tenths, not 1e-10:
    the BVP's tolerance is how well each orbit is solved, not how far the
    ring it starts from is off the manifold.
    """

    @pytest.mark.slow
    def test_the_ring_at_distance_twenty_converges_onto_the_manifold(self):
        coarse = _outer_ring_error(1.5)
        fine = _outer_ring_error(0.75)
        assert coarse < 2.0e-2
        assert fine < 2.5e-3
        assert coarse / fine > 4.0


class TestLorenzManifoldAtThePublishedSettings(TestCase):
    """Krauskopf and Osinga's own growth, reproduced at their settings.

    SIAM J. Appl. Dyn. Syst. 2(4) 2003, Sec. 4.5 and Fig. 4.7: 20 points on a
    circle of radius 1, alpha in (0.3, 0.4), Delta.alpha in (0.1, 1.0),
    Delta >= 0.01; 25 bands reach 62.75, 40 reach 92.75 and 55 reach 122.75,
    and 72 circles reach 151.75, the last of 2,300 points. The IJBC 2005
    survey (Sec. 2.3) starts Delta at 0.25 and spaces points 0.67 to 2.0;
    those are the settings here. The band distances are reproduced exactly:
    doubling from 0.25 gives the crochet pattern's 2.75, 4.75 and 8.75 too.

    At the SIAM paper's own spacing, 0.25 to 1.0, the circle at 150.75 has
    4,524 points: it is 3,217 long, and no circle that long carries only
    2,300 points 1.0 apart at most. At the survey's spacing it has 2,330 at
    152.75, within 1.3% of the published count, so that count is the
    survey's spacing - and 151.75 plus three bands of 1.0 is the survey's
    154.75, so both papers likely report one computation.
    """

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
            GeodesicSettings(
                initial_radius=1.0,
                step=0.25,
                spacing=(0.67, 2.0),
                step_range=(0.01, math.inf),
                angles=(0.3, 0.4),
                step_angles=(0.1, 1.0),
                initial_points=20,
            ),
        ).compute(function, np.zeros(3), jacobian, StableManifold(), distance=160.0)

    @pytest.mark.slow
    def test_the_bands_reach_the_published_distances(self):
        distances = self.mesh.ring_distances
        assert distances[25] == pytest.approx(62.75, abs=1.0e-9)
        assert distances[40] == pytest.approx(92.75, abs=1.0e-9)
        assert distances[55] == pytest.approx(122.75, abs=1.0e-9)

    def test_the_last_circle_is_the_published_one_within_ten_percent(self):
        distances = np.array(self.mesh.ring_distances)
        assert distances[-1] >= 151.75
        circle = int(np.searchsorted(distances, 151.75 - 1.0e-9))
        assert abs(circle - 72) <= 7
        assert abs(len(self.mesh.rings[circle]) - 2_300) <= 230
