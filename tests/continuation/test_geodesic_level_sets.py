"""Two-dimensional manifolds grown as geodesic level sets.

The oracles are the ones a geodesic level set can be held to exactly. A plane
is the stable manifold of a linear saddle, and on a plane geodesic distance is
Euclidean distance, so every level set is a circle about the equilibrium in the
plane's own metric - an ellipse once it is written in the (non-orthogonal)
eigenbasis coordinates. The tilted homoclinic system's stable manifold is a
cylinder over a planar curve, which is developable, so its geodesic distance
is ``sqrt(s^2 + w^2)`` with ``s`` the arclength along the curve. The Lorenz
manifold has no closed form, but it does contain the z-axis, which is a
geodesic of it (the fixed set of the symmetry ``(x, y, z) -> (-x, -y, z)``),
so every level set must cross the axis at ``z = +-r``.

The long Lorenz computation, with its measured mesh size, lives in
``test_lorenz_manifold.py`` so that this file stays a feedback-loop test.
"""

import math
from itertools import pairwise
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.geodesic_level_sets import (
    GeodesicLevelSets,
    GeodesicSettings,
    _advance,
    _first_guess,
    _first_landing,
    _Front,
    _ring_tangents,
    _thin,
    _trace_back,
)
from discrecontinual_equations.continuation.manifold import (
    StableManifold,
    UnstableManifold,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import (
    _TILT,
    Alpha,
    LorenzField,
    State,
    TiltedHomoclinicField,
)

# A non-normal linear saddle: stable eigenvalues -1 and -3, unstable 2, with
# eigenvectors that are neither orthogonal nor of equal length. The chart's
# coordinates are then a skewed frame of the stable plane, which is what turns
# the plane's circles into ellipses in those coordinates.
_EIGENVECTORS = np.array(
    [
        [1.0, 0.6, 0.2],
        [0.3, 1.0, -0.4],
        [-0.2, 0.5, 1.0],
    ],
)
_SADDLE = _EIGENVECTORS @ np.diag([-1.0, -3.0, 2.0]) @ np.linalg.inv(_EIGENVECTORS)


def _field(kind: type[DeterministicFunction]) -> DeterministicFunction:
    return kind(
        variables=[State(), State(), State()],
        parameters=[Alpha(value=0.0)],
        results=[State(), State(), State()],
        time=None,
    )


class LinearSaddle(DeterministicFunction):
    """x' = A x with the skewed saddle above."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        return list(_SADDLE @ np.asarray(point))


def _jacobian(function, point):
    return AutomaticDifferentiation().jacobian(function, np.asarray(point), 0.0)


def _linear_mesh(step: float = 0.1, distance: float = 2.0):
    function = _field(LinearSaddle)
    settings = GeodesicSettings(
        initial_radius=0.1,
        step=step,
        spacing=(0.05, 0.15),
    )
    return GeodesicLevelSets(settings).compute(
        function,
        np.zeros(3),
        _jacobian(function, np.zeros(3)),
        StableManifold(),
        distance=distance,
    )


class TestLinearSaddle(TestCase):
    """Oracle 1: the stable plane of a linear saddle, ring by exact ring."""

    @classmethod
    def setUpClass(cls):
        cls.mesh = _linear_mesh()
        stable = _EIGENVECTORS[:, :2]
        cls.normal = np.cross(stable[:, 0], stable[:, 1])
        cls.normal /= np.linalg.norm(cls.normal)

    def test_every_vertex_lies_on_the_stable_plane(self):
        # A sanity check, not the oracle: any orbit method keeps a linear
        # field's invariant plane, so this would pass for a wrong level set.
        # The radii below are what a geodesic level set is held to.
        offsets = np.abs(self.mesh.vertices @ self.normal)
        assert float(np.max(offsets)) < 1.0e-9

    def test_rings_are_circles_of_their_geodesic_radius(self):
        assert self.mesh.ring_distances[-1] >= 2.0 - 1.0e-12
        for ring, distance in zip(
            self.mesh.rings,
            self.mesh.ring_distances,
            strict=True,
        ):
            radii = np.linalg.norm(self.mesh.vertices[ring], axis=1)
            assert float(np.max(np.abs(radii - distance))) < 1.0e-8

    def test_rings_are_the_ellipses_of_the_eigenbasis(self):
        # In eigen-coordinates theta (x = V theta), a circle of radius r in the
        # plane is the ellipse theta^T (V^T V) theta = r^2.
        stable = _EIGENVECTORS[:, :2]
        metric = stable.T @ stable
        for ring, distance in zip(
            self.mesh.rings,
            self.mesh.ring_distances,
            strict=True,
        ):
            theta = np.linalg.lstsq(stable, self.mesh.vertices[ring].T, rcond=None)[0]
            quadratic = np.einsum("in,ij,jn->n", theta, metric, theta)
            assert float(np.max(np.abs(quadratic - distance**2))) < 1.0e-7

    def test_each_vertex_carries_its_rings_distance(self):
        for ring, distance in zip(
            self.mesh.rings,
            self.mesh.ring_distances,
            strict=True,
        ):
            assert np.allclose(self.mesh.distances[ring], distance)
        assert self.mesh.distances[0] == 0.0
        assert np.allclose(self.mesh.vertices[0], 0.0)

    def test_spacing_stays_within_its_bounds(self):
        for ring in self.mesh.rings[1:]:
            points = self.mesh.vertices[ring]
            gaps = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
            assert float(np.max(gaps)) <= 0.15 + 1.0e-9
            assert float(np.min(gaps)) >= 0.05 - 1.0e-9

    def test_the_rings_triangulate_into_a_disc(self):
        triangles = self.mesh.triangles
        assert triangles.shape[1] == 3
        assert int(triangles.max()) < len(self.mesh.vertices)
        # Every edge of a triangulated disc is shared by two triangles except
        # those on its boundary, which is the outermost ring.
        edges: dict[tuple[int, int], int] = {}
        for a, b, c in triangles:
            for u, v in ((a, b), (b, c), (c, a)):
                key = (min(u, v), max(u, v))
                edges[key] = edges.get(key, 0) + 1
        boundary = [edge for edge, count in edges.items() if count == 1]
        assert all(count <= 2 for count in edges.values())
        assert len(boundary) == len(self.mesh.rings[-1])
        # Euler characteristic of a disc: V - E + F = 1.
        vertices = len(self.mesh.vertices)
        assert vertices - len(edges) + len(triangles) == 1
        # And the triangles cover the disc: their area is pi r^2 less the
        # chords' segments, so within the spacing's sagitta of it.
        corners = self.mesh.vertices[triangles]
        area = 0.5 * np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=1,
        )
        radius = self.mesh.ring_distances[-1]
        assert abs(float(area.sum()) - math.pi * radius**2) < 0.01 * radius**2

    def test_the_crochet_pattern_counts_the_stitches(self):
        pattern = self.mesh.crochet_pattern()
        assert len(pattern) == len(self.mesh.rings)
        for previous, current in pairwise(pattern):
            assert current.stitches == previous.stitches + (
                current.increases - current.decreases
            )
        for row, ring in zip(pattern, self.mesh.rings, strict=True):
            assert row.stitches == len(ring)
        # A circle's circumference grows with its radius, so stitches are added.
        assert sum(row.increases for row in pattern) > 0

    def test_the_seed_is_the_taylor_chart(self):
        assert self.mesh.seed == "taylor"

    def test_a_resonant_chart_falls_back_to_the_tangent_plane(self):
        # Eigenvalues -1 and -2 exactly: the monomial theta_1^2 decays at
        # rate -2, which is in the spectrum, and the chart's solve is singular.
        class Resonant(DeterministicFunction):
            def eval(self, point, time=None):  # noqa: ARG002 (base signature)
                return [-point[0], -2.0 * point[1], point[2]]

        function = _field(Resonant)
        mesh = GeodesicLevelSets(GeodesicSettings(0.1, 0.1, (0.05, 0.15))).compute(
            function,
            np.zeros(3),
            _jacobian(function, np.zeros(3)),
            StableManifold(),
            distance=0.5,
        )
        assert mesh.seed == "eigenspace"
        assert float(np.max(np.abs(mesh.vertices[:, 2]))) < 1.0e-12

    def test_the_unstable_manifold_of_a_curve_is_refused(self):
        function = _field(LinearSaddle)
        with pytest.raises(ValueError, match="two-dimensional"):
            GeodesicLevelSets(GeodesicSettings(0.1, 0.1, (0.05, 0.15))).compute(
                function,
                np.zeros(3),
                _jacobian(function, np.zeros(3)),
                UnstableManifold(),
                distance=1.0,
            )


# Linear saddles the plane oracle must hold for, not only the one above: rate
# ratios up to 20 on the stable plane (Lorenz's is 8.6), eigenvectors close to
# parallel, another seed radius and step, a coarse step, and an unstable
# plane. Each is (eigenvectors, eigenvalues, selection, seed, step, spacing,
# distance). These are the cases an independent review found losing hundreds
# of points before the BVP was solved from the landing end.
_PARALLEL = np.array([[1.0, 1.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 1.0]])
_MILD = np.array([[1.0, 1.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 1.0]])
_SADDLES = {
    "ratio 3": (_EIGENVECTORS, (-1.0, -3.0, 2.0), True, 0.1, 0.1, (0.05, 0.15), 1.5),
    "ratio 8.6": (_EIGENVECTORS, (-1.0, -8.6, 2.0), True, 0.1, 0.1, (0.05, 0.15), 1.5),
    "ratio 12": (_EIGENVECTORS, (-1.0, -12.0, 2.0), True, 0.1, 0.1, (0.05, 0.15), 1.5),
    "ratio 20": (_EIGENVECTORS, (-1.0, -20.0, 2.0), True, 0.1, 0.1, (0.05, 0.15), 1.5),
    "mild skew": (_MILD, (-1.0, -2.5, 1.0), True, 0.1, 0.1, (0.05, 0.15), 1.5),
    "near parallel": (_PARALLEL, (-1.0, -2.5, 1.0), True, 0.1, 0.1, (0.05, 0.15), 1.5),
    "seed 0.37 step 0.13": (
        _EIGENVECTORS,
        (-1.0, -3.0, 2.0),
        True,
        0.37,
        0.13,
        (0.05, 0.15),
        3.0,
    ),
    "coarse": (_EIGENVECTORS, (-1.0, -3.0, 2.0), True, 0.1, 0.5, (0.25, 0.75), 10.0),
    "unstable": (_EIGENVECTORS, (1.0, 3.0, -2.0), False, 0.1, 0.1, (0.05, 0.15), 1.5),
}


def _saddle_function(matrix: np.ndarray) -> DeterministicFunction:
    class Linear(DeterministicFunction):
        def eval(self, point, time=None):  # noqa: ARG002 (base signature)
            return list(matrix @ np.asarray(point))

    return _field(Linear)


@pytest.mark.parametrize("name", list(_SADDLES))
def test_the_plane_oracle_holds_for_every_saddle(name):
    vectors, rates, stable, seed, step, spacing, distance = _SADDLES[name]
    matrix = vectors @ np.diag(rates) @ np.linalg.inv(vectors)
    function = _saddle_function(matrix)
    selection = StableManifold() if stable else UnstableManifold()
    mesh = GeodesicLevelSets(GeodesicSettings(seed, step, spacing)).compute(
        function,
        np.zeros(3),
        _jacobian(function, np.zeros(3)),
        selection,
        distance=distance,
    )
    assert mesh.stopped is None
    assert mesh.lost == 0
    assert mesh.ring_distances[-1] == pytest.approx(distance)
    for ring, radius in zip(mesh.rings, mesh.ring_distances, strict=True):
        radii = np.linalg.norm(mesh.vertices[ring], axis=1)
        assert float(np.max(np.abs(radii - radius))) < 1.0e-9
    for ring in mesh.rings[1:]:
        points = mesh.vertices[ring]
        gaps = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        assert spacing[0] - 1.0e-9 <= float(np.min(gaps))
        assert float(np.max(gaps)) <= spacing[1] + 1.0e-9
    # No triangle is flipped: all face the same way across the plane.
    chosen = [i for i, rate in enumerate(rates) if (rate < 0.0) == stable]
    normal = np.cross(vectors[:, chosen[0]], vectors[:, chosen[1]])
    corners = mesh.vertices[mesh.triangles]
    facing = (
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]) @ normal
    )
    assert np.all(facing > 0.0) or np.all(facing < 0.0)
    pattern = mesh.crochet_pattern()
    for previous, current in pairwise(pattern):
        assert current.stitches == previous.stitches + (
            current.increases - current.decreases
        )


_ADVANCE = "discrecontinual_equations.continuation.geodesic_level_sets._advance"
_FIRST_GUESS = "discrecontinual_equations.continuation.geodesic_level_sets._first_guess"


def _swirl(states: np.ndarray) -> np.ndarray:
    """A rotation in z = 0 whose radial part changes sign around the circle.

    ``rho = -0.3 + 0.5 sin(theta)`` is positive for theta in (0.64, 2.50):
    there the field points out of the unit circle, so flowing towards the
    ring (forward, for a stable manifold) a point just outside it moves
    further out first and comes back to the ring only after turning round.
    """
    x, y, z = states.T
    rho = -0.3 + 0.5 * np.sin(np.arctan2(y, x))
    return np.column_stack([-y + rho * x, x + rho * y, -z])


class TestFirstLanding(TestCase):
    """The continuation in the flight time from zero, the paper's own BVP solve.

    The plane z = 0 is invariant under :func:`_swirl` and the unit circle is
    its front, so the landing Delta beyond each ring point is exactly
    ``(1 + Delta) r`` - whatever the orbit does on its way to the ring.
    """

    def setUp(self):
        angles = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False)
        self.ring = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(40)])
        self.settings = GeodesicSettings(0.1, 0.1, (0.05, 0.3))

    def _land(self, delta):
        return _first_landing(
            _swirl,
            -1.0,
            self.ring,
            np.arange(40, dtype=float),
            (self.ring, _ring_tangents(self.ring), self.ring, delta),
            self.settings,
        )

    def test_every_landing_is_delta_straight_out(self):
        landing = self._land(0.1)
        assert np.all(landing.valid)
        assert np.max(np.abs(landing.points - 1.1 * self.ring)) < 1.0e-12

    def test_an_inward_first_crossing_is_passed_over(self):
        # Where the field points out of the ring, the landings of small tau
        # are inside it, and the paper's first crossing at 0.9 r with them.
        # The outward one is reached after the orbit has gone round (tau
        # 0.8 to 2.9 there).
        outward = np.einsum("ij,ij->i", _swirl(self.ring), self.ring) > 0.0
        landing = self._land(0.1)
        assert np.count_nonzero(outward) == 11
        assert np.max(np.abs(landing.points[outward] - 1.1 * self.ring[outward])) < (
            1.0e-12
        )

    def test_a_landing_the_first_guess_misses_is_retried_alone(self):
        # Two first guesses in seven are spoiled - half the ring away, which
        # Newton, a ring segment a step, cannot cross - or not a number. The
        # continuation lands those points at the same Delta, and the ring is
        # not recomputed: one ring solve per ring, and the same mesh.
        real = _first_guess

        def spoiled(field, sense, front, *rest):
            places, times = real(field, sense, front, *rest)
            places[::7] += 0.5 * len(front.points)
            times[3::7] = np.nan
            return places, times

        solves = []

        def counted(*args):
            solves.append(args[3])
            return _advance(*args)

        clean = _linear_mesh(distance=0.6)
        with patch(_FIRST_GUESS, spoiled), patch(_ADVANCE, counted):
            mesh = _linear_mesh(distance=0.6)
        assert mesh.lost == 0
        assert len(solves) == len(mesh.rings) - 1
        assert mesh.ring_distances == clean.ring_distances
        assert np.max(np.abs(mesh.vertices - clean.vertices)) < 1.0e-8


class TestTraceBack(TestCase):
    """The first guess's back-trace counts only crossings of the ring itself."""

    def test_a_sign_change_far_from_the_ring_is_not_a_crossing(self):
        # The unit circle in z = 0 with outward radial directions, and a
        # field that carries (1.5, 0, 0) up and over the ring's centre without
        # ever meeting the ring: it passes (1, 0, 3.3) and (0, 0, 10). Past
        # x = 1 the point is "inside" by the nearest ring point's outward
        # direction while 3.3 or more from the ring. On the Lorenz manifold
        # such a false crossing 28 to 37 from the ring (spacing 4.5) sent
        # Newton to a start 37 ring points from the true one, and the strict
        # growth stopped at 142.87 with "landings failed".
        angles = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False)
        ring = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(40)])
        front = _Front(ring, ring.copy(), np.arange(40))

        def drift(states):
            return np.tile([-0.15, 0.0, 1.0], (len(states), 1))

        crossed, _, _ = _trace_back(
            drift,
            -1.0,
            front,
            np.array([[1.5, 0.0, 0.0]]),
            0.5,
        )
        assert not crossed[0]

    def test_a_crossing_of_the_ring_is_found(self):
        angles = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False)
        ring = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(40)])
        front = _Front(ring, ring.copy(), np.arange(40))

        def inward(states):
            return np.tile([-1.0, 0.0, 0.0], (len(states), 1))

        crossed, place, flight = _trace_back(
            inward,
            -1.0,
            front,
            np.array([[1.5, 0.0, 0.0]]),
            0.5,
        )
        assert crossed[0]
        assert min(place[0], 40.0 - place[0]) < 1.0e-6
        assert flight[0] == pytest.approx(0.5, rel=1.0e-6)


class TestStopping(TestCase):
    """A growth that cannot go on says where and why, rather than going on."""

    def _grow(self, advance):
        function = _field(LinearSaddle)
        with patch(_ADVANCE, advance):
            return GeodesicLevelSets(
                GeodesicSettings(0.1, 0.1, (0.05, 0.15)),
            ).compute(
                function,
                np.zeros(3),
                _jacobian(function, np.zeros(3)),
                StableManifold(),
                distance=1.0,
            )

    def test_a_ring_that_keeps_too_few_points_stops_the_growth(self):
        # A ring of fewer than six points cannot bound the band below it;
        # this used to reach the triangulation and raise an IndexError there.
        real = _advance

        def starved(field, sense, front, delta, settings):
            ring = real(field, sense, front, delta, settings)
            if len(front.points) > 20:
                ring.points, ring.directions = ring.points[:0], ring.directions[:0]
                ring.keys = ring.keys[:0]
            return ring

        mesh = self._grow(starved)
        assert mesh.stopped is not None
        assert "fewer than" in mesh.stopped
        assert mesh.ring_distances[-1] < 1.0
        assert all(len(ring) >= 6 for ring in mesh.rings)

    def test_failed_landings_at_the_smallest_step_stop_the_growth(self):
        real = _advance

        def failing(field, sense, front, delta, settings):
            ring = real(field, sense, front, delta, settings)
            if len(front.points) > 20:
                ring.lost = 3
            return ring

        mesh = self._grow(failing)
        assert mesh.stopped is not None
        assert "landings failed" in mesh.stopped
        assert mesh.lost == 0
        assert mesh.ring_distances[-1] < 1.0


def _arclength(u: float) -> float:
    """Signed arclength from the origin along the planar stable curve.

    The curve is the zero energy level of ``u' = v, v' = u - u^2``. Its loop
    side is ``u = 1.5 sech^2(t/2)`` with ``t > 0`` for ``v < 0``, which is all
    of it a distance of 1.2 reaches (the loop's tip is 2.04 along); the other
    branch is ``v = -u sqrt(1 - 2u/3)`` for ``u < 0``.
    """
    if u >= 0.0:
        start = 2.0 * math.acosh(math.sqrt(1.5 / max(u, 1.0e-300)))

        def speed(t):
            decay = math.exp(-abs(t) / 2.0)
            s, c = 2.0 * decay / (1.0 + decay * decay), math.tanh(t / 2.0)
            du = -1.5 * s * s * c
            dv = -0.75 * s * s * (1.0 - 3.0 * c * c)
            return math.hypot(du, dv)

        return quad(speed, start, math.inf, limit=200)[0]

    def element(x):
        root = math.sqrt(1.0 - 2.0 * x / 3.0)
        slope = -root + x / (3.0 * root)
        return math.hypot(1.0, slope)

    return -quad(element, u, 0.0, limit=200)[0]


def _cylinder_error(step: float) -> float:
    function = _field(TiltedHomoclinicField)
    settings = GeodesicSettings(
        initial_radius=0.05,
        step=step,
        spacing=(step / 2.0, 1.5 * step),
    )
    mesh = GeodesicLevelSets(settings).compute(
        function,
        np.zeros(3),
        _jacobian(function, np.zeros(3)),
        StableManifold(),
        distance=1.2,
    )
    worst = 0.0
    for index in range(1, len(mesh.vertices)):
        u, _, w = _TILT.T @ mesh.vertices[index]
        geodesic = math.hypot(_arclength(float(u)), float(w))
        worst = max(worst, abs(geodesic - float(mesh.distances[index])))
    return worst


class TestTiltedCylinder(TestCase):
    """A curved manifold whose geodesic distance is still known exactly.

    The level sets are approximate here: each step measures Delta as a chord
    in the plane normal to the ring rather than as an arc, which is K&O's
    method and is second order. So the test is that the error is small and
    falls like Delta^2, not that it vanishes.
    """

    def test_vertices_lie_on_the_zero_energy_cylinder(self):
        function = _field(TiltedHomoclinicField)
        mesh = GeodesicLevelSets(
            GeodesicSettings(0.05, 0.05, (0.025, 0.075)),
        ).compute(
            function,
            np.zeros(3),
            _jacobian(function, np.zeros(3)),
            StableManifold(),
            distance=1.2,
        )
        u, v, _ = _TILT.T @ mesh.vertices.T
        energy = 0.5 * v * v - 0.5 * u * u + u**3 / 3.0
        # The energy is conserved, so nothing damps an orbit started off the
        # cylinder: what is left is the ring's interpolation between its
        # points, third order in the spacing. Measured 4.2e-5, 4.8e-6 and
        # 7.9e-7 at steps 0.1, 0.05 and 0.025; the BVP itself is at 1e-10.
        assert float(np.max(np.abs(energy))) < 1.0e-5

    def test_geodesic_distance_converges_at_second_order(self):
        coarse = _cylinder_error(0.1)
        fine = _cylinder_error(0.05)
        assert fine < 2.0e-3
        assert coarse / fine > 2.5


def _lorenz_mesh(distance: float):
    function = _field(LorenzField)
    settings = GeodesicSettings(
        initial_radius=1.0,
        step=0.5,
        spacing=(0.5, 1.5),
    )
    return GeodesicLevelSets(settings).compute(
        function,
        np.zeros(3),
        _jacobian(function, np.zeros(3)),
        StableManifold(),
        distance=distance,
    )


class TestLorenzManifoldNearTheOrigin(TestCase):
    """Oracle 2, the part that is quick: the first 15 units of W^s(0)."""

    @classmethod
    def setUpClass(cls):
        cls.mesh = _lorenz_mesh(15.0)

    def test_every_level_set_crosses_the_z_axis_at_its_distance(self):
        for ring, distance in zip(
            self.mesh.rings[1:],
            self.mesh.ring_distances[1:],
            strict=True,
        ):
            points = self.mesh.vertices[ring]
            for sign in (1.0, -1.0):
                side = points[sign * points[:, 2] > 0.0]
                nearest = side[np.argmin(np.hypot(side[:, 0], side[:, 1]))]
                # The axis is a geodesic and a chord along it is the arc, so
                # this is exact to the BVP tolerance: measured 1e-11.
                assert abs(abs(nearest[2]) - distance) < 1.0e-8 * distance

    def test_the_manifold_is_symmetric_under_the_rotation_about_z(self):
        mirrored = self.mesh.vertices * np.array([-1.0, -1.0, 1.0])
        for point in mirrored[:: max(1, len(mirrored) // 200)]:
            nearest = float(np.min(np.linalg.norm(self.mesh.vertices - point, axis=1)))
            assert nearest < 1.0

    def test_vertices_flow_into_the_origin(self):
        # Forward in time the manifold flows to the origin while an error eps
        # off it grows at the unstable rate 11.8, so a vertex at distance r
        # comes no closer than about r (eps / r)^(2.67 / 14.5) - the slow
        # stable rate against the two together - before it leaves. That
        # exponent makes this a coarse probe: a quarter of r means eps below
        # about 1e-3. Measured: closest approaches 0.17 to 1.23 from r = 7.3,
        # 0.06 to 0.93 at half the step.
        function = _field(LorenzField)

        def rhs(_t, state):
            return function.eval(point=list(state), time=None)

        middle = len(self.mesh.rings) // 2
        distance = self.mesh.ring_distances[middle]
        for index in self.mesh.rings[middle][::7]:
            path = solve_ivp(
                rhs,
                (0.0, 3.0),
                self.mesh.vertices[index],
                rtol=1.0e-11,
                atol=1.0e-12,
                max_step=0.01,
            )
            closest = float(np.min(np.linalg.norm(path.y, axis=0)))
            assert closest < 0.25 * distance

    def test_the_bvp_residual_is_at_tolerance(self):
        assert self.mesh.residual < 1.0e-8

    def test_each_ring_is_its_step_from_the_ring_before(self):
        # The geodesic property, checked against a spline through the ring
        # before rather than the grower's own interpolant: measured within
        # 1.1% of the step (near the seed, where the rings bend most) and
        # 0.03% beyond r = 5.
        assert max(_ring_gaps(self.mesh)) < 0.03


# A doubly curved manifold known exactly. Conjugating the linear saddle
# x' = A x (A keeps {x3 = 0} invariant) by y = (x1, x2, x3 + h(x1, x2)) gives
# a nonlinear field whose stable manifold is the graph z = h(x, y). The rates
# on it are -1 and -5 along skewed eigenvectors, and h has degree 6, beyond
# the order-5 Taylor seed, so neither the chart nor the flow is exact. Its
# geodesic distance from the origin is computed independently of the grower,
# by shooting geodesics of the graph's metric from the origin (the
# exponential map) with DOP853.
_GRAPH_RATES = np.array([[1.0, 0.7], [0.2, 1.0]])
_GRAPH_LINEAR = np.zeros((3, 3))
_GRAPH_LINEAR[:2, :2] = (
    _GRAPH_RATES @ np.diag([-1.0, -5.0]) @ np.linalg.inv(_GRAPH_RATES)
)
_GRAPH_LINEAR[:2, 2] = [0.4, -0.3]
_GRAPH_LINEAR[2, 2] = 2.0
_C1, _C2, _C3, _C4, _C5 = 0.3, -0.2, 0.25, 0.1, 0.03


def _height(x, y):
    return _C1 * x * x + _C2 * x * y + _C3 * y * y + _C4 * x**3 + _C5 * x * x * y**4


def _slope(x, y):
    hx = 2 * _C1 * x + _C2 * y + 3 * _C4 * x * x + 2 * _C5 * x * y**4
    hy = _C2 * x + 2 * _C3 * y + 4 * _C5 * x * x * y**3
    return hx, hy


def _curvature(x, y):
    hxx = 2 * _C1 + 6 * _C4 * x + 2 * _C5 * y**4
    hxy = _C2 + 8 * _C5 * x * y**3
    hyy = 2 * _C3 + 12 * _C5 * x * x * y * y
    return hxx, hxy, hyy


class GraphSaddle(DeterministicFunction):
    """The linear saddle bent so that its stable plane becomes z = h(x, y)."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        x, y = point[0], point[1]
        off = point[2] - _height(x, y)
        a = _GRAPH_LINEAR
        dx = a[0, 0] * x + a[0, 1] * y + a[0, 2] * off
        dy = a[1, 0] * x + a[1, 1] * y + a[1, 2] * off
        hx, hy = _slope(x, y)
        return [dx, dy, a[2, 2] * off + hx * dx + hy * dy]


def _graph_offset(vertices: np.ndarray) -> np.ndarray:
    """Each vertex's distance from the graph, to first order in that distance."""
    x, y, z = vertices.T
    hx, hy = _slope(x, y)
    return np.abs(z - _height(x, y)) / np.sqrt(1.0 + hx * hx + hy * hy)


def _metric_root() -> np.ndarray:
    hx, hy = _slope(0.0, 0.0)
    metric = np.array([[1 + hx * hx, hx * hy], [hx * hy, 1 + hy * hy]])
    values, vectors = np.linalg.eigh(metric)
    return vectors @ np.diag(np.sqrt(values)) @ vectors.T


def _geodesic(_s, state):
    x, y, vx, vy = state
    hx, hy = _slope(x, y)
    hxx, hxy, hyy = _curvature(x, y)
    bend = (hxx * vx * vx + 2 * hxy * vx * vy + hyy * vy * vy) / (1 + hx * hx + hy * hy)
    return [vx, vy, -hx * bend, -hy * bend]


def _graph_geodesic_distance(point: np.ndarray, guess: float) -> float:
    """Geodesic distance from the origin on the graph, by the exponential map."""
    root = _metric_root()
    inverse = np.linalg.inv(root)

    def landing(unknowns):
        angle, length = unknowns
        start = inverse @ np.array([math.cos(angle), math.sin(angle)])
        path = solve_ivp(
            _geodesic,
            (0.0, length),
            [0.0, 0.0, *start],
            method="DOP853",
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        return path.y[:2, -1] - point[:2]

    direction = root @ point[:2]
    solution = least_squares(
        landing,
        [math.atan2(direction[1], direction[0]), guess],
        xtol=1.0e-14,
        ftol=1.0e-14,
        gtol=1.0e-14,
    )
    assert float(np.max(np.abs(solution.fun))) < 1.0e-11
    return float(solution.x[1])


def _dense_ring(points: np.ndarray) -> np.ndarray:
    """A periodic cubic spline through a ring's points, densely sampled."""
    closed = np.vstack([points, points[:1]])
    knots = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))],
    )
    spline = CubicSpline(knots, closed, bc_type="periodic", axis=0)
    return spline(np.linspace(0.0, knots[-1], 60 * len(points), endpoint=False))


def _ring_gaps(mesh) -> list[float]:
    """How far each ring's points are from the ring before, less the step between.

    The ring before is a periodic cubic spline through its points, so this
    measures the grower against the step it reports, not its own interpolant.
    """
    gaps = []
    for inner, outer, (near, far) in zip(
        mesh.rings,
        mesh.rings[1:],
        pairwise(mesh.ring_distances),
        strict=False,
    ):
        dense = _dense_ring(mesh.vertices[inner])
        distance, _ = cKDTree(dense).query(mesh.vertices[outer])
        gaps.append(float(np.max(np.abs(distance - (far - near)))) / (far - near))
    return gaps


def _graph_mesh(step: float, spacing: tuple[float, float], distance: float):
    function = _field(GraphSaddle)
    return GeodesicLevelSets(GeodesicSettings(0.1, step, spacing)).compute(
        function,
        np.zeros(3),
        _jacobian(function, np.zeros(3)),
        StableManifold(),
        distance=distance,
    )


class TestCurvedGraph(TestCase):
    """A doubly curved manifold, held to its surface and its geodesic distance.

    Measured independently (DOP853 geodesics, the graph's own equation) at
    step 0.1, spacing (0.05, 0.15), out to 1.5: vertices within 9.4e-5 of the
    surface, the seed ring within 4.4e-9; geodesic distance within 2.3e-4 of
    the ring each vertex is on; every vertex is its ring's step from the ring
    before to 1.2e-3 of the step. Halving step and spacing together brought
    the surface error from 7.2e-4 to 1.0e-4 to 3.6e-5 (steps 0.2, 0.1, 0.05
    to distance 2) - third order in the spacing, since it is the ring's
    interpolation the next orbits start from - and the geodesic error from
    1.2e-3 to 2.7e-4 to 6.1e-5, second order in the step.
    """

    @classmethod
    def setUpClass(cls):
        cls.mesh = _graph_mesh(0.1, (0.05, 0.15), 1.5)

    def test_it_reaches_the_distance(self):
        assert self.mesh.stopped is None
        assert self.mesh.lost == 0
        assert self.mesh.ring_distances[-1] == pytest.approx(1.5)

    def test_the_seed_ring_is_on_the_surface(self):
        # The order-5 chart: a tangent-plane seed would be 3e-3 off here.
        assert float(np.max(_graph_offset(self.mesh.vertices[self.mesh.rings[0]]))) < (
            1.0e-7
        )

    def test_every_vertex_is_on_the_surface(self):
        assert float(np.max(_graph_offset(self.mesh.vertices[1:]))) < 2.0e-4

    def test_every_vertex_is_at_its_rings_geodesic_distance(self):
        worst = 0.0
        for ring, distance in zip(
            self.mesh.rings,
            self.mesh.ring_distances,
            strict=True,
        ):
            for index in ring[::4]:
                point = self.mesh.vertices[index]
                worst = max(
                    worst,
                    abs(_graph_geodesic_distance(point, distance) - distance),
                )
        assert worst < 5.0e-4

    def test_each_ring_is_its_step_from_the_ring_before(self):
        assert max(_ring_gaps(self.mesh)) < 3.0e-3


def _turns(mesh) -> list[float]:
    """Per ring from the third on: how sharply the approximate geodesics turn.

    Each point is followed back to its foot on the ring before and that foot
    to its foot on the ring before that (both rings splined), and the angle
    is the one between the two steps - the grower's ``alpha``, rebuilt from
    the mesh alone.
    """
    angles = []
    for older, inner, outer in zip(
        mesh.rings,
        mesh.rings[1:],
        mesh.rings[2:],
        strict=False,
    ):
        feet = _dense_ring(mesh.vertices[inner])
        before = _dense_ring(mesh.vertices[older])
        points = mesh.vertices[outer]
        foot = feet[cKDTree(feet).query(points)[1]]
        heel = before[cKDTree(before).query(foot)[1]]
        out, back = points - foot, foot - heel
        cosine = np.einsum("ij,ij->i", out, back) / (
            np.linalg.norm(out, axis=1) * np.linalg.norm(back, axis=1)
        )
        angles.append(float(np.max(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    return angles


class TestStepControl(TestCase):
    """Delta is halved where the geodesics turn and grown only where they don't.

    Krauskopf and Osinga accept a ring only if every ``alpha_r`` is below
    ``alpha_max`` and grow Delta only if every one is below ``alpha_min``
    (SIADS 2003, Sec. 3.2). Both are checked here on the turning angles
    rebuilt from the mesh, not on the grower's own.
    """

    def test_no_accepted_step_turns_more_than_the_upper_angle(self):
        # Measured 0.238 against the bound 0.3; never shrinking, or shrinking
        # on the mean turn rather than the largest, gives 0.342.
        function = _field(TiltedHomoclinicField)
        mesh = GeodesicLevelSets(
            GeodesicSettings(0.05, 0.6, (0.05, 0.15), angles=(0.1, 0.3)),
        ).compute(
            function,
            np.zeros(3),
            _jacobian(function, np.zeros(3)),
            StableManifold(),
            distance=1.2,
        )
        assert max(_turns(mesh)) < 0.3
        assert min(np.diff(mesh.ring_distances)) < 0.6

    def test_delta_grows_only_after_a_straight_step(self):
        # Measured: Delta grew after turns of 0.049, 0.074 and 0.089, all
        # below the lower angle 0.1; growing regardless, or on the mean turn,
        # grows it after turns of up to 0.199 and 0.157.
        function = _field(GraphSaddle)
        mesh = GeodesicLevelSets(
            GeodesicSettings(
                0.1,
                0.05,
                (0.05, 0.15),
                step_range=(0.01, 1.0),
                angles=(0.1, 0.3),
            ),
        ).compute(
            function,
            np.zeros(3),
            _jacobian(function, np.zeros(3)),
            StableManifold(),
            distance=2.0,
        )
        steps = np.diff(mesh.ring_distances)
        turns = _turns(mesh)
        grown = [
            turns[i - 1]
            for i in range(1, len(steps) - 1)
            if steps[i + 1] > 1.01 * steps[i]
        ]
        assert grown
        assert max(grown) < 0.1


def _graph_steps(**options):
    function = _field(GraphSaddle)
    options.setdefault("step_range", (0.01, 1.0))
    return GeodesicLevelSets(
        GeodesicSettings(0.1, 0.2, (0.05, 0.15), **options),
    ).compute(
        function,
        np.zeros(3),
        _jacobian(function, np.zeros(3)),
        StableManifold(),
        distance=2.0,
    )


_FIRST_LANDING = (
    "discrecontinual_equations.continuation.geodesic_level_sets._first_landing"
)


def test_newton_walks_along_a_finely_spaced_ring_in_lengths_not_segments():
    # At spacing 3e-4 the first guess is up to 23 segments (7e-3) along the
    # ring from the start Newton converges to. One segment a step, rows
    # that needed more than 25 failed - 51 of 2,011 on the first ring - and
    # went to the continuation; half of Delta a step, none does.
    function = _field(LinearSaddle)
    retried = []

    def spy(*args):
        retried.append(len(args[3]))
        return _first_landing(*args)

    with patch(_FIRST_LANDING, spy):
        mesh = GeodesicLevelSets(
            GeodesicSettings(0.1, 0.1, (1.0 / 6400.0, 1.5 / 3200.0)),
        ).compute(
            function,
            np.zeros(3),
            _jacobian(function, np.zeros(3)),
            StableManifold(),
            distance=0.2,
        )
    assert len(mesh.rings[1]) > 2000
    assert mesh.lost == 0
    assert retried == []


class TestPaperStepControl(TestCase):
    """The rest of SIADS 2003, Sec. 3.2: the bound on Delta times alpha,
    doubling, and the warning at the smallest step."""

    def test_delta_times_the_turn_is_bounded_too(self):
        # Measured: the angles alone (0.3, 0.4) accept Delta = 0.2 throughout,
        # where Delta alpha is 0.02 to 0.03; bounding it by 0.01 halves Delta
        # at the first ring and keeps it at 0.1 (Delta alpha 0.005 to 0.008).
        solves = []

        def counted(*args):
            solves.append(args[3])
            return _advance(*args)

        free = _graph_steps()
        with patch(_ADVANCE, counted):
            bounded = _graph_steps(step_angles=(0.002, 0.01))
        assert min(np.diff(free.ring_distances)) == pytest.approx(0.2)
        steps = np.diff(bounded.ring_distances)
        assert max(steps) == pytest.approx(0.1)
        products = np.array(_turns(bounded)) * steps[1 : len(_turns(bounded)) + 1]
        assert max(products) < 0.01
        assert bounded.warnings == []
        # Delta alpha never fell below 0.002, so Delta was never doubled:
        # the one ring solve thrown away is the first, at 0.2.
        assert len(solves) == len(bounded.rings)

    def test_delta_doubles_after_a_straight_ring(self):
        # The plane never turns: from 0.025 Delta doubles every ring up to
        # its bound, 0.025, 0.05, 0.1, 0.2, 0.4.
        function = _field(LinearSaddle)
        mesh = GeodesicLevelSets(
            GeodesicSettings(0.1, 0.025, (0.05, 0.15), step_range=(0.01, 0.4)),
        ).compute(
            function,
            np.zeros(3),
            _jacobian(function, np.zeros(3)),
            StableManifold(),
            distance=1.5,
        )
        steps = np.diff(mesh.ring_distances)
        assert np.allclose(steps[:5], [0.025, 0.05, 0.1, 0.2, 0.4])

    def test_a_sharp_ring_at_the_smallest_step_is_accepted_with_a_warning(self):
        # No step may shrink (the smallest is the step), and every ring of
        # the graph turns by more than 0.02: each is accepted, and said so.
        mesh = _graph_steps(angles=(0.01, 0.02), step_range=(0.2, 0.2))
        assert mesh.stopped is None
        assert len(mesh.warnings) == len(mesh.rings) - 1
        assert "accepted at the smallest step 0.2" in mesh.warnings[0]


def test_thinning_never_opens_a_gap_wider_than_the_spacing():
    # Removing the middle point would leave 0.18 between its neighbours,
    # wider than the 0.15 allowed, so it stays though it is 0.04 from one.
    points = np.array(
        [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.18, 0.0, 0.0]]
        + [[0.18 + 0.1 * k, 0.1 * (k % 2), 0.0] for k in range(1, 8)],
    )
    keep = _thin(points, 0.05, 0.15)
    assert keep[1]
