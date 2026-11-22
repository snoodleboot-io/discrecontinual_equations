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

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.geodesic_level_sets import (
    GeodesicLevelSets,
    GeodesicSettings,
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


def _linear_mesh(step: float = 0.1):
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
        distance=2.0,
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
                assert abs(abs(nearest[2]) - distance) < 0.05 * distance

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
