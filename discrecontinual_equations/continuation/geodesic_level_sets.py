"""Two-dimensional invariant manifolds grown as geodesic level sets.

:class:`~.manifold.TaylorManifold` gives a manifold's chart near its
equilibrium. Further out a two-dimensional manifold has to be grown, and doing
that by integrating a fan of trajectories gives curves that crowd where the
flow converges and spread apart where it diverges - the familiar failure of
drawing a surface as trajectories. Krauskopf and Osinga's method grows it as
level sets of geodesic distance from the equilibrium instead:

* start from a closed curve on the Taylor chart, every point of it at geodesic
  distance ``delta`` from the equilibrium (the chart supplies the curvature);
* for every point ``p`` of the outermost ring, find the point ``q`` of the
  manifold that lies in the plane through ``p`` normal to the ring and at
  distance ``Delta`` from ``p``. The manifold there is made of the orbits
  through the ring, so ``q`` is the end of an orbit segment that starts on the
  ring (at an unknown place along it, after an unknown time) and lands in that
  plane at that distance: a two-point boundary-value problem, solved here by
  Newton's method from the landing end (see :func:`_land`);
* insert points where the new ring's spacing grows too wide and remove them
  where it grows too tight, and shrink ``Delta`` where consecutive radial
  steps turn too sharply - where the manifold curves - and let it grow back
  where they are straight;
* zip each pair of consecutive rings into triangles.

Rings are therefore at equal geodesic distance by construction, and the mesh
is one round per level set with stitches added where the surface flares: the
crochet pattern Osinga and Krauskopf crocheted the Lorenz manifold from, which
:meth:`GeodesicMesh.crochet_pattern` reports.

Each step measures ``Delta`` as a chord in the normal plane rather than as an
arc of the manifold, so on a curved manifold the distances are second-order
accurate in ``Delta``; on a plane they are exact.

References: B. Krauskopf and H. M. Osinga, "Computing geodesic level sets on
global (un)stable manifolds of vector fields", SIAM J. Appl. Dyn. Syst. 2
(2003); "The Lorenz manifold as a collection of geodesic level sets",
Nonlinearity 17 (2004); B. Krauskopf et al., "A survey of methods for
computing (un)stable manifolds of vector fields", Int. J. Bifurcation and
Chaos 15 (2005).
"""

import math
from collections.abc import Callable

import numpy as np
from scipy.spatial import cKDTree

from discrecontinual_equations.continuation.manifold import (
    ManifoldSelection,
    TaylorManifold,
    UnstableManifold,
)
from discrecontinual_equations.function.function import Function

_SURFACE = 2
# An eigenvalue with an imaginary part this big is half of a complex pair.
_COMPLEX = 1.0e-9
# Fewest points a ring is allowed to have; fewer cannot bound a disc.
_MINIMUM_RING = 6
# Newton iterations allowed for one landing before the point is declared lost.
_NEWTON_ITERATIONS = 25
# Finite-difference step along the ring for the shooting map's s-derivative,
# in units of one ring segment.
_RING_STEP = 1.0e-6
# Cap on one Newton step along the ring, in segments: a step that wants to
# jump further than this is crossing to a different orbit, not converging.
_RING_JUMP = 1.0
# A Newton step may shrink the flight time to no less than this fraction of
# itself, and grow it to no more than its inverse.
_TIME_FLOOR = 0.25
# Below this fraction of the speed, the outward component of the field is too
# weak to give the first time guess; the speed itself is used instead.
_OUTWARD_FLOOR = 0.2
# Points of three nearly collinear ring points have no circle through them.
_COLLINEAR = 1.0e-12
# Gauss-Legendre nodes for the chart rays' arclength, and fixed-point passes
# that set the seed ring at geodesic distance ``initial_radius``.
_QUADRATURE = 8
_SEED_PASSES = 6
_DIFFERENCE = 1.0e-6
# How many times one ring may be re-filled where its spacing is too wide.
_INSERT_PASSES = 6
# How much Delta grows by where the rings are straight, and shrinks by where
# they turn too sharply or a landing fails.
_GROW = 1.5
_SHRINK = 0.5


class GeodesicSettings:
    """How finely the manifold is grown.

    ``initial_radius`` is the geodesic radius of the seed ring on the Taylor
    chart, and ``step`` the distance between consecutive rings, which may be
    shrunk to ``step_range[0]`` where the manifold curves and is never more
    than ``step_range[1]`` (by default ``step`` itself, so a plane is grown at
    exactly ``step``). ``spacing`` bounds the distance between neighbouring
    points on a ring: wider gaps get a point inserted, tighter ones lose one.
    ``angles`` bounds, in radians, how sharply consecutive radial steps may
    turn: above the upper bound ``Delta`` is halved and the ring recomputed,
    below the lower it grows. ``tolerance`` is the BVP's: both endpoint
    conditions are met to it. ``integration_steps`` is the number of RK4 steps
    per orbit segment, and ``max_vertices`` stops a growth that has become
    larger than anyone will draw.

    ``loss`` is the fraction of a ring's points whose BVP may fail. More
    than that and ``Delta`` is shrunk and the ring recomputed; if it still
    fails at ``step_range[0]`` the growth stops there, and the mesh it
    returns says so in :attr:`GeodesicMesh.stopped` - a manifold traced as
    far as it can be trusted, the way a continuation ends a branch it can no
    longer correct. It is zero by default: a lost point is a hole the next
    ring has to bridge. A small ``loss`` lets a growth go on past a few such
    holes, and counts them in :attr:`GeodesicMesh.lost`.
    """

    __slots__ = [
        "angles",
        "initial_points",
        "initial_radius",
        "integration_steps",
        "loss",
        "max_vertices",
        "spacing",
        "step",
        "step_range",
        "tolerance",
    ]

    def __init__(  # noqa: PLR0913 (one setting per knob of the method)
        self,
        initial_radius: float,
        step: float,
        spacing: tuple[float, float],
        step_range: tuple[float, float] | None = None,
        angles: tuple[float, float] = (0.1, 0.3),
        tolerance: float = 1.0e-10,
        integration_steps: int = 24,
        max_vertices: int = 250_000,
        initial_points: int | None = None,
        loss: float = 0.0,
    ) -> None:
        self.initial_radius = initial_radius
        self.step = step
        self.spacing = spacing
        self.step_range = step_range or (step / 16.0, step)
        self.angles = angles
        self.tolerance = tolerance
        self.integration_steps = integration_steps
        self.max_vertices = max_vertices
        self.initial_points = initial_points
        self.loss = loss


class CrochetRound:
    """One round of the crochet pattern: one geodesic level set.

    ``stitches`` is the number of points on the ring; ``increases`` and
    ``decreases`` are how many were inserted and removed against the round
    before it, so ``stitches`` is the previous round's plus ``increases``
    minus ``decreases``. A point of the round before whose BVP failed is a
    decrease; an insertion whose BVP failed was never a stitch, and is
    counted only in :attr:`GeodesicMesh.lost`.
    """

    __slots__ = ["decreases", "distance", "increases", "number", "stitches"]

    def __init__(
        self,
        number: int,
        distance: float,
        stitches: int,
        increases: int,
        decreases: int,
    ) -> None:
        self.number = number
        self.distance = distance
        self.stitches = stitches
        self.increases = increases
        self.decreases = decreases


class GeodesicMesh:
    """A triangulated two-dimensional manifold and its geodesic level sets.

    ``vertices`` is ``(N, n)``, vertex 0 being the equilibrium; ``triangles``
    is ``(M, 3)`` indices into it; ``distances`` is each vertex's geodesic
    distance from the equilibrium. ``rings`` lists each level set's vertex
    indices in order around it, ``rings[0]`` being the seed on the chart, and
    ``ring_distances`` their distances. ``residual`` is the largest endpoint
    residual of any BVP solved, ``lost`` the number of points whose BVP did
    not converge and were dropped, and ``seed`` says what the first ring came
    from: ``"taylor"`` for the parameterisation-method chart, ``"eigenspace"``
    for the tangent plane where the chart does not exist (a complex pair, or a
    resonance). ``stopped`` is ``None`` when the growth reached the distance
    asked for, and otherwise says why it stopped where it did; the last of
    ``ring_distances`` is how far it got.
    """

    __slots__ = [
        "_decreases",
        "_increases",
        "distances",
        "lost",
        "residual",
        "ring_distances",
        "rings",
        "seed",
        "stopped",
        "triangles",
        "vertices",
    ]

    def __init__(  # noqa: PLR0913 (the mesh and what it is made of)
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        distances: np.ndarray,
        rings: list[np.ndarray],
        ring_distances: list[float],
        changes: tuple[list[int], list[int]],
        seed: str,
    ) -> None:
        self.vertices = vertices
        self.triangles = triangles
        self.distances = distances
        self.rings = rings
        self.ring_distances = ring_distances
        self._increases, self._decreases = changes
        self.seed = seed
        self.residual = 0.0
        self.lost = 0
        self.stopped: str | None = None

    def crochet_pattern(self) -> list[CrochetRound]:
        """One round per level set, with its stitches, increases and decreases."""
        return [
            CrochetRound(
                number,
                float(distance),
                len(ring),
                self._increases[number],
                self._decreases[number],
            )
            for number, (ring, distance) in enumerate(
                zip(self.rings, self.ring_distances, strict=True),
            )
        ]


class GeodesicLevelSets:
    """Grow a two-dimensional (un)stable manifold ring by ring."""

    __slots__ = ["_order", "_settings"]

    def __init__(self, settings: GeodesicSettings, order: int = 5) -> None:
        self._settings = settings
        self._order = order

    def compute(
        self,
        function: Function,
        equilibrium: np.ndarray,
        jacobian: np.ndarray,
        selection: ManifoldSelection,
        distance: float,
    ) -> GeodesicMesh:
        """Grow the selected manifold of ``equilibrium`` to geodesic ``distance``.

        The selected eigenspace must be two-dimensional; a curve is
        :class:`~.manifold.TaylorManifold`'s, and three dimensions or more
        are not level *curves*.
        """
        settings = self._settings
        equilibrium = np.asarray(equilibrium, dtype=float)
        values = np.linalg.eigvals(jacobian)
        if len(selection.indices(values)) != _SURFACE:
            message = (
                "geodesic level sets grow a two-dimensional manifold; the "
                f"selected eigenspace has {len(selection.indices(values))}"
            )
            raise ValueError(message)
        field = _batched(function, equilibrium)
        sense = 1.0 if isinstance(selection, UnstableManifold) else -1.0
        points, directions, seed = self._seed(
            function,
            equilibrium,
            jacobian,
            selection,
        )
        growth = _Growth(equilibrium, points, directions, settings.initial_radius)
        radius = settings.initial_radius
        step = settings.step
        low, high = settings.step_range
        stopped = None
        while radius < distance - 1.0e-12:
            if len(growth.vertices) >= settings.max_vertices:
                stopped = f"max_vertices ({settings.max_vertices}) reached"
                break
            delta = min(step, distance - radius)
            ring = _advance(field, sense, growth.front, delta, settings)
            lost = ring.lost > settings.loss * len(growth.front.points)
            floor = delta <= low * (1 + 1e-9)
            if (lost or ring.turn > settings.angles[1]) and not floor:
                step = max(low, _SHRINK * delta)
                continue
            if lost:
                stopped = (
                    f"{ring.lost} of {len(growth.front.points)} landings failed "
                    f"at the smallest step {delta:.3g}, beyond distance {radius:.6g}"
                )
                break
            if len(ring.points) < _MINIMUM_RING:
                stopped = (
                    f"the ring beyond distance {radius:.6g} kept "
                    f"{len(ring.points)} points, fewer than {_MINIMUM_RING}"
                )
                break
            radius += delta
            growth.add(ring, radius)
            if ring.turn < settings.angles[0]:
                step = min(high, _GROW * step)
        mesh = growth.mesh(seed)
        mesh.stopped = stopped
        return mesh

    def _seed(
        self,
        function: Function,
        equilibrium: np.ndarray,
        jacobian: np.ndarray,
        selection: ManifoldSelection,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """The first ring, at geodesic distance ``initial_radius`` on the chart.

        Rays of the chart from the equilibrium are walked out until their
        arclength is the radius - the chart's own curvature, not its tangent
        plane - and the rays are spread evenly in an orthonormal frame of the
        tangent plane, since the chart's coordinates need not be orthogonal.
        Where there is no Taylor chart the tangent plane stands in for it
        (see :meth:`_chart`).
        """
        settings = self._settings
        chart, seed = self._chart(function, equilibrium, jacobian, selection)
        linear = np.column_stack(
            [
                (chart(_DIFFERENCE * axis) - chart(-_DIFFERENCE * axis))
                / (2.0 * _DIFFERENCE)
                for axis in np.eye(_SURFACE)
            ],
        )
        frame, _ = np.linalg.qr(linear)
        radius = settings.initial_radius
        count = settings.initial_points or max(
            8,
            math.ceil(2.0 * math.pi * radius / (0.5 * sum(settings.spacing))),
        )
        nodes, weights = np.polynomial.legendre.leggauss(_QUADRATURE)
        points, directions = [], []
        for angle in np.linspace(0.0, 2.0 * math.pi, count, endpoint=False):
            target = frame @ np.array([math.cos(angle), math.sin(angle)])
            ray = np.linalg.lstsq(linear, target, rcond=None)[0]

            def velocity(rho: float, ray: np.ndarray = ray) -> np.ndarray:
                h = _DIFFERENCE * max(rho, radius)
                return (chart((rho + h) * ray) - chart((rho - h) * ray)) / (2.0 * h)

            rho = radius
            for _ in range(_SEED_PASSES):
                length = sum(
                    weight * np.linalg.norm(velocity(0.5 * rho * (node + 1.0)))
                    for node, weight in zip(nodes, weights, strict=True)
                )
                rho *= radius / (0.5 * rho * length)
            points.append(chart(rho * ray))
            outward = velocity(rho)
            directions.append(outward / np.linalg.norm(outward))
        return np.array(points), np.array(directions), seed

    def _chart(
        self,
        function: Function,
        equilibrium: np.ndarray,
        jacobian: np.ndarray,
        selection: ManifoldSelection,
    ) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
        """The Taylor chart, or the tangent plane where there is none.

        A complex pair spans a plane the Taylor chart does not parameterise
        (see :mod:`~.manifold`), and a resonance - an exact ``k . lambda`` in
        the spectrum - leaves its order-by-order solve singular. Either way the
        tangent plane is the chart, which is first-order accurate, so the seed
        radius should then be small.
        """
        values, vectors = np.linalg.eig(jacobian)
        chosen = selection.indices(values)
        if all(abs(values[i].imag) <= _COMPLEX for i in chosen):
            try:
                taylor = TaylorManifold(order=self._order).compute(
                    function,
                    equilibrium,
                    jacobian,
                    selection,
                )
            except np.linalg.LinAlgError:
                pass
            else:

                def curved(theta: np.ndarray) -> np.ndarray:
                    return np.asarray(taylor.point(list(theta)), dtype=float)

                return curved, "taylor"
            frame = np.column_stack([vectors[:, i].real for i in chosen])
        else:
            first = vectors[:, chosen[0]]
            frame = np.column_stack([first.real, first.imag])

        def flat(theta: np.ndarray) -> np.ndarray:
            return equilibrium + frame @ theta

        return flat, "eigenspace"


def _batched(
    function: Function,
    equilibrium: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """The field evaluated on a ``(k, n)`` stack of states at once.

    A field written component by component (``x * y - z``) evaluates on whole
    columns as readily as on numbers, which is what makes growing a mesh of
    thousands of points affordable. That is checked once against the field
    point by point; a field that does not survive it is looped over instead.
    """

    def looped(states: np.ndarray) -> np.ndarray:
        return np.array(
            [function.eval(point=list(state), time=None) for state in states],
            dtype=float,
        )

    def columns(states: np.ndarray) -> np.ndarray:
        values = function.eval(point=states.T, time=None)
        return np.stack(
            [
                np.broadcast_to(np.asarray(v, dtype=float), states.shape[:1])
                for v in values
            ],
            axis=1,
        )

    generator = np.random.default_rng(0)
    probe = equilibrium + generator.uniform(-1.0, 1.0, (3, equilibrium.size))
    try:
        agrees = np.allclose(columns(probe), looped(probe), rtol=1e-12, atol=1e-12)
    except (TypeError, ValueError, IndexError, AttributeError):
        agrees = False
    return columns if agrees else looped


def _flow(
    field: Callable[[np.ndarray], np.ndarray],
    sense: float,
    starts: np.ndarray,
    times: np.ndarray,
    steps: int,
) -> np.ndarray:
    """RK4 from each start for its own time, all starts at once."""
    dt = (sense * times / steps)[:, None]
    state = starts.copy()
    # A Newton iterate with a wild flight time can overflow; the caller
    # discards that row, so the overflow itself is not worth a warning.
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(steps):
            k1 = field(state)
            k2 = field(state + 0.5 * dt * k1)
            k3 = field(state + 0.5 * dt * k2)
            k4 = field(state + dt * k3)
            state = state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


def _circle_centres(
    before: np.ndarray,
    at: np.ndarray,
    after: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The centre of the circle through each triple, and where one exists."""
    a, b = before - after, at - after
    cross = np.cross(a, b)
    square = np.einsum("ij,ij->i", cross, cross)
    scale = np.einsum("ij,ij->i", a, a) * np.einsum("ij,ij->i", b, b)
    valid = square > _COLLINEAR * np.maximum(scale, 1.0e-300)
    numerator = np.cross(
        np.einsum("ij,ij->i", a, a)[:, None] * b
        - np.einsum("ij,ij->i", b, b)[:, None] * a,
        cross,
    )
    centre = after + numerator / (2.0 * np.where(valid, square, 1.0))[:, None]
    return centre, valid


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _ring_tangents(ring: np.ndarray) -> np.ndarray:
    """The tangent at each ring point, from the circle through it and its neighbours.

    A circle through three points is exact on a circle and second order on any
    smooth curve, which a central difference on unevenly spaced points is not.
    """
    before, after = np.roll(ring, 1, axis=0), np.roll(ring, -1, axis=0)
    chord = after - before
    centre, valid = _circle_centres(before, ring, after)
    radial = _unit(np.where(valid[:, None], ring - centre, chord))
    along = np.einsum("ij,ij->i", chord, radial)[:, None] * radial
    tangent = np.where(valid[:, None], chord - along, chord)
    return _unit(tangent)


def _arc(
    centre: np.ndarray,
    valid: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The point a ``fraction`` of the way along the arc, and its tangent."""
    u, w = start - centre, end - centre
    cosine = np.einsum("ij,ij->i", _unit(u), _unit(w))
    theta = np.arccos(np.clip(cosine, -1.0, 1.0))
    sine = np.sin(theta)
    usable = valid & (sine > _COLLINEAR)
    safe = np.where(usable, sine, 1.0)
    t = fraction
    point = centre + (
        (np.sin((1.0 - t) * theta) / safe)[:, None] * u
        + (np.sin(t * theta) / safe)[:, None] * w
    )
    tangent = (-np.cos((1.0 - t) * theta))[:, None] * u + np.cos(t * theta)[:, None] * w
    linear = start + t[:, None] * (end - start)
    point = np.where(usable[:, None], point, linear)
    tangent = np.where(usable[:, None], tangent, end - start)
    return point, _unit(tangent)


def _interpolate(
    ring: np.ndarray,
    keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Points and tangents on the ring at fractional indices ``keys``.

    Each is the average of the arcs of the two circles through the segment
    and one neighbour either side: exact on a circle, as the plane's level
    sets are, and second order on any smooth ring.
    """
    count = len(ring)
    index = np.floor(keys).astype(int) % count
    fraction = keys - np.floor(keys)
    p0 = ring[(index - 1) % count]
    p1 = ring[index]
    p2 = ring[(index + 1) % count]
    p3 = ring[(index + 2) % count]
    first, valid_first = _circle_centres(p0, p1, p2)
    second, valid_second = _circle_centres(p1, p2, p3)
    point_a, tangent_a = _arc(first, valid_first, p1, p2, fraction)
    point_b, tangent_b = _arc(second, valid_second, p1, p2, fraction)
    return 0.5 * (point_a + point_b), _unit(tangent_a + tangent_b)


# The back-trace walks along its orbit in steps of this fraction of Delta (or
# of the widest ring spacing, if that is larger: Delta may have been shrunk to
# its floor), and gives up after this many steps - 250 such lengths of
# arclength, far longer than any orbit that reaches the ring at all takes.
_TRACE_STEP = 0.125
_TRACE_LIMIT = 2000
# RK4 steps are kept to this many inverse rates of the local field, which is
# both stable and accurate to the BVP tolerance on a step of that size.
_STEP_RATE = 0.5
_MAXIMUM_STEPS = 4000
# Relative finite-difference step for the flow's Jacobian in the landing point.
_SHIFT = 1.0e-7


def _first_guess(  # noqa: PLR0913 (the BVP's data)
    field: Callable[[np.ndarray], np.ndarray],
    sense: float,
    front: "_Front",
    centres: np.ndarray,
    keys: np.ndarray,
    outward: np.ndarray,
    delta: float,
    settings: GeodesicSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Where on the ring each orbit starts, and how long it flows: a guess.

    Traced back from ``delta`` straight out to where that crosses the ring;
    otherwise the centre's own place and the time the outward part of the
    field would take to cover ``delta``.
    """
    speed = sense * field(centres)
    radial = np.einsum("ij,ij->i", speed, outward)
    norm = np.linalg.norm(speed, axis=1)
    times = delta / np.maximum(radial, _OUTWARD_FLOOR * norm)
    places = keys.astype(float).copy()
    traced, crossing, flight = _trace_back(
        field,
        sense,
        front,
        centres + delta * outward,
        max(delta, settings.spacing[1]),
    )
    places[traced] = crossing[traced]
    times[traced] = flight[traced]
    return places, times


def _trace_back(
    field: Callable[[np.ndarray], np.ndarray],
    sense: float,
    front: "_Front",
    targets: np.ndarray,
    length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flow each target back to the ring: where it crosses it, and after how long.

    Returns which targets crossed, the ring parameter at the crossing and the
    flight time. A crossing is where the signed distance outside the ring -
    measured along the nearest ring point's outward direction - changes sign
    within ``length`` of the ring. Further out that sign says nothing: on the
    Lorenz scrolls an orbit traced back from beside the ring runs off along
    the scroll and passes 28 to 37 from the ring where its nearest point's
    outward direction already says "inside", and counting that as a crossing
    started Newton 37 ring points from the true orbit.

    The trace runs until it crosses, not for a fixed multiple of a guessed
    time: where one stable rate is many times the other, the orbit that lands
    beside a centre can take several times longer than the outward speed
    suggests, and starts several ring points away. Each step covers a fixed
    fraction of ``length`` of arclength, capped where the field is slow (near
    the equilibrium) by the local rate, so RK4 stays stable.
    """
    ring, outward = front.points, front.directions
    count, rows = len(ring), len(targets)
    tree = cKDTree(ring)
    state = targets.copy()
    nearest = tree.query(state)[1]
    side = np.einsum("ij,ij->i", state - ring[nearest], outward[nearest])
    crossed = np.zeros(rows, dtype=bool)
    alive = side > 0.0
    start = np.zeros(rows)
    flight = np.zeros(rows)
    elapsed = np.zeros(rows)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(_TRACE_LIMIT):
            going = np.flatnonzero(alive & ~crossed)
            if going.size == 0:
                break
            here = state[going]
            speed = np.linalg.norm(field(here), axis=1)
            rate = _local_rate(field, here)
            dt = np.minimum(
                _TRACE_STEP * length / np.maximum(speed, 1e-300),
                _STEP_RATE / np.maximum(rate, 1e-300),
            )[:, None]
            k1 = -sense * field(here)
            k2 = -sense * field(here + 0.5 * dt * k1)
            k3 = -sense * field(here + 0.5 * dt * k2)
            k4 = -sense * field(here + dt * k3)
            moved = here + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            finite = np.all(np.isfinite(moved), axis=1)
            alive[going[~finite]] = False
            moved = np.where(finite[:, None], moved, here)
            near = tree.query(moved)[1]
            now = np.einsum("ij,ij->i", moved - ring[near], outward[near])
            gap = np.linalg.norm(moved - ring[near], axis=1)
            hit = finite & (now <= 0.0) & (side[going] > 0.0) & (gap <= length)
            if np.any(hit):
                rows_hit = going[hit]
                fraction = side[rows_hit] / (side[rows_hit] - now[hit])
                point = here[hit] + fraction[:, None] * (moved[hit] - here[hit])
                start[rows_hit] = _ring_parameter(ring, near[hit], point)
                flight[rows_hit] = elapsed[rows_hit] + fraction * dt[hit, 0]
                crossed[rows_hit] = True
            state[going] = moved
            side[going] = now
            elapsed[going] += dt[:, 0]
    return crossed & (flight > 0.0), start % count, flight


def _local_rate(
    field: Callable[[np.ndarray], np.ndarray],
    states: np.ndarray,
) -> np.ndarray:
    """A bound on the field's fastest local rate: the Jacobian's largest row sum."""
    base = field(states)
    size = states.shape[1]
    total = np.zeros((len(states), size))
    for axis in range(size):
        h = _SHIFT * (1.0 + np.abs(states[:, axis]))
        shifted = states.copy()
        shifted[:, axis] += h
        total += np.abs((field(shifted) - base) / h[:, None])
    return np.max(total, axis=1)


def _ring_parameter(
    ring: np.ndarray,
    near: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """The fractional ring index of each point, from its nearest ring point."""
    count = len(ring)
    best = near.astype(float)
    closest = np.linalg.norm(points - ring[near], axis=1)
    for first in (near - 1, near):
        a, b = ring[first % count], ring[(first + 1) % count]
        segment = b - a
        length = np.maximum(np.einsum("ij,ij->i", segment, segment), 1e-300)
        t = np.clip(np.einsum("ij,ij->i", points - a, segment) / length, 0.0, 1.0)
        distance = np.linalg.norm(a + t[:, None] * segment - points, axis=1)
        better = distance < closest
        best = np.where(better, first + t, best)
        closest = np.minimum(closest, distance)
    return best


class _Landing:
    """Where the orbits from one ring land, and how well."""

    __slots__ = ["points", "residual", "times", "valid"]

    def __init__(
        self,
        points: np.ndarray,
        times: np.ndarray,
        residual: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        self.points = points
        self.times = times
        self.residual = residual
        self.valid = valid


def _land(  # noqa: PLR0913 (the BVP's data)
    field: Callable[[np.ndarray], np.ndarray],
    sense: float,
    front: "_Front",
    centres: np.ndarray,
    tangents: np.ndarray,
    keys: np.ndarray,
    outward: np.ndarray,
    delta: float,
    settings: GeodesicSettings,
) -> _Landing:
    """Solve each centre's BVP: the orbit from the ring that lands at ``delta``.

    The orbit segment runs from ``C(s)`` on the ring for a time ``tau`` to the
    new point ``q``, which must lie in the plane through the centre normal to
    the ring's tangent and at distance ``delta`` from it. It is solved from
    the landing end: the unknowns are ``q``, ``tau`` and ``s``, and the
    conditions are those two on ``q`` and that ``q`` flowed back for ``tau``
    arrives at ``C(s)``. That is the same orbit segment as shooting out from
    the ring, but a far better conditioned problem. Shooting out, where one
    rate on the manifold is many times the other, the landing point moves by
    the ratio's power for every move of ``s``, and at a ratio of 20 no
    ``s`` in double precision lands within the tolerance. Flowing back, the
    manifold's directions contract, so ``s`` is pinned without amplification,
    and it is only the direction off the manifold that grows - which is
    exactly what pins ``q`` onto it.

    The first guess is traced: the point ``delta`` straight out from the
    centre is flowed back until it crosses the ring, and where and when it
    crosses is where Newton starts. The flow's Jacobian in ``q`` is by finite
    differences, all rows integrated together, with as many RK4 steps as the
    local rate needs.

    The ring between its points is the same circular-arc interpolation the
    centres are placed with, not its chords. A chord cuts inside a curved ring
    by its sagitta, and the flow need not carry that error away: on the tilted
    homoclinic system's cylinder the energy is conserved, so an orbit started
    off the manifold by a chord's sagitta stays off it by that much for good.

    A converged landing that is not outward of its centre is a solution of
    the equations on the wrong side - an orbit that crossed back inside the
    ring - and is counted lost.
    """
    ring = front.points
    rows, size = centres.shape
    s, times = _first_guess(
        field,
        sense,
        front,
        centres,
        keys,
        outward,
        delta,
        settings,
    )
    points = centres + delta * outward
    residual = np.full(rows, np.inf)
    valid = np.zeros(rows, dtype=bool)
    active = np.arange(rows)
    # A row Newton has sent the wrong way can overflow; it is dropped below,
    # so its arithmetic is not worth a warning.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(_NEWTON_ITERATIONS):
            if active.size == 0:
                break
            values, jacobian, usable = _shooting_system(
                field,
                sense,
                ring,
                points[active],
                s[active],
                times[active],
                (centres[active], tangents[active], delta),
                settings.integration_steps,
            )
            residual[active] = np.max(np.abs(values), axis=1)
            done = usable & (residual[active] < settings.tolerance)
            valid[active[done]] = True
            update = _newton_update(jacobian, values, usable)
            dy, dt, ds = update[:, :size], update[:, size], update[:, size + 1]
            current = times[active]
            # One damping factor for every unknown, so the step stays a
            # Newton direction: no more than a ring segment along the ring,
            # half of delta for the point, and a bounded change of time.
            damping = np.minimum(1.0, _RING_JUMP / np.maximum(np.abs(ds), 1e-300))
            damping = np.minimum(
                damping,
                0.5 * delta / np.maximum(np.linalg.norm(dy, axis=1), 1e-300),
            )
            shrink = (1.0 - _TIME_FLOOR) * current / np.maximum(dt, 1e-300)
            stretch = (1.0 / _TIME_FLOOR - 1.0) * current / np.maximum(-dt, 1e-300)
            damping = np.minimum(damping, np.where(dt > 0.0, shrink, stretch))
            keep = ~done & usable & np.all(np.isfinite(update), axis=1)
            rows_kept = active[keep]
            points[rows_kept] -= damping[keep, None] * dy[keep]
            times[rows_kept] = current[keep] - damping[keep] * dt[keep]
            s[rows_kept] -= damping[keep] * ds[keep]
            active = rows_kept
    outward_side = np.einsum("ij,ij->i", points - centres, outward) > 0.0
    return _Landing(points, times, residual, valid & outward_side)


def _shooting_system(  # noqa: PLR0913 (the BVP's data)
    field: Callable[[np.ndarray], np.ndarray],
    sense: float,
    ring: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    times: np.ndarray,
    target: tuple[np.ndarray, np.ndarray, float],
    least_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The BVP residuals at ``(y, tau, s)``, their Jacobian, and the usable rows.

    Rows are ``(y - P) . T``, ``|y - P| - delta`` and ``phi_tau(y) - C(s)``
    with ``phi`` the flow towards the ring; columns are ``y``, ``tau``, ``s``.
    """
    centres, tangents, delta = target
    count, m, size = len(ring), len(y), y.shape[1]
    place = np.mod(s, count)
    hit, _ = _interpolate(ring, place)
    nudged, _ = _interpolate(ring, place + _RING_STEP)
    slope = (nudged - hit) / _RING_STEP
    h = _SHIFT * (1.0 + np.linalg.norm(y, axis=1))
    identity = np.eye(size)
    stack = np.vstack([y, *(y + h[:, None] * identity[k] for k in range(size))])
    steps = int(
        np.clip(
            np.ceil(np.max(times * _local_rate(field, y)) / _STEP_RATE),
            least_steps,
            _MAXIMUM_STEPS,
        ),
    )
    ends = _flow(field, -sense, stack, np.tile(times, size + 1), steps)
    finite = np.all(np.isfinite(ends.reshape(size + 1, m, size)), axis=(0, 2))
    ends = np.where(np.isfinite(ends), ends, 0.0)
    back = ends[:m]
    offset = y - centres
    length = np.linalg.norm(offset, axis=1)
    values = np.column_stack(
        [np.einsum("ij,ij->i", offset, tangents), length - delta, back - hit],
    )
    jacobian = np.zeros((m, size + 2, size + 2))
    jacobian[:, 0, :size] = tangents
    jacobian[:, 1, :size] = offset / np.maximum(length, 1e-300)[:, None]
    for k in range(size):
        jacobian[:, 2:, k] = (ends[(k + 1) * m : (k + 2) * m] - back) / h[:, None]
    jacobian[:, 2:, size] = -sense * field(back)
    jacobian[:, 2:, size + 1] = -slope
    usable = finite & np.all(np.isfinite(jacobian), axis=(1, 2))
    return values, jacobian, usable


def _newton_update(
    jacobian: np.ndarray,
    values: np.ndarray,
    usable: np.ndarray,
) -> np.ndarray:
    """Solve every row's Newton system, by least squares where one is singular."""
    jacobian = np.where(usable[:, None, None], jacobian, np.eye(jacobian.shape[1]))
    values = np.where(usable[:, None], values, 0.0)
    try:
        return np.linalg.solve(jacobian, values[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        return np.stack(
            [
                np.linalg.lstsq(matrix, value, rcond=None)[0]
                for matrix, value in zip(jacobian, values, strict=True)
            ],
        )


class _Ring:
    """A new level set, before it is added to the mesh."""

    __slots__ = [
        "decreases",
        "directions",
        "increases",
        "keys",
        "lost",
        "points",
        "residual",
        "turn",
    ]

    def __init__(self) -> None:
        self.points = np.zeros((0, 3))
        self.directions = np.zeros((0, 3))
        self.keys = np.zeros(0)
        self.increases = 0
        self.decreases = 0
        self.lost = 0
        self.residual = 0.0
        self.turn = 0.0


class _Front:
    """The outermost ring: its points and the radial directions they arrived along."""

    __slots__ = ["directions", "indices", "points"]

    def __init__(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.points = points
        self.directions = directions
        self.indices = indices


def _solve(  # noqa: PLR0913 (the BVP's data)
    field: Callable[[np.ndarray], np.ndarray],
    sense: float,
    front: _Front,
    keys: np.ndarray,
    delta: float,
    settings: GeodesicSettings,
) -> tuple[_Landing, np.ndarray]:
    """Land the orbits for the centres at ring parameters ``keys``."""
    count = len(front.points)
    whole = np.isclose(keys, np.round(keys)) & (np.round(keys) < count)
    centres, tangents = _interpolate(front.points, keys)
    exact = np.round(keys[whole]).astype(int)
    centres[whole] = front.points[exact]
    tangents[whole] = _ring_tangents(front.points)[exact]
    index = np.floor(keys).astype(int) % count
    fraction = (keys - np.floor(keys))[:, None]
    outward = _unit(
        (1.0 - fraction) * front.directions[index]
        + fraction * front.directions[(index + 1) % count],
    )
    landing = _land(
        field,
        sense,
        front,
        centres,
        tangents,
        keys,
        outward,
        delta,
        settings,
    )
    steps = np.where(landing.valid[:, None], landing.points - centres, outward)
    return landing, _unit(steps)


def _advance(
    field: Callable[[np.ndarray], np.ndarray],
    sense: float,
    front: _Front,
    delta: float,
    settings: GeodesicSettings,
) -> _Ring:
    """The next level set, ``delta`` beyond the front, spaced within bounds."""
    count = len(front.points)
    keys = np.arange(count, dtype=float)
    landing, steps = _solve(field, sense, front, keys, delta, settings)
    ring = _Ring()
    ring.lost = int(np.count_nonzero(~landing.valid))
    dropped = ring.lost
    turns = [_turn(front, keys, steps, landing.valid)]
    points = landing.points[landing.valid]
    directions = steps[landing.valid]
    keys = keys[landing.valid]
    residuals = [landing.residual[landing.valid]]
    low, high = settings.spacing
    for _ in range(_INSERT_PASSES):
        following = np.roll(points, -1, axis=0)
        gaps = np.linalg.norm(following - points, axis=1)
        wide = np.flatnonzero(gaps > high)
        if wide.size == 0:
            break
        upper = np.where(
            wide == len(keys) - 1,
            keys[0] + count,
            keys[(wide + 1) % len(keys)],
        )
        middle = 0.5 * (keys[wide] + upper)
        middle = np.where(middle >= count, middle - count, middle)
        extra, extra_steps = _solve(field, sense, front, middle, delta, settings)
        good = extra.valid
        ring.lost += int(np.count_nonzero(~good))
        turns.append(_turn(front, middle, extra_steps, good))
        residuals.append(extra.residual[good])
        ring.increases += int(np.count_nonzero(good))
        keys = np.concatenate([keys, middle[good]])
        points = np.vstack([points, extra.points[good]])
        directions = np.vstack([directions, extra_steps[good]])
        order = np.argsort(keys, kind="stable")
        keys, points, directions = keys[order], points[order], directions[order]
        if not np.any(good):
            break
    keep = _thin(points, low, high)
    ring.decreases = dropped + int(np.count_nonzero(~keep))
    ring.points = points[keep]
    ring.directions = directions[keep]
    ring.keys = keys[keep]
    ring.residual = float(
        max((float(np.max(r)) for r in residuals if r.size), default=0.0),
    )
    ring.turn = max(turns)
    return ring


def _turn(
    front: _Front,
    keys: np.ndarray,
    steps: np.ndarray,
    valid: np.ndarray,
) -> float:
    """How sharply the radial direction turns from one ring to the next."""
    if not np.any(valid):
        return 0.0
    count = len(front.points)
    index = np.floor(keys).astype(int) % count
    fraction = (keys - np.floor(keys))[:, None]
    incoming = _unit(
        (1.0 - fraction) * front.directions[index]
        + fraction * front.directions[(index + 1) % count],
    )
    cosine = np.einsum("ij,ij->i", incoming[valid], steps[valid])
    return float(np.max(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _thin(points: np.ndarray, low: float, high: float) -> np.ndarray:
    """Which points to keep so that no two neighbours are closer than ``low``.

    A point goes only if the gap it leaves behind is still within ``high``,
    so thinning never undoes what insertion has just done.
    """
    count = len(points)
    keep = np.ones(count, dtype=bool)
    previous = 0
    for index in range(1, count):
        if count - np.count_nonzero(~keep) <= _MINIMUM_RING:
            break
        # Points after ``index`` are not yet thinned, so its neighbour is kept.
        following = index + 1
        near = np.linalg.norm(points[index] - points[previous]) < low
        bridged = np.linalg.norm(points[following % count] - points[previous])
        if near and bridged <= high:
            keep[index] = False
        else:
            previous = index
    return keep


class _Growth:
    """The mesh as it is grown: vertices, triangles, rings."""

    __slots__ = [
        "decreases",
        "distances",
        "front",
        "increases",
        "lost",
        "residual",
        "ring_distances",
        "rings",
        "triangles",
        "vertices",
    ]

    def __init__(
        self,
        equilibrium: np.ndarray,
        points: np.ndarray,
        directions: np.ndarray,
        radius: float,
    ) -> None:
        count = len(points)
        self.vertices: list[np.ndarray] = [equilibrium, *points]
        self.distances: list[float] = [0.0] + [radius] * count
        indices = np.arange(1, count + 1)
        self.rings = [indices]
        self.ring_distances = [radius]
        self.triangles: list[tuple[int, int, int]] = [
            (0, int(indices[(i + 1) % count]), int(indices[i])) for i in range(count)
        ]
        self.front = _Front(points, directions, indices)
        self.increases = [0]
        self.decreases = [0]
        self.residual = 0.0
        self.lost = 0

    def add(self, ring: _Ring, radius: float) -> None:
        """Append the ring's vertices and zip it to the front."""
        first = len(self.vertices)
        self.vertices.extend(ring.points)
        self.distances.extend([radius] * len(ring.points))
        indices = np.arange(first, first + len(ring.points))
        self.triangles.extend(_zip(self.front.indices, indices, ring.keys))
        self.rings.append(indices)
        self.ring_distances.append(radius)
        self.increases.append(ring.increases)
        self.decreases.append(ring.decreases)
        self.residual = max(self.residual, ring.residual)
        self.lost += ring.lost
        self.front = _Front(ring.points, ring.directions, indices)

    def mesh(self, seed: str) -> GeodesicMesh:
        mesh = GeodesicMesh(
            np.array(self.vertices),
            np.array(self.triangles, dtype=int),
            np.array(self.distances),
            self.rings,
            self.ring_distances,
            (self.increases, self.decreases),
            seed,
        )
        mesh.residual = self.residual
        mesh.lost = self.lost
        return mesh


def _zip(
    old: np.ndarray,
    new: np.ndarray,
    keys: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Triangulate the band between two consecutive rings.

    Each new point carries the ring parameter of the old point (or the place
    between two) it was grown from, so walking both rings in that parameter
    and always advancing whichever ring's next point comes first stitches the
    band without crossings - the way a crochet round is worked into the last.
    """
    count, size = len(old), len(new)
    triangles: list[tuple[int, int, int]] = []
    i = j = 0
    while i < count or j < size:
        next_old = float(i + 1)
        next_new = float(keys[j + 1]) if j + 1 < size else float(keys[0]) + count
        a, b = int(old[i % count]), int(new[j % size])
        if j >= size or (i < count and next_old <= next_new):
            triangles.append((a, int(old[(i + 1) % count]), b))
            i += 1
        else:
            triangles.append((a, int(new[(j + 1) % size]), b))
            j += 1
    return triangles
