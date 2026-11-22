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
  plane at that distance: a two-point boundary-value problem in two unknowns,
  solved here by Newton's method on a shooting map;
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
# A Newton determinant smaller than this is a singular shooting map.
_SINGULAR = 1.0e-300
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

    ``loss`` is the fraction of a ring's points whose BVP may fail before
    ``Delta`` is shrunk and the ring recomputed. It is zero by default: a
    lost point is a hole the next ring has to bridge. Where the flow runs
    along a level set - on the Lorenz manifold beyond a geodesic distance of
    about 105, measured - the orbits that land beside a point start far
    along the ring and some BVPs have no nearby solution at any ``Delta``,
    so a strict growth stalls there with ``Delta`` at its floor and the
    rings multiplying. A small ``loss`` lets it go on, at the price of
    those holes.
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
    minus ``decreases``.
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
    resonance).
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
        while radius < distance - 1.0e-12:
            if len(growth.vertices) >= settings.max_vertices:
                break
            delta = min(step, distance - radius)
            ring = _advance(field, sense, growth.front, delta, settings)
            turn = ring.turn
            low, high = settings.step_range
            lost = ring.lost > settings.loss * len(growth.front.points)
            if (lost or turn > settings.angles[1]) and delta > low * (1 + 1e-9):
                step = max(low, _SHRINK * delta)
                continue
            radius += delta
            growth.add(ring, radius)
            if turn < settings.angles[0]:
                step = min(high, _GROW * step)
        return growth.mesh(seed)

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


# Steps the back-trace takes, and how many first-guess flight times it may
# run for before giving up on finding the ring.
_TRACE_STEPS = 48
_TRACE_SPAN = 4.0


def _first_guess(  # noqa: PLR0913 (the BVP's data)
    field: Callable[[np.ndarray], np.ndarray],
    sense: float,
    front: "_Front",
    centres: np.ndarray,
    keys: np.ndarray,
    outward: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Where on the ring each orbit starts, and how long it flows: a guess.

    Traced back from ``delta`` straight out where that crosses the ring;
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
        times,
    )
    places[traced] = crossing[traced]
    times[traced] = flight[traced]
    return places, times


def _trace_back(
    field: Callable[[np.ndarray], np.ndarray],
    sense: float,
    front: "_Front",
    targets: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flow each target back to the ring: where it crosses it, and after how long.

    Returns which targets crossed, the ring parameter at the crossing and the
    flight time. A crossing is where the signed distance outside the ring -
    measured along the nearest ring point's outward direction - changes sign.
    """
    ring, outward = front.points, front.directions
    count, rows = len(ring), len(targets)
    tree = cKDTree(ring)
    dt = (_TRACE_SPAN * times / _TRACE_STEPS)[:, None]
    state = targets.copy()
    nearest = tree.query(state)[1]
    side = np.einsum("ij,ij->i", state - ring[nearest], outward[nearest])
    crossed = np.zeros(rows, dtype=bool)
    start = np.zeros(rows)
    flight = np.zeros(rows)
    with np.errstate(over="ignore", invalid="ignore"):
        for step in range(1, _TRACE_STEPS + 1):
            k1 = -sense * field(state)
            k2 = -sense * field(state + 0.5 * dt * k1)
            k3 = -sense * field(state + 0.5 * dt * k2)
            k4 = -sense * field(state + dt * k3)
            moved = state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            moved = np.where(np.isfinite(moved), moved, state)
            near = tree.query(moved)[1]
            now = np.einsum("ij,ij->i", moved - ring[near], outward[near])
            hit = ~crossed & (side > 0.0) & (now <= 0.0)
            if np.any(hit):
                fraction = side[hit] / (side[hit] - now[hit])
                point = state[hit] + fraction[:, None] * (moved[hit] - state[hit])
                start[hit] = _ring_parameter(ring, near[hit], point)
                flight[hit] = (step - 1.0 + fraction) * dt[hit, 0]
                crossed |= hit
            state, side = moved, now
    return crossed & (flight > 0.0), start % count, flight


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

    The unknowns are ``s``, where along the ring the orbit starts, and
    ``tau``, how long it flows; the conditions are that it ends in
    the plane through the centre normal to the ring's tangent, and at distance
    ``delta`` from the centre. Newton's method on that two-by-two shooting map
    starts from the centre's own place on the ring and the time the outward
    part of the field would take to cover ``delta``, with the s-derivative by
    a finite difference along the ring and the tau-derivative the field.

    Where the flow runs along the ring rather than out of it - near the
    Lorenz origin, where the strong stable rate is 8.6 times the weak one -
    the orbit that lands beside a centre starts far from it along the ring,
    and Newton from the centre's own place does not find it. So the first
    guess is traced: the point ``delta`` straight out from the centre is
    flowed back until it crosses the ring, and where and when it crosses is
    where Newton starts.

    The ring between its points is the same circular-arc interpolation the
    centres are placed with, not its chords. A chord cuts inside a curved ring
    by its sagitta, and the flow need not carry that error away: on the tilted
    homoclinic system's cylinder the energy is conserved, so an orbit started
    off the manifold by a chord's sagitta stays off it by that much for good.
    """
    ring = front.points
    count, rows = len(ring), len(centres)
    s, times = _first_guess(field, sense, front, centres, keys, outward, delta)
    points = centres.copy()
    residual = np.full(rows, np.inf)
    valid = np.zeros(rows, dtype=bool)
    active = np.arange(rows)
    # A row Newton has sent the wrong way can overflow; it is dropped below,
    # so its arithmetic is not worth a warning.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(_NEWTON_ITERATIONS):
            if active.size == 0:
                break
            place = np.mod(s[active], count)
            start, _ = _interpolate(ring, place)
            nudged, _ = _interpolate(ring, place + _RING_STEP)
            ends = _flow(
                field,
                sense,
                np.vstack([start, nudged]),
                np.concatenate([times[active], times[active]]),
                settings.integration_steps,
            )
            end, shifted = ends[: active.size], ends[active.size :]
            # An orbit Newton has sent the wrong way overflows rather than lands;
            # that point is lost, not the ring.
            finite = np.all(np.isfinite(ends[: active.size]), axis=1) & np.all(
                np.isfinite(shifted),
                axis=1,
            )
            end = np.where(finite[:, None], end, centres[active])
            shifted = np.where(finite[:, None], shifted, centres[active])
            offset = end - centres[active]
            length = np.linalg.norm(offset, axis=1)
            plane = np.einsum("ij,ij->i", offset, tangents[active])
            reach = length - delta
            points[active] = end
            residual[active] = np.maximum(np.abs(plane), np.abs(reach))
            done = residual[active] < settings.tolerance
            valid[active[done]] = True
            direction = offset / np.maximum(length, 1.0e-300)[:, None]
            along = (shifted - end) / _RING_STEP
            flow = sense * field(end)
            a11 = np.einsum("ij,ij->i", along, tangents[active])
            a12 = np.einsum("ij,ij->i", flow, tangents[active])
            a21 = np.einsum("ij,ij->i", along, direction)
            a22 = np.einsum("ij,ij->i", flow, direction)
            det = a11 * a22 - a12 * a21
            singular = np.abs(det) < _SINGULAR
            det = np.where(singular, 1.0, det)
            ds = (plane * a22 - a12 * reach) / det
            dt = (a11 * reach - a21 * plane) / det
            current = times[active]
            # One damping factor for both unknowns, so the step stays a
            # Newton direction: clipping s alone and not tau sent tau
            # downhill until it reached zero wherever the flow ran along the
            # ring and the start had to be found more than a segment away.
            damping = np.minimum(1.0, _RING_JUMP / np.maximum(np.abs(ds), 1e-300))
            shrink = (1.0 - _TIME_FLOOR) * current / np.maximum(dt, 1e-300)
            stretch = (1.0 / _TIME_FLOOR - 1.0) * current / np.maximum(-dt, 1e-300)
            damping = np.minimum(damping, np.where(dt > 0.0, shrink, stretch))
            ds = damping * ds
            updated = current - damping * dt
            keep = ~done & ~singular & finite & np.isfinite(ds) & np.isfinite(updated)
            s[active[keep]] -= ds[keep]
            times[active[keep]] = updated[keep]
            active = active[keep]
    return _Landing(points, times, residual, valid)


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
            wide == len(keys) - 1, keys[0] + count, keys[(wide + 1) % len(keys)],
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
    ring.decreases = int(np.count_nonzero(~keep))
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
        following = index + 1
        while following < count and not keep[following % count]:
            following += 1
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
        self.decreases.append(ring.decreases + ring.lost)
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
