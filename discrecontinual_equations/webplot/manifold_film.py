"""A two-dimensional manifold as a film whose clock is geodesic distance.

The stage films scrub a parameter. A manifold grown by
:class:`~discrecontinual_equations.continuation.geodesic_level_sets.GeodesicLevelSets`
has a parameter of its own - the geodesic distance from the equilibrium - and
its level sets are the rings it was grown in. So this film scrubs that: the
surface grows ring by ring as the viewer drags, coloured by distance, with the
current ring drawn as the front, and can be turned in three dimensions.

The scene is only data: the mesh, the rings, the crochet pattern they make, and
whatever curves (an orbit that lies in the surface, say) and points belong
beside it. :class:`~.manifold_film_renderer.ManifoldFilmRenderer` draws it.
"""

from collections.abc import Sequence
from itertools import pairwise

import numpy as np

from discrecontinual_equations.continuation.geodesic_level_sets import (
    CrochetRound,
    GeodesicMesh,
)

# Digits kept for a vertex in the page. The mesh is drawn at a few hundred
# pixels across, so four significant digits past the largest coordinate's
# magnitude is far below a pixel and keeps a 100,000-coordinate page small.
_DIGITS = 4


class FilmCurve:
    """A curve drawn with the surface, in full state coordinates.

    ``kind`` names its colour - ``"stable"``, ``"unstable"``, ``"orbit"`` - and
    ``label`` is what the legend calls it.
    """

    __slots__ = ["kind", "label", "points"]

    def __init__(
        self,
        label: str,
        kind: str,
        points: Sequence[Sequence[float]],
    ) -> None:
        self.label = label
        self.kind = kind
        self.points = [[float(v) for v in point] for point in points]


class FilmMarker:
    """A labelled point, an equilibrium say."""

    __slots__ = ["kind", "label", "point"]

    def __init__(self, label: str, kind: str, point: Sequence[float]) -> None:
        self.label = label
        self.kind = kind
        self.point = [float(v) for v in point]


class ManifoldFilm:
    """A grown manifold, ready to be drawn ring by ring.

    ``axes`` names the three coordinates drawn; ``projection`` maps a full
    state onto them, three rows of the state's length, so a manifold of a
    four-dimensional system can still be filmed. ``note`` is the page's
    closing paragraph: how the surface was computed, and what it took.
    """

    __slots__ = [
        "axes",
        "curves",
        "equation",
        "markers",
        "mesh",
        "note",
        "projection",
        "subtitle",
        "title",
    ]

    def __init__(  # noqa: PLR0913 (a page's worth of content)
        self,
        title: str,
        subtitle: str,
        mesh: GeodesicMesh,
        axes: tuple[str, str, str] = ("x", "y", "z"),
        projection: Sequence[Sequence[float]] | None = None,
        equation: str = "",
        note: str = "",
    ) -> None:
        self.title = title
        self.subtitle = subtitle
        self.mesh = mesh
        self.axes = axes
        size = mesh.vertices.shape[1]
        self.projection = (
            np.eye(3, size) if projection is None else np.asarray(projection, float)
        )
        self.equation = equation
        self.note = note
        self.curves: list[FilmCurve] = []
        self.markers: list[FilmMarker] = []


def film_payload(film: ManifoldFilm) -> dict:
    """The film as plain JSON: the mesh packed (see :func:`_mesh_payload`)."""
    mesh = film.mesh
    vertices = mesh.vertices @ film.projection.T
    scale = float(np.max(np.abs(vertices))) or 1.0
    digits = max(0, _DIGITS - int(np.floor(np.log10(scale))))
    return {
        "title": film.title,
        "axes": list(film.axes),
        **_mesh_payload(mesh, vertices, digits),
        "ringDistances": [round(float(r), 6) for r in mesh.ring_distances],
        "pattern": [_round(row) for row in mesh.crochet_pattern()],
        "curves": [
            {
                "label": curve.label,
                "kind": curve.kind,
                "points": np.round(
                    np.asarray(curve.points) @ film.projection.T,
                    digits,
                ).tolist(),
            }
            for curve in film.curves
        ],
        "markers": [
            {
                "label": marker.label,
                "kind": marker.kind,
                "point": np.round(
                    film.projection @ np.asarray(marker.point),
                    digits,
                ).tolist(),
            }
            for marker in film.markers
        ],
        "seed": mesh.seed,
        "residual": float(mesh.residual),
        "stopped": mesh.stopped,
        "lost": int(mesh.lost),
        "warnings": list(mesh.warnings),
    }


def _mesh_payload(mesh: GeodesicMesh, vertices: np.ndarray, digits: int) -> dict:
    """The mesh as the page draws it, in about a quarter of the JSON.

    Nothing drawn is dropped or moved. Vertices are integer steps from the
    vertex before, in units of the last digit kept: neighbours on a ring are a
    few hundred units apart, so a step is three or four characters where a
    coordinate was six or seven, and the page's running sum gives back the
    rounded coordinates exactly. Rings are vertex ranges appended in order, so
    their sizes say which vertices they hold. Each band between two rings is
    the zip of :func:`~.geodesic_level_sets._zip`, so it is one character per
    triangle - whether the zip advanced on the inner ring or the outer - where
    it was three indices. The per-vertex distances are the rings' and were
    never read by the page. A mesh not built that way keeps its rings and
    triangles as they are.

    Lorenz to r = 100 (10,289 vertices, 20,154 triangles): 746 kB of JSON
    became 222 kB (3.4 times smaller), the page 778 kB to 255 kB.
    """
    units = np.round(vertices * 10.0**digits).astype(np.int64)
    payload: dict = {
        "vertexDigits": digits,
        "vertexSteps": np.diff(units, axis=0, prepend=0).ravel().tolist(),
    }
    sizes = [len(ring) for ring in mesh.rings]
    bands = _bands(mesh, sizes)
    if bands is None:
        payload["rings"] = [ring.astype(int).tolist() for ring in mesh.rings]
        payload["triangles"] = mesh.triangles.ravel().astype(int).tolist()
    else:
        payload["ringSizes"] = sizes
        payload["bands"] = bands
    return payload


def _bands(mesh: GeodesicMesh, sizes: list[int]) -> str | None:
    """Every band's zip, a ``0`` or ``1`` per triangle; ``None`` if not zipped.

    ``0`` is a triangle that advanced on the inner ring - ``(inner i, inner
    i + 1, outer j)`` - and ``1`` one that advanced on the outer - ``(inner
    i, outer j + 1, outer j)``. The first ring's fan about the equilibrium
    is implied by its size.
    """
    first = 1
    for ring, size in zip(mesh.rings, sizes, strict=True):
        if not np.array_equal(ring, np.arange(first, first + size)):
            return None
        first += size
    if first != len(mesh.vertices):
        return None
    triangles = [tuple(int(v) for v in t) for t in mesh.triangles]
    seed = sizes[0]
    if triangles[:seed] != [(0, 1 + (i + 1) % seed, 1 + i) for i in range(seed)]:
        return None
    bits: list[str] = []
    position, inner = seed, 1
    for old, new in pairwise(sizes):
        outer = inner + old
        i = j = 0
        for _ in range(old + new):
            if position >= len(triangles):
                return None
            here = (inner + i % old, outer + j % new)
            if triangles[position] == (here[0], inner + (i + 1) % old, here[1]):
                bits.append("0")
                i += 1
            elif triangles[position] == (here[0], outer + (j + 1) % new, here[1]):
                bits.append("1")
                j += 1
            else:
                return None
            position += 1
        inner = outer
    return "".join(bits) if position == len(triangles) else None


def _round(row: CrochetRound) -> list[float]:
    return [
        row.number,
        round(row.distance, 6),
        row.stitches,
        row.increases,
        row.decreases,
    ]
