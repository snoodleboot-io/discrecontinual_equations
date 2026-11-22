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
    """The film as plain JSON: flat arrays, rounded, rings by index."""
    mesh = film.mesh
    vertices = mesh.vertices @ film.projection.T
    scale = float(np.max(np.abs(vertices))) or 1.0
    digits = max(0, _DIGITS - int(np.floor(np.log10(scale))))
    return {
        "title": film.title,
        "axes": list(film.axes),
        "vertices": np.round(vertices, digits).ravel().tolist(),
        "triangles": mesh.triangles.ravel().astype(int).tolist(),
        "distances": np.round(mesh.distances, 6).tolist(),
        "rings": [ring.astype(int).tolist() for ring in mesh.rings],
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
    }


def _round(row: CrochetRound) -> list[float]:
    return [
        row.number,
        round(row.distance, 6),
        row.stitches,
        row.increases,
        row.decreases,
    ]
