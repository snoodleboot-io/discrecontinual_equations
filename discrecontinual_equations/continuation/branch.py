"""Ordered collection of continuation points forming a solution branch."""

from discrecontinual_equations.continuation.continuation_point import ContinuationPoint


class Branch:
    """A solution branch: an ordered list of points plus detected bifurcations.

    Mirrors the container style of :class:`discrecontinual_equations.curve.Curve`
    with ``append`` semantics and index access, but holds
    :class:`~.continuation_point.ContinuationPoint` records rather than raw
    discretization arrays.
    """

    __slots__ = ["_points", "_special_points"]

    def __init__(self) -> None:
        self._points: list[ContinuationPoint] = []
        self._special_points: list[ContinuationPoint] = []

    @property
    def points(self) -> list[ContinuationPoint]:
        """Regular continuation points in the order they were computed."""
        return self._points

    @property
    def special_points(self) -> list[ContinuationPoint]:
        """Detected bifurcation points along the branch."""
        return self._special_points

    def append_point(self, point: ContinuationPoint) -> None:
        """Append a regular continuation point."""
        self._points.append(point)

    def append_special_point(self, point: ContinuationPoint) -> None:
        """Append a detected bifurcation point."""
        self._special_points.append(point)

    def __getitem__(self, index: int) -> ContinuationPoint:
        return self._points[index]

    def __len__(self) -> int:
        return len(self._points)
