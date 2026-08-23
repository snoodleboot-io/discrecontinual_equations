"""Shape a branch into plottable data without any plotting dependency.

Kept free of plotly so it can be unit-tested and reused by any renderer. A
:class:`BifurcationDiagram` splits a branch into stability-labelled polyline
segments (so a renderer can draw stable and unstable arcs differently) and
exposes the detected bifurcations as markers.
"""

from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.continuation_point import ContinuationPoint


class StabilitySegment:
    """A run of consecutive points sharing one stability label."""

    __slots__ = ["_measures", "_parameters", "_stability"]

    def __init__(
        self,
        stability: str,
        parameters: list[float],
        measures: list[float],
    ) -> None:
        self._stability = stability
        self._parameters = parameters
        self._measures = measures

    @property
    def stability(self) -> str:
        """Shared stability label of the segment."""
        return self._stability

    @property
    def parameters(self) -> list[float]:
        """Parameter values along the segment."""
        return self._parameters

    @property
    def measures(self) -> list[float]:
        """Measure values along the segment."""
        return self._measures


class BifurcationDiagram:
    """Render-agnostic view of a branch as segments plus bifurcation markers."""

    __slots__ = ["_branch"]

    def __init__(self, branch: Branch) -> None:
        self._branch = branch

    def special_points(self) -> list[ContinuationPoint]:
        """Detected bifurcations, to be drawn as markers."""
        return self._branch.special_points

    def segments(self) -> list[StabilitySegment]:
        """Split the branch into stability-labelled polylines.

        Adjacent segments share the transition point so a renderer draws an
        unbroken curve while still switching style at each stability change.
        """
        segments: list[StabilitySegment] = []
        label: str | None = None
        parameters: list[float] = []
        measures: list[float] = []
        for point in self._branch.points:
            if label is None:
                label = point.stability
            if point.stability != label:
                parameters.append(point.parameter)
                measures.append(point.measure)
                segments.append(StabilitySegment(label, parameters, measures))
                label = point.stability
                parameters = []
                measures = []
            parameters.append(point.parameter)
            measures.append(point.measure)
        if label is not None:
            segments.append(StabilitySegment(label, parameters, measures))
        return segments
