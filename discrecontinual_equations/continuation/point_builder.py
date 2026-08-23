"""Assemble finished ``ContinuationPoint`` records from working states."""

from discrecontinual_equations.continuation.continuation_point import ContinuationPoint
from discrecontinual_equations.continuation.continuation_state import ContinuationState
from discrecontinual_equations.continuation.jacobian import JacobianProvider
from discrecontinual_equations.continuation.measure import Measure
from discrecontinual_equations.continuation.stability import StabilityAnalyzer


class ContinuationPointBuilder:
    """Turn a working state into a serializable point with stability and measure.

    Depends only on abstractions: the Jacobian source, the stability analyzer, and
    the measure. It performs no numerical policy of its own.
    """

    __slots__ = ["_jacobian", "_measure", "_stability_analyzer"]

    def __init__(
        self,
        jacobian: JacobianProvider,
        stability_analyzer: StabilityAnalyzer,
        measure: Measure,
    ) -> None:
        self._jacobian = jacobian
        self._stability_analyzer = stability_analyzer
        self._measure = measure

    def build(
        self,
        state: ContinuationState,
        kind: str = "regular",
        frequency: float | None = None,
    ) -> ContinuationPoint:
        """Build a point, defaulting to a regular (non-bifurcation) record."""
        jacobian = self._jacobian.state_jacobian(state.state, state.parameter)
        stability = self._stability_analyzer.analyze(jacobian)
        return ContinuationPoint(
            arclength=state.arclength,
            state=[float(component) for component in state.state],
            parameter=float(state.parameter),
            tangent=[float(component) for component in state.tangent],
            eigenvalues=stability.eigenvalues,
            unstable_dimension=stability.unstable_dimension,
            stability=stability.label,  # type: ignore[arg-type]
            measure=self._measure.of(state.state),
            kind=kind,
            frequency=frequency,
        )
