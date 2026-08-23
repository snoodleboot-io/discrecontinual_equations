"""The continuation driver: a thin orchestrator over injected collaborators.

``Continuer`` owns no numerical algorithm itself. It sequences the predictor,
corrector, tangent computer, detectors, localizer and step controller supplied in
its :class:`~.components.ContinuationComponents`, following the
:class:`discrecontinual_equations.solver.solver.Solver` interface.
"""

import numpy as np

from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.components import ContinuationComponents
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.continuation.continuation_point import ContinuationPoint
from discrecontinual_equations.continuation.continuation_state import ContinuationState
from discrecontinual_equations.continuation.detection import DetectionContext
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver import Solver


class Continuer(Solver):
    """Trace equilibria of an equation as one parameter varies."""

    def __init__(
        self,
        solver_config: ContinuationConfig,
        components: ContinuationComponents,
    ) -> None:
        super().__init__(solver_config=solver_config)
        self._components = components

    def solve(
        self,
        equation: DifferentialEquation,
        initial_values: list[float],
    ) -> Branch:
        """Continue equilibria from ``initial_values``; store and return a Branch.

        The problem (equation and continuation parameter) is bound to the injected
        components by the factory; ``equation`` is used here only to validate the
        seed dimension, keeping the :class:`Solver` contract.
        """
        self._check_dimension(equation, initial_values)
        config = self._configuration()
        components = self._components

        state = components.seed_refiner.refine(
            np.asarray(initial_values, dtype=float),
            config.initial_parameter,
        )
        tangent = components.tangent_computer.compute(
            state,
            config.initial_parameter,
            None,
        )
        current = ContinuationState(state, config.initial_parameter, tangent, 0.0)

        branch = Branch()
        branch.append_point(components.point_builder.build(current))
        previous_values = self._detector_values(current)

        step = config.initial_step
        for _ in range(config.maximum_points):
            advanced = self._step(current, step)
            if advanced is None:
                step = components.step_controller.on_failure(step)
                if components.step_controller.is_exhausted(step):
                    break
                continue
            nxt, iterations = advanced
            current_values = self._detector_values(nxt)
            self._detect(branch, current, nxt, previous_values, current_values)
            branch.append_point(components.point_builder.build(nxt))
            current = nxt
            previous_values = current_values
            step = components.step_controller.on_success(step, iterations)
            if self._out_of_bounds(nxt.parameter):
                break

        self.solution = branch
        return branch

    def seed_secondary_branch(
        self,
        branch_point: ContinuationPoint,
    ) -> tuple[np.ndarray, float]:
        """Delegate to the injected branch switcher for a bifurcating-branch seed."""
        return self._components.branch_switcher.secondary_seed(branch_point)

    def _step(
        self,
        current: ContinuationState,
        step: float,
    ) -> tuple[ContinuationState, int] | None:
        components = self._components
        predicted_state, predicted_parameter = components.predictor.predict(
            current,
            step,
        )
        result = components.corrector.correct(
            predicted_state,
            predicted_parameter,
            current.tangent,
        )
        if not result.converged:
            return None
        tangent = components.tangent_computer.compute(
            result.state,
            result.parameter,
            current.tangent,
        )
        nxt = ContinuationState(
            result.state,
            result.parameter,
            tangent,
            current.arclength + step,
        )
        return nxt, result.iterations

    def _detector_values(self, state: ContinuationState) -> dict[str, float]:
        context = DetectionContext(state, self._components.jacobian)
        return {
            detector.kind: detector.test_value(context)
            for detector in self._components.detectors
        }

    def _detect(
        self,
        branch: Branch,
        lower: ContinuationState,
        upper: ContinuationState,
        previous_values: dict[str, float],
        current_values: dict[str, float],
    ) -> None:
        for detector in self._components.detectors:
            earlier = previous_values.get(detector.kind)
            later = current_values.get(detector.kind)
            if earlier is None or later is None:
                continue
            if earlier * later < 0:
                located = self._components.localizer.localize(lower, upper, detector)
                if located is not None:
                    branch.append_special_point(
                        self._components.point_builder.build(
                            located.state,
                            located.kind,
                            located.frequency,
                        ),
                    )

    def _out_of_bounds(self, parameter: float) -> bool:
        config = self._configuration()
        lower = config.parameter_lower_bound
        upper = config.parameter_upper_bound
        below = lower is not None and parameter < lower
        above = upper is not None and parameter > upper
        return bool(below or above)

    def _check_dimension(
        self,
        equation: DifferentialEquation,
        initial_values: list[float],
    ) -> None:
        # Codim-1 seeds match the variable count; two-parameter curve seeds are
        # augmented and larger, so require at least the variable count.
        if len(initial_values) < len(equation.variables):
            message = "initial_values must have at least one entry per variable"
            raise ValueError(message)

    def _configuration(self) -> ContinuationConfig:
        config = self.solver_config
        if not isinstance(config, ContinuationConfig):  # pragma: no cover
            message = "Continuer requires a ContinuationConfig"
            raise TypeError(message)
        return config
