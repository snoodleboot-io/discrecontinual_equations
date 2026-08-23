"""Composition root: a builder that assembles a :class:`~.continuer.Continuer`.

The builder imports abstractions and ``sweet_tea``'s factories only - never a
concrete strategy. Every collaborator is named by a registry key in the
:class:`~.continuation_config.ContinuationConfig` and created dynamically with
``AbstractFactory[Interface].create(key, configuration=...)``. Adding or swapping
an implementation is a config change plus a registered class; this file does not
change.
"""

from typing import Any

from sweet_tea.abstract_factory import AbstractFactory
from sweet_tea.factory import Factory

from discrecontinual_equations.continuation.components import ContinuationComponents
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.continuation.continuer import Continuer
from discrecontinual_equations.continuation.corrector import Corrector
from discrecontinual_equations.continuation.detection import BifurcationDetector
from discrecontinual_equations.continuation.jacobian import JacobianProvider
from discrecontinual_equations.continuation.linear_solver import LinearSolver
from discrecontinual_equations.continuation.localizer import Localizer
from discrecontinual_equations.continuation.measure import Measure
from discrecontinual_equations.continuation.parameterization import Parameterization
from discrecontinual_equations.continuation.predictor import Predictor
from discrecontinual_equations.continuation.residual import ResidualFunction
from discrecontinual_equations.continuation.root_finder import ScalarRootFinder
from discrecontinual_equations.continuation.seed_refiner import SeedRefiner
from discrecontinual_equations.continuation.stability import StabilityAnalyzer
from discrecontinual_equations.continuation.step_controller import StepController
from discrecontinual_equations.continuation.switching import BranchSwitcher
from discrecontinual_equations.continuation.tangent import TangentComputer
from discrecontinual_equations.differential_equation import DifferentialEquation


class ContinuerBuilder:
    """Assemble a wired continuer from a typed configuration and an equation."""

    @staticmethod
    def build(
        config: ContinuationConfig,
        equation: DifferentialEquation,
    ) -> Continuer:
        """Resolve the equation residual by key, then assemble the continuer."""
        residual = AbstractFactory[ResidualFunction].create(
            config.residual,
            configuration={
                "equation": equation,
                "parameter_index": config.continuation_parameter_index,
                "evaluation_time": config.evaluation_time,
            },
        )
        return ContinuerBuilder.assemble(config, residual)

    @staticmethod
    def assemble(
        config: ContinuationConfig,
        residual: ResidualFunction,
    ) -> Continuer:
        """Assemble a continuer around an already-built residual.

        Reused by two-parameter continuation, where ``residual`` is an augmented
        bifurcation-curve system rather than the plain equation residual.
        """
        linear_solver = AbstractFactory[LinearSolver].create(config.linear_solver)
        jacobian = AbstractFactory[JacobianProvider].create(
            config.jacobian,
            configuration={
                "residual": residual,
                "epsilon": config.finite_difference_epsilon,
            },
        )
        corrector = AbstractFactory[Corrector].create(
            config.corrector,
            configuration={
                "residual": residual,
                "jacobian": jacobian,
                "linear_solver": linear_solver,
                "parameterization": AbstractFactory[Parameterization].create(
                    config.parameterization,
                ),
                "settings": Factory.create(
                    "newton_settings",
                    configuration={
                        "tolerance": config.newton_tolerance,
                        "max_iterations": config.maximum_newton_iterations,
                    },
                ),
            },
        )
        tangent_computer = AbstractFactory[TangentComputer].create(
            config.tangent_computer,
            configuration={
                "jacobian": jacobian,
                "linear_solver": linear_solver,
                "direction": config.direction,
            },
        )
        components = ContinuationComponents(
            jacobian=jacobian,
            seed_refiner=AbstractFactory[SeedRefiner].create(
                config.seed_refiner,
                configuration={
                    "residual": residual,
                    "jacobian": jacobian,
                    "linear_solver": linear_solver,
                    "iterations": config.seed_polish_iterations,
                    "tolerance": config.newton_tolerance,
                },
            ),
            predictor=AbstractFactory[Predictor].create(config.predictor),
            corrector=corrector,
            tangent_computer=tangent_computer,
            step_controller=ContinuerBuilder._step_controller(config),
            detectors=ContinuerBuilder._detectors(config),
            localizer=AbstractFactory[Localizer].create(
                config.localizer,
                configuration={
                    "corrector": corrector,
                    "tangent_computer": tangent_computer,
                    "jacobian": jacobian,
                    "root_finder": ContinuerBuilder._root_finder(config),
                },
            ),
            point_builder=Factory.create(
                config.point_builder,
                configuration={
                    "jacobian": jacobian,
                    "stability_analyzer": AbstractFactory[StabilityAnalyzer].create(
                        config.stability_analyzer,
                        configuration={"tolerance": config.stability_tolerance},
                    ),
                    "measure": ContinuerBuilder._measure(config),
                },
            ),
            branch_switcher=AbstractFactory[BranchSwitcher].create(
                config.branch_switcher,
                configuration={
                    "jacobian": jacobian,
                    "kick": config.initial_step,
                    "singular_tolerance": config.singular_direction_tolerance,
                },
            ),
        )
        return Continuer(solver_config=config, components=components)

    @staticmethod
    def _root_finder(config: ContinuationConfig) -> ScalarRootFinder:
        return AbstractFactory[ScalarRootFinder].create(
            config.root_finder,
            configuration={
                "settings": Factory.create(
                    "root_finder_settings",
                    configuration={
                        "value_tolerance": config.localization_tolerance,
                        "fraction_tolerance": config.fraction_tolerance,
                        "max_iterations": config.maximum_localization_iterations,
                    },
                ),
            },
        )

    @staticmethod
    def _detectors(config: ContinuationConfig) -> list[BifurcationDetector]:
        return [
            AbstractFactory[BifurcationDetector].create(key) for key in config.detectors
        ]

    @staticmethod
    def _measure(config: ContinuationConfig) -> Measure:
        configuration: dict[str, Any] = (
            {"index": config.measure_index} if config.measure == "component" else {}
        )
        return AbstractFactory[Measure].create(
            config.measure,
            configuration=configuration,
        )

    @staticmethod
    def _step_controller(config: ContinuationConfig) -> StepController:
        return AbstractFactory[StepController].create(
            config.step_controller,
            configuration={
                "target": config.target_newton_iterations,
                "growth": config.step_growth,
                "shrink": config.step_shrink,
                "bounds": (config.minimum_step, config.maximum_step),
                "slack": config.newton_iteration_slack,
            },
        )
