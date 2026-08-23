"""Configuration for numerical continuation of equilibria.

``ContinuationConfig`` follows the same pattern as the solver configurations in
:mod:`discrecontinual_equations.solver.solver_config`: it is a pydantic model
whose fields carry sensible defaults, so a bare ``ContinuationConfig()`` traces a
robust equilibrium branch with pseudo-arclength continuation and codim-1
bifurcation detection. Every algorithmic choice is a field and can be overridden.
"""

from typing import Literal

from pydantic import ConfigDict, Field

from discrecontinual_equations.solver.solver_config import SolverConfig


class ContinuationConfig(SolverConfig):
    """Configuration object for :class:`~.continuer.Continuer`.

    The continuation parameter is one of the equation's own ``Parameter`` objects,
    identified by ``continuation_parameter_index``. Its value is swept while the
    equilibrium condition ``f(u, lambda) = 0`` is tracked as a solution curve.
    """

    continuation_parameter_index: int = Field(
        default=0,
        description="Index into the equation's parameter list of the parameter to vary",
    )
    initial_parameter: float = Field(
        default=0.0,
        description="Starting value of the continuation parameter",
    )
    evaluation_time: float = Field(
        default=0.0,
        description="Time passed to the derivative (autonomous systems ignore it)",
    )

    corrector: str = Field(
        default="newton_corrector",
        description="Registry key of the corrector implementation",
    )
    parameterization: Literal["pseudo_arclength", "natural"] = Field(
        default="pseudo_arclength",
        description="Registry key of the corrector constraint",
    )
    predictor: str = Field(
        default="tangent_predictor",
        description="Registry key of the predictor implementation",
    )
    jacobian: str = Field(
        default="finite_difference_jacobian",
        description="Registry key of the Jacobian provider",
    )
    residual: str = Field(
        default="equation_residual",
        description="Registry key of the residual adapter",
    )
    linear_solver: str = Field(
        default="least_squares_fallback_solver",
        description="Registry key of the linear solver",
    )
    seed_refiner: str = Field(
        default="newton_seed_refiner",
        description="Registry key of the seed refiner",
    )
    tangent_computer: str = Field(
        default="bordered_tangent_computer",
        description="Registry key of the tangent computer",
    )
    step_controller: str = Field(
        default="adaptive_step_controller",
        description="Registry key of the step controller",
    )
    stability_analyzer: str = Field(
        default="eigenvalue_stability_analyzer",
        description="Registry key of the stability analyzer",
    )
    localizer: str = Field(
        default="refining_localizer",
        description="Registry key of the localizer",
    )
    branch_switcher: str = Field(
        default="null_space_branch_switcher",
        description="Registry key of the branch switcher",
    )
    point_builder: str = Field(
        default="continuation_point_builder",
        description="Registry key of the point builder",
    )
    root_finder: str = Field(
        default="bisection",
        description="Registry key of the scalar root-finder for localization",
    )
    finite_difference_epsilon: float = Field(
        default=1e-7,
        description="Step used for central finite-difference derivatives",
    )

    initial_step: float = Field(
        default=0.05,
        description="Initial arclength step magnitude",
    )
    minimum_step: float = Field(
        default=1e-4,
        description="Smallest arclength step before continuation stops",
    )
    maximum_step: float = Field(
        default=0.25,
        description="Largest permitted arclength step",
    )
    maximum_points: int = Field(
        default=4000,
        description="Maximum number of continuation points to compute",
    )
    direction: Literal[1, -1] = Field(
        default=1,
        description="Initial direction of travel along the branch",
    )
    parameter_lower_bound: float | None = Field(
        default=None,
        description="Continuation stops if the parameter drops below this value",
    )
    parameter_upper_bound: float | None = Field(
        default=None,
        description="Continuation stops if the parameter rises above this value",
    )
    target_newton_iterations: int = Field(
        default=3,
        description="Target corrector iteration count for step-size adaption",
    )
    step_growth: float = Field(
        default=1.3,
        description="Factor the step grows by when the corrector converges quickly",
    )
    step_shrink: float = Field(
        default=0.6,
        description="Factor the step shrinks by when the corrector is slow",
    )
    newton_iteration_slack: int = Field(
        default=2,
        description="Iterations above target before the step is shrunk",
    )
    seed_polish_iterations: int = Field(
        default=50,
        description="Maximum fixed-parameter Newton iterations for the seed",
    )
    fraction_tolerance: float = Field(
        default=1e-12,
        description="Bracket-width tolerance for the localizer root-finder",
    )
    singular_direction_tolerance: float = Field(
        default=1e-6,
        description="Threshold below which a branch-switch direction is degenerate",
    )
    stability_tolerance: float = Field(
        default=1e-9,
        description="Real-part threshold for counting unstable eigenvalues",
    )

    newton_tolerance: float = Field(
        default=1e-9,
        description="Residual norm tolerance for the corrector",
    )
    maximum_newton_iterations: int = Field(
        default=12,
        description="Maximum corrector iterations per step",
    )

    detectors: list[str] = Field(
        default_factory=lambda: ["fold", "branch_point", "hopf"],
        description="Registry keys of the bifurcation detectors to run",
    )
    localization_tolerance: float = Field(
        default=1e-8,
        description="Test-function tolerance when localizing a bifurcation",
    )
    maximum_localization_iterations: int = Field(
        default=40,
        description="Maximum bisection iterations when localizing a bifurcation",
    )

    measure: Literal["norm", "component"] = Field(
        default="norm",
        description="Scalar plotted against the parameter on a diagram",
    )
    measure_index: int = Field(
        default=0,
        description="State component used when measure='component'",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
