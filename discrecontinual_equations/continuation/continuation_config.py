"""Configuration for numerical continuation of equilibria.

``ContinuationConfig`` follows the same pattern as the solver configurations in
:mod:`discrecontinual_equations.solver.solver_config`: it is a pydantic model
whose fields carry sensible defaults, so a bare ``ContinuationConfig()`` traces a
robust equilibrium branch with pseudo-arclength continuation and codim-1
bifurcation detection. Every algorithmic choice is a field and can be overridden.

The defaults assemble the classical method: Newton seed polish at
``initial_parameter``; the tangent from the bordered system ``[F_y; v_prev^T] v =
e_(n+1)``; an Euler tangent predictor of arclength ``initial_step``; a Newton
corrector on ``F = 0`` plus Keller's pseudo-arclength constraint (the hyperplane
through the prediction normal to the tangent); step control by corrector iteration
count; fold, branch-point and Hopf test functions checked for sign changes and
located by bisection along the chord (Kuznetsov (1998), Sections 10.2.1-10.2.2;
Keller (1977), who introduced pseudo-arclength continuation). ``parameterization
= "natural"`` fixes the parameter instead (Kuznetsov's (10.33) with the parameter
as the fixed coordinate), which fails at every fold.

The Jacobian is by central differences, unless the residual supplies its own,
with a *relative* step: component ``j`` steps ``finite_difference_epsilon
max(|x_j|, typical_j)`` and the parameter ``finite_difference_epsilon
max(|lambda|, parameter_typical)`` (Dennis and Schnabel 1983, Sec. 5.4). The
typical sizes are one by default, so a component of size ``1e8`` is stepped in
proportion to itself, but one that lives near ``1e-8`` is stepped as if it were
of size one unless ``typical`` says otherwise (DEQ-158).

References:
- Kuznetsov, Yu. A. "Elements of Applied Bifurcation Theory", 2nd ed., Applied
  Mathematical Sciences 112, Springer, 1998, doi:10.1007/b98848, Sections
  10.2.1-10.2.2, eqs. (10.29)-(10.36).
- Keller, H. B. "Numerical solution of bifurcation and nonlinear eigenvalue
  problems", in P. H. Rabinowitz (ed.), Applications of Bifurcation Theory,
  Academic Press, 1977, pp. 359-384.
- Allgower, E. L. and Georg, K. "Numerical Continuation Methods: An Introduction",
  Springer, 1990, doi:10.1007/978-3-642-61257-2, Chapters 2, 3 and 6.
- Dennis, J. E. and Schnabel, R. B. "Numerical Methods for Unconstrained
  Optimization and Nonlinear Equations", Prentice-Hall, 1983, Sec. 5.4 and
  Algorithm A5.4.1: the step ``r max(|x_j|, typx_j)``.
"""

from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field, field_validator

from discrecontinual_equations.finite_difference import typical_sizes
from discrecontinual_equations.solver.solver_config import SolverConfig


class ContinuationConfig(SolverConfig):
    """Configuration object for :class:`~.continuer.Continuer`.

    The continuation parameter is one of the equation's own ``Parameter`` objects,
    identified by ``continuation_parameter_index``. Its value is swept while the
    equilibrium condition ``f(u, lambda) = 0`` is tracked as a solution curve.

    Every field is documented by its ``description``; registry keys name the
    strategy classes registered with ``sweet_tea`` (for example
    ``"newton_corrector"`` for :class:`~.corrector.NewtonCorrector`).

    References:
    - Kuznetsov (1998), Section 10.2; Keller (1977).
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
        description=(
            "Relative step of the central finite-difference Jacobian: state "
            "component j steps this times max(|x_j|, typical_j) and the "
            "parameter this times max(|lambda|, parameter_typical), both typical "
            "sizes one by default, so on a component of size one or less the "
            "step is this value exactly"
        ),
    )
    typical: float | list[float] | None = Field(
        default=None,
        description=(
            "Typical size of each state component (one per component, or one "
            "for all) for the finite-difference Jacobian's step "
            "finite_difference_epsilon * max(|x_j|, typical_j); None for one. "
            "Set it to the scale of a component that lives far below one, "
            "near zero or in small units, or the step swamps it"
        ),
    )
    parameter_typical: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Typical size of the continuation parameter for the parameter "
            "column's step finite_difference_epsilon * max(|lambda|, "
            "parameter_typical); None for one"
        ),
    )
    jacobian_sparsity: Literal["dense", "detect"] | list[list[int]] = Field(
        default="dense",
        description=(
            "Sparsity of the state Jacobian: 'dense' differences every column; "
            "'detect' finds the non-zero pattern once, at the first point; a list "
            "gives, for each equation, the state indices it depends on. A known "
            "pattern lets the provider difference structurally orthogonal columns "
            "together (Curtis-Powell-Reid), so a ring costs 2-3 grouped passes "
            "instead of n columns, with bit-identical entries"
        ),
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

    arclength_state_scale: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "State size that one unit of pseudo-arclength stands for: the state "
            "enters the arclength divided by it, as AUTO's weights THU do, so a "
            "system whose state is measured in other units is continued alike "
            "once this is rescaled with it; 1 is the unweighted arclength"
        ),
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
        description=(
            "Test-function tolerance when localizing a bifurcation, as a fraction "
            "of the larger test value at the two bracketing points, so the "
            "located point does not depend on the units of time or state"
        ),
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

    @field_validator("typical")
    @classmethod
    def _positive_typical(
        cls,
        value: float | list[float] | None,
    ) -> float | list[float] | None:
        """Refuse a typical size that is not positive."""
        if value is not None:
            sizes = np.atleast_1d(np.asarray(value, dtype=float))
            typical_sizes(sizes, sizes.size)
        return value
