# CONTINUATION

## Overview

The continuation subsystem traces a branch of equilibria of a differential
equation as one of its parameters varies, and detects codimension-one
bifurcations along the way. It uses **pseudo-arclength continuation** (Keller),
which follows the solution curve by arclength rather than by the parameter, so it
turns smoothly through folds where the parameter itself is not a valid coordinate.

It reuses the existing domain model: the object being continued is an ordinary
`DifferentialEquation`, and the continuation parameter is one of its own
`Parameter` objects, selected by index. Equilibria are the zeros of the
derivative function, `f(u, lambda) = 0`.

## Architecture

The driver owns no numerical algorithm. Each capability is a small class behind
an abstraction, and a composition root injects them into the driver. Adding a new
predictor, corrector, or bifurcation detector means adding a class, not editing
the loop.

```
ContinuerBuilder.build(config, equation)   # composition root (wiring)
        │  builds and injects
        ▼
Continuer(Solver)                            # thin orchestrator
        │  depends only on abstractions, bundled in
        ▼
ContinuationComponents
├── ResidualFunction        (EquationResidual)          # sole domain coupling
├── JacobianProvider        (FiniteDifferenceJacobian)
├── LinearSolver            (LeastSquaresFallbackSolver)
├── SeedRefiner             (NewtonSeedRefiner)
├── Predictor               (TangentPredictor)
├── Corrector               (NewtonCorrector)
│     └── Parameterization  (PseudoArclength | Natural)   # injected constraint
├── TangentComputer         (BorderedTangentComputer)
├── StepController          (AdaptiveStepController)
├── StabilityAnalyzer       (EigenvalueStabilityAnalyzer)
├── Measure                 (NormMeasure | ComponentMeasure)
├── BifurcationDetector[]   (FoldDetector, BranchPointDetector, HopfDetector)
├── Localizer               (RefiningLocalizer)
│     └── ScalarRootFinder  (Bisection | Secant)          # injected root-find
├── ContinuationPointBuilder
└── BranchSwitcher          (NullSpaceBranchSwitcher)
```

Note the nested injections: the corrector does not know *which* continuation
constraint it enforces (that is a `Parameterization`), and the localizer does not
know *how* to root-find the test function (that is a `ScalarRootFinder`). Both are
families of interchangeable solutions, so they are injected, not hardcoded.

### Where injection stops

A capability becomes an injection point when there is a real, named family of
alternatives a user would swap: parameterization, corrector, predictor, detector,
root-finder, measure, step control. It stays concrete when the choice *is* the
class's identity - `FoldDetector`'s test function is what makes it the fold
detector, and `NormMeasure` computing a norm is not a swappable sub-decision. The
rule avoids infinite regress: inject alternative algorithms, not the arithmetic
inside a leaf.

Every arrow is an abstract base class (`abc.ABC`); the names in parentheses are
the concrete implementations the factory selects from the config. Because the
driver depends on the interfaces, any of them can be replaced — in tests a custom
`BifurcationDetector` is injected via `components.model_copy(update=...)` with no
change to `Continuer`.

## SOLID notes

- **Single responsibility**: each class does one thing (predict, correct, detect,
  classify, measure, locate, switch, build a record).
- **Open/closed**: new bifurcation types or predictors are new classes; the loop
  is untouched. `ContinuationPoint.kind` is an open string for the same reason.
- **Liskov**: `Continuer` is a `Solver`; every concrete strategy is substitutable
  for its abstraction.
- **Interface segregation**: abstractions are narrow (`Predictor.predict`,
  `Corrector.correct`, `Measure.of`), so implementations carry no dead methods.
- **Dependency inversion**: the driver depends on abstractions; `ContinuerBuilder`
  is the only place that names concretions.

## Executive Summary

**Purpose**: Equilibrium continuation and codim-1 bifurcation detection
**Key Features**: Fold-tracking corrector, configurable detectors, finite-difference Jacobian, per-point stability
**Method**: Pseudo-arclength predictor-corrector with bordered Newton
**Use Cases**: Bifurcation diagrams, stability boundaries, normal-form validation

## Core Classes

### ContinuationConfig

```python
class ContinuationConfig(SolverConfig):
    """Configuration for the Continuer (defaults give a working run)."""

    continuation_parameter_index: int = 0
    initial_parameter: float = 0.0
    corrector: Literal["pseudo_arclength", "natural"] = "pseudo_arclength"
    jacobian: Literal["finite_difference"] = "finite_difference"
    initial_step: float = 0.05
    direction: Literal[1, -1] = 1
    detect_folds: bool = True
    detect_branch_points: bool = True
    detect_hopf: bool = True
    measure: Literal["norm", "component"] = "norm"
```

### Continuer

```python
class Continuer(Solver):
    """Thin orchestrator; collaborators are injected via ContinuationComponents."""

    def __init__(
        self,
        solver_config: ContinuationConfig,
        components: ContinuationComponents,
    ) -> None: ...

    def solve(
        self,
        equation: DifferentialEquation,
        initial_values: list[float],
    ) -> Branch: ...
```

Build one with the factory rather than wiring by hand:

```python
continuer = ContinuerBuilder.build(config, equation)
branch = continuer.solve(equation, initial_values)
```

## Method

The zero set `f(u, lambda) = 0` with `u` in R^n and `lambda` in R is a curve in
R^(n+1). Continuation follows it by:

1. **Predict** along the unit tangent `t` of the curve: `w_pred = w + ds * t`.
2. **Correct** back onto the curve with Newton on the bordered system

   ```
   f(u, lambda)              = 0        (n equations)
   t . (w - w_pred)          = 0        (arclength constraint)
   ```

   whose Jacobian is `[[f_u | f_lambda]; t^T]`, an (n+1) x (n+1) system.
3. **Update** the tangent as the oriented null vector of `[f_u | f_lambda]`.

### Detection

Between accepted points, sign changes of scalar test functions bracket a
bifurcation, which is then localized by bisection:

- **Fold**: the parameter component of the tangent, `t[-1]`, passes through zero.
- **Branch point**: the determinant of the bordered extended Jacobian changes
  sign.
- **Hopf**: the determinant of the bialternate product `2A (.) I` of the state
  Jacobian `A` changes sign, which happens exactly when `A` has a pair of
  eigenvalues summing to zero (a purely imaginary pair).

Stability at each point is read from the eigenvalues of the state Jacobian.

## Requirements on the derivative function

The continued `Function` must read its continuation parameter live from
`self.parameters[index].value` inside `eval` (as `SimpleODEFunction` does) rather
than caching it in `__init__`, because the driver sweeps that parameter's value
between evaluations. `eval` should be free of side effects.

## Branch switching

`Continuer.seed_secondary_branch(branch_point)` returns a seed on the branch that
crosses at a detected branch point, using the second null direction of the
extended Jacobian (with a transverse fallback for pitchforks, where both branches
share a tangent). Continue from that seed to trace the bifurcating branch.

## Example

```python
config = ContinuationConfig(
    initial_parameter=3.0,
    direction=-1,
    measure="component",
    parameter_lower_bound=-0.6,
    parameter_upper_bound=3.2,
)
continuer = ContinuerBuilder.create(config, saddle_node_equation)
branch = continuer.solve(saddle_node_equation, initial_values=[math.sqrt(3.0)])
folds = [p for p in branch.special_points if p.kind == "fold"]
```

See `examples/continuation/bifurcation_examples.py` for a runnable gallery
(saddle-node, transcritical, pitchfork with branch switching, cusp/hysteresis,
planar Hopf, van der Pol, and the Selkov glycolysis model).

## References

- Keller, H. B. "Numerical solution of bifurcation and nonlinear eigenvalue
  problems" (1977).
- Kuznetsov, Y. A. *Elements of Applied Bifurcation Theory* (bialternate product
  and codim-1 normal forms).
- Allgower, E. L. and Georg, K. *Numerical Continuation Methods* (1990).

## Builder vs. factory, and how strategies are resolved

`ContinuerBuilder` is a **builder**: it wires the infrastructure (residual,
Jacobian, linear solver, settings) and assembles the component bundle. It is not
a factory - it does not create instances by key from a registry.

The interchangeable strategies (parameterization, root-finder, detectors,
measure) are resolved through `sweet_tea`'s `AbstractFactory[T].create(key, ...)`,
which looks each one up by key in the class registry. Registration is automatic:
the package is scanned once by `sweet_tea`'s `fill_registry` (invoked from the
distribution's root package init), so concrete strategies are registered under
their class name with no manual `register()` calls. Adding a new strategy is just
writing the class (a subclass of the relevant abstract base) and referencing its
key from the config - the builder and registry are untouched.
