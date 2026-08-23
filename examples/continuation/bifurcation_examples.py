"""Runnable gallery of continuation and bifurcation examples.

Each example builds an ordinary ``DifferentialEquation`` whose derivative reads
its continuation parameter live, wires a continuer with ``ContinuerBuilder``, and
prints the bifurcations detected along the branch. Run with::

    python -m examples.continuation.bifurcation_examples
"""

from pathlib import Path

from sweet_tea.registry import Registry

import discrecontinual_equations
from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class Mu(Parameter, name="Continuation parameter", abbreviation="mu"):
    pass


class State(Variable, name="State", abbreviation="x"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


class SaddleNode(DeterministicFunction):
    """x' = mu - x^2."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return [self.parameters[0].value - point[0] * point[0]]


class Transcritical(DeterministicFunction):
    """x' = mu x - x^2."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x = point[0]
        return [mu * x - x * x]


class Pitchfork(DeterministicFunction):
    """x' = mu x - x^3."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x = point[0]
        return [mu * x - x * x * x]


class Cusp(DeterministicFunction):
    """x' = mu + 3x - x^3 (hysteresis S-curve, two folds)."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x = point[0]
        return [mu + 3.0 * x - x * x * x]


class Hopf(DeterministicFunction):
    """Planar Andronov-Hopf normal form; Hopf at mu = 0."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        radius_squared = x * x + y * y
        return [mu * x - y - x * radius_squared, x + mu * y - y * radius_squared]


class VanDerPol(DeterministicFunction):
    """Van der Pol oscillator x'' - mu(1 - x^2)x' + x = 0; Hopf at mu = 0."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        return [y, mu * (1.0 - x * x) * y - x]


class Selkov(DeterministicFunction):
    """Selkov glycolysis model with a non-origin equilibrium; Hopf as b varies."""

    _A = 0.08

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        b = self.parameters[0].value
        x, y = point[0], point[1]
        return [-x + self._A * y + x * x * y, b - self._A * y - x * x * y]


def build_equation(
    function_type: type[DeterministicFunction],
    variables: list[Variable],
) -> DifferentialEquation:
    parameter = Mu(value=0.0)
    results = [State() for _ in variables]
    derivative = function_type(
        variables=variables,
        parameters=[parameter],
        results=results,
        time=None,
    )
    return DifferentialEquation(
        variables=variables,
        time=Time(),
        parameters=[parameter],
        derivative=derivative,
    )


def report(name: str, branch: Branch) -> None:
    print(f"\n{name}")
    print(f"  points on branch: {len(branch)}")
    if not branch.special_points:
        print("  no bifurcations detected")
        return
    for point in branch.special_points:
        frequency = (
            "" if point.frequency is None else f", frequency={point.frequency:.3f}"
        )
        print(
            f"  {point.kind:>12s} at mu={point.parameter:+.4f}, "
            f"|x|={point.measure:.4f}{frequency}",
        )


def run_saddle_node() -> None:
    equation = build_equation(SaddleNode, [State()])
    config = ContinuationConfig(
        detectors=["fold"],
        initial_parameter=3.0,
        direction=-1,
        measure="component",
        parameter_lower_bound=-0.6,
        parameter_upper_bound=3.2,
    )
    continuer = ContinuerBuilder.build(config, equation)
    report("Saddle-node  x' = mu - x^2", continuer.solve(equation, [3.0**0.5]))


def run_transcritical() -> None:
    equation = build_equation(Transcritical, [State()])
    config = ContinuationConfig(
        detectors=["branch_point"],
        initial_parameter=-2.0,
        measure="component",
        parameter_lower_bound=-2.0,
        parameter_upper_bound=2.0,
    )
    continuer = ContinuerBuilder.build(config, equation)
    report("Transcritical  x' = mu x - x^2", continuer.solve(equation, [0.0]))


def run_pitchfork() -> None:
    equation = build_equation(Pitchfork, [State()])
    config = ContinuationConfig(
        detectors=["branch_point"],
        initial_parameter=-2.0,
        measure="component",
        parameter_lower_bound=-2.0,
        parameter_upper_bound=2.0,
    )
    continuer = ContinuerBuilder.build(config, equation)
    branch = continuer.solve(equation, [0.0])
    report("Pitchfork  x' = mu x - x^3", branch)
    if branch.special_points:
        seed_state, seed_parameter = continuer.seed_secondary_branch(
            branch.special_points[0],
        )
        print(
            f"  secondary-branch seed: "
            f"x={seed_state[0]:+.4f}, mu={seed_parameter:+.4f}",
        )


def run_cusp() -> None:
    equation = build_equation(Cusp, [State()])
    config = ContinuationConfig(
        detectors=["fold"],
        initial_parameter=-2.0,
        direction=1,
        measure="component",
        initial_step=0.03,
        maximum_step=0.1,
        parameter_lower_bound=-3.0,
        parameter_upper_bound=3.0,
    )
    continuer = ContinuerBuilder.build(config, equation)
    report("Cusp / hysteresis  x' = mu + 3x - x^3", continuer.solve(equation, [-2.0]))


def run_hopf() -> None:
    equation = build_equation(Hopf, [State(), State()])
    config = ContinuationConfig(
        detectors=["hopf"],
        initial_parameter=-1.0,
        measure="norm",
        parameter_lower_bound=-1.0,
        parameter_upper_bound=1.0,
    )
    continuer = ContinuerBuilder.build(config, equation)
    report("Hopf normal form (planar)", continuer.solve(equation, [0.0, 0.0]))


def run_van_der_pol() -> None:
    equation = build_equation(VanDerPol, [State(), State()])
    config = ContinuationConfig(
        detectors=["hopf"],
        initial_parameter=-1.0,
        measure="norm",
        parameter_lower_bound=-1.0,
        parameter_upper_bound=1.0,
    )
    continuer = ContinuerBuilder.build(config, equation)
    report("Van der Pol oscillator", continuer.solve(equation, [0.0, 0.0]))


def run_selkov() -> None:
    equation = build_equation(Selkov, [State(), State()])
    config = ContinuationConfig(
        detectors=["hopf"],
        initial_parameter=0.9,
        direction=-1,
        measure="norm",
        finite_difference_epsilon=1e-6,
        initial_step=0.02,
        maximum_step=0.05,
        parameter_lower_bound=0.1,
        parameter_upper_bound=0.9,
    )
    continuer = ContinuerBuilder.build(config, equation)
    report(
        "Selkov glycolysis (non-origin equilibrium)",
        continuer.solve(equation, [0.9, 1.35]),
    )


def main() -> None:
    Registry.fill_registry(
        path=str(Path(discrecontinual_equations.__file__).parent),
        module="discrecontinual_equations",
        exclude=["*.tests", "*.examples", "*.plot"],
    )
    print("Continuation & bifurcation examples")
    print("=" * 40)
    run_saddle_node()
    run_transcritical()
    run_pitchfork()
    run_cusp()
    run_hopf()
    run_van_der_pol()
    run_selkov()


if __name__ == "__main__":
    main()
