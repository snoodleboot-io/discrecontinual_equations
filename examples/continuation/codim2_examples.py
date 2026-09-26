"""Two-parameter continuation examples: Bogdanov-Takens, zero-Hopf, Hopf-Hopf.

Run with ``python -m examples.continuation.codim2_examples``. Each system has an
analytically known codim-2 point; the script continues the relevant curve from a
codim-1 seed and prints the detected location.
"""

from discrecontinual_equations.continuation.codim2_builder import (
    Codim2Driver,
    CurveSeed,
)
from discrecontinual_equations.continuation.codim2_config import Codim2Config
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

try:  # python -m examples.continuation.codim2_examples
    from examples.continuation.component_registry import ensure_registry
except ImportError:  # run as a script path: only this directory is on sys.path
    from component_registry import ensure_registry


class First(Parameter, name="First parameter", abbreviation="p1"):
    pass


class Second(Parameter, name="Second parameter", abbreviation="p2"):
    pass


class State(Variable, name="State", abbreviation="s"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


def _variables(count: int) -> list[Variable]:
    return [State() for _ in range(count)]


def _equation(
    function_type: type[DeterministicFunction],
    count: int,
    first: float,
    second: float,
) -> DifferentialEquation:
    parameters = [First(value=first), Second(value=second)]
    derivative = function_type(
        variables=_variables(count),
        parameters=parameters,
        results=_variables(count),
        time=None,
    )
    return DifferentialEquation(
        variables=_variables(count),
        time=Time(),
        parameters=parameters,
        derivative=derivative,
    )


class BogdanovTakensNormalForm(DeterministicFunction):
    """x' = y, y' = b1 + b2 y + x^2 - x y; Bogdanov-Takens at (0, 0)."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        x, y = point[0], point[1]
        return [y, b1 + b2 * y + x * x - x * y]


class ZeroHopfSystem(DeterministicFunction):
    """Fold in x coupled to a Hopf in (y, z); zero-Hopf at mu1 = 0."""

    frequency = 1.0

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu1 = self.parameters[0].value
        mu2 = self.parameters[1].value
        x, y, z = point[0], point[1], point[2]
        radius = y * y + z * z
        return [
            mu1 - x * x,
            mu2 * y - self.frequency * z - y * radius,
            self.frequency * y + mu2 * z - z * radius,
        ]


class HopfHopfSystem(DeterministicFunction):
    """Two decoupled Hopf blocks; Hopf-Hopf at mu2 = 0."""

    first_frequency = 1.0
    second_frequency = 2.0

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu1 = self.parameters[0].value
        mu2 = self.parameters[1].value
        x, y, z, w = point[0], point[1], point[2], point[3]
        first = x * x + y * y
        second = z * z + w * w
        return [
            mu1 * x - self.first_frequency * y - x * first,
            self.first_frequency * x + mu1 * y - y * first,
            mu2 * z - self.second_frequency * w - z * second,
            self.second_frequency * z + mu2 * w - w * second,
        ]


def _report(title: str, result: object, expectation: str) -> None:
    print(title)
    for point in result.points:
        frequency = "" if point.omega is None else f", omega={point.omega:+.4f}"
        print(
            f"  {point.kind} at "
            f"(p1={point.parameter_a:+.4f}, p2={point.parameter_b:+.4f})"
            f"{frequency}",
        )
    print(f"  expected: {expectation}\n")


def main() -> None:
    ensure_registry()
    print("Codim-2 bifurcation examples\n")

    equation = _equation(BogdanovTakensNormalForm, 2, first=-0.25, second=-0.5)
    seed = CurveSeed(state=[-0.5, 0.0], parameter_a=-0.25, frequency=1.0)
    config = Codim2Config(
        continuation_parameter_index=0,
        second_parameter_index=1,
        curve="hopf",
        codim2_detectors=["bogdanov_takens"],
        initial_parameter=-0.5,
        direction=1,
        parameter_lower_bound=-0.6,
        parameter_upper_bound=0.05,
    )
    _report(
        "Bogdanov-Takens (Hopf curve of the BT normal form)",
        Codim2Driver.run(config, equation, seed),
        "(0, 0)",
    )

    equation = _equation(ZeroHopfSystem, 3, first=0.5, second=0.0)
    seed = CurveSeed(state=[0.5**0.5, 0.0, 0.0], parameter_a=0.0, frequency=1.0)
    config = Codim2Config(
        continuation_parameter_index=1,
        second_parameter_index=0,
        curve="hopf",
        codim2_detectors=["zero_hopf"],
        initial_parameter=0.5,
        direction=-1,
        parameter_lower_bound=-0.1,
        parameter_upper_bound=0.6,
    )
    _report(
        "Zero-Hopf (fold-Hopf, decoupled system)",
        Codim2Driver.run(config, equation, seed),
        "mu1 = 0",
    )

    equation = _equation(HopfHopfSystem, 4, first=0.0, second=0.5)
    seed = CurveSeed(state=[0.0, 0.0, 0.0, 0.0], parameter_a=0.0, frequency=1.0)
    config = Codim2Config(
        continuation_parameter_index=0,
        second_parameter_index=1,
        curve="hopf",
        codim2_detectors=["hopf_hopf"],
        initial_parameter=0.5,
        direction=-1,
        parameter_lower_bound=-0.3,
        parameter_upper_bound=0.6,
    )
    _report(
        "Hopf-Hopf (two decoupled Hopf blocks)",
        Codim2Driver.run(config, equation, seed),
        "mu2 = 0",
    )


if __name__ == "__main__":
    main()
