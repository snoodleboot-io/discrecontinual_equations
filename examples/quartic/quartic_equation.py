#!/usr/bin/env python3
"""
Example: Solving a Quartic Equation Potential

This example demonstrates solving a differential equation with a quartic potential.
The energy function is E(x) = -a * x^4 + b * x^2, representing a double-well potential.
"""

import os

from discrecontinual_equations.function.function import Function
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.plot.line_plot import LinePlot
from discrecontinual_equations.variable import Variable


class X(Variable, name="Displacement", abbreviation="x"):
    pass


class QuarticAmplitude(Parameter, name="Quartic Amplitude", abbreviation="A"):
    """Corresponds to 'a' in the quartic potential E(x) = -a x^4 + b x^2"""


class QuadraticOffset(Parameter, name="Quadratic Offset", abbreviation="B"):
    """Corresponds to 'b' in the quartic potential E(x) = -a x^4 + b x^2"""


class QuarticPotential(Function):
    """A function representing a quartic potential: E(x) = -a * x^4 + b * x^2"""

    __slots__ = ["a", "b"]

    class Energy(Variable, name="Energy", abbreviation="E"):
        pass

    def __init__(
        self,
        variables: list[Variable],
        parameters: list[Parameter],
        time: Variable | None = None,
    ):
        super().__init__(variables, parameters, [self.Energy()], time)
        self.a = parameters[0].value
        self.b = parameters[1].value

    def eval(
        self,
        point: list[float],
        time: float | None = None,
    ) -> list[float]:
        x = point[0]
        # Quartic potential: E(x) = -a * x^4 + b * x^2
        result = -self.a * x**4 + self.b * x**2
        self._curve.append([point, [result]])
        return [result]


def main():
    """Run the quartic equation example."""
    print("Solving Quartic Equation Potential")

    # Create variables and parameters
    x_axis = X()
    params = [QuarticAmplitude(value=0.1), QuadraticOffset(value=0.5)]  # A=0.1, B=0.5

    # Create the quartic potential function
    quartic = QuarticPotential(variables=[x_axis], parameters=params, time=None)

    # Evaluate over a range of x values
    x_values = [value / 1000.0 for value in range(-1500, 1501)]  # -1.5 to 1.5
    print(f"Evaluating potential for {len(x_values)} points...")

    for x_val in x_values:
        quartic.eval(point=[x_val])

    print("Evaluation complete.")
    print(f"Curve has {len(quartic.curve)} points")

    # Plot the results
    plot = LinePlot(output_dir=os.path.join(os.path.dirname(__file__), "images"))
    plot.plot(
        x=quartic.curve.variables[0],
        y=quartic.curve.results[0],
        title="Quartic Potential: E(x) = -0.1*x^4 + 0.5*x^2",
    )

    print("Plot saved as PNG. The quartic potential shows a double-well structure.")


if __name__ == "__main__":
    main()
