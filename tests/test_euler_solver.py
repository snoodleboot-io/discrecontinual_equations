import math
from unittest import TestCase

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.function import Function
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.solver.deterministic.euler.euler_config import (
    EulerConfig,
)
from discrecontinual_equations.solver.deterministic.euler.euler_solver import (
    EulerSolver,
)
from discrecontinual_equations.variable import Variable


class TestEulerSolver(TestCase):
    def test_solve(self):
        class X(Variable, name="Displacement", abbreviation="x"):
            pass

        class Time(Variable, name="Time", abbreviation="t"):
            pass

        class Steepness(Parameter, name="Steepness", abbreviation="a"):
            pass

        class Separation(Parameter, name="Separation", abbreviation="b"):
            pass

        class Amplitude(Parameter, name="Amplitude", abbreviation="A"):
            pass

        class Omega(Parameter, name="Frequency", abbreviation="ω"):
            pass

        class Oscillator(Function):

            __slots__ = ["a", "amplitude", "b", "frequency"]

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
                self.frequency = parameters[2].value * 2.0 * math.pi
                self.amplitude = parameters[3].value

            def eval(
                self, point: list[float], time: float | None = None,
            ) -> list[float]:
                x = point[0]
                result = (
                    -self.a * x**3
                    + self.b * x
                    + self.amplitude * math.sin(self.frequency * time)
                )
                self._curve.append([time, point, [result]])
                return [result]

        x_axis = X()
        t = Time()
        params = [
            Steepness(value=1.0),
            Separation(value=1.0),
            Omega(value=1.0),
            Amplitude(value=0.03),
        ]

        oscillator = Oscillator(variables=[x_axis], parameters=params, time=t)
        ode = DifferentialEquation(derivative=oscillator, time=t, parameters=params, variables=[x_axis])

        dt = 0.01

        config = EulerConfig(step_size=dt, start_time = 0.0, end_time = 1000.0 * dt)


        solver = EulerSolver(solver_config=config)

        solver.solve(equation=ode, initial_values=[0])
        print(1)
        #
        # [
        #     oscillator.eval(point=[value / 1000.0], time=dt * index)
        #     for index, value in zip(range(2002), range(-1001, 1001))
        # ]
        # plot = LinePlot()
        # plot.plot(
        #     x=oscillator.curve.variables[0],
        #     y=oscillator.curve.results[0],
        #     title="Oscillator",
        # )
        #
        # print(oscillator.curve)
