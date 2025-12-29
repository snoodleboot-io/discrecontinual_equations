from unittest import TestCase
from unittest.mock import patch

from discrecontinual_equations.function.function import Function
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.plot.line_plot import LinePlot
from discrecontinual_equations.variable import Variable


class TestPlot(TestCase):
    @patch("plotly.graph_objects.Figure.write_image")
    def test_plot(self, mock_write):
        class X(Variable, name="Displacement", abbreviation="x"):
            pass

        class Time(Variable, name="Time", abbreviation="t"):
            pass

        class A(Parameter, name="Amplitude", abbreviation="a"):
            pass

        class B(Parameter, name="Offset", abbreviation="b"):
            pass

        class Oscillator(Function):
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
                result = -self.a * x**3 + self.b * x
                self._curve.append([point, [result]])
                return [result]

        x_axis = X()
        params = [A(value=0.1), B(value=0.023)]

        oscillator = Oscillator(variables=[x_axis], parameters=params, time=None)

        [oscillator.eval(point=[value / 1000.0]) for value in range(-1001, 1001)]
        plot = LinePlot()
        plot.plot(
            x=oscillator.curve.variables[0],
            y=oscillator.curve.results[0],
            title="Oscillator",
        )
        print(oscillator.curve)
