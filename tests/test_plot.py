import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pytest

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.function.function import Function
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.plot.bifurcation_plot import BifurcationPlot
from discrecontinual_equations.plot.delay_plot import DelayPlot
from discrecontinual_equations.plot.line_plot import LinePlot
from discrecontinual_equations.plot.phase_plot import PhasePlot
from discrecontinual_equations.plot.spectrum_plot import SpectrumPlot
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


def _line() -> tuple[Variable, Variable]:
    class X(Variable, name="Position", abbreviation="x"):
        pass

    class Y(Variable, name="Height", abbreviation="y"):
        pass

    x, y = X(), Y()
    x.discretization = [0.0, 1.0, 2.0]
    y.discretization = [0.0, 1.0, 4.0]
    return x, y


def _is_png(path: Path) -> bool:
    return path.is_file() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


class TestLinePlotWrites(TestCase):
    """A LinePlot that says it saved a file has written that file (DEQ-79)."""

    def test_png_output_writes_a_png(self):
        x, y = _line()
        with TemporaryDirectory() as out:
            LinePlot(output_dir=out, output_format="png").plot(x, y, title="Line")
            assert _is_png(Path(out) / "Line.png")

    def test_plot_3d_png_writes_a_png_and_html_writes_html(self):
        x, y = _line()
        with TemporaryDirectory() as out, patch("plotly.graph_objects.Figure.show"):
            plot = LinePlot(output_dir=out)
            plot.plot_3d(x, y, y, title="Png", output_format="png")
            plot.plot_3d(x, y, y, title="Html", output_format="html")
            assert _is_png(Path(out) / "Png.png")
            assert not (Path(out) / "Png.html").exists()
            assert "<html>" in (Path(out) / "Html.html").read_text(encoding="utf-8")
            assert not (Path(out) / "Html.png").exists()

    def test_png_without_kaleido_raises_and_claims_nothing(self):
        x, y = _line()
        with (
            TemporaryDirectory() as out,
            patch.dict(sys.modules, {"kaleido": None}),
            patch("builtins.print") as printed,
        ):
            plot = LinePlot(output_dir=out, output_format="png")
            for draw in (
                lambda: plot.plot(x, y, title="Line"),
                lambda: plot.plot_curves(Curve([x], [y], None), title="Line"),
                lambda: plot.plot_3d(x, y, y, title="Line", output_format="png"),
                lambda: plot.plot_3d(x, y, y, title="Line", output_format="both"),
            ):
                with pytest.raises(ImportError, match=r"kaleido.*html"):
                    draw()
            printed.assert_not_called()
            assert list(Path(out).iterdir()) == []

    def test_an_unknown_format_is_refused(self):
        x, y = _line()
        with TemporaryDirectory() as out:
            with pytest.raises(ValueError, match="svg"):
                LinePlot(output_dir=out, output_format="svg").plot(x, y)
            with pytest.raises(ValueError, match="svg"):
                LinePlot(output_dir=out).plot_3d(x, y, y, output_format="svg")


class TestOutputDirectory(TestCase):
    """A plotter creates where it was told to write, rather than failing there."""

    def test_every_plotter_creates_a_missing_directory(self):
        plotters = (LinePlot, PhasePlot, DelayPlot, SpectrumPlot, BifurcationPlot)
        for plotter in plotters:
            with TemporaryDirectory() as parent:
                target = Path(parent) / "images" / "nested"
                plotter(output_dir=str(target))
                assert target.is_dir(), plotter.__name__
