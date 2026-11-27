"""Time series and trajectories drawn with Plotly.

:class:`LinePlot` draws one variable against another (a time series is a variable
against time), every result of a :class:`~discrecontinual_equations.curve.Curve`
against its variables, or a trajectory in three dimensions, and writes the figure
as PNG, HTML or both. There is no algorithm here; it is a thin layer over
``plotly.graph_objects``.

Limits: traces accumulate on the one figure across :meth:`LinePlot.plot` and
:meth:`LinePlot.plot_curves` calls (only :meth:`LinePlot.plot_3d` starts afresh),
and PNG output needs ``kaleido``.
"""

import importlib
from pathlib import Path

import plotly.graph_objects as go

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.variable import Variable

_FORMATS = ("png", "html", "both")


def _require_kaleido() -> None:
    """Plotly writes PNG through kaleido; if it is missing, say how to get round it."""
    try:
        importlib.import_module("kaleido")
    except ImportError as error:
        raise ImportError(
            "LinePlot writes PNG through kaleido, which is not installed: "
            "`pip install kaleido`, or use output_format='html'"
        ) from error


class LinePlot:
    """Plot variables against each other and save the figure.

    Args:
        output_dir: Directory the figures are written to; created if missing.
        output_format: ``"png"``, ``"html"`` or ``"both"``, used by
            :meth:`plot_curves` and :meth:`plot`.
    """

    def __init__(self, output_dir: str = ".", output_format: str = "png"):
        self.output_dir = output_dir
        # A plotter is told where to write; make sure that place exists,
        # so an example whose output directory is not in the tree still runs.
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_format = output_format
        self.figure: go.Figure = go.Figure()

    def plot_curves(self, curve: Curve, title: str = "Curve Plot"):
        """Plot every result of ``curve`` against every variable, on one figure.

        Args:
            curve: The curve whose ``results`` are drawn against its ``variables``.
            title: Figure title, also used for the file name.

        Raises:
            ValueError: If ``output_format`` is not ``png``, ``html`` or ``both``.
            ImportError: If PNG output is asked for and kaleido is missing.
        """

        # Plot variables vs results
        for variable in curve.variables:
            for result in curve.results:
                self.figure.add_trace(
                    go.Scatter(
                        x=variable.discretization,
                        y=result.discretization,
                        mode="lines+markers",
                        name=result.name if hasattr(variable, "name") else "Variable",
                        line=dict(width=0.5),
                        # marker=dict(size=4),
                    ),
                )

        self.figure.update_layout(
            title=title,
            xaxis_title="X",
            yaxis_title="Result",
            hovermode="x unified",
        )

        self._save(title, self.output_format)

    def plot(self, x: Variable, y: Variable, title: str = "Curve Plot"):
        """Plot ``y`` against ``x`` and save the figure.

        The data is drawn twice, as a wide line and a thin red one over it.

        Args:
            x: Horizontal variable (time, for a time series).
            y: Vertical variable.
            title: Figure title, also used for the file name.

        Raises:
            ValueError: If ``output_format`` is not ``png``, ``html`` or ``both``.
            ImportError: If PNG output is asked for and kaleido is missing.
        """

        # Plot variables vs results
        self.figure.add_trace(
            go.Scatter(
                x=x.discretization,
                y=y.discretization,
                mode="lines+markers",
                name=y.name if hasattr(y, "name") else "Variable",
                line=dict(width=10),
            ),
        )

        self.figure.add_trace(
            go.Scatter(
                x=x.discretization,
                y=y.discretization,
                mode="lines+markers",
                name=y.name if hasattr(y, "name") else "Variable",
                line=dict(width=0.01, color="red"),
            ),
        )

        self.figure.update_layout(
            title=title,
            xaxis_title="X",
            yaxis_title="Result",
            hovermode="x unified",
        )

        self._save(title, self.output_format)

    def plot_3d(
        self,
        x: Variable,
        y: Variable,
        z: Variable,
        title: str = "3D Plot",
        output_format: str = "both",
    ):
        """Plot a trajectory through three variables on a fresh figure.

        Args:
            x: First coordinate.
            y: Second coordinate.
            z: Third coordinate.
            title: Figure title, also used for the file name.
            output_format: ``"png"``, ``"html"`` or ``"both"``; this method uses
                its own argument, not the constructor's.

        Raises:
            ValueError: If ``output_format`` is not ``png``, ``html`` or ``both``.
            ImportError: If PNG output is asked for and kaleido is missing.
        """
        self.figure = go.Figure()

        self.figure.add_trace(
            go.Scatter3d(
                x=x.discretization,
                y=y.discretization,
                z=z.discretization,
                mode="lines",
                line=dict(width=1),
                name="Trajectory",
            ),
        )

        self.figure.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x.name or "X",
                yaxis_title=y.name or "Y",
                zaxis_title=z.name or "Z",
            ),
        )

        self._save(title, output_format)

    def _save(self, title: str, output_format: str) -> None:
        """Write the figure as ``png``, ``html`` or ``both``, then say where."""
        if output_format not in _FORMATS:
            message = f"output_format must be one of {_FORMATS}, not {output_format!r}"
            raise ValueError(message)
        safe_title = (
            "".join(c for c in title if c.isalnum() or c in " _-")
            .strip()
            .replace(" ", "_")
        )
        written: list[Path] = []
        if output_format in ("png", "both"):
            # Before anything is written, so a missing kaleido leaves no HTML
            # behind from a "both" that did not finish.
            _require_kaleido()
            written.append(Path(self.output_dir) / f"{safe_title}.png")
            self.figure.write_image(str(written[-1]))
        if output_format in ("html", "both"):
            written.append(Path(self.output_dir) / f"{safe_title}.html")
            self.figure.write_html(str(written[-1]))
        print(f"Plot saved as {' and '.join(str(path) for path in written)}")
