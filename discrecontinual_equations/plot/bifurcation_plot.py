"""Plotly renderer for continuation branches and their bifurcations.

Consumes the render-agnostic
:class:`discrecontinual_equations.continuation.diagram.BifurcationDiagram`, so all
data shaping is tested independently of plotly. Follows the shape of the other
plot classes in this package (an ``output_dir``/``output_format`` constructor and
a ``go.Figure``), but returns saved paths rather than printing so it stays within
the project's lint rules.
"""

from pathlib import Path

import plotly.graph_objects as go

from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.diagram import BifurcationDiagram

_LINE_STYLES = {"stable": "solid", "unstable": "dash", "saddle": "dot"}
_MARKER_COLORS = {"fold": "red", "branch_point": "green", "hopf": "purple"}
_MARKER_SIZE = 10
_DEFAULT_STYLE = "solid"
_DEFAULT_COLOR = "black"


class BifurcationPlot:
    """Draw a bifurcation diagram: measure versus continuation parameter."""

    def __init__(self, output_dir: str = ".", output_format: str = "png") -> None:
        self.output_dir = output_dir
        # A plotter is told where to write; make sure that place exists,
        # so an example whose output directory is not in the tree still runs.
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_format = output_format
        self.figure: go.Figure = go.Figure()

    def plot(self, branch: Branch, title: str = "Bifurcation Diagram") -> go.Figure:
        """Populate the figure from a branch and return it."""
        diagram = BifurcationDiagram(branch)
        self._draw_segments(diagram)
        self._draw_markers(diagram)
        self.figure.update_layout(
            title=title,
            xaxis_title="parameter",
            yaxis_title="measure",
            hovermode="x unified",
        )
        return self.figure

    def save(self, title: str) -> Path:
        """Write the figure to ``output_dir`` and return the path."""
        safe_title = (
            "".join(
                character
                for character in title
                if character.isalnum() or character in " _-"
            )
            .strip()
            .replace(" ", "_")
        )
        if self.output_format == "html":
            path = Path(self.output_dir) / f"{safe_title}.html"
            self.figure.write_html(str(path))
            return path
        path = Path(self.output_dir) / f"{safe_title}.png"
        self.figure.write_image(str(path), engine="kaleido")
        return path

    def _draw_segments(self, diagram: BifurcationDiagram) -> None:
        seen: set[str] = set()
        for segment in diagram.segments():
            style = _LINE_STYLES.get(segment.stability, _DEFAULT_STYLE)
            self.figure.add_trace(
                go.Scatter(
                    x=segment.parameters,
                    y=segment.measures,
                    mode="lines",
                    line={"dash": style, "width": 2},
                    name=segment.stability,
                    legendgroup=segment.stability,
                    showlegend=segment.stability not in seen,
                ),
            )
            seen.add(segment.stability)

    def _draw_markers(self, diagram: BifurcationDiagram) -> None:
        for point in diagram.special_points():
            color = _MARKER_COLORS.get(point.kind, _DEFAULT_COLOR)
            self.figure.add_trace(
                go.Scatter(
                    x=[point.parameter],
                    y=[point.measure],
                    mode="markers",
                    marker={"size": _MARKER_SIZE, "color": color, "symbol": "circle"},
                    name=point.kind,
                ),
            )
