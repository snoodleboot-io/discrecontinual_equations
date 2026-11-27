"""Two-dimensional phase portraits drawn with Plotly.

:class:`PhasePlot` draws one or several trajectories in the plane of two state
variables and writes the figure as PNG or HTML. It is a thin layer over
``plotly.graph_objects``, with no algorithm of its own. Traces accumulate on the
one figure across calls.
"""

from pathlib import Path

import plotly.graph_objects as go

from discrecontinual_equations.variable import Variable

_TRAJECTORY_COLORS = [
    "blue",
    "red",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]


class PhasePlot:
    """Plot class for 2D phase space diagrams.

    Args:
        output_dir: Directory the figures are written to; created if missing.
        output_format: ``"png"`` writes a PNG through kaleido; anything else
            writes HTML.
    """

    def __init__(self, output_dir: str = ".", output_format: str = "png") -> None:
        self.output_dir = output_dir
        # A plotter is told where to write; make sure that place exists,
        # so an example whose output directory is not in the tree still runs.
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_format = output_format
        self.figure: go.Figure = go.Figure()

    def plot(self, x: Variable, y: Variable, title: str = "Phase Space Plot") -> str:
        """Plot a single 2D phase-space trajectory and save it.

        Args:
            x: Horizontal state variable.
            y: Vertical state variable.
            title: Figure title, also used for the file name.

        Returns:
            The path of the file written.
        """
        self.figure.add_trace(
            go.Scatter(
                x=x.discretization,
                y=y.discretization,
                mode="lines",
                name="Trajectory",
                line={"color": "blue", "width": 1},
                showlegend=False,
            ),
        )
        self.figure.update_layout(
            title=title,
            xaxis_title=x.name or "X",
            yaxis_title=y.name or "Y",
            hovermode="closest",
            width=800,
            height=600,
        )
        return self._save_plot(title)

    def plot_multiple(
        self,
        trajectories: list[tuple[Variable, Variable]],
        labels: list[str] | None = None,
        title: str = "Phase Space Plot",
    ) -> str:
        """Plot several trajectories in phase space and save the figure.

        Args:
            trajectories: ``(x, y)`` pairs of variables, one per trajectory; the
                first pair's names label the axes.
            labels: Legend labels; missing ones become ``Trajectory k``.
            title: Figure title, also used for the file name.

        Returns:
            The path of the file written.
        """
        for index, (x, y) in enumerate(trajectories):
            color = _TRAJECTORY_COLORS[index % len(_TRAJECTORY_COLORS)]
            if labels and index < len(labels):
                label = labels[index]
            else:
                label = f"Trajectory {index + 1}"
            self.figure.add_trace(
                go.Scatter(
                    x=x.discretization,
                    y=y.discretization,
                    mode="lines",
                    name=label,
                    line={"color": color, "width": 1},
                ),
            )
        self.figure.update_layout(
            title=title,
            xaxis_title=trajectories[0][0].name or "X",
            yaxis_title=trajectories[0][1].name or "Y",
            hovermode="closest",
            showlegend=True,
            width=800,
            height=600,
        )
        return self._save_plot(title)

    def _save_plot(self, title: str) -> str:
        """Save the plot to file and return the path written."""
        safe_title = (
            "".join(c for c in title if c.isalnum() or c in " _-")
            .strip()
            .replace(" ", "_")
        )
        directory = Path(self.output_dir)
        if self.output_format == "png":
            filename = directory / f"{safe_title}.png"
            self.figure.write_image(str(filename))
        else:
            filename = directory / f"{safe_title}.html"
            self.figure.write_html(str(filename))
        return str(filename)

    def show(self) -> None:
        """Display the plot."""
        self.figure.show()
