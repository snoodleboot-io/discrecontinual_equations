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
    """Plot class for 2D phase space diagrams."""

    def __init__(self, output_dir: str = ".", output_format: str = "png") -> None:
        self.output_dir = output_dir
        self.output_format = output_format
        self.figure: go.Figure = go.Figure()

    def plot(self, x: Variable, y: Variable, title: str = "Phase Space Plot") -> str:
        """Plot a single 2D phase-space trajectory and save it."""
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
        """Plot several trajectories in phase space and save the figure."""
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
