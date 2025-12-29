import os

import plotly.graph_objects as go

from discrecontinual_equations.variable import Variable


class PhasePlot:
    """Plot class for 2D phase space diagrams."""

    def __init__(self, output_dir: str = ".", output_format: str = "png"):
        self.output_dir = output_dir
        self.output_format = output_format
        self.figure: go.Figure = go.Figure()

    def plot(self, x: Variable, y: Variable, title: str = "Phase Space Plot"):
        """Plot 2D phase space diagram."""
        self.figure.add_trace(
            go.Scatter(
                x=x.discretization,
                y=y.discretization,
                mode="lines",
                name="Trajectory",
                line=dict(color="blue", width=1),
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

        self._save_plot(title)

    def plot_multiple(
        self,
        trajectories: list[tuple[Variable, Variable]],
        labels: list[str] | None = None,
        title: str = "Phase Space Plot",
    ):
        """Plot multiple trajectories in phase space."""
        colors = [
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

        for i, (x, y) in enumerate(trajectories):
            color = colors[i % len(colors)]
            label = labels[i] if labels and i < len(labels) else f"Trajectory {i + 1}"

            self.figure.add_trace(
                go.Scatter(
                    x=x.discretization,
                    y=y.discretization,
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=1),
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

        self._save_plot(title)

    def _save_plot(self, title: str):
        """Save the plot to file."""
        safe_title = (
            "".join(c for c in title if c.isalnum() or c in " _-")
            .strip()
            .replace(" ", "_")
        )
        if self.output_format == "png":
            filename = os.path.join(self.output_dir, f"{safe_title}.png")
            self.figure.write_image(filename)
        elif self.output_format == "html":
            filename = os.path.join(self.output_dir, f"{safe_title}.html")
            self.figure.write_html(filename)
        print(f"Phase plot saved as {filename}")

    def show(self):
        """Display the plot."""
        self.figure.show()
