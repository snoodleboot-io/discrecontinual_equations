import os

import plotly.graph_objects as go

from discrecontinual_equations.variable import Variable


class DelayPlot:
    """Plot class for delay embeddings and attractor reconstruction."""

    def __init__(self, output_dir: str = ".", output_format: str = "png"):
        self.output_dir = output_dir
        self.output_format = output_format
        self.figure: go.Figure = go.Figure()

    def save(self, title: str):
        """Save the plot in both PNG and HTML formats."""
        safe_title = (
            "".join(c for c in title if c.isalnum() or c in " _-")
            .strip()
            .replace(" ", "_")
        )

        # Save PNG
        png_filename = os.path.join(self.output_dir, f"{safe_title}.png")
        self.figure.write_image(png_filename)
        print(f"Delay plot saved as {png_filename}")

        # Save HTML
        html_filename = os.path.join(self.output_dir, f"{safe_title}.html")
        self.figure.write_html(html_filename)
        print(f"Delay plot saved as {html_filename}")

    def plot(self, x: Variable, delay: int = 1, title: str = "Delay Plot"):
        """Plot delay embedding x(t) vs x(t+delay)."""
        if len(x.discretization) <= delay:
            raise ValueError("Delay is too large for the available data")

        x_current = x.discretization[:-delay]
        x_delayed = x.discretization[delay:]

        self.figure.add_trace(
            go.Scatter(
                x=x_current,
                y=x_delayed,
                mode="markers",
                name=f"x(t) vs x(t+{delay})",
                marker=dict(color="blue", size=2, opacity=0.6),
                showlegend=False,
            ),
        )

        self.figure.update_layout(
            title=title,
            xaxis_title=f"{x.name}(t)" if x.name else "x(t)",
            yaxis_title=f"{x.name}(t+{delay})" if x.name else f"x(t+{delay})",
            hovermode="closest",
            width=800,
            height=600,
        )

        self.save(title)

    def plot_3d(
        self,
        x: Variable,
        delay1: int = 1,
        delay2: int = 2,
        title: str = "3D Delay Plot",
    ):
        """Plot 3D delay embedding [x(t), x(t+d1), x(t+d2)]."""
        if len(x.discretization) <= max(delay1, delay2):
            raise ValueError("Delays are too large for the available data")

        x_current = x.discretization[: -max(delay1, delay2)]
        x_d1 = x.discretization[
            delay1 : len(x.discretization) - max(0, delay2 - delay1)
        ]
        x_d2 = x.discretization[delay2:]

        # Ensure all arrays have the same length
        min_len = min(len(x_current), len(x_d1), len(x_d2))
        x_current = x_current[:min_len]
        x_d1 = x_d1[:min_len]
        x_d2 = x_d2[:min_len]

        self.figure.add_trace(
            go.Scatter3d(
                x=x_current,
                y=x_d1,
                z=x_d2,
                mode="lines",
                name=f"x(t) vs x(t+{delay1}) vs x(t+{delay2})",
                line=dict(color="blue", width=1),
                showlegend=False,
            ),
        )

        self.figure.update_layout(
            title=title,
            scene=dict(
                xaxis_title=f"{x.name}(t)" if x.name else "x(t)",
                yaxis_title=f"{x.name}(t+{delay1})" if x.name else f"x(t+{delay1})",
                zaxis_title=f"{x.name}(t+{delay2})" if x.name else f"x(t+{delay2})",
            ),
            width=800,
            height=600,
        )

        self.save(title)

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
        print(f"Delay plot saved as {filename}")

    def show(self):
        """Display the plot."""
        self.figure.show()
