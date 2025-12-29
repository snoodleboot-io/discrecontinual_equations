import plotly.graph_objects as go

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.variable import Variable


class LinePlot:
    def __init__(self, output_dir: str = ".", output_format: str = "png"):
        self.output_dir = output_dir
        self.output_format = output_format
        self.figure: go.Figure = go.Figure()

    def plot_curves(self, curve: Curve, title: str = "Curve Plot"):
        """Plot all variables and results on a single plot."""

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

        # Save plot
        import os

        safe_title = (
            "".join(c for c in title if c.isalnum() or c in " _-")
            .strip()
            .replace(" ", "_")
        )
        if self.output_format == "png":
            filename = os.path.join(self.output_dir, f"{safe_title}.png")
            self.figure.write_image(filename, engine="kaleido")
        elif self.output_format == "html":
            filename = os.path.join(self.output_dir, f"{safe_title}.html")
            self.figure.write_html(filename)
        print(f"Plot saved as {filename}")

    def plot(self, x: Variable, y: Variable, title: str = "Curve Plot"):
        """Plot all variables and results on a single plot."""

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

        # Save plot
        import os

        safe_title = (
            "".join(c for c in title if c.isalnum() or c in " _-")
            .strip()
            .replace(" ", "_")
        )
        if self.output_format == "png":
            filename = os.path.join(self.output_dir, f"{safe_title}.png")
            self.figure.write_image(filename, engine="kaleido")
        elif self.output_format == "html":
            filename = os.path.join(self.output_dir, f"{safe_title}.html")
            self.figure.write_html(filename)
        print(f"Plot saved as {filename}")

    def plot_3d(
        self,
        x: Variable,
        y: Variable,
        z: Variable,
        title: str = "3D Plot",
        output_format: str = "both",
    ):
        """Plot 3D scatter plot."""
        import plotly.graph_objects as go

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

        # Save plot
        import os

        safe_title = (
            "".join(c for c in title if c.isalnum() or c in " _-")
            .strip()
            .replace(" ", "_")
        )
        if output_format == "png":
            filename = os.path.join(self.output_dir, f"{safe_title}.html")
            self.figure.write_html(filename)
            print(f"Plot saved as {filename}")
        elif output_format == "html":
            filename = os.path.join(self.output_dir, f"{safe_title}.png")
            self.figure.show()
            # self.figure.write_image(filename, engine="kaleido")
            print(f"Plot saved as {filename}")
        elif output_format == "both":
            png_filename = os.path.join(self.output_dir, f"{safe_title}.png")
            self.figure.write_image(png_filename, engine="kaleido")
            html_filename = os.path.join(self.output_dir, f"{safe_title}.html")
            self.figure.write_html(html_filename)
            print(f"Plot saved as {png_filename} and {html_filename}")
