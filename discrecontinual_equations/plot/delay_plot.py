"""Delay embeddings of a scalar time series, drawn with Plotly.

A single observed coordinate ``x(t)`` of a deterministic system can stand in for
the whole state: the delay vectors ``(x(t), x(t + tau), ..., x(t + (m - 1) tau))``
trace out an image of the attractor. Takens (1981) shows that for a generic
smooth observable the map to ``2d + 1`` delays, with ``d`` the dimension of the
compact manifold the dynamics lives on, is an embedding - Theorem 1 for a
diffeomorphism, Theorem 2 for the time-one map of a flow (any fixed delay, by
rescaling time) - so the picture has the same topology as the original
attractor. :class:`DelayPlot` draws the two- and three-dimensional cases,
``m = 2`` and ``m = 3``.

The delay is counted in samples, not in time: ``delay = k`` pairs
``x[i]`` with ``x[i + k]``, so the time delay is ``k`` times the step of the
series, and a non-uniform time grid gives a delay that is not constant in time.

Limits: two or three delays are a picture, not a guaranteed embedding - for an
attractor in more than one dimension Takens' bound asks for more coordinates than
can be drawn, and folds and crossings in the plot can be projections rather than
dynamics. Nothing here chooses the delay (a first zero of the autocorrelation or
a first minimum of the mutual information are the usual rules); a delay much
shorter than the series' correlation time collapses the cloud onto the diagonal,
and Theorem 2's genericity excludes a delay that resonates with a cycle (a
periodic orbit whose period is ``k`` delays, ``k <= 2d + 1``).

References:
    Takens, F. (1981). Detecting strange attractors in turbulence. In D. Rand
    and L.-S. Young (eds.), *Dynamical Systems and Turbulence, Warwick 1980*,
    Lecture Notes in Mathematics 898, pp. 366-381. Springer.
    DOI 10.1007/BFb0091924. Theorems 1 and 2 (delay embeddings of a
    diffeomorphism and of a flow, into ``2d + 1`` dimensions).
"""

import os
from pathlib import Path

import plotly.graph_objects as go

from discrecontinual_equations.variable import Variable


class DelayPlot:
    """Plot class for delay embeddings and attractor reconstruction.

    Each call adds a trace to one ``go.Figure`` and writes it out. Whatever
    ``output_format`` says, :meth:`plot` and :meth:`plot_3d` write both a PNG
    (through kaleido) and an HTML file to ``output_dir``, and print their paths.

    Args:
        output_dir: Directory the figures are written to; created if missing.
        output_format: ``"png"`` or ``"html"``. Kept for symmetry with the other
            plotters; the plotting methods currently write both formats.

    References:
        Takens (1981), Theorems 1 and 2: see the module docstring.
    """

    def __init__(self, output_dir: str = ".", output_format: str = "png"):
        self.output_dir = output_dir
        # A plotter is told where to write; make sure that place exists,
        # so an example whose output directory is not in the tree still runs.
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_format = output_format
        self.figure: go.Figure = go.Figure()

    def save(self, title: str):
        """Save the plot in both PNG and HTML formats.

        Args:
            title: Figure title; its letters, digits, spaces, ``_`` and ``-``
                make the file name, with spaces turned into ``_``.
        """
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
        """Plot the two-dimensional delay embedding ``x[i]`` against ``x[i + delay]``.

        Args:
            x: The observed series; its ``discretization`` is the samples.
            delay: Delay in samples (not in time).
            title: Figure title, also used for the file name.

        Raises:
            ValueError: If the series has no more than ``delay`` samples.
        """
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
        """Plot the three-dimensional delay embedding ``[x[i], x[i + d1], x[i + d2]]``.

        Args:
            x: The observed series; its ``discretization`` is the samples.
            delay1: First delay, in samples.
            delay2: Second delay, in samples.
            title: Figure title, also used for the file name.

        Raises:
            ValueError: If the series has no more samples than the larger delay.
        """
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
