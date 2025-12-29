import os

import numpy as np
import plotly.graph_objects as go

from discrecontinual_equations.variable import Variable


class SpectrumPlot:
    """Plot class for power spectra and frequency analysis."""

    def __init__(self, output_dir: str = ".", output_format: str = "png"):
        self.output_dir = output_dir
        self.output_format = output_format
        self.figure: go.Figure = go.Figure()

    def plot_fft(self, x: Variable, dt: float = 1.0, title: str = "Power Spectrum"):
        """Plot power spectrum using FFT."""
        signal = np.array(x.discretization)

        # Remove DC component
        signal = signal - np.mean(signal)

        # Compute FFT
        n = len(signal)
        freqs = np.fft.fftfreq(n, dt)
        fft_vals = np.fft.fft(signal)
        power = np.abs(fft_vals) ** 2

        # Only positive frequencies
        positive_freq_mask = freqs > 0
        freqs_pos = freqs[positive_freq_mask]
        power_pos = power[positive_freq_mask]

        self.figure.add_trace(
            go.Scatter(
                x=freqs_pos,
                y=power_pos,
                mode="lines",
                name="Power Spectrum",
                line=dict(color="blue", width=2),
                showlegend=False,
            ),
        )

        self.figure.update_layout(
            title=title,
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power",
            hovermode="x",
            width=800,
            height=600,
            xaxis=dict(type="log"),
            yaxis=dict(type="log"),
        )

        self._save_plot(title)

    def plot_periodogram(
        self,
        x: Variable,
        dt: float = 1.0,
        title: str = "Periodogram",
    ):
        """Plot periodogram (smoothed power spectrum)."""
        from scipy.signal import periodogram

        signal = np.array(x.discretization)
        signal = signal - np.mean(signal)

        freqs, power = periodogram(signal, fs=1.0 / dt, scaling="spectrum")

        self.figure.add_trace(
            go.Scatter(
                x=freqs,
                y=power,
                mode="lines",
                name="Periodogram",
                line=dict(color="red", width=2),
                showlegend=False,
            ),
        )

        self.figure.update_layout(
            title=title,
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power Spectral Density",
            hovermode="x",
            width=800,
            height=600,
        )

        self._save_plot(title)

    def plot_multiple(
        self,
        signals: list[tuple[Variable, float]],
        labels: list[str] | None = None,
        title: str = "Multiple Spectra",
    ):
        """Plot power spectra for multiple signals."""
        colors = ["blue", "red", "green", "orange", "purple", "brown", "pink"]

        for i, (x, dt) in enumerate(signals):
            signal = np.array(x.discretization)
            signal = signal - np.mean(signal)

            n = len(signal)
            freqs = np.fft.fftfreq(n, dt)
            fft_vals = np.fft.fft(signal)
            power = np.abs(fft_vals) ** 2

            positive_freq_mask = freqs > 0
            freqs_pos = freqs[positive_freq_mask]
            power_pos = power[positive_freq_mask]

            color = colors[i % len(colors)]
            label = labels[i] if labels and i < len(labels) else f"Signal {i + 1}"

            self.figure.add_trace(
                go.Scatter(
                    x=freqs_pos,
                    y=power_pos,
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=2),
                ),
            )

        self.figure.update_layout(
            title=title,
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power",
            hovermode="x",
            showlegend=True,
            width=800,
            height=600,
            xaxis=dict(type="log"),
            yaxis=dict(type="log"),
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
        print(f"Spectrum plot saved as {filename}")

    def show(self):
        """Display the plot."""
        self.figure.show()
