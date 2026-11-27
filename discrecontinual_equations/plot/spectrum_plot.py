"""Power spectra of a sampled signal, drawn with Plotly.

A periodic signal puts its power at a few frequencies and their harmonics; a
quasiperiodic one at integer combinations of two or more incommensurate ones; a
chaotic one spreads it over a broad band. The spectrum is the quickest way to
tell those apart, and :class:`SpectrumPlot` draws it two ways, both from the
mean-removed samples ``x_0, ..., x_{N-1}`` at spacing ``dt``:

* :meth:`SpectrumPlot.plot_fft` plots ``|X_k|^2`` with
  ``X_k = sum_n x_n exp(-2 pi i k n / N)`` (``numpy.fft.fft``) at the positive
  frequencies ``k / (N dt)``, on log-log axes. It is not normalised, so only its
  shape is meaningful, not its level.
* :meth:`SpectrumPlot.plot_periodogram` uses ``scipy.signal.periodogram`` with a
  rectangular window and ``scaling="spectrum"``, which is the power spectrum in
  squared signal units (not a density per hertz, whatever the axis title says).

Limits: both are single raw periodograms, with no averaging or windowing beyond
the rectangle, so each bin's estimate has a standard deviation as large as its
mean however long the series, and a peak between bins leaks into its
neighbours. For a smoother estimate average segments (``scipy.signal.welch``)
before plotting. The frequency unit is the inverse of ``dt``'s, labelled "Hz".
"""

import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from discrecontinual_equations.variable import Variable

# What ``scipy.signal.periodogram`` returns for each ``scaling``, as an axis title.
# "spectrum" is power per frequency bin; "density" is power per unit frequency.
PERIODOGRAM_LABELS = {
    "spectrum": "Power spectrum (amplitude²)",
    "density": "Power spectral density (amplitude²/Hz)",
}


class SpectrumPlot:
    """Plot class for power spectra and frequency analysis.

    Args:
        output_dir: Directory the figures are written to; created if missing.
        output_format: ``"png"`` or ``"html"``; no other value can be saved.
    """

    def __init__(self, output_dir: str = ".", output_format: str = "png"):
        self.output_dir = output_dir
        # A plotter is told where to write; make sure that place exists,
        # so an example whose output directory is not in the tree still runs.
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_format = output_format
        self.figure: go.Figure = go.Figure()

    def plot_fft(self, x: Variable, dt: float = 1.0, title: str = "Power Spectrum"):
        """Plot the unnormalised power ``|FFT|^2`` of the mean-removed signal.

        Args:
            x: The sampled signal.
            dt: Sample spacing, which sets the frequency axis.
            title: Figure title, also used for the file name.
        """
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
        scaling: str = "spectrum",
    ):
        """Plot the raw periodogram of a sampled signal, labelled by its scaling.

        The signal's mean is removed and its one-sided periodogram is taken with
        ``scipy.signal.periodogram``: a boxcar window (no taper), no averaging and
        no smoothing, so each value is the squared modulus of one DFT bin. The
        ``scaling`` decides what that value means, and the y-axis says so:

        - ``"spectrum"`` (default): power per frequency bin, in amplitude² (V² for
          a signal in volts). A sine on a bin frequency peaks at its mean square,
          ``A² / 2``, and the bins sum to the signal's variance.
        - ``"density"``: power per unit frequency, in amplitude²/Hz (with ``dt``
          in seconds). The bins times the bin width ``1 / (N dt)`` sum to the
          variance; use it for broadband noise, whose level should not depend on
          ``N``.

        Limits: the raw periodogram is not a consistent estimate of a noise
        spectrum - its variance does not fall as ``N`` grows - and a boxcar leaks
        power from a sine between bins into its neighbours. Welch averaging or a
        taper would address both and is not applied here.

        Args:
            x: The sampled signal.
            dt: The sampling interval; the frequency axis is ``1 / dt`` wide.
            title: The plot title, also the file name.
            scaling: ``"spectrum"`` or ``"density"``.

        Raises:
            ValueError: If ``scaling`` is neither.

        References:
        - SciPy ``scipy.signal.periodogram`` documentation (v1.18.0): parameter
          ``scaling`` ("density", V²/Hz; "spectrum", V²), default window
          ``"boxcar"``, default ``detrend="constant"``.
        """
        from scipy.signal import periodogram

        if scaling not in PERIODOGRAM_LABELS:
            message = (
                f"scaling must be one of {sorted(PERIODOGRAM_LABELS)}, not {scaling!r}"
            )
            raise ValueError(message)

        signal = np.array(x.discretization)
        signal = signal - np.mean(signal)

        freqs, power = periodogram(signal, fs=1.0 / dt, scaling=scaling)

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
            yaxis_title=PERIODOGRAM_LABELS[scaling],
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
        """Plot the ``|FFT|^2`` power of several signals on one figure.

        Args:
            signals: ``(x, dt)`` pairs, each signal with its own sample spacing.
            labels: Legend labels; missing ones become ``Signal k``.
            title: Figure title, also used for the file name.
        """
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
