"""Power spectra of a sampled signal, drawn with Plotly.

A periodic signal puts its power at a few frequencies and their harmonics; a
quasiperiodic one at integer combinations of two or more incommensurate ones; a
chaotic one spreads it over a broad band. The spectrum is the quickest way to
tell those apart, and :class:`SpectrumPlot` draws it two ways, both from the
mean-removed samples ``x_0, ..., x_{N-1}`` at spacing ``dt``:

* :meth:`SpectrumPlot.plot_fft` plots the one-sided power spectrum
  ``2 |X_k|^2 / N^2`` with ``X_k = sum_n x_n exp(-2 pi i k n / N)``
  (``numpy.fft.fft``) at the positive frequencies ``k / (N dt)`` below Nyquist,
  on log-log axes. A sine of amplitude ``A`` on a bin frequency peaks at its mean
  square ``A^2 / 2`` whatever ``N``, and the bins sum to the variance less the
  zero and Nyquist bins (Parseval). Until DEQ-173 it plotted the unnormalised
  ``|X_k|^2``, which grows as ``N^2``, under the axis title "Power".
* :meth:`SpectrumPlot.plot_periodogram` uses ``scipy.signal.periodogram`` with a
  rectangular window and ``scaling="spectrum"`` by default: the same power
  spectrum in squared signal units, on the same scale as ``plot_fft``, with the
  Nyquist bin included. ``scaling="density"`` gives power per hertz instead.

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


def one_sided_power(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """The one-sided power spectrum of a mean-removed signal, in amplitude².

    With ``X_k = sum_n x_n exp(-2 pi i k n / N)``, the two-sided power in bin
    ``k`` is ``|X_k|^2 / N^2`` (Parseval: the bins sum to the mean square). The
    bins ``k`` and ``N - k`` carry equal power for a real signal, so the positive
    frequencies below Nyquist are doubled. This is ``scipy.signal.periodogram``
    with ``scaling="spectrum"`` and a boxcar window, without its zero and Nyquist
    bins.

    Args:
        signal: The samples ``x_0, ..., x_{N-1}``.
        dt: The sampling interval.

    Returns:
        The frequencies ``k / (N dt)`` for ``0 < k < N / 2`` and their power.

    References:
        SciPy 1.18.1, ``scipy/signal/_spectral_py.py``, ``_spectral_helper``:
        ``scaling="spectrum"`` multiplies ``|X_k|^2`` by ``1 / win.sum()**2``
        (``1 / N^2`` for the boxcar) and a one-sided result doubles every bin but
        the zero and an even ``N``'s Nyquist one. See ``docs/references.md``
        (SciPy2026).
    """
    centred = np.asarray(signal, dtype=float) - np.mean(signal)
    n = len(centred)
    freqs = np.fft.fftfreq(n, dt)
    power = 2.0 * np.abs(np.fft.fft(centred)) ** 2 / n**2
    positive = freqs > 0  # fftfreq puts an even N's Nyquist bin at -1 / (2 dt)
    return freqs[positive], power[positive]


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
        """Plot the one-sided power spectrum of the mean-removed signal.

        The power in each positive frequency bin below Nyquist is
        ``2 |X_k|^2 / N^2`` (see :func:`one_sided_power`), in squared signal
        units, so a sine of amplitude ``A`` on a bin frequency peaks at ``A^2 / 2``
        however long the record. The axis is labelled as
        :meth:`plot_periodogram`'s default, whose values these are.

        Args:
            x: The sampled signal.
            dt: Sample spacing, which sets the frequency axis.
            title: Figure title, also used for the file name.

        References:
            SciPy 1.18.1, ``scipy.signal.periodogram`` and ``_spectral_helper``;
            see ``docs/references.md`` (SciPy2026).
        """
        freqs_pos, power_pos = one_sided_power(np.array(x.discretization), dt)

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
            yaxis_title=PERIODOGRAM_LABELS["spectrum"],
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
        """Plot the one-sided power spectra of several signals on one figure.

        Each is :func:`one_sided_power` of its mean-removed samples, as in
        :meth:`plot_fft`.

        Args:
            signals: ``(x, dt)`` pairs, each signal with its own sample spacing.
            labels: Legend labels; missing ones become ``Signal k``.
            title: Figure title, also used for the file name.
        """
        colors = ["blue", "red", "green", "orange", "purple", "brown", "pink"]

        for i, (x, dt) in enumerate(signals):
            freqs_pos, power_pos = one_sided_power(np.array(x.discretization), dt)

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
            yaxis_title=PERIODOGRAM_LABELS["spectrum"],
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
