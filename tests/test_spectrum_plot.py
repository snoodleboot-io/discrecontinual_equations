"""The periodogram's axis says what its values are (DEQ-132).

``scipy.signal.periodogram`` with ``scaling="spectrum"`` returns power per
frequency bin (amplitude squared); with ``scaling="density"`` it returns power per
unit frequency (amplitude squared per Hz). The checks here use Parseval's theorem
and a sine's known power, not scipy: for a sine of amplitude ``A`` on a bin
frequency, the power spectrum peaks at the mean square ``A^2 / 2`` and its bins
sum to the signal's variance, while the density's bins sum to the variance only
after multiplying by the bin width ``df = 1 / (N dt)``.
"""

from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.plot.spectrum_plot import SpectrumPlot
from discrecontinual_equations.variable import Variable

_SAMPLES = 64
_STEP = 0.1
_AMPLITUDE = 2.0
_BIN = 5  # the sine completes exactly five periods: no leakage


def _sine() -> Variable:
    times = np.arange(_SAMPLES) * _STEP
    frequency = _BIN / (_SAMPLES * _STEP)
    values = _AMPLITUDE * np.sin(2 * np.pi * frequency * times)
    return Variable(discretization=list(values))


def _periodogram(**options) -> tuple[np.ndarray, str]:
    with TemporaryDirectory() as directory:
        plot = SpectrumPlot(output_dir=directory, output_format="html")
        plot.plot_periodogram(_sine(), dt=_STEP, **options)
    trace = plot.figure.data[0]
    return np.asarray(trace.y), plot.figure.layout.yaxis.title.text


class TestPeriodogramLabel(TestCase):
    def test_default_is_a_power_spectrum_labelled_as_one(self):
        power, label = _periodogram()
        mean_square = _AMPLITUDE**2 / 2
        assert power.max() == pytest.approx(mean_square)
        assert power.sum() == pytest.approx(mean_square)
        assert label == "Power spectrum (amplitude²)"
        assert "density" not in label.lower()

    def test_density_is_labelled_per_hertz_and_integrates_to_the_variance(self):
        power, label = _periodogram(scaling="density")
        bin_width = 1.0 / (_SAMPLES * _STEP)
        assert (power * bin_width).sum() == pytest.approx(_AMPLITUDE**2 / 2)
        assert power.max() != pytest.approx(_AMPLITUDE**2 / 2)
        assert label == "Power spectral density (amplitude²/Hz)"

    def test_an_unknown_scaling_is_refused(self):
        with pytest.raises(ValueError, match="scaling must be one of"):
            _periodogram(scaling="psd")


def _fft_trace(samples: int) -> tuple[np.ndarray, np.ndarray, str]:
    times = np.arange(samples) * _STEP
    values = _AMPLITUDE * np.sin(2 * np.pi * _BIN / (samples * _STEP) * times)
    with TemporaryDirectory() as directory:
        plot = SpectrumPlot(output_dir=directory, output_format="html")
        plot.plot_fft(Variable(discretization=list(values)), dt=_STEP)
    trace = plot.figure.data[0]
    return np.asarray(trace.x), np.asarray(trace.y), plot.figure.layout.yaxis.title.text


class TestFftPowerSpectrum:
    """``plot_fft`` is the one-sided power spectrum its axis names (DEQ-173).

    A sine of amplitude ``A`` on a bin frequency has mean square ``A^2 / 2``, all
    in that one bin, at any record length; the unnormalised ``|X_k|^2`` it used to
    plot was ``(A N / 2)^2`` there.
    """

    @pytest.mark.parametrize("samples", [64, 65, 256])
    def test_a_sine_peaks_at_its_mean_square_at_any_length(self, samples):
        freqs, power, label = _fft_trace(samples)
        assert power.max() == pytest.approx(_AMPLITUDE**2 / 2)
        assert freqs[power.argmax()] == pytest.approx(_BIN / (samples * _STEP))
        assert power.sum() == pytest.approx(_AMPLITUDE**2 / 2)
        assert label == "Power spectrum (amplitude²)"

    def test_it_matches_the_periodogram_without_its_zero_and_nyquist_bins(self):
        rng = np.random.default_rng(7)
        signal = Variable(discretization=list(rng.standard_normal(_SAMPLES)))
        with TemporaryDirectory() as directory:
            plot = SpectrumPlot(output_dir=directory, output_format="html")
            plot.plot_fft(signal, dt=_STEP)
            plot.plot_periodogram(signal, dt=_STEP)
        fft, periodogram = plot.figure.data
        np.testing.assert_allclose(fft.x, periodogram.x[1:-1])
        np.testing.assert_allclose(fft.y, periodogram.y[1:-1], rtol=1e-12)
