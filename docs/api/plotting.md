# Plotting API

Two packages draw results. `webplot` writes self-contained D3 pages: stills,
playable stages (films) and the atlas that links them. `plot` writes Plotly
figures of time series, phase portraits, delay embeddings and spectra. The
[Atlas pages and films](../user-guide/pages-and-films.md) guide explains how
they fit together.

## Pages and films (`webplot`)

::: discrecontinual_equations.webplot
    options:
      members: false

### Stills

::: discrecontinual_equations.webplot.scene

::: discrecontinual_equations.webplot.scene_builder

### Stages

::: discrecontinual_equations.webplot.stage

::: discrecontinual_equations.webplot.stage_builder

::: discrecontinual_equations.webplot.manifold_flow

### Stochastic stages

::: discrecontinual_equations.webplot.stochastic_stage_builder

::: discrecontinual_equations.webplot.quasipotential_stage

### Manifold films

::: discrecontinual_equations.webplot.manifold_film

### Renderers and the atlas

::: discrecontinual_equations.webplot.renderer

::: discrecontinual_equations.webplot.stage_renderer

::: discrecontinual_equations.webplot.stochastic_stage_renderer

::: discrecontinual_equations.webplot.manifold_film_renderer

::: discrecontinual_equations.webplot.report

::: discrecontinual_equations.webplot.latex

## Figures (`plot`)

::: discrecontinual_equations.plot.bifurcation_plot

::: discrecontinual_equations.plot.line_plot

::: discrecontinual_equations.plot.phase_plot

::: discrecontinual_equations.plot.delay_plot

### Spectra: power spectrum or power spectral density

`SpectrumPlot.plot_periodogram` shows how a sampled signal's power is spread over
frequency. It takes the raw periodogram: the mean is removed, no window taper is
applied (a boxcar), and nothing is averaged or smoothed. Each value is the
squared modulus of one DFT bin, and `scaling` says how it is normalised
([SciPy2026](../references.md#pages-films-and-numerics-infrastructure)):

| `scaling` | What a value is | Units | Use it for |
|---|---|---|---|
| `"spectrum"` (default) | power in one frequency bin | amplitude² | a sine's power: it peaks at the mean square $A^2/2$ |
| `"density"` | power per unit frequency | amplitude²/Hz | broadband noise: its level does not depend on $N$ |

The y-axis title states which one is plotted. The bins of the power spectrum sum
to the signal's variance; the density's bins sum to it after multiplying by the
bin width $1/(N\,\Delta t)$.

```python
import tempfile

import numpy as np

from discrecontinual_equations.plot.spectrum_plot import SpectrumPlot
from discrecontinual_equations.variable import Variable

dt, n = 0.1, 64
times = np.arange(n) * dt
x = Variable(discretization=list(2.0 * np.sin(2 * np.pi * 5 / (n * dt) * times)))

plot = SpectrumPlot(output_dir=tempfile.mkdtemp(), output_format="html")
plot.plot_periodogram(x, dt=dt)
print(plot.figure.layout.yaxis.title.text, max(plot.figure.data[0].y))
```

prints the file it wrote, then `Power spectrum (amplitude²) 2.0`: the
sine of amplitude 2 has mean square 2.

**Limits.** The raw periodogram is not a consistent estimate of a noise
spectrum: its variance does not shrink as the record grows. A boxcar also leaks
a sine that falls between bins into its neighbours, so the peak height is exact
only on a bin frequency. Welch averaging and tapered windows address both and
are not applied.

`plot_fft` and `plot_multiple` plot the same power spectrum, $2|X_k|^2/N^2$ at the
positive frequencies below Nyquist, under the same axis title, so their level no
longer grows with the record length. Before DEQ-173 they plotted $|X_k|^2$
unnormalised, labelled "Power", which grows as $N^2$. Continuing the example:

```python
plot.plot_fft(x, dt=dt)
print(plot.figure.layout.yaxis.title.text, max(plot.figure.data[1].y))
```

prints `Power spectrum (amplitude²) 2.0`, the periodogram's peak; the unnormalised
value was 4096 here. Source: [SciPy `periodogram` and `_spectral_helper`](../references.md#pages-films-and-numerics-infrastructure).

::: discrecontinual_equations.plot.spectrum_plot
