# PLOT

## Overview

The `plot` submodule provides comprehensive visualization utilities for differential equation solutions and dynamical systems analysis. It uses Plotly to create interactive 2D and 3D plots, supporting both static image export and interactive HTML output for time series, phase portraits, delay embeddings, and spectral analysis.

## Architecture

```
Plot (Base Classes)
├── LinePlot - Time series and general 2D/3D plotting
│   ├── plot_curves(curve, title)
│   ├── plot(x, y, title)
│   └── plot_3d(x, y, z, title, output_format)
│
├── PhasePlot - 2D phase space diagrams
│   ├── plot(x, y, title)
│   └── plot_multiple(trajectories, labels, title)
│
├── DelayPlot - Delay embeddings and attractor reconstruction
│   ├── plot(x, delay, title)
│   └── plot_3d(x, delay1, delay2, title)
│
└── SpectrumPlot - Power spectra and frequency analysis
    ├── plot_fft(x, dt, title)
    ├── plot_periodogram(x, dt, title, scaling)
    └── plot_multiple(signals, labels, title)
```

## Core Classes

Every class takes `output_dir` and `output_format` and draws on a Plotly figure kept
as `figure`; each plotting call writes the figure to `output_dir` under a file name
made from the title.

- **`LinePlot`**: `plot(x, y, title)` draws one `Variable` against another (a time
  series is a variable against time); `plot_curves(curve, title)` draws every result
  of a `Curve` against its variables; `plot_3d(x, y, z, title, output_format="both")`
  draws a trajectory on a fresh figure and uses its own `output_format`, not the
  constructor's. `output_format` is `"png"`, `"html"` or `"both"`. Traces accumulate
  on the one figure across `plot` and `plot_curves` calls.
- **`PhasePlot`**: `plot(x, y, title)` and `plot_multiple(trajectories, labels,
  title)` with `(x, y)` pairs; both return the path written.
- **`DelayPlot`**: `plot(x, delay, title)` draws `x[i + delay]` against `x[i]`;
  `plot_3d(x, delay1, delay2, title)` the three-dimensional embedding. Delays are in
  samples. Whatever `output_format` says, both methods write a PNG (through kaleido,
  which starts a headless browser and takes seconds) and an HTML file.
- **`SpectrumPlot`**: `plot_fft(x, dt, title)` draws the one-sided power spectrum
  `2 |X_k|^2 / N^2` at the positive frequencies below Nyquist, in squared signal
  units, so a sine of amplitude `A` on a bin frequency peaks at its mean square
  `A^2 / 2` whatever the record length. `plot_periodogram(x, dt, title, scaling)` is
  SciPy's raw periodogram: `scaling="spectrum"` (the default) is the same power
  spectrum, Nyquist bin included, and `scaling="density"` is power per hertz.
  `plot_multiple(signals, labels, title)` takes `(x, dt)` pairs and draws each one's
  `plot_fft` spectrum. The y-axis title says which quantity is drawn: "Power spectrum
  (amplitude²)" or "Power spectral density (amplitude²/Hz)". Both spectra are single
  raw estimates, with no averaging or taper.

Three of them in one go, writing HTML to a scratch directory (`DelayPlot` is left
out only because it always writes a PNG as well):

```python
import tempfile

import numpy as np

from discrecontinual_equations.plot.line_plot import LinePlot
from discrecontinual_equations.plot.phase_plot import PhasePlot
from discrecontinual_equations.plot.spectrum_plot import SpectrumPlot
from discrecontinual_equations.variable import Variable

out = tempfile.mkdtemp()
dt, n = 0.1, 64
times = np.arange(n) * dt
t = Variable(name="Time", abbreviation="t", discretization=list(times))
x = Variable(name="x", discretization=list(2.0 * np.sin(2 * np.pi * 5 / (n * dt) * times)))
v = Variable(name="v", discretization=list(np.gradient(x.discretization, dt)))

LinePlot(output_dir=out, output_format="html").plot(t, x, title="Sine")
PhasePlot(output_dir=out, output_format="html").plot(x, v, title="Sine phase")
spectrum = SpectrumPlot(output_dir=out, output_format="html")
spectrum.plot_fft(x, dt=dt, title="Sine spectrum")
print(spectrum.figure.layout.yaxis.title.text, f"{max(spectrum.figure.data[0].y):.4f}")
```

Each call prints the file it wrote; the last line prints
`Power spectrum (amplitude²) 2.0000`, the sine's mean square `2^2 / 2`.

## Usage Examples

### Basic 2D Plotting

```python
time = Variable(name="Time", abbreviation="t")
time.discretization = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

concentration = Variable(name="Concentration", abbreviation="c")
concentration.discretization = [1.0, 0.9048, 0.8187, 0.7408, 0.6703, 0.6065]

# Writes Exponential_Decay.html; output_format="png" writes a PNG through kaleido
plot = LinePlot(output_dir=out, output_format="html")
plot.plot(time, concentration, title="Exponential Decay")
```

### Plotting Solution Curves

A solve returns a `Curve`: `time` holds the time nodes and `results` the state at
each. Here a short Lorenz run with the Euler method:

```python
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.solver.deterministic.euler.euler_config import EulerConfig
from discrecontinual_equations.solver.deterministic.euler.euler_solver import EulerSolver


class Lorenz(DeterministicFunction):
    def eval(self, point, time=None):
        x, y, z = point
        return [10.0 * (y - x), x * (28.0 - z) - y, x * y - 8.0 / 3.0 * z]


state = [Variable(name=name) for name in ("x", "y", "z")]
clock = Variable(name="t")
field = Lorenz(state, [], [Variable(name=f"{v.name}'") for v in state], clock)
equation = DifferentialEquation(variables=state, time=clock, parameters=[], derivative=field)
solution = EulerSolver(EulerConfig(start_time=0.0, end_time=5.0, step_size=0.005)).solve(
    equation, [1.0, 1.0, 1.0]
)

plot = LinePlot(output_dir=out, output_format="html")
plot.plot_curves(solution, title="ODE Solution")
```

### 3D Trajectory Plotting

```python
x_vals, y_vals, z_vals = solution.results
plot = LinePlot(output_dir=out)
plot.plot_3d(x_vals, y_vals, z_vals, title="Lorenz Attractor", output_format="html")
```

`output_format="both"` (the default for `plot_3d`) writes a PNG and an HTML file.

### Advanced Plotting

The figure is a plain `plotly.graph_objects.Figure`, so it can be built up by hand:

```python
from pathlib import Path

import plotly.graph_objects as go

plot = LinePlot(output_dir=out)
for i, start in enumerate(([1.0, 1.0, 1.0], [1.0, 1.0, 1.001])):
    run = EulerSolver(EulerConfig(start_time=0.0, end_time=5.0, step_size=0.005)).solve(
        equation, start
    )
    plot.figure.add_trace(
        go.Scatter(
            x=run.time.discretization,
            y=run.results[0].discretization,
            mode="lines",
            name=f"Solution {i + 1}",
            line=dict(width=2),
        )
    )

plot.figure.update_layout(
    title="Multiple Solutions Comparison",
    xaxis_title="Time",
    yaxis_title="x",
)
plot.figure.write_html(str(Path(out) / "comparison.html"))
```

## Output Formats

### PNG Images
- **Format**: Static raster images
- **Use**: Publications, presentations, documentation
- **Engine**: Kaleido (headless browser)

### HTML Files
- **Format**: Interactive web-based plots
- **Use**: Analysis, sharing, web deployment
- **Features**: Zoom, pan, hover tooltips, export options

### Combined Output
- **Option**: `output_format="both"`
- **Creates**: Both PNG and HTML versions

## Plot Customization

### Layout Options

```python
plot.figure.update_layout(
    title="Custom Plot Title",
    xaxis_title="Time (s)",
    yaxis_title="Concentration (mol/L)",
    hovermode="x unified",
    template="plotly_white"
)
```

### Styling Traces

```python
plot.figure.add_trace(
    go.Scatter(
        x=time.discretization,
        y=concentration.discretization,
        mode="lines+markers",
        name="Solution",
        line=dict(
            color="red",
            width=3,
            dash="dash"
        ),
        marker=dict(
            size=6,
            symbol="circle"
        )
    )
)
```

### 3D Plot Options

```python
plot = LinePlot(output_dir=out)
plot.plot_3d(x_vals, y_vals, z_vals, title="Lorenz Attractor", output_format="html")
plot.figure.update_scenes(
    xaxis_title="X Position",
    yaxis_title="Y Position",
    zaxis_title="Z Position",
    camera=dict(
        eye=dict(x=1.5, y=1.5, z=1.5)
    )
)
```

## Integration with Solvers

The plot module works on the `Curve` a solve returns:

```python
t = solution.time
x, y, z = solution.results

plot = LinePlot(output_dir=out, output_format="html")

# Time series plots
plot.plot(t, x, "X vs Time")
plot.figure = go.Figure()  # Reset for new plot
plot.plot(t, y, "Y vs Time")

# Phase space plot
plot.figure = go.Figure()
plot.plot(x, y, "Phase Space X vs Y")

# 3D trajectory (always on a fresh figure)
plot.plot_3d(x, y, z, "3D Trajectory", output_format="html")
```

## Performance Considerations

- **Large Datasets**: Use PNG for static plots, HTML for interactive exploration
- **Memory Usage**: Large HTML files may be slow to load
- **File Sizes**: PNG files are smaller but less flexible
- **Batch Processing**: Reuse LinePlot instances for multiple plots

## Error Handling

An unknown `output_format` is refused with `ValueError`, and PNG output without
kaleido raises an `ImportError` that says to install it or use `"html"`:

```python
try:
    LinePlot(output_dir=out, output_format="svg").plot(t, x, "Trajectory")
except ValueError as error:
    print(f"Plotting failed: {error}")
```

## Dependencies

- **plotly**: Core plotting library
- **kaleido** (optional): For PNG export in headless environments

## File Naming

Plots are automatically saved with safe filenames:

Every character that is not a letter, digit, space, `_` or `-` is dropped, and spaces
become underscores. Letters include non-ASCII ones:

```text
Title:    "Lorenz Attractor (σ=10, ρ=28, β=8/3)"
Becomes:  "Lorenz_Attractor_σ10_ρ28_β83.png"
```

---

**Parent Module:** [DISCRECONTINUAL_EQUATIONS](../DISCRECONTINUAL_EQUATIONS.md)

**Related Modules:**
- [SOLVER](../solver/SOLVER.md) - Provides solution data for plotting
