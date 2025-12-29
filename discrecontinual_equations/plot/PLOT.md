# PLOT

## Overview

The `plot` submodule provides comprehensive visualization utilities for differential equation solutions and dynamical systems analysis. It uses Plotly to create interactive 2D and 3D plots, supporting both static image export and interactive HTML output for time series, phase portraits, delay embeddings, and spectral analysis.

## Architecture

```
Plot (Base Classes)
├── LinePlot - Time series and general 2D/3D plotting
│   ├── plot_curves(curve, title)
│   ├── plot(x, y, title)
│   ├── plot_3d(x, y, z, title)
│   └── plot_multiple(trajectories, labels, title)
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
    ├── plot_periodogram(x, dt, title)
    └── plot_multiple(signals, labels, title)
```

## Core Classes

### LinePlot - Time Series and General Plotting

```python
class LinePlot:
    """Plotting utility for time series and general 2D/3D plots."""

    def __init__(self, output_dir: str = ".", output_format: str = "png"):
        """Initialize plotter with output settings."""

    def plot_curves(self, curve: Curve, title: str = "Curve Plot"):
        """Plot all variables and results from a Curve object."""

    def plot(self, x: Variable, y: Variable, title: str = "Line Plot"):
        """Plot x vs y variables as a line."""

    def plot_3d(self, x: Variable, y: Variable, z: Variable, title: str = "3D Plot"):
        """Create 3D scatter/line plots."""

    def plot_multiple(self, trajectories: list[tuple], labels: list[str] = None, title: str = "Multiple Trajectories"):
        """Plot multiple trajectories on the same axes."""
```

### PhasePlot - Phase Space Analysis

```python
class PhasePlot:
    """Plotting utility for 2D phase space diagrams."""

    def __init__(self, output_dir: str = ".", output_format: str = "png"):
        """Initialize phase plotter."""

    def plot(self, x: Variable, y: Variable, title: str = "Phase Space Plot"):
        """Plot 2D phase space diagram x vs y."""

    def plot_multiple(self, trajectories: list[tuple], labels: list[str] = None, title: str = "Phase Space Plot"):
        """Plot multiple trajectories in phase space."""
```

### DelayPlot - Delay Embeddings

```python
class DelayPlot:
    """Plotting utility for delay embeddings and attractor reconstruction."""

    def __init__(self, output_dir: str = ".", output_format: str = "png"):
        """Initialize delay plotter."""

    def plot(self, x: Variable, delay: int = 1, title: str = "Delay Plot"):
        """Plot delay embedding x(t) vs x(t+delay)."""

    def plot_3d(self, x: Variable, delay1: int = 1, delay2: int = 2, title: str = "3D Delay Plot"):
        """Plot 3D delay embedding [x(t), x(t+d1), x(t+d2)]."""
```

### SpectrumPlot - Frequency Analysis

```python
class SpectrumPlot:
    """Plotting utility for power spectra and frequency analysis."""

    def __init__(self, output_dir: str = ".", output_format: str = "png"):
        """Initialize spectrum plotter."""

    def plot_fft(self, x: Variable, dt: float = 1.0, title: str = "Power Spectrum"):
        """Plot power spectrum using FFT."""

    def plot_periodogram(self, x: Variable, dt: float = 1.0, title: str = "Periodogram"):
        """Plot periodogram (smoothed power spectrum)."""

    def plot_multiple(self, signals: list[tuple], labels: list[str] = None, title: str = "Multiple Spectra"):
        """Plot power spectra for multiple signals."""
```

## Usage Examples

### Basic 2D Plotting

```python
from discrecontinual_equations.plot import LinePlot
from discrecontinual_equations import Variable

# Create variables with solution data
time = Variable(name="Time", abbreviation="t")
time.discretization = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

concentration = Variable(name="Concentration", abbreviation="c")
concentration.discretization = [1.0, 0.9048, 0.8187, 0.7408, 0.6703, 0.6065]

# Create plot
plot = LinePlot(output_dir=".", output_format="png")
plot.plot(time, concentration, title="Exponential Decay")
# Saves plot as PNG file
```

### Plotting Solution Curves

```python
from discrecontinual_equations.plot import LinePlot

# After solving a differential equation
solver.solve(equation, initial_values)

# Plot the solution curve
plot = LinePlot(output_dir="results")
plot.plot_curves(solver.solution, title="ODE Solution")
```

### 3D Trajectory Plotting

```python
from discrecontinual_equations.plot import LinePlot

# For systems like Lorenz equations with 3 variables
plot = LinePlot(output_dir="results", output_format="both")

# Extract solution variables
x_vals = solver.solution.variables[0]
y_vals = solver.solution.variables[1]
z_vals = solver.solution.variables[2]

plot.plot_3d(x_vals, y_vals, z_vals,
             title="Lorenz Attractor",
             output_format="both")
# Saves both PNG and HTML files
```

### Advanced Plotting

```python
import plotly.graph_objects as go
from discrecontinual_equations.plot import LinePlot

# Create custom plot with multiple traces
plot = LinePlot()

# Add multiple solution curves
for i, solution in enumerate(solutions):
    x_data = solution.variables[0].discretization
    y_data = solution.results[0].discretization

    plot.figure.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="lines",
            name=f"Solution {i+1}",
            line=dict(width=2)
        )
    )

plot.figure.update_layout(
    title="Multiple Solutions Comparison",
    xaxis_title="Time",
    yaxis_title="Value"
)

# Custom save
plot.figure.write_image("comparison.png")
plot.figure.write_html("comparison.html")
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
        x=x_data,
        y=y_data,
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

The plot module integrates seamlessly with solver results:

```python
# Solve equation
solver.solve(equation, [1.0, 0.0, 0.0])

# Extract variables for plotting
t = solver.solution.time
x = solver.solution.variables[0]
y = solver.solution.variables[1]
z = solver.solution.variables[2]

# Create various plots
plot = LinePlot()

# Time series plots
plot.plot(t, x, "X vs Time")
plot.figure = go.Figure()  # Reset for new plot
plot.plot(t, y, "Y vs Time")

# Phase space plot
plot.figure = go.Figure()
plot.plot(x, y, "Phase Space: X vs Y")

# 3D trajectory
plot.figure = go.Figure()
plot.plot_3d(x, y, z, "3D Trajectory")
```

## Performance Considerations

- **Large Datasets**: Use PNG for static plots, HTML for interactive exploration
- **Memory Usage**: Large HTML files may be slow to load
- **File Sizes**: PNG files are smaller but less flexible
- **Batch Processing**: Reuse LinePlot instances for multiple plots

## Error Handling

```python
try:
    plot.plot_3d(x, y, z, "Trajectory")
except Exception as e:
    print(f"Plotting failed: {e}")
    # Fallback to 2D plots
    plot.figure = go.Figure()
    plot.plot(x, y, "2D Projection")
```

## Dependencies

- **plotly**: Core plotting library
- **kaleido** (optional): For PNG export in headless environments

## File Naming

Plots are automatically saved with safe filenames:

```python
# Title: "Lorenz Attractor (σ=10, ρ=28, β=8/3)"
# Becomes: "Lorenz_Attractor_sigma10_rho28_beta83.png"
```

---

**Parent Module:** [DISCRECONTINUAL_EQUATIONS](../DISCRECONTINUAL_EQUATIONS.md)

**Related Modules:**
- [SOLVER](../SOLVER.md) - Provides solution data for plotting
