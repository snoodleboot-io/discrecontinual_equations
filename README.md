
# Discrecontinual Equations

A comprehensive Python library for solving differential equations using advanced numerical methods, with full support for both deterministic and stochastic systems.

## Features

### 🔢 **Deterministic ODE Solvers**
- **Euler Method**: Basic first-order method
- **Adams-Bashforth 2 & 3**: Multi-step explicit methods
- **Runge-Kutta Methods**: Cash-Karp, Dormand-Prince, Fehlberg variants
- **Adaptive Time Stepping**: Automatic step size control for precision

### 🎲 **Stochastic SDE Solvers**
- **Euler-Maruyama**: Fundamental stochastic method (strong order 0.5)
- **Milstein**: Higher-order with diffusion correction (strong order 1.0)
- **Stochastic Runge-Kutta**: SRK2, SRK3, SRK4, SRK5 (orders 1.0 to 2.5)

### 📊 **Calculus Support**
- **Ito Calculus**: Standard interpretation for SDEs
- **Stratonovich Calculus**: Alternative interpretation with drift corrections
- Automatic conversion between interpretations

### 📈 **Advanced Plotting**
- **LinePlot**: Time series and general 2D/3D plotting
- **PhasePlot**: 2D phase space diagrams
- **DelayPlot**: Delay embeddings and attractor reconstruction
- **SpectrumPlot**: Power spectra and frequency analysis

### 🎯 **Key Capabilities**
- Systems of equations (arbitrary dimension)
- Reproducible stochastic simulations (random seeds)
- Extensible architecture for custom solvers
- Interactive and static plot outputs (PNG/HTML)

## Installation

Install with pip or uv:

```bash
pip install discrecontinual-equations
```

or

```bash
uv add discrecontinual-equations
```

## Quick Start

### Solving ODEs

```python
from discrecontinual_equations import *
from discrecontinual_equations.solver.deterministic.euler import EulerConfig, EulerSolver

# Define variables and parameters
x = Variable(name="Position", abbreviation="x")
t = Variable(name="Time", abbreviation="t")
k = Parameter(name="Spring Constant", value=1.0)

# Create harmonic oscillator function
class HarmonicOscillator(DeterministicFunction):
    def eval(self, point, time=None):
        x_val, v_val = point  # position and velocity
        return [v_val, -self.parameters[0].value * x_val]  # [dx/dt, dv/dt]

# Set up and solve
config = EulerConfig(start_time=0, end_time=10, step_size=0.01)
solver = EulerSolver(config)
# ... solve equation
```

### Solving SDEs

```python
from discrecontinual_equations.solver.stochastic.srk2 import SRK2Config, SRK2Solver

# Geometric Brownian motion with Stratonovich calculus
config = SRK2Config(
    start_time=0, end_time=1, step_size=0.001,
    calculus="stratonovich", random_seed=42
)
solver = SRK2Solver(config)
# ... solve stochastic equation
```

### Advanced Plotting

```python
from discrecontinual_equations.plot import PhasePlot, SpectrumPlot

# Phase space analysis
phase_plot = PhasePlot(output_dir="./plots")
phase_plot.plot(x_variable, y_variable, title="Phase Portrait")

# Frequency analysis
spectrum_plot = SpectrumPlot(output_dir="./plots")
spectrum_plot.plot_fft(signal, dt=0.01, title="Power Spectrum")
```

## Architecture

### Core Components

**Variables**: Represent dependent variables with discretization and metadata
**Parameters**: Constants used in equations with validation
**Functions**: Define equation right-hand sides (deterministic/stochastic)
**Differential Equations**: Encapsulate complete equation systems
**Solvers**: Numerical integration algorithms with configuration
**Plots**: Visualization classes for different analysis types

### Solver Hierarchy

```
Solver (base)
├── Deterministic Solvers
│   ├── Euler, Adams-Bashforth, Runge-Kutta variants
│   └── Adaptive step size controllers
└── Stochastic Solvers
    ├── Euler-Maruyama (order 0.5)
    ├── Milstein (order 1.0)
    └── SRK2-5 (orders 1.0-2.5)
```

### Calculus Interpretations

**Ito SDEs**: `dX = μ(X,t)dt + σ(X,t)dW`
- Standard for financial modeling
- No drift correction needed

**Stratonovich SDEs**: `dX = μ(X,t)dt + σ(X,t) ∘ dW`
- Equivalent to Ito with modified drift: `μ_corrected = μ - (1/2)σ ∂σ/∂x`
- Natural for physical systems

## Examples

Comprehensive examples in the `examples/` directory:

- **lorenz_equation.py**: Lorenz attractor with multiple solvers
- **brownian_motion.py**: Standard Brownian motion simulation
- **geometric_brownian_motion.py**: Financial modeling with GBM
- **ccefs_example.py**: Coupled electric field sensors

## Documentation

Detailed documentation available in markdown files:
- `DISCRECONTINUAL_EQUATIONS.md`: Main library overview
- `SOLVER.md`: Complete solver documentation
- `FUNCTION.md`: Function definition guide
- `PLOT.md`: Plotting capabilities

## Contributing

The library uses an extensible architecture. New solvers can be added by:
1. Creating config and solver classes inheriting from base classes
2. Implementing the required methods
3. Adding appropriate docstrings and tests

## License

See LICENSE file for details.
