
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
- **Stochastic Runge-Kutta**: SRK2 (strong order 1.0), SRK3 (strong order 1.5, scalar noise), SRK4 (weak order 2.0, scalar noise); SRK5 refuses to run - see DEQ-25

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

A harmonic oscillator, $x'' = -kx$, written as the first-order system
$x' = v$, $v' = -kx$ and solved with the forward Euler method:

```python
import math

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.solver.deterministic.euler.euler_config import EulerConfig
from discrecontinual_equations.solver.deterministic.euler.euler_solver import EulerSolver
from discrecontinual_equations.variable import Variable


class HarmonicOscillator(DeterministicFunction):
    def eval(self, point, time=None):
        position, velocity = point
        return [velocity, -self.parameters[0].value * position]  # [dx/dt, dv/dt]


x = Variable(name="Position", abbreviation="x")
v = Variable(name="Velocity", abbreviation="v")
t = Variable(name="Time", abbreviation="t")
k = Parameter(name="Spring Constant", value=1.0)
oscillator = HarmonicOscillator(
    variables=[x, v], parameters=[k], results=[Variable(name="x'"), Variable(name="v'")], time=t
)
equation = DifferentialEquation(variables=[x, v], time=t, parameters=[k], derivative=oscillator)

config = EulerConfig(start_time=0.0, end_time=10.0, step_size=0.01)
solution = EulerSolver(config).solve(equation, [1.0, 0.0])  # x(0) = 1, v(0) = 0
position = solution.results[0].discretization
print(f"x(10) = {position[-1]:.4f}, exact {math.cos(10.0):.4f}")
```

prints `x(10) = -0.8823, exact -0.8391`: forward Euler is first order, and on an
oscillator it slowly gains energy.

### Solving SDEs

Geometric Brownian motion $dX = \mu X\,dt + \sigma X \circ dW$ in the
Stratonovich sense, with the two-stage stochastic Runge-Kutta scheme:

```python
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.solver.stochastic.srk2.srk2_config import SRK2Config
from discrecontinual_equations.solver.stochastic.srk2.srk2_solver import SRK2Solver


class GeometricBrownianMotion(StochasticFunction):
    def eval(self, point, time=None):  # the drift
        return [self.parameters[0].value * point[0]]

    def diffusion(self, point, time=None):  # the noise coefficient
        return [self.parameters[1].value * point[0]]


X = Variable(name="Price", abbreviation="X")
mu, sigma = Parameter(name="mu", value=0.1), Parameter(name="sigma", value=0.2)
gbm = GeometricBrownianMotion(
    variables=[X], parameters=[mu, sigma], results=[Variable(name="dX")], time=t
)
sde = DifferentialEquation(variables=[X], time=t, parameters=[mu, sigma], derivative=gbm)

config = SRK2Config(
    start_time=0.0, end_time=1.0, step_size=0.001,
    calculus="stratonovich", random_seed=42,
)
path = SRK2Solver(config).solve(sde, [1.0])
print(f"X(1) = {path.results[0].discretization[-1]:.4f}")
```

prints `X(1) = 0.7036`. Run again with the same `random_seed` and the
path repeats.

### Advanced Plotting

Continuing the ODE example: the oscillator's phase portrait, and the power
spectrum of its position.

```python
from discrecontinual_equations.plot.phase_plot import PhasePlot
from discrecontinual_equations.plot.spectrum_plot import SpectrumPlot

# Phase space analysis: position against velocity
phase_plot = PhasePlot(output_dir="./plots", output_format="html")
phase_plot.plot(solution.results[0], solution.results[1], title="Phase Portrait")

# Frequency analysis: one-sided power spectrum, in squared signal units
spectrum_plot = SpectrumPlot(output_dir="./plots", output_format="html")
spectrum_plot.plot_fft(solution.results[0], dt=0.01, title="Power Spectrum")
peak = max(spectrum_plot.figure.data[0].y)
print(f"peak power {peak:.3f}")
```

writes both figures as HTML under `./plots` (the default `output_format="png"`
needs kaleido), prints where it saved the spectrum, then `peak power 0.355`. A sine
of amplitude $A$ on a frequency bin would peak at its mean square $A^2/2$, about
0.5 here; the oscillator's $1/2\pi \approx 0.159$ Hz falls between bins 0.1 Hz
apart, so its power spreads over the neighbours.

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
    └── SRK2-4 (measured orders: strong 1.0, strong 1.5, weak 2.0); SRK5 refused
```

### Calculus Interpretations

**Ito SDEs**: `dX = μ(X,t)dt + σ(X,t)dW`
- Standard for financial modeling
- No drift correction needed

**Stratonovich SDEs**: `dX = μ(X,t)dt + σ(X,t) ∘ dW`
- Equivalent to Ito with modified drift: `μ_Ito = μ + (1/2)σ ∂σ/∂x`
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
