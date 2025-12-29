# GOTTWALD_MELBOURNE

## Overview

The Gottwald-Melbourne method is a deterministic homogenisation approach for solving stochastic differential equations (SDEs) driven by α-stable Lévy processes with non-Lipschitz coefficients. It uses the Thaler map to generate α-stable increments through deterministic fast-slow dynamics, avoiding direct simulation of heavy-tailed Lévy noise.

## Architecture

```
GottwaldMelbourneSolver
├── Config: alpha, eta, beta, epsilon, start_time, end_time, step_size, random_seed
├── Method: Deterministic homogenisation with Thaler map
├── Order: Weak convergence to Marcus SDE solution
└── Range: α ∈ (1,2), β ∈ [-1,1]
```

## Executive Summary

**Purpose**: Solve Marcus SDEs with α-stable Lévy noise and non-Lipschitz coefficients
**Key Features**: Deterministic noise generation, handles non-Lipschitz terms, preserves natural boundaries
**Performance**: Computationally intensive due to fast-slow coupling
**Use Cases**: Heavy-tailed noise processes, systems with natural boundaries, non-Lipschitz SDEs

## Core Classes

### GottwaldMelbourneConfig

```python
class GottwaldMelbourneConfig(StochasticConfig):
    """Configuration for Gottwald-Melbourne homogenisation method."""

    alpha: float = 1.5      # Stability parameter (1 < α < 2)
    eta: float = 1.0        # Scale parameter
    beta: float = 0.0       # Skewness parameter (-1 ≤ β ≤ 1)
    epsilon: float = 0.01   # Homogenisation parameter

    @property
    def gamma(self) -> float:
        return 1.0 / self.alpha
```

### GottwaldMelbourneSolver

```python
class GottwaldMelbourneSolver(Solver):
    """Gottwald-Melbourne method for α-stable SDEs."""

    def __init__(self, solver_config: GottwaldMelbourneConfig)

    def solve(self, equation: DifferentialEquation, initial_values: list[float]):
        """Solve SDE using Gottwald-Melbourne homogenisation."""

    def _thaler_map(self, x: float) -> float:
        """Compute Thaler map T(x)."""

    def _compute_observable(self, x: float, chi: int) -> float:
        """Compute observable v(x) for noise generation."""
```

## Mathematical Foundation

The Gottwald-Melbourne method approximates Marcus SDEs of the form:

```
dZ = a(Z) dt + b(Z)  dW^α,η,β
```

where `W^α,η,β` is an α-stable Lévy process, using a deterministic fast-slow system:

```
z_{n+1} = z_n + ε a(z_n) + ε^γ b(z_n) v^{(n)}
x_{n+1} = T(x_n)
```

Here `T` is the Thaler map, `v^{(n)}` is an observable generating α-stable increments, and `ε` is a small homogenisation parameter.

### Thaler Map

The Thaler map is defined as:

```
T(x) = [x^{1-γ} + (1+x)^{1-γ} - 1]^{1/(1-γ)} mod 1
```

where `γ = 1/α`. It has an indifferent fixed point at x=0 and generates intermittent dynamics that produce α-stable statistics.

### Convergence

As `ε → 0`, the slow variable `z_n` converges weakly to the solution of the Marcus SDE in the Skorohod topology.

## Examples

### Basic α-Stable Diffusion

```python
from discrecontinual_equations.solver.stochastic.gottwald_melbourne import GottwaldMelbourneConfig, GottwaldMelbourneSolver

# dZ = b(Z)  dW^α,η,β (pure diffusion)
class AlphaStableDiffusion(StochasticFunction):
    def eval(self, point, time=None):
        return [0.0]  # No drift

    def diffusion(self, point, time=None):
        return [1.0]  # Constant diffusion

config = GottwaldMelbourneConfig(
    alpha=1.5, eta=1.0, beta=0.0, epsilon=0.01,
    start_time=0, end_time=1, step_size=0.01,
    random_seed=42
)

solver = GottwaldMelbourneSolver(config)
solver.solve(equation, [0.0])  # Start at 0
```

### Non-Lipschitz SDE with Natural Boundary

```python
# dZ = Z^3 dt + Z^2  dW^α,η,β (natural boundary at Z=0)
class BoundarySDE(StochasticFunction):
    def eval(self, point, time=None):
        z = point[0]
        return [-z**3]  # Drift with natural boundary

    def diffusion(self, point, time=None):
        z = point[0]
        return [z**2]   # Non-Lipschitz diffusion

config = GottwaldMelbourneConfig(
    alpha=1.5, eta=0.5, beta=0.5, epsilon=0.005,
    start_time=0, end_time=2, step_size=0.001
)

solver = GottwaldMelbourneSolver(config)
solver.solve(equation, [0.1])  # Start near boundary
```

## Algorithm Details

### Fast-Slow Coupling

```python
def solve(self, equation, initial_values):
    z = initial_values[0]  # Slow variable
    x = 0.5               # Fast variable on [0,1]
    chi = ±1              # Skewness indicator

    for each time step:
        for _ in range(n_steps_per_epsilon):
            v = self._compute_observable(x, chi)
            drift = equation.eval([z], t)
            diffusion = equation.diffusion([z], t)

            z += epsilon * drift + epsilon**gamma * diffusion * v
            x = self._thaler_map(x)

            if x > x_star:  # In hyperbolic region
                chi = sample ±1 with bias beta
```

### Observable Construction

The observable `v(x)` generates α-stable increments:

```python
def _compute_observable(self, x, chi):
    if x <= x_star:
        v_tilde = 1.0
    else:
        v_tilde = (1 - 2**(gamma-1))**(-1) * chi

    return eta * d_alpha**(-gamma) * v_tilde
```

## Algorithm Details

### Fast-Slow Coupling

```python
def solve(self, equation, initial_values):
    z = initial_values[0]  # Slow variable
    x = 0.5               # Fast variable on [0,1]
    chi = ±1              # Skewness indicator

    for each time step:
        for _ in range(n_steps_per_epsilon):
            v = self._compute_observable(x, chi)
            drift = equation.eval([z], t)
            diffusion = equation.diffusion([z], t)

            z += epsilon * drift + epsilon**gamma * diffusion * v
            x = self._thaler_map(x)

            if x > x_star:  # In hyperbolic region
                chi = sample ±1 with bias beta
```

### Observable Construction

The observable `v(x)` generates α-stable increments:

```python
def _compute_observable(self, x, chi):
    if x <= x_star:
        v_tilde = 1.0
    else:
        v_tilde = (1 - 2**(gamma-1))**(-1) * chi

    return eta * d_alpha**(-gamma) * v_tilde
```

## Performance Characteristics

| Aspect | Performance |
|--------|-------------|
| **Computational Cost** | High (O(1/ε) per time step) |
| **Memory Usage** | Moderate |
| **Accuracy** | Weak convergence |
| **Stability** | Numerically stable |
| **Parallelization** | Limited by deterministic coupling |

## Advantages over Traditional Methods

1. **Handles Non-Lipschitz Coefficients**: No taming or truncation needed
2. **Preserves Natural Boundaries**: Correctly enforces physical constraints
3. **Deterministic Noise Generation**: Avoids sampling heavy tails directly
4. **Marcus Interpretation**: Naturally implements the correct stochastic calculus

## Limitations

1. **Computational Intensity**: Requires many fast iterations per slow step
2. **Parameter Range**: Restricted to α ∈ (1,2)
3. **Convergence Rate**: Unknown rigorous rates
4. **Single Dimension**: Currently implemented for 1D only

## References

- Gottwald, G.A. & Melbourne, I. (2020). "Simulation of non-Lipschitz α-stable stochastic differential equations: a method based on deterministic homogenisation"
- Thaler, M. (1980). "Estimates of the invariant densities of endomorphisms with indifferent fixed points"
- Applebaum, D. (2009). "Lévy Processes and Stochastic Calculus"

---

**Parent Module:** [STOCHASTIC](../STOCHASTIC.md)

**Related Modules:**
- [EULER_MARUYAMA](../euler_maruyama/EULER_MARUYAMA.md) - Standard Brownian motion solver
- [MILSTEIN](../milstein/MILSTEIN.md) - Higher-order Brownian motion method
