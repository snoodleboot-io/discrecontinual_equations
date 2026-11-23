# GOTTWALD_MELBOURNE

## Overview

The Gottwald-Melbourne method is a deterministic homogenisation approach for solving stochastic differential equations (SDEs) driven by α-stable Lévy processes with non-Lipschitz coefficients. It uses the Thaler map to generate α-stable increments through deterministic fast-slow dynamics, avoiding direct simulation of heavy-tailed Lévy noise.

## Architecture

```
GottwaldMelbourneSolver
├── Config: alpha, eta, beta, epsilon, start_time, end_time, step_size, random_seed
├── Method: Deterministic homogenisation with Thaler map
├── Order: Weak convergence to Marcus SDE solution
└── Range: α ∈ (1,2), β ∈ [-1,1], 0 < ε ≤ step_size
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

As `ε → 0`, the slow variable `z_n` converges weakly to the solution of the Marcus SDE in the Skorohod topology. The law of `ẑ_ε(1)` on `dZ = dW` and on `dZ = -Z dt + dW` approaches the stable law at about Remark 2.6's indicative rate `n^{-(2/α-1)}`, `n = 1/ε`; the measurement is in the DEQ-108 verification.

Each fast step advances slow time by `ε`: the path at time `t` is `z_⌊t/ε⌋` (Theorem 4.1 of the paper). A grid step shorter than `ε` cannot be resolved, so `ε` may not exceed `step_size`, and both the config and the solver raise a `ValueError` naming the two values when it does. Before this check (DEQ-86) the solver still took one fast step per grid step, so the path ran `ε/step_size` times faster than its own clock: `ε = 1` on a grid of `1/20000` ran 20,000 units of slow time and overflowed in the z update.

The z update is an explicit step of length `ε` in the drift plus a bounded kick `ε^γ b(z) v`, so at finite `ε` it carries the usual explicit-step restriction on `ε |a'(z)|`, which depends on the equation and is not checked. The paper notes the spurious unstable fixed points this introduces at finite `ε` (Section 4.3); they move away as `ε → 0`.

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

The scheme is that of Gottwald & Melbourne (arXiv:2004.09914, MMS 19(2), 2021), Sections 2, 4 and 5. With `γ = 1/α`:

| Piece | Formula | Paper |
|-------|---------|-------|
| `x*` | `x*^{1-γ} + (1+x*)^{1-γ} = 2` | (2.2) |
| `g_α` | `Γ(1-α) cos(απ/2)` | (2.5) |
| `d_α` | `α^α (1-γ)/(2^{1-γ}-1) · g_α` | (2.4) |
| `v(x)` | `η d_α^{-γ} (1-2^{γ-1})^{-γ} ṽ(x)`, `ṽ = 1` on `[0, x*]`, `ṽ = (1-2^{1-γ})^{-1}` on `(x*, 1]` | §4.1 |
| sign `χ` | a fresh `δ = ±1`, `P(δ = ±1) = (1 ± β)/2`, on each entry to `(x*, 1]`, signing that hyperbolic step and the laminar steps after it | §4.1, proof of Thm 4.1 |
| `x_0` | `T^10000 x_0'`, `x_0'` uniform | Remark 4.4 |
| first sign | the excursion under way at `x_0` gets a `δ` of its own | §5.3 (`µ_Y × λ`) |
| clock | the value at grid time `t_i` is `z_{n_i}`, `n_i = ⌊(t_i - t_0)/ε⌋` | Thm 4.1 |
| perturbation | `N(0, 10^-20)` added to `x` mod 1 on entering `(x*, 1]`, at least `10^4` iterations after the last | §4.2 |

`ṽ` is negative on `(x*, 1]`, so `v` has mean zero under the invariant measure. Section 4.1 prints the sign as the product `χ^(n) = χ_{n-1}···χ_0`; taken literally, that product gives the symmetric law whatever `β` (its marginal is ±1 with probability ½ each). The proof of Theorem 4.1 (Section 5.3) gives each excursion its own `δ` (the induced observable `V(y, ±1)`), and that is what the solver does: it converges to the requested `β` (DEQ-108). Read literally, the empty product also gives the excursion under way at `x_0` the sign +1 on every path; that does not change the limit, but at finite `ε` it biases `ẑ_ε(1)` upwards (median +0.53 at `ε = 0.01` on `dZ = dW`, `β = 0`), so the solver signs it with a `δ` like any other.

```python
def solve(self, equation, initial_values):
    x = T^10000(uniform)          # Remark 4.4
    chi = delta()
    for n in range(n_last):       # n_i = floor((t_i - t_0) / eps) at grid time t_i
        if x > x_star:
            chi = delta()         # a new excursion, a new sign
        v = chi * v(x)            # both branches
        t_n = t_0 + n * eps       # the drift's time is the fast step's own
        z += eps * a(z, t_n) + eps**gamma * b(z, t_n) * v
        x = T(x)                  # perturbed per Section 4.2
```

Before DEQ-108 the solver dropped the `1/(2^{1-γ}-1)` and `(1-2^{γ-1})^{-γ}` factors, divided by `g_α`, gave `ṽ` the wrong sign on `(x*, 1]`, signed only that branch, started every path at `x_0 = 0.5`, took `int(h/ε)` fast steps per grid step and had no perturbation. Its observable had nonzero mean, so the path drifted: on `dZ = dW` (`α = 1.5`, `β = 0`) the median of `ẑ_ε(1)` was +2.2, +6.7 and +22 at `ε = 0.1`, `0.01` and `0.001`.

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
3. **Convergence Rate**: No rigorous rate; Remark 2.6 indicates `n^{-(2/α-1)}`, slow as `α → 2`
4. **Single Dimension**: Currently implemented for 1D only
5. **Start-up**: each path first iterates the map 10,000 times (Remark 4.4)

## References

- Gottwald, G.A. & Melbourne, I. (2021). "Simulation of non-Lipschitz stochastic differential equations driven by α-stable noise: a method based on deterministic homogenisation", Multiscale Model. Simul. 19(2); arXiv:2004.09914
- Thaler, M. (1980). "Estimates of the invariant densities of endomorphisms with indifferent fixed points"
- Applebaum, D. (2009). "Lévy Processes and Stochastic Calculus"

---

**Parent Module:** [STOCHASTIC](../STOCHASTIC.md)

**Related Modules:**
- [EULER_MARUYAMA](../euler_maruyama/EULER_MARUYAMA.md) - Standard Brownian motion solver
- [MILSTEIN](../milstein/MILSTEIN.md) - Higher-order Brownian motion method
