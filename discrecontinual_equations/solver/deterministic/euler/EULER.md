# EULER

## Overview

The Euler method is the fundamental first-order numerical algorithm for solving ordinary differential equations. It represents the simplest approach to numerical integration, using forward differences to approximate derivatives.

## Architecture

```
EulerSolver
├── Config: start_time, end_time, step_size
├── Method: First-order explicit forward Euler
├── Order: 1st order, O(h)
└── Stability: Conditionally stable
```

## Executive Summary

**Purpose**: Basic ODE solver for educational and baseline comparisons
**Key Features**: Simple implementation, explicit formulation, foundation method
**Performance**: Fast execution, acceptable for rough approximations
**Use Cases**: Educational examples, baseline accuracy comparisons, simple problems

## Core Classes

### EulerConfig

```python
from discrecontinual_equations.solver.solver_config import DeterministicConfig
from discrecontinual_equations.solver.deterministic.euler.euler_config import EulerConfig

config = EulerConfig(start_time=0.0, end_time=1.0, step_size=0.25)
assert isinstance(config, DeterministicConfig)
print(config.times.tolist())  # [0.0, 0.25, 0.5, 0.75, 1.0]
```

### EulerSolver

```python
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.solver.deterministic.euler.euler_solver import EulerSolver

# Built from an EulerConfig; solve(equation, initial_values) returns a Curve.
assert issubclass(EulerSolver, Solver)
```

## UML Class Diagram

```mermaid
classDiagram
    class Solver {
        +solution: Curve | None
        +solve(equation, initial_values)*
    }

    class DeterministicSolver {
        +config: DeterministicConfig
    }

    class FixedStepSolver {
        +step_size: float
    }

    class EulerSolver {
        +solve(equation, initial_values)
    }

    Solver <|-- DeterministicSolver
    DeterministicSolver <|-- FixedStepSolver
    FixedStepSolver <|-- EulerSolver
```

## Sequence Diagram - Euler Step

```mermaid
sequenceDiagram
    participant Solver
    participant Equation
    participant Curve

    Solver->>Equation: eval(current_state, time) - get f(t,y)
    Solver->>Solver: Compute y_new = y + h * f(t,y)
    Solver->>Curve: append(y_new, time + h)
    Solver->>Solver: Update time = time + h
```

## Folder Structure

```
euler/
├── __init__.py
├── euler_config.py
├── euler_solver.py
└── EULER.md
```

## Examples

### Exponential Decay

```python
import math

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


# dy/dt = -k*y
class ExponentialDecay(DeterministicFunction):
    def eval(self, point, time=None):
        y = point[0]
        k = self.parameters[0].value
        return [-k * y]


y, t, k = Variable(name="y"), Variable(name="t"), Parameter(value=1.0, name="k")
decay = ExponentialDecay([y], [k], [Variable(name="dy")], t)
equation = DifferentialEquation(variables=[y], time=t, parameters=[k], derivative=decay)

config = EulerConfig(start_time=0, end_time=5, step_size=0.1)
solver = EulerSolver(config)
solver.solve(equation, [1.0])  # y(0) = 1
final = solver.solution.results[0].discretization[-1]
print(f"y(5) = {final:.5f}, exact {math.exp(-5):.5f}")  # y(5) = 0.00515, exact 0.00674
```

### Harmonic Oscillator

```python
# d²x/dt² = -ω²x (system of 2 first-order ODEs)
class HarmonicOscillator(DeterministicFunction):
    def eval(self, point, time=None):
        x, v = point  # position, velocity
        omega = self.parameters[0].value
        return [v, -omega**2 * x]  # [dx/dt, dv/dt]


x, v, omega = Variable(name="x"), Variable(name="v"), Parameter(value=1.0, name="omega")
oscillator = HarmonicOscillator([x, v], [omega], [Variable(name="dx"), Variable(name="dv")], t)
equation = DifferentialEquation(variables=[x, v], time=t, parameters=[omega], derivative=oscillator)

config = EulerConfig(start_time=0, end_time=10, step_size=0.01)
solver = EulerSolver(config)
solver.solve(equation, [1.0, 0.0])  # x=1, v=0
x10, v10 = (r.discretization[-1] for r in solver.solution.results)
print(f"energy {0.5 * (x10**2 + v10**2):.4f}, exact 0.5")  # 0.5526: Euler gains energy
```

## Functionality Explanation

### Mathematical Foundation

For an ODE `dy/dt = f(t,y)` with initial condition `y(t₀) = y₀`, the Euler method approximates:

```
y_{n+1} = y_n + h * f(t_n, y_n)
```

Where `h` is the step size and `t_{n+1} = t_n + h`.

### Convergence Properties

- **Order**: 1st order - `|y(t) - ŷ(t)| = O(h)`
- **Local Error**: O(h²) per step
- **Global Error**: O(h) over the interval
- **Consistency**: Method is consistent with the ODE

### Stability Analysis

- **Region of Stability**: the disc `|1 + hλ| ≤ 1`, centred at `hλ = -1` with radius 1 (only part of the left half-plane)
- **Dahlquist Barrier**: Cannot be A-stable beyond order 2
- **Practical Stability**: Requires small step sizes for stiff problems

## Algorithm Details

### Single Time Step

```python
import numpy as np


def euler_step(y, t, h, equation):
    """One Euler step: the field at the start of the step, (t_n, y_n)."""
    derivative = equation.derivative.eval(point=list(y), time=t)
    return y + h * np.array(derivative)
```

### Complete Solving Process

```python
# The whole solve on the decay equation, by hand: step along the config's grid.
decay_equation = DifferentialEquation(variables=[y], time=t, parameters=[k], derivative=decay)
grid = EulerConfig(start_time=0, end_time=5, step_size=0.1).times
state = np.array([1.0])
for t_n, t_next in zip(grid[:-1], grid[1:]):
    state = euler_step(state, t_n, t_next - t_n, decay_equation)

print(f"{state[0]:.5f}")  # 0.00515, as EulerSolver gave
```

## Error Analysis

### Local Truncation Error

The local error arises from truncating the Taylor series:
```
y(t+h) = y(t) + h y'(t) + (h²/2) y''(t) + O(h³)
ŷ(t+h) = y(t) + h f(t,y(t))
```

Local error: `τ(h) = (h²/2) y''(t) + O(h³) = O(h²)`

### Global Error Accumulation

For smooth problems, global error accumulates as:
```
|y(t_N) - ŷ(t_N)| ≤ (h/2) * max|y''(t)| * T + O(h²)
```

Where T is the total integration time.

## Performance Characteristics

| Aspect | Euler Method |
|--------|--------------|
| **Computational Cost** | O(N) function evaluations |
| **Memory Usage** | Minimal O(1) |
| **Accuracy** | Basic O(h) global error |
| **Stability** | Conditionally stable |
| **Implementation** | Very simple |

## Applications

### Educational Use
- **Teaching**: Introduction to numerical methods
- **Understanding**: Basic discretization concepts
- **Comparison**: Baseline for more advanced methods

### Simple Physical Systems
- **Free fall**: `dv/dt = -g` (no air resistance)
- **Radioactive decay**: `dN/dt = -λN`
- **RC circuits**: Simple electrical circuits

### Baseline Comparisons
- **Method comparison**: Reference for accuracy studies
- **Step size studies**: Demonstrate convergence behavior
- **Stability analysis**: Show limitations of explicit methods

## Limitations and Considerations

### Accuracy Limitations
- **Low order**: Requires small step sizes for accuracy
- **Error accumulation**: Global error grows linearly with T
- **Stiff problems**: May require impractically small steps

### Stability Issues
- **Explicit method**: Cannot handle very stiff problems
- **Step size restriction**: CFL-like condition for stability
- **Nonlinear problems**: May require adaptive stepping

## References

- Euler, Leonhard (1768). "Institutionum calculi integralis"
- Butcher, J.C. (2008). "Numerical Methods for Ordinary Differential Equations"
- Hairer, E., et al. (1993). "Solving Ordinary Differential Equations I"

---

**Parent Module:** [DETERMINISTIC](../DETERMINISTIC.md)

**Related Modules:**
- [ADAMS_BASHFORTH2](../adams_bashforth2/ADAMS_BASHFORTH2.md) - Higher-order multistep method
- [`CashKarpSolver`](../cash_karp/cash_karp_solver.py) - Adaptive Runge-Kutta method
