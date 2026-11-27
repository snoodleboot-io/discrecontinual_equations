# Getting Started

Welcome to Discrecontinual Equations! This guide solves a first differential
equation with the library.

## Installation

Install the package using pip:

```bash
pip install discrecontinual-equations
```

## Where things are imported from

The package root holds only the version. Every class is imported from its own
module, as the examples on every page do:

| Name | Import from |
|---|---|
| `Variable` | `discrecontinual_equations.variable` |
| `Parameter` | `discrecontinual_equations.parameter` |
| `DifferentialEquation` | `discrecontinual_equations.differential_equation` |
| `DeterministicFunction` | `discrecontinual_equations.function.deterministic` |
| `StochasticFunction` | `discrecontinual_equations.function.stochastic` |
| `Curve` (what a solve returns) | `discrecontinual_equations.curve` |
| `EulerConfig`, `EulerSolver` | `discrecontinual_equations.solver.deterministic.euler.euler_config`, `.euler_solver` |

## Quick Start

Exponential decay, $x' = -k x$ with $x(0) = 1$, has the exact solution
$x(t) = e^{-kt}$. Solve it with the forward Euler method and compare:

```python
import math

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.solver.deterministic.euler.euler_config import EulerConfig
from discrecontinual_equations.solver.deterministic.euler.euler_solver import EulerSolver
from discrecontinual_equations.variable import Variable


class Decay(DeterministicFunction):
    """The right-hand side f(t, x) = -k x."""

    def eval(self, point, time=None):
        k = self.parameters[0].value
        return [-k * point[0]]


# Variables, a parameter, and the right-hand side that uses them
x = Variable(name="position", abbreviation="x")
t = Variable(name="time", abbreviation="t")
k = Parameter(value=1.0, name="decay constant", abbreviation="k")
decay = Decay(variables=[x], parameters=[k], results=[Variable(name="x'")], time=t)
equation = DifferentialEquation(variables=[x], time=t, parameters=[k], derivative=decay)

# Solve from t = 0 to t = 5 in steps of 0.1
config = EulerConfig(start_time=0.0, end_time=5.0, step_size=0.1)
solution = EulerSolver(config).solve(equation, [1.0])  # x(0) = 1

final = solution.results[0].discretization[-1]
print(f"x(5) = {final:.5f}, exact {math.exp(-5.0):.5f}")
```

prints `x(5) = 0.00515, exact 0.00674`: Euler's method is first order, so halving
the step roughly halves that error.

The solution is a `Curve`: `solution.time.discretization` holds the time nodes and
`solution.results[0].discretization` the state at each. Next:

- [Basic Usage](basic-usage.md) - variables, parameters, functions and plots.
- [Deterministic solvers](deterministic-solvers.md) - every ODE method, its order
  and its limits.
