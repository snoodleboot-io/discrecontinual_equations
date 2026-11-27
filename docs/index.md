# Discrecontinual Equations

A comprehensive Python library for solving differential equations with advanced numerical methods.

## Features

- **Multiple Solvers**: Euler, Runge-Kutta, Adams-Bashforth, and stochastic methods
- **Stochastic Differential Equations**: Support for SDEs with various numerical schemes
- **Visualization Tools**: Interactive plotting with time series, phase portraits, and delay embeddings
- **Extensible Architecture**: Easy to add new solvers and equation types
- **Professional Documentation**: Complete API reference and user guides

## Quick Start

```python
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.solver.deterministic.euler.euler_config import EulerConfig
from discrecontinual_equations.solver.deterministic.euler.euler_solver import EulerSolver
from discrecontinual_equations.variable import Variable


# Define your differential equation: dx/dt = -x (exponential decay)
class MyODE(DeterministicFunction):
    def eval(self, point, time=None):
        return [-point[0]]


x, t = Variable(name="x"), Variable(name="t")
equation = DifferentialEquation(
    variables=[x], time=t, parameters=[], derivative=MyODE([x], [], [Variable(name="x'")], t)
)

# Set up and solve
config = EulerConfig(start_time=0, end_time=5, step_size=0.1)
solver = EulerSolver(config)

# Get results
solution = solver.solve(equation, [1.0])  # Initial condition x(0) = 1
print(solution.results[0].discretization[-1])  # x(5), near exp(-5)
```

Classes are imported from their modules; the package root holds only the
version. [Getting Started](user-guide/getting-started.md) lists where each lives.

## Installation

```bash
pip install discrecontinual-equations
```

## Documentation

- [Getting Started](user-guide/getting-started.md) - Basic usage tutorial
- [API Reference](api/core.md) - Complete API documentation
- [Examples](https://github.com/snoodleboot-io/discrecontinual_equations/tree/main/examples) - Working examples for different equation types

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
