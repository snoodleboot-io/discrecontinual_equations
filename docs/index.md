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
from discrecontinual_equations.solver.deterministic.euler import EulerConfig, EulerSolver
from discrecontinual_equations.variable import Variable
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.function import Function

# Define your differential equation
class MyODE(Function):
    def eval(self, point, time=None):
        return [-point[0]]  # dx/dt = -x (exponential decay)

# Set up and solve
config = EulerConfig(start_time=0, end_time=5, step_size=0.1)
solver = EulerSolver(config)

# Get results
solution = solver.solve(equation, [1.0])  # Initial condition x(0) = 1
```

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
