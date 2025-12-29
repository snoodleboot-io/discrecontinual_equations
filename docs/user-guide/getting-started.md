# Getting Started

Welcome to Discrecontinual Equations! This guide will help you get started with solving differential equations using our library.

## Installation

Install the package using pip:

```bash
pip install discrecontinual-equations
```

## Quick Start

Here's a simple example of solving an ODE using the Euler method:

```python
from discrecontinual_equations import DifferentialEquation, Variable, Parameter
from discrecontinual_equations.solver.deterministic.euler import EulerConfig, EulerSolver

# Define variables and parameters
x = Variable(name="position")
t = Variable(name="time")
k = Parameter(value=1.0)

# Create equation: dx/dt = -k*x (exponential decay)

# Solve using Euler method
config = EulerConfig(start_time=0, end_time=5, step_size=0.1)
solver = EulerSolver(config)

initial_values = [1.0]  # x(0) = 1
solution = solver.solve(equation, initial_values)
