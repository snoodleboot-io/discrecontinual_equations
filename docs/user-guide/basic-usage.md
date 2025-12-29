# Basic Usage

## Defining Variables and Parameters

```python
from discrecontinual_equations import Variable, Parameter

# Create variables
x = Variable(name="position", abbreviation="x")
t = Variable(name="time", abbreviation="t")

# Create parameters
k = Parameter(value=1.0, name="decay constant")
```

## Creating Functions

```python
from discrecontinual_equations.function.function import Function

class MyFunction(Function):
    def eval(self, point, time=None):
        return [-self.parameters[0].value * point[0]]  # dx/dt = -k*x
```

## Solving Equations

```python
from discrecontinual_equations import DifferentialEquation
from discrecontinual_equations.solver.deterministic.euler import EulerConfig, EulerSolver

# Create equation
equation = DifferentialEquation(...)

# Configure solver
config = EulerConfig(start_time=0, end_time=10, step_size=0.1)
solver = EulerSolver(config)

# Solve
solution = solver.solve(equation, initial_values=[1.0])
