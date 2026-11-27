# Basic Usage

The examples on this page run in order, in one session.

## Defining Variables and Parameters

A `Variable` is a quantity that changes along a solution; a `Parameter` is a
constant fixed before the solve. Both carry a `name` and a plotting
`abbreviation`.

```python
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

# Create variables
x = Variable(name="position", abbreviation="x")
t = Variable(name="time", abbreviation="t")

# Create parameters
k = Parameter(value=1.0, name="decay constant", abbreviation="k")
```

## Creating Functions

The right-hand side $f(t, x)$ is a `DeterministicFunction` (a `StochasticFunction`
for an SDE). Its `eval(point, time)` returns the derivative as a list, and it reads
its parameters from `self.parameters`.

```python
from discrecontinual_equations.function.deterministic import DeterministicFunction


class MyFunction(DeterministicFunction):
    def eval(self, point, time=None):
        return [-self.parameters[0].value * point[0]]  # dx/dt = -k*x


function = MyFunction(variables=[x], parameters=[k], results=[Variable(name="x'")], time=t)
```

## Solving Equations

A `DifferentialEquation` bundles the function with its variables, time and
parameters; a solver, configured with a time span and a step, returns a `Curve`.

```python
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.deterministic.euler.euler_config import EulerConfig
from discrecontinual_equations.solver.deterministic.euler.euler_solver import EulerSolver

# Create equation
equation = DifferentialEquation(variables=[x], time=t, parameters=[k], derivative=function)

# Configure solver
config = EulerConfig(start_time=0, end_time=10, step_size=0.1)
solver = EulerSolver(config)

# Solve
solution = solver.solve(equation, initial_values=[1.0])
print(len(solution.time.discretization), solution.results[0].discretization[-1])
```

## Fixed-step time grids

Every fixed-step solver (Euler, Adams-Bashforth, Euler-Maruyama, Milstein, the
stochastic Runge-Kutta family, Gottwald-Melbourne, Zhan-Duan-Li-Li and the
stochastic ensemble) steps on the grid its config's `times` gives: the nodes
`start_time + k * step_size` for `k = 0, ..., n`, with the last node equal to
`end_time`. The final state of a solve is the state at `end_time`.

The span `end_time - start_time` must be a whole number `n` of steps. Decimal
inputs such as `end_time=10, step_size=0.005` are whole numbers of steps only up
to binary rounding ([PythonDocs2026](../references.md#deterministic-solvers)),
and those are accepted. A span that is a real fraction of a step off, such as
`end_time=0.8, step_size=0.015` (53 1/3 steps), is refused with a `ValueError`
when the grid is first read:

```python
from discrecontinual_equations.solver.deterministic.euler.euler_config import EulerConfig

config = EulerConfig(start_time=0.0, end_time=10.0, step_size=0.005)
times = config.times
assert len(times) == 2001 and times[-1] == 10.0
assert config.n_steps == 2000

try:
    EulerConfig(start_time=0.0, end_time=0.8, step_size=0.015).times
except ValueError as error:
    print(error)  # ... is not a whole number of steps of 0.015 ...
```

Why refuse rather than end with a shorter last step: each solver takes the same
`dt` every step. The Adams-Bashforth coefficients assume an equidistant grid
([HairerNorsettWanner1993](../references.md#deterministic-solvers), §III.1), the
Wiener increments are drawn with variance `dt`, and a fixed `BrownianPath` is read
a whole number of its cells at a time, so a shorter last step would be a
different scheme. Pick `end_time = start_time + n * step_size`, or a step that
divides the span.

Limits and history:

- The check accepts a gap between the span and `n * step_size` of up to `1e-9`
  of the larger of `|start_time|` and `|end_time|`
  (`solver_config.GRID_TOLERANCE`), and never more than `1e-3` of a step
  (`solver_config.GRID_STEP_FRACTION`): far from zero the first bound alone
  would exceed the step, so that at `start_time = 1e6` with `step_size = 0.001`
  a span half a step off would be accepted with a last step 1.5 times the others.
- `start_time == end_time` gives the one-node grid `[start_time]`; a step that is
  not positive and finite, or an `end_time` before `start_time`, is refused.
- Before DEQ-129 the grid was `np.arange(start, end + h, h)`, whose length
  rounding decides ([NumPyDevelopers2026](../references.md#deterministic-solvers)):
  `(0, 0.8, 0.015)` ran to 0.81, and the whole spans `(0, 10, 0.005)` and
  `(0, 20, 0.01)` took one step past the end. `AdamsBashforth3Config` with a
  `step_size` used `np.arange(start, end, h)` and stopped a step short. Solves
  whose old grid had the right number of nodes are unchanged to rounding.

## Plotting Results

`LinePlot` writes one result against time. It writes PNG through `kaleido` by
default; `output_format="html"` needs nothing extra.

```python
from discrecontinual_equations.plot.line_plot import LinePlot

plot = LinePlot(output_dir="plots", output_format="html")
plot.plot(x=solution.time, y=solution.results[0], title="Decay")
```
