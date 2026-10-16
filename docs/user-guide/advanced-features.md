# Advanced Features

## Stochastic Differential Equations

```python
from discrecontinual_equations.function.stochastic import StochasticFunction

class MySDE(StochasticFunction):
    def eval(self, point, time=None):
        return [-self.alpha * point[0]]  # Drift term

    def diffusion(self, point, time=None):
        return [self.sigma]  # Diffusion term
```

A vector from `diffusion()` is diagonal noise: one Wiener process per component.
Return an `n x m` nested list instead and the noise is a full matrix `G` with `m`
independent drivers, stepped as `G dW` - a rotation, a shear, one driver shared by
several components. `EulerMaruyamaSolver` takes any `G`; `MilsteinSolver` takes a
`G` whose columns commute and refuses one that does not; the `srk*` solvers take
the vector form only.

```python
class ConformalNoise(StochasticFunction):
    def eval(self, point, time=None):
        return [self.a * point[0], self.a * point[1]]

    def diffusion(self, point, time=None):
        x, y = point
        return [[self.sigma * x, -self.sigma * y], [self.sigma * y, self.sigma * x]]
```

## Custom Solvers

Extend the base `Solver` class to implement custom numerical methods.

## Plotting Results

```python
from discrecontinual_equations.plot.line_plot import LinePlot

plot = LinePlot()
plot.plot(x=solution.time, y=solution.results[0])
