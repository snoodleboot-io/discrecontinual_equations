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

### Ensembles of paths

For statistics - a stationary covariance, an escape rate, a residence-time
density - step thousands of paths at once with `StochasticEnsemble` instead of
looping over a per-path solver:

```python
from discrecontinual_equations.solver.stochastic.ensemble import StochasticEnsemble

config = EulerMaruyamaConfig(start_time=0, end_time=10, step_size=0.005, random_seed=3)
result = StochasticEnsemble(config).solve(equation, [0.0, 0.0], paths=20000, keep_every=200)
result.final      # (paths, n) at the end time
result.samples    # (kept, paths, n) every 200 steps
```

Every `StochasticFunction` works in an ensemble; one that overrides `eval_batch`
and `diffusion_batch` with array arithmetic over the `(paths, n)` argument runs
about a hundred times faster than the per-point default.

```python
class ConformalNoise(StochasticFunction):
    ...
    def eval_batch(self, points, time=None):
        return self.a * points

    def diffusion_batch(self, points, time=None):
        x, y = points[:, 0], points[:, 1]
        s = self.sigma
        return s * np.stack([np.stack([x, -y], 1), np.stack([y, x], 1)], 1)
```

## Custom Solvers

Extend the base `Solver` class to implement custom numerical methods.

## Plotting Results

```python
from discrecontinual_equations.plot.line_plot import LinePlot

plot = LinePlot()
plot.plot(x=solution.time, y=solution.results[0])
