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
several components. `EulerMaruyamaSolver` takes any `G`; so does `MilsteinSolver`,
at strong order 1.0: when the columns do not commute it draws the Levy areas
between drivers (`LevyAreaSampler`, DEQ-36) and warns once about their cost; the
`srk*` solvers take the vector form only.

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

### Quasipotential and minimum action paths

How a planar system gets from one attractor to another under small noise is
answered by the Freidlin-Wentzell quasipotential `U`: escapes cost
`exp(-U / eps)`, and the minimiser of the action is the most probable path.
`OrderedLineIntegral` computes `U` on a `DensityGrid` by the ordered line
integral method (OLIM-MID), for a drift `Function` and a `NoiseMatrix`, from a
stable equilibrium or a limit cycle:

```python
from discrecontinual_equations.continuation import (
    DensityGrid, EquilibriumAttractor, GeometricMinimumAction, OrderedLineIntegral,
)

grid = DensityGrid.box([-2.0, -1.5], [1.0, 1.5], [161, 161])
solver = OrderedLineIntegral(drift, noise, grid)
field = solver.solve(EquilibriumAttractor([-1.0, 0.0]))
field.values                                   # U on the grid, shaped like it
path = solver.minimum_action_path(field, [-0.3, 0.6], end=[-1.0, 0.0])
path.states, path.action                       # back along -(f + D grad U)

between = GeometricMinimumAction(drift, noise).path(initial_chain)  # across a saddle
```

`U` is normalised as the rest of the library writes `dx = f dt + G dW`:
`D = G G^T`, so a gradient drift `-grad V` with `G = I` has `U = 2 V`, and the
grid density of `StationaryFokkerPlanck` is `exp(-U)` to leading order. The
method is planar and time-independent and needs `D` non-singular at every grid
point; it is first order in the spacing (measured, and the bound the method's
authors give). `GeometricMinimumAction` needs a constant `D`. The module
docstring of `continuation/quasipotential.py` has the measured errors.

## Custom Solvers

Extend the base `Solver` class to implement custom numerical methods.

## Plotting Results

```python
from discrecontinual_equations.plot.line_plot import LinePlot

plot = LinePlot()
plot.plot(x=solution.time, y=solution.results[0])
