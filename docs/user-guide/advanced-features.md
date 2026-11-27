# Advanced Features

The examples on this page run in order, in one session.

## Stochastic Differential Equations

An SDE $dx = f(x, t)\,dt + g(x, t)\,dW$ is a `StochasticFunction`: `eval` is the
drift $f$ and `diffusion` the noise amplitude $g$.

```python
import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config import (
    EulerMaruyamaConfig,
)
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_solver import (
    EulerMaruyamaSolver,
)
from discrecontinual_equations.variable import Variable


class MySDE(StochasticFunction):
    alpha = 1.0
    sigma = 0.3

    def eval(self, point, time=None):
        return [-self.alpha * point[0]]  # Drift term

    def diffusion(self, point, time=None):
        return [self.sigma]  # Diffusion term


def sde_equation(function_type, dimension):
    """Wrap a StochasticFunction class as an equation in ``dimension`` states."""
    t = Variable(name="t")
    states = [Variable(name=f"x{i}") for i in range(dimension)]
    function = function_type(states, [], [Variable() for _ in states], t)
    return DifferentialEquation(variables=states, time=t, parameters=[], derivative=function)


config = EulerMaruyamaConfig(start_time=0, end_time=1, step_size=0.01, random_seed=3)
path = EulerMaruyamaSolver(config).solve(sde_equation(MySDE, 1), [1.0])
print(len(path.time.discretization))  # 101 nodes
```

A vector from `diffusion()` is diagonal noise: one Wiener process per component.
Return an `n x m` nested list instead and the noise is a full matrix `G` with `m`
independent drivers, stepped as `G dW` - a rotation, a shear, one driver shared by
several components. `EulerMaruyamaSolver` takes any `G`; so does `MilsteinSolver`,
at strong order 1.0: when the columns do not commute it draws the Levy areas
between drivers (`LevyAreaSampler`, DEQ-36) and warns once about their cost; the
`srk*` solvers take the vector form only. `MilsteinConfig(levy_area_accuracy=c1)`
sets the areas' error bound `c1 dt^(3/2)` per step (default 1, the cheapest that
keeps order 1.0); at moderate `dt` the default measures a slope nearer 0.75, and
`0.05` measures 0.97 for some 15 times the area normals (DEQ-67).

```python
class ConformalNoise(StochasticFunction):
    a = -1.0
    sigma = 0.3

    def eval(self, point, time=None):
        return [self.a * point[0], self.a * point[1]]

    def diffusion(self, point, time=None):
        x, y = point
        return [[self.sigma * x, -self.sigma * y], [self.sigma * y, self.sigma * x]]
```

### Declaring diagonal noise

When every `b_i` depends on `x_i` (and time) alone, say so, and the solvers read the
slopes `db_i/dx_i` from two evaluations of `diffusion` instead of `2 n`; a Milstein
step at `n = 128` took 1.9 ms declared against 38 ms undeclared:

```python
class IndependentNoise(StochasticFunction):
    diagonal_noise = True
    sigma = 0.3

    def eval(self, point, time=None):
        return [-x for x in point]

    def diffusion(self, point, time=None):
        return [self.sigma * x for x in point]
```

The promise is checked once per solve, at the initial state, at a cost of
`1 + 2 n` evaluations: if some `b_i` reads another component there, you get a
`UserWarning` naming the function, and the solve runs as if nothing had been
declared. Only the initial state is checked, so a coupling that is zero there
(`x_0 * x_1` started at `x_0 = 0`) goes unnoticed and gives wrong steps. Declare it
only when it holds everywhere.

### Ensembles of paths

For statistics - a stationary covariance, an escape rate, a residence-time
density - step thousands of paths at once with `StochasticEnsemble` instead of
looping over a per-path solver:

```python
from discrecontinual_equations.solver.stochastic.ensemble import StochasticEnsemble

equation = sde_equation(ConformalNoise, 2)
config = EulerMaruyamaConfig(start_time=0, end_time=1, step_size=0.01, random_seed=3)
result = StochasticEnsemble(config).solve(equation, [1.0, 0.0], paths=200, keep_every=20)
print(result.final.shape)    # (paths, n) at the end time
print(result.samples.shape)  # (kept, paths, n) every 20 steps
```

A study of a stationary law wants far more: `paths=20000` over `end_time=10` is
typical, and that is where the batched methods below pay.

Every `StochasticFunction` works in an ensemble; one that overrides `eval_batch`
and `diffusion_batch` with array arithmetic over the `(paths, n)` argument runs
about a hundred times faster than the per-point default.

```python
class BatchedConformalNoise(ConformalNoise):
    def eval_batch(self, points, time=None):
        return self.a * points

    def diffusion_batch(self, points, time=None):
        x, y = points[:, 0], points[:, 1]
        s = self.sigma
        return s * np.stack([np.stack([x, -y], 1), np.stack([y, x], 1)], 1)


fast = StochasticEnsemble(config).solve(
    sde_equation(BatchedConformalNoise, 2), [1.0, 0.0], paths=200, keep_every=20
)
print(fast.final.shape)
```

`StochasticEnsemble(config, scheme="milstein")` takes the same noise as
`MilsteinSolver`: when the columns do not commute it draws the Levy areas of every
path at once (DEQ-65) and warns once about their cost.

### Quasipotential and minimum action paths

How a planar system gets from one attractor to another under small noise is
answered by the Freidlin-Wentzell quasipotential `U`: escapes cost
`exp(-U / eps)`, and the minimiser of the action is the most probable path.
`OrderedLineIntegral` computes `U` on a `DensityGrid` by the ordered line
integral method (OLIM-MID: midpoint quadrature, `U` linear along the accepted
front), for a drift `Function` and a `NoiseMatrix`, from a
stable equilibrium or a limit cycle:

Here the drift is the double well $f = -\nabla V$, $V = x^4/4 - x^2/2 + y^2/2$,
with $G = I$, so the exact answer is $U = 2V$ less its value at the well.

```python
from discrecontinual_equations.continuation import (
    DensityGrid, EquilibriumAttractor, GeometricMinimumAction, OrderedLineIntegral,
)
from discrecontinual_equations.continuation.noise import NoiseMatrix
from discrecontinual_equations.function.deterministic import DeterministicFunction


class DoubleWell(DeterministicFunction):
    def eval(self, point, time=None):
        x, y = point
        return [x - x**3, -y]


class ConstantColumn(DeterministicFunction):
    column = (1.0, 0.0)

    def eval(self, point, time=None):
        return list(self.column)


class SecondColumn(ConstantColumn):
    column = (0.0, 1.0)


def planar(function_type):
    return function_type([Variable(), Variable()], [], [Variable(), Variable()], None)


drift = planar(DoubleWell)
noise = NoiseMatrix([planar(ConstantColumn), planar(SecondColumn)])  # G = I

grid = DensityGrid.box([-2.0, -1.5], [1.0, 1.5], [41, 41])
solver = OrderedLineIntegral(drift, noise, grid)
field = solver.solve(EquilibriumAttractor([-1.0, 0.0]))
field.values                                   # U on the grid, shaped like it
path = solver.minimum_action_path(field, [-0.3, 0.6], end=[-1.0, 0.0])
path.states, path.action                       # back along -(f + D grad U)

initial_chain = np.linspace([-1.0, 0.0], [0.0, 0.0], 21)  # well to saddle
gmam = GeometricMinimumAction(drift, noise, tolerance=1e-6)
between = gmam.path(initial_chain)             # across a saddle
print(f"{between.action:.3f}")                 # 2 (V(0) - V(-1)) = 0.5

point = int(np.ravel_multi_index((13, 27), grid.shape))  # (-1.025, 0.525)
checked = solver.check(field, indices=[point])  # optional, about a second a point
checked.errors, checked.status                 # estimated error; GRID, CHECKED or REFINED
print(f"{field.values.ravel()[point]:.4f} +- {checked.errors.ravel()[point]:.4f}")
```

prints `0.501` for the saddle crossing and `0.2789 +- 0.0020` for the grid value,
whose exact value is $2(V(-1.025, 0.525) - V(-1, 0)) = 0.2769$. The chain's
default `tolerance=1e-10` asks for far more relaxation than an example needs.

The march says nothing about how accurate a value is. `check` estimates it at
chosen points against a value computed without the grid, by the geometric
minimum action method from a small surface about the attractor, and with a
`tolerance` replaces the grid values that miss it.

`U` is normalised as the rest of the library writes `dx = f dt + G dW`:
`D = G G^T`, so a gradient drift `-grad V` with `G = I` has `U = 2 V`, and the
grid density of `StationaryFokkerPlanck` is `exp(-U)` to leading order. The
method is planar and time-independent and needs `D` non-singular at every grid
point. With the default straight base of OLIM-MID it is first order in the
spacing, the bound the method's authors give, and second order where the
quadrature dominates. `update_order="quadratic"` takes `U` as a parabola
along the front instead; it is not a published method, it costs 2.3 to 2.8
times the march's time, and it pays only on a drift that is linear or nearly
so (94 times closer on the rotating Ornstein-Uhlenbeck process, 3.6 times on
Dahiya and Cameron's SDE (21) at 321 points); on the nonlinear problems
measured the straight base wins at equal cost by 3 to 6 times. The update
radius is 8 cells on every grid unless `update_radius` is given;
`default_radius(grid)` in `continuation.quasipotential` is the authors' rule
of thumb, growing with the grid, which pays on a strongly rotating drift and
costs three to four times the error on one that is nearly a gradient.
`GeometricMinimumAction` takes a constant or a state-dependent `D`. The module
docstring of `continuation/quasipotential.py` has the measured errors.

### First-passage and residence times

The quasipotential says how hard an escape is; `MeanFirstPassage` says how
long it takes. It solves the backward Kolmogorov problem `L tau = -1` off a
target set on a `DensityGrid` of any dimension, with the density solve's own
stencil, and an ensemble gives the same time with an error bar:

```python
from discrecontinual_equations.continuation import (
    Ball, DensityGrid, EnsembleFirstPassage, EnsembleResidence, MeanFirstPassage,
    eyring_kramers_time,
)

grid = DensityGrid.box([-2.2, -1.8], [2.2, 1.8], [161, 161])
times = MeanFirstPassage(drift, noise, Ito(), grid).solve(Ball([1.0, 0.0], 0.3).on(grid))
times.at([-1.0, 0.0])                          # mean time from the left well

sample = EnsembleFirstPassage(StochasticConfig(step_size=0.0025, random_seed=1)).solve(
    equation, [-1.0, 0.0], 2000, Ball([1.0, 0.0], 0.3), horizon=1e5,
)
sample.mean, sample.confidence_interval()      # the same time, by simulation

residences = EnsembleResidence(config).solve(
    equation, (left, right), 400, (Ball(left, 0.3), Ball(right, 0.3)), horizon=2000.0,
)
residences.asymmetry_interval()                # a residence-time sensor's reading
```

`eyring_kramers_time` is the small-noise limit (Eyring-Kramers, and Bouchet
and Reygner's non-gradient form) for overdamped dynamics with one
non-degenerate saddle between two point attractors; it is a limit, and the
grid shows where it starts to hold. The module docstring of
`continuation/first_passage.py` has the measured errors and the limits of
both routes.

## Custom Solvers

Extend the base `Solver` class to implement custom numerical methods.

## Plotting Results

```python
from discrecontinual_equations.plot.line_plot import LinePlot

solution = EulerMaruyamaSolver(config).solve(sde_equation(MySDE, 1), [1.0])
plot = LinePlot(output_dir="plots", output_format="html")
plot.plot(x=solution.time, y=solution.results[0])
```
