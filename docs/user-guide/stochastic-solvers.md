# Stochastic solvers and iterated integrals

The library integrates Itô stochastic differential equations

$$
dX = a(X, t)\,dt + G(X, t)\,dW, \qquad X \in \mathbb{R}^n,\quad W \in \mathbb{R}^m,
$$

where $a$ is the drift, $G$ the diffusion and $W$ an $m$-dimensional Wiener process. A one-step scheme approximates
the Itô-Taylor expansion of the solution over one step $[t, t+h]$. Each term of that expansion multiplies a
functional of the Brownian path over the step: the increment $\Delta W_j$, the time integral
$I_{(j,0)} = \int_t^{t+h} (W_j(s) - W_j(t))\,ds$, the twofold integrals $I_{(j,k)} = \int_t^{t+h}\int_t^s dW_j\,dW_k$,
and so on. A scheme reaches the order it was derived for only when it is given these functionals with their exact
joint law. That is why the solvers take their noise from a *Wiener source* rather than drawing one normal per stage.

Two orders are quoted throughout:

- the **strong order** $\gamma$: $\mathbb{E}\,|X_h(T) - X(T)| = O(h^\gamma)$, where both are driven by the same
  Brownian path. It measures how well a scheme follows paths.
- the **weak order** $\beta$: $|\mathbb{E}\,f(X_h(T)) - \mathbb{E}\,f(X(T))| = O(h^\beta)$ for smooth $f$. It
  measures how well a scheme gets expectations.

| Solver | Strong order | Weak order | Noise it takes |
|---|---|---|---|
| [Euler-Maruyama](#euler-maruyama) | 0.5 | 1.0 | any: vector or $n \times m$ matrix |
| [Milstein](#milstein-and-levy-areas) | 1.0 | 1.0 | any; draws Lévy areas when the noise does not commute |
| [srk2](#srk2-derivative-free-milstein) | 1.0 | 1.0 | scalar or diagonal (vector) |
| [srk3](#srk3-strong-order-15) | 1.5 | not claimed | scalar, diagonal or matrix |
| [srk4](#srk4-weak-order-20) | 1.0 | 2.0 | one variable |
| [srk5](#srk5-refuses-to-run) | none | none | refuses to run |
| [Gottwald-Melbourne](#gottwald-melbourne-stable-noise) | none (weak convergence as $\varepsilon \to 0$) | none | scalar, α-stable noise |
| [Zhan-Duan-Li-Li](#zhan-duan-li-li-theta-step) | see its limits | see its limits | additive noise only, for an Itô answer |

The numbers in the table are the ones measured in the solvers' docstrings, on geometric Brownian motion and on the
non-commuting systems described there.

## Defining a stochastic equation

A stochastic equation is a `StochasticFunction`. Its `eval` returns the drift $a$ and its `diffusion` returns $G$.
`diffusion` may return a vector, with one coefficient per component, which is diagonal noise: component $i$ has its
own $W_i$. It may also return an $n \times m$ nested list, a full matrix with $m$ drivers. The examples on this page
build on this one, geometric Brownian motion $dX = \mu X\,dt + \sigma X\,dW$:

```python
import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class GeometricBrownianMotion(StochasticFunction):
    """dX = mu X dt + sigma X dW."""

    def __init__(self, x, mu, sigma, t):
        super().__init__([x], [mu, sigma], [Variable(name="dx")], t)

    def eval(self, point, time=None):  # the drift a(x, t)
        return [self.parameters[0].value * point[0]]

    def diffusion(self, point, time=None):  # the diffusion b(x, t)
        return [self.parameters[1].value * point[0]]


x, t = Variable(name="x"), Variable(name="t")
mu, sigma = Parameter(value=1.0, name="mu"), Parameter(value=0.5, name="sigma")
gbm = DifferentialEquation(
    variables=[x], time=t, parameters=[mu, sigma],
    derivative=GeometricBrownianMotion(x, mu, sigma, t),
)
```

**Itô or Stratonovich.** Every configuration has `calculus="ito"` (the default) or `"stratonovich"`. A Stratonovich
equation is integrated as its Itô equivalent, whose drift is
$a_i + \tfrac12 \sum_{k} \sum_l G_{lk}\,\partial_l G_{ik}$. For scalar noise that is $a + \tfrac12 b b'$, and the
correction is added, not subtracted. The derivatives are central differences
([`ItoCoefficients`](../api/stochastic-solvers.md#discrecontinual_equations.solver.stochastic.coefficients.ItoCoefficients)).
A function may set `diagonal_noise = True` to promise that each $b_i$ depends on $x_i$ alone. Each solve checks that
promise once, at the initial state, and then reads all the slopes from two evaluations.

## Euler-Maruyama

**Idea.** Freeze $a$ and $G$ at the start of the step and add one increment per driver:

$$
X_{n+1} = X_n + a(X_n, t_n)\,h + G(X_n, t_n)\,\Delta W_n, \qquad \Delta W_n \sim N(0, h I_m).
$$

The step keeps only the first two terms of the Itô-Taylor expansion. That costs half an order in the strong sense,
but it needs no derivative and no iterated integral, so it accepts any noise shape.

```python
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config import (
    EulerMaruyamaConfig,
)
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_solver import (
    EulerMaruyamaSolver,
)

config = EulerMaruyamaConfig(start_time=0.0, end_time=1.0, step_size=0.01, random_seed=7)
curve = EulerMaruyamaSolver(config).solve(gbm, [1.0])
time, _, state = curve[-1]
print(f"X({time:.2f}) = {state[0]:.4f}")  # X(1.00) = 1.0398
```

**Limits.** The strong order is 0.5, so halving the path error takes four times the steps. The method is explicit,
so a stiff drift needs $h$ below its stability limit, as in the deterministic Euler method. A `random_seed` makes a
run repeatable.

**Sources.** [Maruyama1955], [KloedenPlaten1992] on the [References](../references.md) page. API:
[`EulerMaruyamaSolver`](../api/stochastic-solvers.md#discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_solver.EulerMaruyamaSolver).

## Measuring an order: a Brownian path fixed in advance

A strong error compares a scheme with the exact solution *on the same Brownian path*. That path has to be the same
at every step size, so it must exist before the solver runs.
[`BrownianPath`](../api/stochastic-solvers.md#discrecontinual_equations.solver.stochastic.wiener.BrownianPath)
holds a path on a fine grid. It hands each step its exact $\Delta W$, the trapezoidal $I_{(j,0)}$ and the Lévy areas.
Each step must be a whole number of grid cells. Geometric Brownian motion has the closed form
$X(t) = X_0 \exp((\mu - \sigma^2/2)t + \sigma W(t))$:

```python
from discrecontinual_equations.solver.stochastic.milstein.milstein_config import MilsteinConfig
from discrecontinual_equations.solver.stochastic.milstein.milstein_solver import MilsteinSolver
from discrecontinual_equations.solver.stochastic.srk3.srk3_config import SRK3Config
from discrecontinual_equations.solver.stochastic.srk3.srk3_solver import SRK3Solver
from discrecontinual_equations.solver.stochastic.wiener import BrownianPath

schemes = {
    "euler_maruyama": (EulerMaruyamaConfig, EulerMaruyamaSolver),
    "milstein": (MilsteinConfig, MilsteinSolver),
    "srk3": (SRK3Config, SRK3Solver),
}
steps = [2.0**-k for k in range(3, 7)]
errors = {name: np.zeros(len(steps)) for name in schemes}
paths = 50
for seed in range(paths):
    path = BrownianPath.sample(0.0, 1.0, 2.0**-10, dimension=1, seed=seed)
    exact = np.exp((1.0 - 0.5 * 0.5**2) + 0.5 * path.value(1.0)[0])
    for name, (Config, Solver) in schemes.items():
        for i, h in enumerate(steps):
            config = Config(start_time=0.0, end_time=1.0, step_size=h)
            final = Solver(config, wiener=path).solve(gbm, [1.0])[-1][2][0]
            errors[name][i] += abs(final - exact) / paths

for name, error in errors.items():
    slope = np.polyfit(np.log(steps), np.log(error), 1)[0]
    print(f"{name:15s} strong order {slope:.2f}")
```

This prints 0.70, 0.94 and 1.59. Fifty paths over four coarse step sizes resolve the order only roughly. Over
$h = 2^{-3}\ldots2^{-8}$ with 100,000 paths, the solvers' docstrings measure 0.59, 0.97 and 1.48. A real study
should use more paths and finer steps.

## Milstein and Lévy areas

**Idea.** Milstein keeps the next term of the Itô-Taylor expansion,

$$
X_{n+1} = X_n + a\,h + G\,\Delta W + \sum_{j,k} (\partial G_k)\,G_j\; I_{(j,k)},
\qquad
I_{(j,k)} = \tfrac12\left(\Delta W_j \Delta W_k - \delta_{jk} h\right) + A_{(j,k)},
$$

where $G_k$ is column $k$ of $G$, $\partial G_k$ its Jacobian, and $A$ the antisymmetric matrix of **Lévy areas**
$A_{(j,k)} = \tfrac12(I_{(j,k)} - I_{(k,j)})$. The first index of $I_{(j,k)}$ is the inner integral, as in Kloeden and
Platen and in Mrongowius and Rößler's eq. (7).

When the noise *commutes*, $(\partial G_k) G_j = (\partial G_j) G_k$ for every pair, the areas cancel between $(j,k)$
and $(k,j)$, and the step needs only $\Delta W$. Scalar noise and diagonal noise with $b_i = b_i(x_i)$ always commute.
The solver checks commutativity at every step. When the noise does not commute, it draws the areas and warns once
per solve about the cost.

**The areas.** $A_{(j,k)}$ has mean $0$ and variance $h^2/4$. It is uncorrelated with the increments. Given them, its
variance is $h^2/12 + h(\Delta W_j^2 + \Delta W_k^2)/12$ (Lévy's formula). It is not Gaussian and has no closed-form
sampler. [`LevyAreaSampler`](../api/stochastic-solvers.md#discrecontinual_equations.solver.stochastic.wiener.LevyAreaSampler)
draws it by Mrongowius and Rößler's algorithm (their §5.1). The algorithm takes a truncated Fourier series of the
Brownian bridge with $n$ terms. The part of the remainder that is linear in $\Delta W$ is simulated exactly, and the
rest is replaced by a Gaussian with the right covariance. Its first two moments are exact at every $n$. Its L2 error
is at most $h\sqrt{m}/(\sqrt{12}\,\pi n)$ (their Corollary 4.5). Strong order 1.0 needs that error below
$c_1 h^{3/2}$ (Kastner and Rößler (64)), so $n \ge \sqrt{m/(12\pi^2)}/(c_1\sqrt h)$ (their (67)). The constant $c_1$
is `MilsteinConfig.levy_area_accuracy`, default 1.

On $dx = dW_1,\ dy = x\,dW_2$, $y(T)$ is the iterated integral $I_{(1,2)}(0,T)$ itself, and Milstein reproduces the
path's own value:

```python
import warnings


class Area(StochasticFunction):
    """dx = dW_1, dy = x dW_2: y(T) is the iterated integral I_(1,2)(0, T)."""

    def __init__(self, x, y, t):
        super().__init__([x, y], [], [Variable(name="dx"), Variable(name="dy")], t)

    def eval(self, point, time=None):
        return [0.0, 0.0]

    def diffusion(self, point, time=None):  # a 2 x 2 matrix: two drivers
        return [[1.0, 0.0], [0.0, point[0]]]


y = Variable(name="y")
area = DifferentialEquation(variables=[x, y], time=t, parameters=[], derivative=Area(x, y, t))

path = BrownianPath.sample(0.0, 1.0, 2.0**-8, dimension=2, seed=3)
exact = path.increments(0.0, 1.0, 2, levy_area=True).iterated_integrals(1.0)[0, 1]
config = MilsteinConfig(start_time=0.0, end_time=1.0, step_size=2.0**-4)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # the solver warns once that it draws Levy areas
    y_final = MilsteinSolver(config, wiener=path).solve(area, [0.0, 0.0])[-1][2][1]
print(f"Milstein y(1) = {y_final:.6f}, the path's I_(1,2)(0, 1) = {exact:.6f}")
```

The sampler on its own:

```python
from discrecontinual_equations.solver.stochastic.wiener import LevyAreaSampler

h, m = 0.01, 2
print("Fourier terms for c1 = 1:", LevyAreaSampler.terms(h, m))  # 2
print("Fourier terms for c1 = 0.05:", LevyAreaSampler.terms(h, m, accuracy=0.05))  # 26

generator = np.random.default_rng(0)
sampler = LevyAreaSampler(generator)
delta_w = np.sqrt(h) * generator.standard_normal((100_000, m))
areas = sampler.sample(delta_w, h)[:, 0, 1]
print(f"mean {areas.mean():+.1e}, variance / (h^2 / 4) = {areas.var() / (h**2 / 4):.3f}")
```

**Limits.**

- With the default $c_1 = 1$, $n$ stays at 1 for $h \ge 2^{-5}$ with two drivers. Sampler-driven Milstein then measures a
  slope of about 0.75 over $h = 2^{-3}\ldots2^{-10}$, while staying within its error bound. With $c_1 = 0.05$ the
  slope is 0.97, at 8 to 17 times the Fourier terms. Order 1.0 holds as $h \to 0$ for any $c_1$.
- Each step with non-commuting noise costs $2mn + m(m+1)/2$ extra normals, so the cost grows like $m^{3/2}/c_1$.
- The commutativity test is relative, with tolerance $10^{-6}$. Area terms smaller than that fraction of the products
  are dropped.
- A seeded run draws the same increments as Euler-Maruyama with that seed, because the areas come from a stream of
  their own.
- Erratum carried over from DEQ-36: the ticket gave the variance of $A$ as $h^2/12$. That is only the conditional
  variance at $\Delta W = 0$. Averaging Lévy's formula gives $h^2/4$.

**Sources.** [Milstein1995], [KloedenPlaten1992], [MrongowiusRossler2021], [KastnerRossler2022],
[KloedenPlatenWright1992], [Wiktorsson2001]. API:
[`MilsteinSolver`](../api/stochastic-solvers.md#discrecontinual_equations.solver.stochastic.milstein.milstein_solver.MilsteinSolver),
[`MilsteinConfig`](../api/stochastic-solvers.md#discrecontinual_equations.solver.stochastic.milstein.milstein_config.MilsteinConfig),
[`LevyAreaSampler`](../api/stochastic-solvers.md#discrecontinual_equations.solver.stochastic.wiener.LevyAreaSampler).

## srk2: derivative-free Milstein

**Idea.** Platen's scheme, Kloeden and Platen (11.1.7), is the Milstein step with $b'$ replaced by a difference over
a supporting value $\bar Y = Y + a h + b\sqrt h$:

$$
Y_{n+1} = Y + a h + b\,\Delta W + \frac{b(\bar Y) - b}{2\sqrt h}\left(\Delta W^2 - h\right).
$$

Since $(b(\bar Y) - b)/(2\sqrt h) = \tfrac12 b b' + O(\sqrt h)$, the last term is the Milstein correction up to a
strong-order-1.5 remainder. The "2" in the name counts stages. It is not an order.

**Limits.** The measured orders are 1.0 strong and 1.0 weak. For a system, the scheme uses one supporting value per
component and omits the cross terms $b_k\,\partial_k b_j\,I_{(k,j)}$. It keeps order 1.0 only when each $b_j$ depends
on $x_j$ alone. Otherwise it falls to 0.5. It refuses a matrix diffusion. API:
[`SRK2Solver`](../api/stochastic-solvers.md#discrecontinual_equations.solver.stochastic.srk2.srk2_solver.SRK2Solver).
Source: [KloedenPlaten1992].

## srk3: strong order 1.5

**Idea.** Kloeden and Platen (11.2.1) is the order 1.5 strong Itô-Taylor scheme with every derivative of $a$ and $b$
replaced by differences over the supporting values $Y_\pm = Y + ah \pm b\sqrt h$ and
$\Phi_\pm = Y_+ \pm b(Y_+)\sqrt h$. It consumes $\Delta W$ and $\Delta Z = I_{(1,0)}$. These are jointly Gaussian,
with $\operatorname{Var}\Delta Z = h^3/3$ and $\operatorname{Cov}(\Delta W, \Delta Z) = h^2/2$, and the Wiener source
draws them exactly. For several channels the same construction is applied to the several-channel Taylor scheme
(Kuznetsov (5)). That form also needs every $I_{(j,k)}$ and $I_{(j,k,l)}$, drawn jointly; see
[iterated integrals](#iterated-integrals-kuznetsovs-legendre-series) below. A non-commuting two-channel system:

```python
class CrossCosine(StochasticFunction):
    """dx = -x/2 dt + cos(y) dW_1, dy = -y/2 dt + cos(x) dW_2."""

    def __init__(self, x, y, t):
        super().__init__([x, y], [], [Variable(name="dx"), Variable(name="dy")], t)

    def eval(self, point, time=None):
        return [-0.5 * point[0], -0.5 * point[1]]

    def diffusion(self, point, time=None):
        return [[np.cos(point[1]), 0.0], [0.0, np.cos(point[0])]]


cross = DifferentialEquation(variables=[x, y], time=t, parameters=[], derivative=CrossCosine(x, y, t))
config = SRK3Config(start_time=0.0, end_time=1.0, step_size=0.05, random_seed=11)
print(SRK3Solver(config).solve(cross, [0.3, -0.2])[-1][2])
```

**Limits.**

- The measured strong order is 1.48 for scalar noise and 1.43-1.46 for the non-commuting systems in the docstring.
  No weak order is claimed.
- Several channels are expensive. Each step makes $1 + 2m$ drift and $1 + 2m + 2m^2$ diffusion evaluations, and it
  needs the $m^3$ triple integrals plus a twofold series of $O(h^{-2})$ terms (see below).
- A plain `BrownianPath` holds no triple integrals, so it is refused for several channels. Use an
  `IteratedIntegralPath` instead.
- Rößler's order 1.5 SRK methods [Rossler2010] cover scalar, diagonal and additive noise only, so they are not used.

API: [`SRK3Solver`](../api/stochastic-solvers.md#discrecontinual_equations.solver.stochastic.srk3.srk3_solver.SRK3Solver).
Sources: [KloedenPlaten1992], [Kuznetsov2018].

## srk4: weak order 2.0

**Idea.** Kloeden and Platen (15.1.4) is the derivative-free form of the order 2.0 weak Taylor scheme. In place of
both $I_{(1,0)}$ and $I_{(0,1)}$ it uses $\tfrac12 h\,\Delta W$. That stand-in has the right mean and the right
covariance with $\Delta W$, but variance $h^3/4$ instead of $h^3/3$. No expectation at order $h^2$ can see the
difference, but a path can.

**Limits.** The measured orders are 2.0 weak and 1.0 strong. The solver takes one variable only. The several-channel
form needs a two-point variable per pair of channels, which is not implemented, so systems are refused. API:
[`SRK4Solver`](../api/stochastic-solvers.md#discrecontinual_equations.solver.stochastic.srk4.srk4_solver.SRK4Solver).
Sources: [KloedenPlaten1992], [TalayTubaro1990].

## srk5: refuses to run

`SRK5Solver.solve` raises `NotImplementedError`. Its earlier version claimed strong order 2.5 and weak order 5.0, but
drew no random numbers. For multiplicative noise, strong order 2.0 already needs $I_{(1,1,0)}$, $I_{(1,0,1)}$,
$I_{(0,1,1)}$ and $I_{(1,1,1,1)}$ with their joint law. A Runge-Kutta stage count says nothing about stochastic
order, because the noise terms have order conditions of their own ([BurrageBurrage1996], [Rossler2010]). Use srk3
for the highest strong order and srk4 for the highest weak order. API:
[`SRK5Solver`](../api/stochastic-solvers.md#discrecontinual_equations.solver.stochastic.srk5.srk5_solver.SRK5Solver).

## Iterated integrals: Kuznetsov's Legendre series

**Idea.** Expand each Brownian motion on the step in Legendre polynomials $\phi_p$ that are orthonormal on
$[t, t+h]$. The coefficients $\zeta_p^{(i)} = \int \phi_p\,dW_i$ are independent standard normals. The first two fix
the increments exactly: $\Delta W = \sqrt h\,\zeta_0$ and $\Delta Z = \tfrac12 h^{3/2}(\zeta_0 - \zeta_1/\sqrt3)$
(Kuznetsov (42), (43)). Every multiple integral is then a series in the $\zeta$:

$$
I_{(j,k)} = \frac h2\left(\zeta_0^j\zeta_0^k + \sum_{i=1}^{q}
\frac{\zeta_{i-1}^j\zeta_i^k - \zeta_i^j\zeta_{i-1}^k}{\sqrt{4i^2-1}} - \delta_{jk}\right)
\quad\text{(his (44))},
$$

with mean-square error $h^2/(4(2q+1))$ for $j \ne k$ (his (52)). For the triple integrals the series is
$\sum C_{p_3 p_2 p_1}\,\zeta_{p_1}^j\zeta_{p_2}^k\zeta_{p_3}^l$, less the Itô pairings (his (55)). The coefficients
$C$ (his (56)-(57)) are computed exactly by Gauss-Legendre quadrature. The sampler reads $\zeta_0$ and $\zeta_1$ back
from increments a source has already drawn, so the integrals belong to the same path as the increments. Order 1.5
needs a mean-square error of at most $h^4$ per integral (his (66)-(67)). That gives
$q_1 \ge (1/(4h^2) - 1)/2$ for the twofold series and about $0.13/h$ for the threefold one.

```python
from discrecontinual_equations.solver.stochastic.iterated_integrals import (
    GaussianIteratedIntegralSource,
    IteratedIntegralSampler,
)

h = 0.01
q1, q3 = IteratedIntegralSampler.double_terms(h), IteratedIntegralSampler.triple_terms(h)
print(f"h = {h}: q1 = {q1} terms for I_(j,k), q3 = {q3} for I_(j,k,l)")  # 1250 and 13
print("exact error at q = 6, distinct channels:",
      f"{IteratedIntegralSampler.triple_errors(6)['distinct']:.6f}")  # 0.019554, his (63)

source = GaussianIteratedIntegralSource(seed=1)
step = source.iterated_integrals(0.0, h, dimension=2)
# The symmetric part of the twofold integrals is fixed by the increments:
print(np.allclose(step.double + step.double.T, np.outer(step.delta_w, step.delta_w) - h * np.eye(2)))
```

**Limits.**

- The twofold series has no remainder correction, so it needs $O(h^{-2})$ normals per step and channel: 1250 at
  $h = 0.01$. The Lévy-area sampler for Milstein needs only $O(h^{-1/2})$, but it serves order 1.0, not 1.5.
- The triple integrals cost $O(m^3 q_3^3)$ per step.
- Multiplicity four, which order 2.0 needs, is not implemented.
- An `IteratedIntegralPath` holds every integral of every cell of a fine grid, truncated at a fixed `cell_terms`. It
  reads a step by Chen's relation, which keeps every step size on one path. Its accumulated truncation error stays
  within order 1.5 while $\text{resolution} \le 4(2q+1)h^3$.
- **Erratum in the source.** Kuznetsov's (65) prints $0.0236084\,(T-t)^4$ for the fourfold error at $q = 2$. The exact
  value is $234761/10245312 = 0.0229140$, so the printed constant is a summation slip (DEQ-66, DEQ-114). His (63)
  for the threefold error, $\approx 0.01956$, is reproduced (0.019554).

API:
[`IteratedIntegralSampler`](../api/stochastic-solvers.md#discrecontinual_equations.solver.stochastic.iterated_integrals.IteratedIntegralSampler),
[`GaussianIteratedIntegralSource`](../api/stochastic-solvers.md#discrecontinual_equations.solver.stochastic.iterated_integrals.GaussianIteratedIntegralSource),
[`IteratedIntegralPath`](../api/stochastic-solvers.md#discrecontinual_equations.solver.stochastic.iterated_integrals.IteratedIntegralPath).
Sources: [Kuznetsov2018], [KastnerRossler2022], [KloedenPlaten1992].

## Ensembles: many paths at once

**Idea.** The per-path solvers pay Python overhead on every step. `StochasticEnsemble` steps a `(paths, n)` array
at once, with Euler-Maruyama or Milstein. For non-commuting noise, Milstein draws the Lévy areas of every path in one
sampler call. An equation that defines `eval_batch` and `diffusion_batch` on arrays makes the per-step cost a few
array operations, however many paths there are:

```python
from discrecontinual_equations.solver.solver_config import StochasticConfig
from discrecontinual_equations.solver.stochastic.ensemble import StochasticEnsemble


class OrnsteinUhlenbeck(StochasticFunction):
    """dx = -x dt + dW, with array versions for the ensemble."""

    def __init__(self, x, t):
        super().__init__([x], [], [Variable(name="dx")], t)

    def eval(self, point, time=None):
        return [-point[0]]

    def diffusion(self, point, time=None):
        return [1.0]

    def eval_batch(self, points, time=None):
        return -points

    def diffusion_batch(self, points, time=None):
        return np.ones_like(points)


ou = DifferentialEquation(variables=[x], time=t, parameters=[], derivative=OrnsteinUhlenbeck(x, t))
config = StochasticConfig(start_time=0.0, end_time=5.0, step_size=0.01, random_seed=0)
result = StochasticEnsemble(config, scheme="euler_maruyama").solve(ou, [2.0], paths=20_000)
print(f"variance at t = 5: {result.final[:, 0].var():.3f} (stationary 1/2)")
```

**Limits.** Without the batch methods, the defaults loop over the paths. That works, but only as fast as the
per-path solver. With a Wiener source handed in, the source is asked once per path and step. Ensembles return arrays,
not `Curve`s. API:
[`StochasticEnsemble`](../api/stochastic-solvers.md#discrecontinual_equations.solver.stochastic.ensemble.StochasticEnsemble).
Sources: as for Euler-Maruyama and Milstein.

## Gottwald-Melbourne: α-stable noise

**Idea.** The equation is the scalar Marcus SDE $dZ = a(Z)\,dt + b(Z) \diamond dW_{\alpha,\eta,\beta}$, driven by an
α-stable Lévy process with $\alpha \in (1, 2)$. Its coefficients need not be Lipschitz. The method uses no Lévy
increments. It iterates the deterministic fast-slow map (their (4.2)):

$$
z_{n+1} = z_n + \varepsilon\,a(z_n) + \varepsilon^{\gamma}\,b(z_n)\,v^{(n)}, \qquad x_{n+1} = T(x_n), \qquad \gamma = 1/\alpha,
$$

with $T$ the intermittent Thaler map on $[0,1]$. The observable $v$ is constant on the laminar region $[0, x^*]$ and on
the hyperbolic region $(x^*, 1]$. Its sign is drawn with $P(\pm1) = (1 \pm \beta)/2$. The long laminar excursions
near the neutral fixed point 0 build the stable law's jumps. As $\varepsilon \to 0$, $\hat z_\varepsilon(t) = z_{\lfloor t/\varepsilon \rfloor}$
converges weakly to the Marcus solution (their Theorem 4.1).

```python
from discrecontinual_equations.solver.stochastic.gottwald_melbourne.gottwald_melbourne_config import (
    GottwaldMelbourneConfig,
)
from discrecontinual_equations.solver.stochastic.gottwald_melbourne.gottwald_melbourne_solver import (
    GottwaldMelbourneSolver,
)


class StableOU(StochasticFunction):
    """dZ = -Z dt + dW_alpha."""

    def __init__(self, z, t):
        super().__init__([z], [], [Variable(name="dz")], t)

    def eval(self, point, time=None):
        return [-point[0]]

    def diffusion(self, point, time=None):
        return [1.0]


z = Variable(name="z")
stable = DifferentialEquation(variables=[z], time=t, parameters=[], derivative=StableOU(z, t))
config = GottwaldMelbourneConfig(
    start_time=0.0, end_time=1.0, step_size=0.01,
    alpha=1.5, beta=0.0, eta=1.0, epsilon=1e-4, random_seed=5,
)
print(GottwaldMelbourneSolver(config).solve(stable, [0.0])[-1][2][0])
```

**Limits.**

- The method is scalar only ($d = m = 1$), with $\alpha \in (1,2)$.
- $\varepsilon$ must not exceed `step_size`, since each fast step advances slow time by $\varepsilon$ (DEQ-86).
- The convergence is weak (in law), with no rigorous rate. Their Remark 2.6 suggests $n^{-(2/\alpha - 1)}$ with
  $n = 1/\varepsilon$, which is slow as $\alpha \to 2$. A path costs $1/\varepsilon$ map iterations.
- **The current release's observable is wrong.** It drops normalising factors of §4.1 and gives the hyperbolic
  branch the wrong sign. It also signs only that branch, so $v$ has a nonzero mean and the paths drift. On the example above it returns
  about $+29$, where the corrected solver of PR #75 (DEQ-108) returns $-3.9$. PR #75 also adds the burn-in of their
  Remark 4.4 and the perturbation of §4.2. Until it merges, do not use this solver's output quantitatively.
- **Errata in the source,** both followed in the corrected solver:
    - §4.1 prints the sign as the running product $\chi^{(n)} = \chi_{n-1}\cdots\chi_0$. Read literally, that gives
      the symmetric law for every $\beta < 1$. The proof of Theorem 4.1 (§5.3) needs one independent sign per
      excursion into $(x^*, 1]$, and that is what PR #75 implements.
    - Theorem 5.2 prints $\eta = ((c_1+c_2)\,g_\alpha)^{-1/\alpha}$. The stable law's tail
      $P(|X| > x) \sim \eta^\alpha x^{-\alpha}/g_\alpha$ gives $\eta = ((c_1+c_2)\,g_\alpha)^{1/\alpha}$, so the
      exponent's sign is an erratum.

API:
[`GottwaldMelbourneSolver`](../api/stochastic-solvers.md#discrecontinual_equations.solver.stochastic.gottwald_melbourne.gottwald_melbourne_solver.GottwaldMelbourneSolver).
Source: [GottwaldMelbourne2021].

## Zhan-Duan-Li-Li theta step

**What it is.** The solver is a theta-weighted Euler predictor-corrector step that reuses one increment:

$$
\tilde Y = Y + a h + b\,\Delta W, \qquad
Y_{n+1} = Y + \big((1-\theta)a(Y) + \theta a(\tilde Y)\big)h + \big((1-\theta)b(Y) + \theta b(\tilde Y)\big)\Delta W .
$$

Here $\theta$ is `method_parameter`, default $\tfrac12$. With $\theta = 0$ the step is Euler-Maruyama, and with
$\theta = \tfrac12$ it is the stochastic Heun step.

**Limits.**

- **It converges to the Stratonovich solution, not the Itô one.** The configuration fixes `calculus = "ito"`, but for
  multiplicative noise the corrector carries $\theta\,b b'\,\Delta W^2$. The mean of that term, $\theta b b' h$, is a
  drift the Itô equation lacks. At $\theta = \tfrac12$ it is exactly the Stratonovich correction. Other values of
  $\theta > 0$ converge to neither solution. For additive noise every $\theta$ is correct.
- The step is not the cited paper's scheme. arXiv:2010.07691 is a symplectic Euler method for Hamiltonian Marcus SDEs
  with Lévy noise. It contains no theta or trapezoidal method. The solver keeps the name for continuity.

```python
from discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_config import (
    ZhanDuanLiLiConfig,
)
from discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_solver import (
    ZhanDuanLiLiSolver,
)

mu.value, sigma.value = 0.0, 1.0  # dX = X dW: Ito mean 1, Stratonovich mean e^(1/2) = 1.65
finals = [
    ZhanDuanLiLiSolver(
        ZhanDuanLiLiConfig(start_time=0.0, end_time=1.0, step_size=0.05, random_seed=seed)
    ).solve(gbm, [1.0])[-1][2][0]
    for seed in range(2000)
]
print(f"mean X(1) = {np.mean(finals):.2f} +- {np.std(finals) / np.sqrt(2000):.2f}")  # about 1.6
```

For an Itô equation with multiplicative noise, use Euler-Maruyama or Milstein. API:
[`ZhanDuanLiLiSolver`](../api/stochastic-solvers.md#discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_solver.ZhanDuanLiLiSolver).
Source: [ZhanDuanLiLi2020].

## References

The entries cited as [Key] on this page are listed under "Stochastic solvers and iterated integrals" on the
[References](../references.md) page.
