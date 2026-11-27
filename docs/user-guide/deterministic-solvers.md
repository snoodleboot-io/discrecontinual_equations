# Deterministic solvers

The deterministic solvers integrate an initial value problem

$$
y'(t) = f(t, y(t)), \qquad y(t_0) = y_0,
$$

from `start_time` to `end_time`. There are two kinds:

- **Fixed-step methods** step along a grid of equal steps $h$: forward Euler and the
  Adams-Bashforth methods with two and three steps.
- **Embedded Runge-Kutta pairs** choose each step so that an estimate of the local
  error meets a tolerance: Cash-Karp 5(4), Runge-Kutta-Fehlberg 4(5), Dormand-Prince
  5(4), and the thirteen-stage Prince-Dormand 8(7) pair (`PrinceDormand87Solver`).

All of them are explicit. That makes them cheap per step, but on a stiff problem they
are forced to take small steps (see [Limits](#limits-common-to-all-the-solvers)).

!!! note "Defects fixed in DEQ-137 to DEQ-141"
    Checking each solver against its source found defects in five of them. They are
    fixed, and the orders below are measured on the fixed code:

    - **Dormand-Prince 5(4)** had the node $c_5 = 7/9$ instead of $8/9$, so it was
      first order whenever $f$ depends on $t$ (DEQ-137).
    - **Cash-Karp** had its two weight rows swapped and advanced with the
      fourth-order one (DEQ-138).
    - **`DormandPrince89Solver`** was an 8(7) pair with one mistyped eighth-order
      weight; its error stalled between $10^{-7}$ and $10^{-6}$. It is now
      `PrinceDormand87Solver`; the old name still works and warns (DEQ-139).
    - **The adaptive step control** retried a step rejected at `min_step_size`
      forever, and its "PI" controller was an integral one. It now raises
      `StepSizeTooSmallError` and is a real PI controller (DEQ-140).
    - **Adams-Bashforth 3** was second order because of its Euler start, and
      **Euler** evaluated $f(t_{n+1}, y_n)$ (DEQ-141).

    A test now checks every Runge-Kutta tableau against its order conditions in exact
    arithmetic, one condition per rooted tree
    ([Butcher1963](../references.md#deterministic-solvers)), so a wrong coefficient
    cannot come back unnoticed.

| Solver | Kind | Order claimed | Order measured | Field evaluations per step |
|---|---|---|---|---|
| `EulerSolver` | fixed step | 1 | 1 | 1 |
| `AdamsBashforth2Solver` | fixed step, 2 steps | 2 | 2 | 1 |
| `AdamsBashforth3Solver` | fixed step, 3 steps | 3 | 3 | 1 |
| `CashKarpSolver` | adaptive | 5(4) | 5 (embedded row 4) | 6 |
| `FehlbergSolver` | adaptive | 4(5) | 5 (it advances with the fifth-order row) | 6 |
| `DormandPrinceSolver` | adaptive | 5(4) | 5 (embedded row 4) | 7 |
| `PrinceDormand87Solver` | adaptive | 8(7) | 8 (embedded row 7), down to $10^{-16}$ | 13 |

We measured the orders by running each solver with fixed steps on
$y' = -y^2$, $y' = y\cos t$ and $y' = -2ty^2$ up to $t = 2$, over five step sizes.
The adaptive solvers were held to a fixed step by setting
`min_step_size = max_step_size`, and their embedded rows were measured with a separate
Runge-Kutta loop fed the same tableau.

## Setting up an equation

A right-hand side is a `DeterministicFunction` whose `eval(point, time)` returns
$f(t, y)$ as a list. A `DifferentialEquation` bundles it with its variables. The
examples on this page all solve the logistic equation $y' = y(1-y)$ with $y(0) = 0.1$.
Its exact solution is $y(t) = 1/(1 + 9e^{-t})$.

```python
import math

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.variable import Variable


class Logistic(DeterministicFunction):
    def eval(self, point, time=None):
        return [point[0] * (1.0 - point[0])]


def logistic_equation():
    t, y, r = Variable(name="t"), Variable(name="y"), Variable(name="r")
    f = Logistic([y], [], [r], t)
    return DifferentialEquation(variables=[y], time=t, parameters=[], derivative=f)


exact = 1.0 / (1.0 + 9.0 * math.exp(-2.0))  # y(2)
```

Every `solve` returns a [`Curve`](../api/core.md#discrecontinual_equations.curve.Curve).
For component $k$, `curve.results[k].discretization` holds the state at each node, and
`curve.time.discretization` holds the nodes. In the fixed-step solvers,
`curve.variables` holds the state one node earlier; the adaptive solvers fill it with
zeros. A solve does not change the equation, so you can solve the same equation again
or with another solver.

## Forward Euler

Euler's method follows the slope at the current point for one step. In Euler's own
words, treat $V = dy/dx$ as constant over a short interval and step to
$b' = b + A(a' - a)$ ([Euler1768](../references.md#deterministic-solvers),
Problem 85, §650):

$$
y_{n+1} = y_n + h\, f(t_n, y_n).
$$

The local error is $O(h^2)$ and the global error $O(h)$, so halving $h$ halves the
error. Euler himself observes that the per-step errors add up, and that they are
largest where the slope changes fastest (§§652-653).

```python
from discrecontinual_equations.solver.deterministic.euler.euler_config import EulerConfig
from discrecontinual_equations.solver.deterministic.euler.euler_solver import EulerSolver

for h in (0.1, 0.05):
    config = EulerConfig(start_time=0.0, end_time=2.0, step_size=h)
    curve = EulerSolver(config).solve(logistic_equation(), [0.1])
    print(h, curve.results[0].discretization[-1] - exact)
# 0.1 -0.0124...
# 0.05 -0.0062...   (half the error: first order)
```

**Limits.**

- On $y' = \lambda y$, the method is stable only when $|1 + h\lambda| \le 1$.
- Until DEQ-141 the code evaluated $f(t_{n+1}, y_n)$ rather than $f(t_n, y_n)$: the
  same results when $f$ does not depend on $t$, different (still first-order) ones
  when it does.
- The grid comes from the config's `times` (see `solver_config.FixedStepConfig`).

API: [`EulerSolver`](../api/deterministic-solvers.md#discrecontinual_equations.solver.deterministic.euler.euler_solver.EulerSolver).

## Adams-Bashforth 2 and 3

A multistep method reuses derivatives it has already computed, so each step costs one
new evaluation of $f$. J. C. Adams integrated the polynomial that interpolates the past
values $f_n, f_{n-1}, \dots$ over the next step. He wrote the step with backward
differences $\nabla$ ([Bashforth1883](../references.md#deterministic-solvers), p. 18):

$$
y_{n+1} - y_n = h\left(f_n + \tfrac12 \nabla f_n + \tfrac{5}{12} \nabla^2 f_n
+ \tfrac38 \nabla^3 f_n + \cdots\right).
$$

Stopping after $\nabla f_n$ gives the two-step method, and stopping after
$\nabla^2 f_n$ gives the three-step one:

$$
y_{n+1} = y_n + \tfrac{h}{2}\left(3 f_n - f_{n-1}\right), \qquad
y_{n+1} = y_n + \tfrac{h}{12}\left(23 f_n - 16 f_{n-1} + 5 f_{n-2}\right).
$$

A $k$-step method needs $k$ starting values, but the initial condition supplies only
one. These solvers compute the missing $k - 1$ with the classical fourth-order
Runge-Kutta method ([Kutta1901](../references.md#deterministic-solvers)). Its local
error $O(h^5)$, made a fixed number of times, is below the method's own error, so the
start does not lower the order. Each stored derivative is $f(t_n, y_n)$ at the state
the solver computed for node $n$.

```python
from discrecontinual_equations.solver.deterministic.adams_bashforth2.adams_bashforth2_config import (
    AdamsBashforth2Config,
)
from discrecontinual_equations.solver.deterministic.adams_bashforth2.adams_bashforth2_solver import (
    AdamsBashforth2Solver,
)
from discrecontinual_equations.solver.deterministic.adams_bashforth3.adams_bashforth3_config import (
    AdamsBashforth3Config,
)
from discrecontinual_equations.solver.deterministic.adams_bashforth3.adams_bashforth3_solver import (
    AdamsBashforth3Solver,
)

for h in (0.1, 0.05):
    n = round(2.0 / h)
    ab2 = AdamsBashforth2Solver(AdamsBashforth2Config(end_time=2.0, step_size=h))
    # number_of_steps is the node count of the grid; the step is read off it.
    ab3 = AdamsBashforth3Solver(
        AdamsBashforth3Config(end_time=2.0, number_of_steps=n + 1),
    )
    e2 = ab2.solve(logistic_equation(), [0.1]).results[0].discretization[-1] - exact
    e3 = ab3.solve(logistic_equation(), [0.1]).results[0].discretization[-1] - exact
    print(h, e2, e3)
# 0.1 0.000101... 7.44e-05
# 0.05 2.62e-05 9.77e-06   (AB2's error falls by 4, AB3's by 8: orders 2 and 3)
```

**Limits.**

- The step must stay fixed, because the coefficients assume equal steps.
- Explicit Adams methods have small stability regions, so they suit non-stiff
  problems only.
- `AdamsBashforth3Config` builds its grid from `number_of_steps` (`linspace`) or
  `np.arange(start, end, step_size)`, which stops one step short of `end_time`
  (being changed in DEQ-129). The step is read off the grid.
- Until DEQ-141 both solvers started with an Euler step and stored $f(t_1, y_0)$ for
  node 1, which left Adams-Bashforth 3 at second order.

API: [`AdamsBashforth2Solver`](../api/deterministic-solvers.md#discrecontinual_equations.solver.deterministic.adams_bashforth2.adams_bashforth2_solver.AdamsBashforth2Solver),
[`AdamsBashforth3Solver`](../api/deterministic-solvers.md#discrecontinual_equations.solver.deterministic.adams_bashforth3.adams_bashforth3_solver.AdamsBashforth3Solver).

## Embedded Runge-Kutta pairs and step-size control

An explicit Runge-Kutta step with $s$ stages evaluates

$$
k_i = f\Big(t_n + c_i h,\; y_n + h\sum_{j<i} a_{ij} k_j\Big), \qquad i = 1, \dots, s,
$$

and combines the stages with weights $b_i$. An *embedded pair* has two weight rows on
the same stages, one of order $p$ and one of order $\hat p$. The difference
$\mathrm{err} = |y_{n+1} - \hat y_{n+1}|$ estimates the local error of the
lower-order formula at no extra cost
([Fehlberg1969](../references.md#deterministic-solvers), eqs. (2)-(3)). Advancing
with the higher-order row is called *local extrapolation*. All four solvers here do
so: their `b` row (named `b5` or `b8`) is the higher-order one, and `b_hat` (`b4` or
`b7`) the embedded one. The error estimate then belongs to the lower-order solution,
so it is pessimistic for the solution returned.

Every pair uses the step control of Hairer and Wanner's `DOPRI5`
([HairerWanner1996](../references.md#deterministic-solvers), subroutine `DOPCOR`). The
error is measured in the root-mean-square norm,

$$
E = \sqrt{\frac1n \sum_i \left(\frac{\mathrm{err}_i}{\mathrm{atol} + \mathrm{rtol}\,\max(|y_{n,i}|, |y_{n+1,i}|)}\right)^2},
$$

and a step is accepted when $E \le 1$. The next step comes from a
proportional-integral (PI) controller,

$$
h_{\text{new}} = h\, S\, E^{-\alpha}\, E_{\text{old}}^{\beta}, \qquad
\alpha = \frac1q - 0.75\,\beta,
$$

with the change held to a factor between 0.2 and 10 and the step to
`[min_step_size, max_step_size]`. Here $S$ is `safety_factor` (default 0.9), $\beta$ is
`beta` (default 0.04), $E_{\text{old}}$ is the error of the last accepted step (at
least $10^{-4}$), and $q$ is one more than the order of the error estimate: 5 for the
5(4) and 4(5) pairs, 8 for the 8(7) pair. The $E^{-\alpha}$ term drives the error
towards its target; the $E_{\text{old}}^{\beta}$ term damps the step-size oscillation
a purely integral controller shows when stability, not accuracy, limits the step
([Gustafsson1991](../references.md#deterministic-solvers), as cited by `DOPRI5`).
$\beta = 0$ gives the classical $h\,S\,E^{-1/q}$. After a rejection the step shrinks
by $E^{\alpha}/S$, at most fivefold, and the step after a rejection may not grow. The
last step is stretched or shortened to land exactly on `end_time`.

Until DEQ-140 the controller was $h\,S\,r^{-(1+\beta)/q}$ with a max norm: integral
only, with no limit on the change. On a Van der Pol oscillator, the Arenstorf orbit
and $y' = y\cos t$ at four tolerances, the PI controller rejects about half as many
steps and needs 4-9% fewer field evaluations, at the same accuracy.

```python
from discrecontinual_equations.solver.deterministic.dormand_prince.dormand_prince_config import (
    DormandPrinceConfig,
)
from discrecontinual_equations.solver.deterministic.dormand_prince.dormand_prince_solver import (
    DormandPrinceSolver,
)

for tol in (1e-6, 1e-9):
    config = DormandPrinceConfig(
        end_time=2.0, absolute_tolerance=tol, relative_tolerance=tol,
    )
    curve = DormandPrinceSolver(config).solve(logistic_equation(), [0.1])
    print(tol, len(curve) - 1, curve.results[0].discretization[-1] - exact)
# 1e-06 8 2.04e-07    (8 accepted steps)
# 1e-09 19 3.63e-10
```

The tolerance bounds the local error of each step. It does not bound the global
error, which can be larger.

**Limits common to the four pairs.**

- **A step that fails at `min_step_size` stops the solve.** It raises
  `StepSizeTooSmallError` (a `RuntimeError`) naming $t$, $h$ and the error $E$
  reached, and so does a step too small to change $t$ in floating point. The solver
  does not accept the step with a warning: a step that fails its tolerance at the
  smallest size allowed usually means a singularity or stiffness, and the estimate
  has already disowned the value. Lower `min_step_size` or loosen the tolerances to go
  further. For example, $y' = 1/(1-t)$ towards $t = 1$ stops about $2\times10^{-8}$
  before the singularity after some 600 field evaluations; until DEQ-140 it never
  returned.

  ```python
  from discrecontinual_equations.solver.deterministic.embedded_runge_kutta import (
      StepSizeTooSmallError,
  )
  ```
- `beta` must be below $4/(3q)$ (0.27 for the 5(4) pairs, 0.17 for the 8(7) pair), or
  a rejected step would not shrink; a larger value is refused.
- The curve records only the accepted nodes, so its spacing varies. There is no dense
  output to fill in between.
- The pairs are explicit, so a stiff problem forces steps limited by stability, not
  accuracy.

### Cash-Karp 5(4)

Cash and Karp's six-stage tableau
([CashKarp1990](../references.md#deterministic-solvers)) was designed for a
variable-order method on right-hand sides that vary rapidly. This solver uses it as a
plain 5(4) pair with nodes $c = (0, \tfrac15, \tfrac{3}{10}, \tfrac35, 1, \tfrac78)$,
advancing with the fifth-order row
$(\tfrac{37}{378}, 0, \tfrac{250}{621}, \tfrac{125}{594}, 0, \tfrac{512}{1771})$, as
GSL's `rkck` and Boost.Odeint's `runge_kutta_cash_karp54` do
([GSL2024](../references.md#deterministic-solvers),
[BoostOdeint2024](../references.md#deterministic-solvers)).

```python
from discrecontinual_equations.solver.deterministic.cash_karp.cash_karp_config import (
    CashKarpConfig,
)
from discrecontinual_equations.solver.deterministic.cash_karp.cash_karp_solver import (
    CashKarpSolver,
)

config = CashKarpConfig(end_time=2.0, absolute_tolerance=1e-9, relative_tolerance=1e-9)
curve = CashKarpSolver(config).solve(logistic_equation(), [0.1])
print(len(curve) - 1, curve.results[0].discretization[-1] - exact)
# 18 1.04e-10
```

Until DEQ-138 the two weight rows were swapped and the solver advanced with the
fourth-order one: the same example gave an error of $1.6\times10^{-8}$ in 16 steps
(now $1.0\times10^{-10}$ in 18).

API: [`CashKarpSolver`](../api/deterministic-solvers.md#discrecontinual_equations.solver.deterministic.cash_karp.cash_karp_solver.CashKarpSolver).

### Runge-Kutta-Fehlberg 4(5)

Fehlberg's Formula 2 ([Fehlberg1969](../references.md#deterministic-solvers),
Table III, p. 13) has nodes $c = (0, \tfrac14, \tfrac38, \tfrac{12}{13}, 1, \tfrac12)$
and is the RKF45 pair in common use. Every coefficient in the code matches Table III.
Fehlberg made the fourth-order formula accurate and advanced with it. This solver
advances with his fifth-order formula $\hat y$ instead, so it converges at order 5.

```python
from discrecontinual_equations.solver.deterministic.fehlberg.fehlberg_config import (
    FehlbergConfig,
)
from discrecontinual_equations.solver.deterministic.fehlberg.fehlberg_solver import (
    FehlbergSolver,
)

config = FehlbergConfig(end_time=2.0, absolute_tolerance=1e-9, relative_tolerance=1e-9)
curve = FehlbergSolver(config).solve(logistic_equation(), [0.1])
print(len(curve) - 1, curve.results[0].discretization[-1] - exact)
# 21 -5.08e-12
```

API: [`FehlbergSolver`](../api/deterministic-solvers.md#discrecontinual_equations.solver.deterministic.fehlberg.fehlberg_solver.FehlbergSolver).

### Dormand-Prince 5(4)

Dormand and Prince chose their seven-stage tableau so that the *fifth*-order solution
is the accurate one, which is the one to advance with
([DormandPrince1980](../references.md#deterministic-solvers); coefficients as in
Hairer and Wanner's `CDOPRI`). It is the method behind MATLAB's `ode45` and SciPy's
`RK45`. Its last stage is evaluated at the new point, so the next step could reuse it
("first same as last"). This solver does not reuse it, so each step costs seven
evaluations. The step-size example above uses this solver.

Until DEQ-137 the code had $c_5 = 7/9$. The source has $8/9$, which is also the sum
of the row $a_{5j}$, as Butcher's reduction of $t$ to a state variable requires. The
weights and the $a_{ij}$ were right, so a field that does not depend on $t$ (like
the logistic equation) was integrated at fifth order, but one that does was integrated
at first order: on $y' = y\cos t$ with fixed steps 0.1 down to 0.0125, the error at
$t = 2$ fell only from $1.3\times10^{-2}$ to $1.6\times10^{-3}$. It now falls from
$2.9\times10^{-9}$ to $7.3\times10^{-14}$.

API: [`DormandPrinceSolver`](../api/deterministic-solvers.md#discrecontinual_equations.solver.deterministic.dormand_prince.dormand_prince_solver.DormandPrinceSolver).

### Prince-Dormand 8(7): `PrinceDormand87Solver`

This is the thirteen-stage pair of Prince and Dormand
([PrinceDormand1981](../references.md#deterministic-solvers)). It has an eighth-order
formula to advance with and an embedded *seventh*-order formula for the error
estimate. RKSUITE carries the same pair to 30 digits
([BrankinGladwellShampine1991](../references.md#deterministic-solvers)) and calls the
second row "the one of order 7". The solver advances with the eighth-order row `b8`
and estimates the error with the seventh-order `b7`. Until DEQ-139 it was named
`DormandPrince89Solver`, for a ninth-order formula it never had; that name and
`DormandPrince89Config` still work and emit a `DeprecationWarning`. The published
coefficients are rational approximations good to about $10^{-17}$; every one agrees
with RKSUITE to within one rounding.

```python
from discrecontinual_equations.solver.deterministic.prince_dormand87.prince_dormand87_config import (
    PrinceDormand87Config,
)
from discrecontinual_equations.solver.deterministic.prince_dormand87.prince_dormand87_solver import (
    PrinceDormand87Solver,
)

for tol in (1e-6, 1e-9):
    config = PrinceDormand87Config(
        end_time=2.0, absolute_tolerance=tol, relative_tolerance=tol,
    )
    curve = PrinceDormand87Solver(config).solve(logistic_equation(), [0.1])
    print(tol, len(curve) - 1, curve.results[0].discretization[-1] - exact)
# 1e-06 5 -7.17e-10
# 1e-09 8 -7.26e-12   (eight steps of thirteen stages)
```

Until DEQ-139 one eighth-order weight was mistyped: `b8[7]` read
$561292735/797845732$ where RKSUITE has $0.7035106694\ldots = 561292985/797845732$.
The weights summed to $1 - 3.1\times10^{-7}$, so every step lost about
$3.1\times10^{-7}\,h f$ and the error stalled between $10^{-7}$ and $10^{-6}$; at
`1e-9` the example took 194 steps for an error of $1.5\times10^{-7}$.

API: [`PrinceDormand87Solver`](../api/deterministic-solvers.md#discrecontinual_equations.solver.deterministic.prince_dormand87.prince_dormand87_solver.PrinceDormand87Solver).

## Limits common to all the solvers

- **Stiffness.** Every solver here is explicit. On a stiff problem (widely separated
  time scales, or large negative eigenvalues of $\partial f/\partial y$), stability
  rather than accuracy limits the step, and the adaptive solvers take very many tiny
  steps.
- **Smoothness.** The orders assume $f$ is smooth enough: $p$ continuous derivatives
  for order $p$. Near a kink or a jump in $f$, the adaptive solvers shrink the step
  and the fixed-step solvers lose order.
- **Cost.** Each step evaluates $f$ through Python lists, one `eval` call per stage.
  These solvers suit small systems and teaching, not large production runs.

## References

All the sources, and what each was checked against, are on the
[References](../references.md#deterministic-solvers) page: Euler1768, Bashforth1883,
Kutta1901, Butcher1963, Fehlberg1969, CashKarp1990, DormandPrince1980,
PrinceDormand1981, Gustafsson1991, HairerWanner1996, HairerWanner853,
BrankinGladwellShampine1991, GSL2024 and BoostOdeint2024.
