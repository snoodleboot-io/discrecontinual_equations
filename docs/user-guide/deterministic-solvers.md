# Deterministic solvers

The deterministic solvers integrate an initial value problem

$$
y'(t) = f(t, y(t)), \qquad y(t_0) = y_0,
$$

from `start_time` to `end_time`. There are two kinds:

- **Fixed-step methods** step along a grid of equal steps $h$: forward Euler and the
  Adams-Bashforth methods with two and three steps.
- **Embedded Runge-Kutta pairs** choose each step so that an estimate of the local
  error meets a tolerance: Cash-Karp 4(5), Runge-Kutta-Fehlberg 4(5), Dormand-Prince
  5(4), and the thirteen-stage Prince-Dormand 8(7) pair (`DormandPrince89Solver`).

All of them are explicit. That makes them cheap per step, but on a stiff problem they
are forced to take small steps (see [Limits](#limits-common-to-all-the-solvers)).

!!! warning "Known defects"
    We checked each solver against its source while writing these docs and found
    defects in four of them. They are recorded here and in the docstrings, and not yet
    fixed (this pass changed no code):

    - **Dormand-Prince 5(4)** uses the node $c_5 = 7/9$; the source has $8/9$. When
      $f$ depends on $t$, the result is only first order.
    - **Cash-Karp** has its two weight rows swapped, so it advances with the
      *fourth*-order solution.
    - **`DormandPrince89Solver`** is an 8(7) pair, not 8(9), and one of its
      eighth-order weights is mistyped. Its error stalls near $10^{-7}$.
    - **Adams-Bashforth 3** is second order, not third, because of how it is started.

    The table below gives the order each solver actually reaches.

| Solver | Kind | Order claimed | Order measured | Field evaluations per step |
|---|---|---|---|---|
| `EulerSolver` | fixed step | 1 | 1 | 1 |
| `AdamsBashforth2Solver` | fixed step, 2 steps | 2 | 2 | 1 |
| `AdamsBashforth3Solver` | fixed step, 3 steps | 3 | **2** | 1 |
| `CashKarpSolver` | adaptive | 4(5) | **4** | 6 |
| `FehlbergSolver` | adaptive | 4(5) | 5 (it advances with the fifth-order row) | 6 |
| `DormandPrinceSolver` | adaptive | 5(4) | 5 if $f$ does not depend on $t$, **1** if it does | 7 |
| `DormandPrince89Solver` | adaptive | "8(9)" | **none**: the error stalls at about $10^{-7}$ | 13 |

We measured the orders by running each solver with fixed steps on
$y' = -y^2$ and on $y' = y\cos t$ up to $t = 2$, halving $h$ three times. The adaptive
solvers were held to a fixed step by setting `min_step_size = max_step_size`.

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
- It departs from Euler's formula: the code evaluates $f(t_{n+1}, y_n)$, not
  $f(t_n, y_n)$. This makes no difference when $f$ does not depend on $t$. When it
  does, the method is a different one, though still first order.
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
one. These solvers take the first step with Euler. The three-step solver then takes
its second step with Adams-Bashforth 2.

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
    # AdamsBashforth3Config takes number_of_steps (the node count) for its grid,
    # and step_size for the step itself.
    ab3 = AdamsBashforth3Solver(
        AdamsBashforth3Config(end_time=2.0, number_of_steps=n + 1, step_size=h),
    )
    e2 = ab2.solve(logistic_equation(), [0.1]).results[0].discretization[-1] - exact
    e3 = ab3.solve(logistic_equation(), [0.1]).results[0].discretization[-1] - exact
    print(h, e2, e3)
# 0.1 -0.00268... -0.00200...
# 0.05 -0.00069... -0.00052...   (both errors fall by 4: second order)
```

**Limits.**

- **Adams-Bashforth 3 is second order.** A three-step method of order 3 needs
  starting values accurate to $O(h^3)$. The Euler first step is accurate only to
  $O(h^2)$, and so the solver converges at order 2.
- The start is also misaligned. The derivative the solvers store for node 1 is
  $f(t_1, y_0)$, the one the Euler step used, rather than $f(t_1, y_1)$. This adds
  only $O(h^2)$ errors a fixed number of times, so Adams-Bashforth 2 keeps order 2.
- The step must stay fixed, because the coefficients assume equal steps.
- Explicit Adams methods have small stability regions, so they suit non-stiff
  problems only.
- `AdamsBashforth3Config` builds its grid from `number_of_steps` (`linspace`) or
  `np.arange(start, end, step_size)`, which stops one step short of `end_time`
  (being changed in DEQ-129). Its step always comes from `step_size`, so give
  `step_size` in either case.

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
with the higher-order row is called *local extrapolation*. All four solvers here
advance with the row named `b5` (or `b8`), whatever its actual order (see below).

Every pair uses the same step control. A step is accepted when

$$
\max_i \frac{\mathrm{err}_i}{\mathrm{atol} + \mathrm{rtol}\,\max(|y_{n,i}|, |y_{n+1,i}|)} \le 1,
$$

and the next step proposed, accepted or not, is

$$
h_{\text{new}} = h \cdot S \cdot r^{-(1+\beta)/q},
$$

clipped to `[min_step_size, max_step_size]`. Here $r$ is the same ratio with $|y_n|$
alone in the denominator, $S$ is `safety_factor` (default 0.9), $\beta$ is `beta`
(default 0.04), and $q = 5$ for the 4(5) and 5(4) pairs ($q = 8$ for the 8(7) pair).
The last step is shortened so that it lands exactly on `end_time`.

The defaults $S = 0.9$ and $\beta = 0.04$ are those of Hairer and Wanner's `DOPRI5`
([HairerWanner1996](../references.md#deterministic-solvers)). Its controller
differs from this one in three ways:

- It is a PI ("Lund-stabilized") controller,
  $h_{\text{new}} = h\,S\,\mathrm{err}^{-(0.2 - 0.75\beta)}\,\mathrm{err}_{\text{old}}^{\beta}$.
  The controller here never uses the previous error, so it is an integral controller
  with a slightly larger exponent, even though the code calls it PI.
- It measures the error in the RMS norm; this code takes the maximum over components.
- It limits each change of step to a factor between 0.2 and 10.

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
# 1e-06 5 1.25e-06    (5 accepted steps)
# 1e-09 16 7.08e-10
```

The tolerance bounds the local error of each step. It does not bound the global
error, which can be larger.

**Limits common to the four pairs.**

- **A rejected step at `min_step_size` is retried forever.** After a rejection the
  step becomes `max(h_new, min_step_size)`. A step that fails at `min_step_size` is
  therefore tried again at the same size, and the
  `RuntimeError("Step size became too small")` in the code can never be raised. Choose
  `min_step_size` well below the step your tolerance needs, or loosen the tolerance.
- If the error estimate is exactly zero, the next step jumps to `max_step_size`.
- The curve records only the accepted nodes, so its spacing varies. There is no dense
  output to fill in between.
- The pairs are explicit, so a stiff problem forces steps limited by stability, not
  accuracy.

### Cash-Karp 4(5)

Cash and Karp's six-stage tableau
([CashKarp1990](../references.md#deterministic-solvers)) was designed for a
variable-order method on right-hand sides that vary rapidly. This solver uses it as a
plain 4(5) pair with nodes $c = (0, \tfrac15, \tfrac{3}{10}, \tfrac35, 1, \tfrac78)$.

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
# 16 1.56e-08
```

**Limit: the weight rows are swapped.** The row named `b4`,
$(\tfrac{37}{378}, 0, \tfrac{250}{621}, \tfrac{125}{594}, 0, \tfrac{512}{1771})$,
is the fifth-order row: it satisfies the order conditions, solved in exact
arithmetic, through order 5. The row named `b5` satisfies them only through order 4.
The solver advances with `b5`, so it returns the fourth-order solution. You can see
this in the example: at the same tolerance and step count, its error is twenty times
Dormand-Prince's. The error estimate itself is unaffected.

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
# 18 -3.67e-10
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

**Limit: one node is wrong.** The code has $c_5 = 7/9$. The source has $8/9$, which is
also the sum of the row $a_{5j}$, as every explicit Runge-Kutta tableau requires. The
weights and the $a_{ij}$ are right, so when $f$ does not depend on $t$ (like the
logistic equation) the solver is fifth order. When $f$ does depend on $t$, the fifth
stage is evaluated at the wrong time and the result is first order. On
$y' = y\cos t$ with fixed steps from 0.1 down to 0.0125, the error at $t = 2$ only
halves with each halving of $h$, from $1.3\times10^{-2}$ to $1.6\times10^{-3}$. Until
this is fixed, use the Fehlberg pair for a field that depends on $t$.

API: [`DormandPrinceSolver`](../api/deterministic-solvers.md#discrecontinual_equations.solver.deterministic.dormand_prince.dormand_prince_solver.DormandPrinceSolver).

### Prince-Dormand 8(7): `DormandPrince89Solver`

This is the thirteen-stage pair of Prince and Dormand
([PrinceDormand1981](../references.md#deterministic-solvers)). It has an eighth-order
formula to advance with and an embedded *seventh*-order formula for the error
estimate. RKSUITE carries the same pair to 30 digits
([BrankinGladwellShampine1991](../references.md#deterministic-solvers)) and calls the
second row "the one of order 7". The class name and its `b9` row suggest a ninth-order
formula, but there is none. The step exponent $1/8$ is the right one for an 8(7) pair.

```python
from discrecontinual_equations.solver.deterministic.dormand_prince89.dormand_prince89_config import (
    DormandPrince89Config,
)
from discrecontinual_equations.solver.deterministic.dormand_prince89.dormand_prince89_solver import (
    DormandPrince89Solver,
)

for tol in (1e-6, 1e-9):
    config = DormandPrince89Config(
        end_time=2.0, absolute_tolerance=tol, relative_tolerance=tol,
    )
    curve = DormandPrince89Solver(config).solve(logistic_equation(), [0.1])
    print(tol, len(curve) - 1, curve.results[0].discretization[-1] - exact)
# 1e-06 8 -1.41e-07
# 1e-09 194 -1.55e-07   (more steps, no better: see below)
```

**Limit: one eighth-order weight is mistyped.** `b8[7]` is
$561292735/797845732 \approx 0.7035103561$. Prince and Dormand's value, as RKSUITE
carries it, is $0.7035106694\ldots = 561292985/797845732$. Every other coefficient
agrees to $10^{-16}$. Because of the typo the weights sum to $1 - 3.1\times10^{-7}$,
so the formula used to advance is not even consistent. Each step loses about
$3.1\times10^{-7}\, h f$, and the global error stays near $10^{-7}$ whatever the step.
A tight tolerance makes it worse: the error estimate now contains that same
inconsistency, so the controller shrinks the step (194 steps above) without gaining
accuracy. Until the weight is fixed, use Dormand-Prince 5(4) or Fehlberg when you need
accuracy better than about $10^{-6}$.

API: [`DormandPrince89Solver`](../api/deterministic-solvers.md#discrecontinual_equations.solver.deterministic.dormand_prince89.dormand_prince89_solver.DormandPrince89Solver).

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
Fehlberg1969, CashKarp1990, DormandPrince1980, PrinceDormand1981, HairerWanner1996 and
BrankinGladwellShampine1991.
