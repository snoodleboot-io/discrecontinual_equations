# Numerical continuation

Numerical continuation follows a solution of $F(y) = 0$ as a parameter changes.
For equilibria of $\dot u = f(u, \lambda)$, $u \in \mathbb{R}^n$, the unknown is
$y = (u, \lambda) \in \mathbb{R}^{n+1}$ and $F = f$: $n$ equations in $n + 1$
unknowns. Near a point where the $n \times (n+1)$ Jacobian $F_y = [f_u \mid
f_\lambda]$ has full rank, they define a smooth curve (Kuznetsov 1998, eqs.
(10.27)–(10.28)). Continuation computes a sequence of points along that curve,
watches test functions for bifurcations, and locates the ones it finds.

This page covers the machinery: the predictor-corrector loop, the tangent, step
control, Jacobians (dense, sparse and by Taylor jets), bifurcation location,
branch switching, deflation, ring symmetry, focal values, and the drivers built
on top (families of diagrams, automatic exploration). What each bifurcation's
test function is, and the normal forms, are on the
[bifurcations page](normal-forms-detection.md). Sources are on the
[References](../references.md#numerical-continuation-predictor-corrector-jacobians-step-control)
page; the classes are in the
[API reference](../api/continuation.md#numerical-continuation).

All examples below share this setup. Run the blocks in order: later ones reuse
`build`, `Mu`, `X` and `T`.

```python
from pathlib import Path

from sweet_tea.registry import Registry

import discrecontinual_equations
from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

# The builder resolves its strategies by name: fill the registry once.
Registry.fill_registry(
    path=str(Path(discrecontinual_equations.__file__).parent),
    module="discrecontinual_equations",
    exclude=["*.tests", "*.examples", "*.plot"],
)


class Mu(Parameter, name="mu", abbreviation="mu"):
    pass


class X(Variable, name="x", abbreviation="x"):
    pass


class T(Variable, name="t", abbreviation="t"):
    pass


def build(field):
    mu = Mu(value=0.0)
    return DifferentialEquation(
        variables=[X()],
        time=T(),
        parameters=[mu],
        derivative=field(variables=[X()], parameters=[mu], results=[X()], time=None),
    )
```

## The predictor-corrector loop

### The idea

Stepping the parameter and re-solving $f(u, \lambda) = 0$ for $u$ fails at a
fold, where the branch turns back and $\lambda$ stops being a usable coordinate
(Allgower and Georg 1990, Algorithm (1.6) and Fig. 1.b). Continuation steps along
the curve instead. From an accepted point $y_j$ with unit tangent $v_j$:

1. **Predict** with the tangent (Kuznetsov 1998, eq. (10.29); Allgower and Georg,
   eq. (2.2.2)):
   $$\tilde y_{j+1} = y_j + h_j v_j.$$
2. **Correct** by Newton on $F$ plus one scalar equation $g_j(y) = 0$ that picks
   a single point of the curve near the prediction (Kuznetsov, eq. (10.32)). The
   default is Keller's pseudo-arclength condition, the hyperplane through the
   prediction normal to the tangent (Kuznetsov, eq. (10.34); Keller 1977):
   $$g_j(y) = \langle y - \tilde y_{j+1}, v_j\rangle = \langle y - y_j, v_j\rangle - h_j.$$
   Each Newton step solves the bordered system
   $\begin{pmatrix} F_y \\ v_j^\top \end{pmatrix}\delta = -\begin{pmatrix} F \\ g_j\end{pmatrix}$,
   which stays nonsingular through a fold because $v_j$ is not orthogonal to the
   kernel of $F_y$ (Allgower and Georg, discussion of eq. (2.1.5)). The
   alternative `parameterization="natural"` fixes $\lambda$ (Kuznetsov's
   (10.33) with $\lambda$ as the coordinate) and is singular at every fold. The
   constraint itself lives in `parameterization.py`, documented in the
   parameterization-module section of the [manifolds page](manifolds.md).
3. **Tangent** at the new point from the appended system
   $\begin{pmatrix} F_y(y_{j+1}) \\ v_j^\top\end{pmatrix} v = \begin{pmatrix} 0\\ 1\end{pmatrix}$,
   normalised; the last row fixes both the scale and the direction of travel
   (Kuznetsov, after eq. (10.30)). At the seed, $v_j^\top$ is replaced by
   $e_{n+1}^\top$ and `direction` picks the sign.
4. **Adapt the step** by how hard the corrector worked: grow by `step_growth`
   when it took fewer than `target_newton_iterations`, shrink by `step_shrink`
   when it took more than target plus `newton_iteration_slack`, halve on failure,
   stop below `minimum_step` (Kuznetsov, §10.2.1 "Step-size control"; the
   iteration-count idea of Den Heijer and Rheinboldt in Allgower and Georg,
   Algorithm (6.2.1)).

Before the first step the user's seed is polished by Newton at the fixed starting
parameter (Kuznetsov, §10.1.1).

### Example

$\dot x = \mu - x^2$ has a fold at the origin: two equilibria $\pm\sqrt\mu$ for
$\mu > 0$, none for $\mu < 0$. Starting on the upper branch and travelling to
smaller $\mu$, pseudo-arclength rounds the fold and returns along the lower
branch; natural continuation stalls at it.

```python
class Fold(DeterministicFunction):
    """x' = mu - x^2: a fold at (x, mu) = (0, 0)."""

    def eval(self, point, time=None):
        return [self.parameters[0].value - point[0] ** 2]


equation = build(Fold)
config = ContinuationConfig(
    initial_parameter=1.0,
    direction=-1,
    detectors=["fold"],
    measure="component",
    parameter_lower_bound=-0.5,
    parameter_upper_bound=1.0,
)
branch = ContinuerBuilder.build(config, equation).solve(equation, [1.0])

fold = branch.special_points[0]
print(fold.kind, round(fold.parameter, 6), round(fold.state[0], 4))
print(sorted({point.stability for point in branch.points}))

natural = ContinuationConfig(
    initial_parameter=1.0,
    direction=-1,
    detectors=[],
    measure="component",
    parameterization="natural",
    parameter_lower_bound=-0.5,
    parameter_upper_bound=1.0,
)
stuck = ContinuerBuilder.build(natural, equation).solve(equation, [1.0])
for traced in (stuck, branch):
    end = traced.points[-1]
    print(round(end.parameter, 3), round(end.state[0], 3))
```

prints

```text
fold -0.0 0.0
['stable', 'unstable']
0.0 0.0
1.123 -1.06
```

The fold is located at the origin; the branch is stable above it and unstable
below. Natural continuation ends at the fold, $(\mu, x) = (0, 0)$; pseudo-arclength
ends on the lower branch, $x = -\sqrt\mu$.

### Limits

- **Bounds are checked after a step.** The last point can lie beyond
  `parameter_upper_bound` or `parameter_lower_bound` (above: $\mu = 1.123$ against
  a bound of $1$).
- **Branch jumping.** A step large compared with the distance between two
  branches can converge onto the other one; the step controller sees only the
  iteration count and cannot tell.
- **Corrector acceptance.** If Newton does not reach `newton_tolerance` within
  `maximum_newton_iterations`, the point is still accepted when the residual of
  $F$ alone (without the constraint) is below $100\times$ `newton_tolerance`.
- **Singular points.** At a branch point $F_y$ loses rank and the bordered matrix
  is singular there; the linear solver then falls back to a minimum-norm
  least-squares step (Allgower and Georg, Lemma (3.2.3)). An Euler-Newton method
  steps over a simple branch point and continues on the same branch (Allgower and
  Georg, §8.1); to follow the other one, switch branches (below).
- **The arclength is pseudo-arclength**: the sum of accepted steps $h_j$, not the
  length of the curve.
- **The seed polish returns its last iterate** even when it did not converge;
  check the first point's residual when the seed is a rough guess.

Sources: [AllgowerGeorg1990, Keller1977, Kuznetsov1998](../references.md#numerical-continuation-predictor-corrector-jacobians-step-control).
API: [`Continuer`](../api/continuation.md#discrecontinual_equations.continuation.continuer.Continuer),
[`ContinuationConfig`](../api/continuation.md#discrecontinual_equations.continuation.continuation_config.ContinuationConfig),
[`TangentPredictor`](../api/continuation.md#discrecontinual_equations.continuation.predictor.TangentPredictor),
[`NewtonCorrector`](../api/continuation.md#discrecontinual_equations.continuation.corrector.NewtonCorrector),
[`BorderedTangentComputer`](../api/continuation.md#discrecontinual_equations.continuation.tangent.BorderedTangentComputer),
[`AdaptiveStepController`](../api/continuation.md#discrecontinual_equations.continuation.step_controller.AdaptiveStepController),
[`NewtonSeedRefiner`](../api/continuation.md#discrecontinual_equations.continuation.seed_refiner.NewtonSeedRefiner).

## Locating a bifurcation

### The idea

Each detector computes a scalar test function $\psi(y)$ that changes sign at its
bifurcation. A bifurcation is detected when $\psi(y_k)\,\psi(y_{k+1}) < 0$
(Kuznetsov 1998, eq. (10.35)). The localizer then parametrises the chord
$y(\sigma) = y_k + \sigma(y_{k+1} - y_k)$, $\sigma \in [0, 1]$, corrects each trial
point back onto the curve, and finds the zero of $\psi$ in $\sigma$ with a scalar
root-finder: bisection by default (`root_finder="bisection"`), or a clamped secant
(`"secant"`). This is location along the curve (Kuznetsov §10.2.2; Allgower and
Georg §9.2), not Newton on the augmented system $F = 0,\ \psi = 0$ (Kuznetsov's
(10.36)), so $\psi$ need not be differentiable.

The fold above was found this way. Its location is accurate to the corrector and
root-finder tolerances (`newton_tolerance`, `localization_tolerance`,
`fraction_tolerance`). `localization_tolerance` is relative. The localizer stops
once $|\psi|$ is below it times the larger $|\psi|$ at the two bracketing points,
so the located point does not depend on the units of time or state (DEQ-146; see
[Labels that do not depend on the units](../api/continuation.md#labels-that-do-not-depend-on-the-units-end-to-end-deq-146-deq-147)).

### Limits

- Two zeros of $\psi$ inside one step cancel and are missed; so is a zero where
  $\psi$ touches zero without changing sign.
- If the corrector fails at a trial point, the test value at $y_k$ is used
  instead. That keeps the bracket's sign, but bisection can then converge onto
  the failure rather than the zero.
- `Secant` does not bracket: it can stop on the clamp at $\sigma = 0$ or $1$.

Sources: [Kuznetsov1998, AllgowerGeorg1990](../references.md#numerical-continuation-predictor-corrector-jacobians-step-control).
API: [`RefiningLocalizer`](../api/continuation.md#discrecontinual_equations.continuation.localizer.RefiningLocalizer),
[`Bisection`](../api/continuation.md#discrecontinual_equations.continuation.root_finder.Bisection),
[`Secant`](../api/continuation.md#discrecontinual_equations.continuation.root_finder.Secant).

## Switching branches

Continuation follows one curve of equilibria. Where two curves cross - a transcritical or a pitchfork bifurcation, or a symmetry-breaking point in a larger system - the detector `branch_point` marks the crossing, and the curve you were following carries on through it. To follow the *other* curve you need a starting point on it. `Continuer.seed_secondary_branch(point)` returns one.

### The idea

At a simple branch point the extended Jacobian `[F_x | F_mu]` has a two-dimensional kernel that holds both curves' tangents. Which two directions in that plane are tangents is decided by the second derivatives of `F`: projected onto the left null vector `phi`, every tangent `v` satisfies the *algebraic branching equation* `<phi, F''[v, v]> = 0`. In a basis `q1, q2` of the kernel, with `q1` the tangent of the curve you followed, this is a quadratic in two unknowns whose two root directions are the two tangents ([Kuznetsov1998] §10.2.4; [AllgowerGeorg1990] §8.3). The switcher:

1. finds `phi` and `q2` as minimal singular vectors of the extended Jacobian, the latter bordered by `q1`;
2. approximates the quadratic's coefficients by second differences of `phi . F`;
3. takes the root farther from `q1` as the secondary tangent `v2`;
4. steps `initial_step` along `v2` and corrects with Newton's method on the hyperplane normal to `v2`.

It then checks the result is on a different curve: off the primary tangent line, with a tangent closer to `v2` than to `q1`, and within one kick of the prediction. If not, it tries the other side and larger kicks (up to 8 times `initial_step`), and if none passes it raises `BranchSwitchError`. It never returns a point on the curve it started from. Where the secondary curve crosses the parameter, the returned point is on the side that continuation in the configured `direction` leaves by.

### Example

The pitchfork `x' = mu x - x^3`, followed along `x = 0`, switched onto `mu = x^2` at the default settings:

```python
from pathlib import Path

from sweet_tea.registry import Registry

import discrecontinual_equations
from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class Pitchfork(DeterministicFunction):
    def eval(self, point, time=None):
        mu = self.parameters[0].value
        return [mu * point[0] - point[0] ** 3]


# The builders resolve their parts by name, so fill the registry once.
Registry.fill_registry(
    path=str(Path(discrecontinual_equations.__file__).parent),
    module="discrecontinual_equations",
    exclude=["*.tests", "*.examples", "*.plot"],
)

x = Variable(name="x", abbreviation="x")
mu = Parameter(value=0.0, name="mu", abbreviation="mu")
pitchfork = DifferentialEquation(
    variables=[x],
    time=Variable(name="t", abbreviation="t"),
    parameters=[mu],
    derivative=Pitchfork(
        variables=[x],
        parameters=[mu],
        results=[Variable(name="x'", abbreviation="dx")],
        time=None,
    ),
)

config = ContinuationConfig(
    detectors=["branch_point"],
    initial_parameter=-2.0,
    parameter_lower_bound=-2.0,
    parameter_upper_bound=2.0,
)
continuer = ContinuerBuilder.build(config, pitchfork)
trivial = continuer.solve(pitchfork, initial_values=[0.0])

state, seed_mu = continuer.seed_secondary_branch(trivial.special_points[0])
print(state, seed_mu)  # [0.05] 0.0025: on mu = x^2, not polished back to x = 0

secondary = ContinuerBuilder.build(
    config.model_copy(update={"initial_parameter": seed_mu}),
    pitchfork,
).solve(pitchfork, initial_values=list(state))
print(max(point.state[0] for point in secondary.points))  # about 1.4
```

### Limits

- Only *simple* branch points are handled: two curves crossing transversally, a two-dimensional kernel. Where the second derivatives vanish on the kernel (three or more curves crossing, as at `x' = x^3 - x mu^2`), the quadratic is degenerate; the switcher steps along the kernel vector orthogonal to `q1` and, if Newton does not converge there, refuses.
- The tangent `q1` is the one recorded on the located point. In one dimension any error in it does not matter (the kernel is the whole plane); in higher dimensions it tilts the kernel estimate, and the larger retry kicks are what still find the branch.
- The located point must be close to the true crossing compared with `initial_step`: the result is accepted only within one kick of its prediction. A point far from any crossing is refused.
- Before this change the switcher kicked state component 0 and the parameter by `initial_step` and let a fixed-parameter Newton polish the seed. At the pitchfork with the default step of 0.05 that returned the trivial branch `x = 0`; it now returns `(x, mu) = (0.05, 0.0025)`.

Sources: [Kuznetsov1998, AllgowerGeorg1990](../references.md#numerical-continuation-predictor-corrector-jacobians-step-control).
API: [`NullSpaceBranchSwitcher`](../api/continuation.md#discrecontinual_equations.continuation.switching.NullSpaceBranchSwitcher).

## Jacobians

### Finite differences, shared per point

By default the Jacobian is a central difference with an absolute step
$\varepsilon$ = `finite_difference_epsilon` ($10^{-7}$) for every state column and
for $\partial f/\partial\lambda$:
$$J_{:,j} \approx \frac{F(u + \varepsilon e_j) - F(u - \varepsilon e_j)}{2\varepsilon},$$
$2n + 2$ residual evaluations per extended Jacobian. A residual that knows its
own derivatives supplies them instead (`jacobian="analytic_jacobian"` with a
`DifferentiableResidual`). Every collaborator shares one memoizing provider, so
the Jacobian at an accepted point is computed once and read by the tangent, the
detectors and the stability classification.

### Sparse Jacobians by column groups

A field whose component $i$ reads a few states has a Jacobian with $O(n)$
non-zeros, but a column-by-column difference still costs $n$ pairs of
evaluations. Curtis, Powell and Reid (1974) observed that *structurally
orthogonal* columns — no non-zero in a common row — can be differenced together:
perturbing along $d = \sum_{j \in G} e_j$ gives, in each row, the derivative with
respect to the one column of $G$ that row depends on. Coleman and Moré (1983)
recast the grouping as colouring the column intersection graph (Gebremedhin,
Manne and Pothen 2005, §1.2). At least $\rho_M$ groups are needed, the most
non-zeros in a row, and a matrix of bandwidth $b$ needs at most $b$ (Moré 2001,
§3). `SparsityPattern` colours greedily in column order, as Curtis, Powell and
Reid did; on a ring it reaches the optimum, two groups for even $n$ and three for
odd.

```python
from discrecontinual_equations.continuation.sparsity import SparsityPattern

for n in (6, 7):
    ring = SparsityPattern.from_rows([[i, (i + 1) % n] for i in range(n)], n)
    print(n, ring.group_count, ring.groups.tolist())
```

prints

```text
6 2 [0, 1, 0, 1, 0, 1]
7 3 [0, 1, 0, 1, 0, 1, 2]
```

In continuation, set `ContinuationConfig(jacobian_sparsity=...)` to the list of
state indices each equation reads (`[[i, (i + 1) % n] for i in range(n)]` for the
ring), or to `"detect"` to read the pattern from dense Jacobians at the first
point. With a pattern that covers the true one, every entry is bit-identical to
the dense difference.

### Exact derivatives by Taylor jets

Detectors of higher codimension (cusp, generalized Hopf, codim 3) need second to
fifth derivatives, which finite differences cannot deliver. A `Jet` carries the
truncated Taylor coefficients $[f, f', f''/2!, \dots]$ along one direction;
evaluating a field on jets seeded at $x_0 + s\,v$ gives the coefficients of
$f(x_0 + s v)$, i.e. $D^m f(v, \dots, v)/m!$, with no step size (Taylor-mode
algorithmic differentiation, Griewank and Walther 2008, Chapter 13). `Bivariate`
and `Multivariate` do the same with full multivariate polynomials, for the focal
values and the centre manifold reduction.

```python
from discrecontinual_equations.continuation.jet import Jet

x = Jet.seed(2.0, 1.0, 3)  # x = 2 + s
print((x * x * x).coefficients)  # (2 + s)^3 = 8 + 12 s + 6 s^2 + s^3
```

prints `[8.0, 12.0, 6.0, 1.0]`.

### Dividing series

A rational field, such as a predation rate $xy/(1+x)$, evaluates on series too.
Dividing by a number scales every coefficient. Dividing by a series $w$ inverts
the product: from $u = h\,w$, the part of $h$ of exact total degree $k$ is

$$
h_k = \frac{1}{w_0}\Big(u_k - \sum_{j=0}^{k-1} h_j\, w_{k-j}\Big),
$$

where $u_k$, $w_k$, $h_k$ are the homogeneous parts of degree $k$. This is the
one-variable division formula of [Neidinger2010](../references.md#numerical-continuation-predictor-corrector-jacobians-step-control)
(§6, derived from the product rule (6.2)) applied to homogeneous parts, the
extension his §8 describes for series in several variables. Every coefficient up
to the series' `order` is exact.

```python
from discrecontinual_equations.continuation.bivariate import Bivariate

s = Bivariate({(1, 0): 1.0}, 4)
t = Bivariate({(0, 1): 1.0}, 4)
print((s / 2).coefficients)
quotient = 1 / (1 - s - t)  # the sum of (s + t)^n
print(quotient.coefficient(2, 2), quotient.coefficient(1, 3))
try:
    1 / s
except ZeroDivisionError as error:
    print("refused:", str(error).split(":")[0])
```

prints

```text
{(1, 0): 0.5}
6.0 4.0
refused: cannot divide by a series whose constant term is zero
```

The focal values need nothing more: the field's denominator is evaluated at the
equilibrium plus a small offset, so its constant term is the denominator's value
at the equilibrium.

### Limits

- **Absolute difference step.** $\varepsilon = 10^{-7}$ is the same for every
  component. It is too large for components of size $10^{-8}$ and too small for
  components of size $10^{8}$. Scale the problem, or supply analytic derivatives.
- **A sparsity pattern smaller than the truth gives a wrong Jacobian, silently.**
  `"detect"` reads the pattern at the first point and two probes nearby, and can
  miss a dependence that vanishes at all three.
- **Jets and series have arithmetic only.** `Jet`, `Bivariate` and
  `Multivariate` have `+ - * / **`. A field built from `sin`, `exp` and the like
  needs the finite-difference provider. Dividing by a series needs a non-zero
  constant term, the denominator's value at the expansion point; one that
  vanishes there has no power series and raises `ZeroDivisionError` (see
  [Dividing series](#dividing-series)).

Sources: [CurtisPowellReid1974, ColemanMore1983, GebremedhinMannePothen2005, More2001, GriewankWalther2008](../references.md#numerical-continuation-predictor-corrector-jacobians-step-control).
API: [`FiniteDifferenceJacobian`](../api/continuation.md#discrecontinual_equations.continuation.jacobian.FiniteDifferenceJacobian),
[`MemoizedJacobian`](../api/continuation.md#discrecontinual_equations.continuation.jacobian.MemoizedJacobian),
[`SparsityPattern`](../api/continuation.md#discrecontinual_equations.continuation.sparsity.SparsityPattern),
[`Jet`](../api/continuation.md#discrecontinual_equations.continuation.jet.Jet),
[`AutomaticDifferentiation`](../api/continuation.md#discrecontinual_equations.continuation.derivative_provider.AutomaticDifferentiation),
[`Bivariate`](../api/continuation.md#discrecontinual_equations.continuation.bivariate.Bivariate),
[`Multivariate`](../api/continuation.md#discrecontinual_equations.continuation.multivariate.Multivariate).

## Bordered matrices

Fold and Hopf curves in two parameters are continued on *minimally augmented*
systems: the equilibrium equations plus one scalar $g$ that vanishes where a
matrix $L$ is singular. $g$ is the last component of the solution of
$$\begin{pmatrix} L & p_0 \\ q_0^\top & 0\end{pmatrix}\begin{pmatrix} w \\ g\end{pmatrix} = \begin{pmatrix} 0 \\ 1\end{pmatrix},$$
nonsingular even where $L$ is, as long as $q_0$, $p_0$ are near the null vectors
of $L$ and $L^\top$; by Cramer's rule $g = \det L / \det M$, and
$g_z = -v^\top L_z w$ with $v$ from the transposed system (Kuznetsov 1998, eqs.
(10.68)–(10.70); $L = f_u$ for a fold, the bialternate product $2f_u \odot I_n$
for Hopf, eq. (10.71); Govaerts 2000, Chapter 3). `BorderedMatrix` factorises the
bordered matrix once and serves every solve, plain or transposed. The borders are
refreshed at each accepted point. One border cannot remove a two-dimensional
kernel, so the bialternate system is singular at every Hopf-Hopf point whatever
the borders (Kuznetsov's footnote to (10.71)). The curves themselves are
documented on the [bifurcations page](normal-forms-detection.md).

```python
import numpy as np

from discrecontinual_equations.continuation.bordered import BorderedMatrix

singular = np.array([[1.0, 0.0], [0.0, 0.0]])  # null vector e_2
bordered = BorderedMatrix(singular, np.array([0.0, 1.0]), np.array([0.0, 1.0]))
w_and_g = bordered.solve(np.array([0.0, 0.0, 1.0]))
print(w_and_g.round(12))  # w = e_2, g = 0: L is singular
```

prints `[0. 1. 0.]`.

Sources: [Govaerts2000, Kuznetsov1998](../references.md#numerical-continuation-predictor-corrector-jacobians-step-control).
API: [`BorderedMatrix`](../api/continuation.md#discrecontinual_equations.continuation.bordered.BorderedMatrix).

## Finding disconnected equilibria by deflation

### The idea

Continuation only follows the branch it starts on. To find equilibria that
continuation would never reach, `DeflatedSolver` runs Newton on a *deflated*
residual that cannot converge to a root already found. With roots $r_1, \dots,
r_k$ known, it solves
$$\prod_{i=1}^{k}\left(\frac{1}{\lVert x - r_i\rVert^{2}} + 1\right) f(x) = 0,$$
the shifted deflation operator $M_{p,\alpha}(u; r) = I/\lVert u - r\rVert^p +
\alpha I$ of Farrell, Birkisson and Funke (2015, Definition 2.5, eq. (2.8)) with
$p = 2$ and their default shift $\alpha = 1$, applied once per root as in their
(2.2). Far from every known root the factor tends to one, so the residual is not
artificially driven to zero. Each Newton step uses a finite-difference Jacobian of
the deflated residual, solved directly or matrix-free by GMRES
(`matrix_free=True`).

### Example

```python
import numpy as np

from discrecontinual_equations.continuation.deflation import DeflatedSolver


class Cubic(DeterministicFunction):
    """x^3 - x: roots -1, 0 and 1."""

    def eval(self, point, time=None):
        return [point[0] ** 3 - point[0]]


cubic = Cubic(variables=[X()], parameters=[], results=[X()], time=None)
roots = DeflatedSolver(cubic).find([np.array([0.5])])
print(sorted(round(float(root[0]), 6) for root in roots))
```

prints `[-1.0, 0.0, 1.0]`: all three roots from one seed.

### Limits

- **No guarantee of completeness.** Deflation stops when Newton diverges or
  returns a known root; roots outside every seed's reach stay unfound.
- **Fixed thresholds.** A candidate is a root when $\lVert f\rVert < 10^{-6}$,
  and new when it is more than $10^{-5}$ from every known root; Newton gives up
  when $\lVert x\rVert > 10^{6}$. Rescale problems whose roots are tiny or huge.
- **Finite-difference step $10^{-7}$**, absolute, as above. The deflated Jacobian
  is formed from differences of the whole product, not by the Sherman-Morrison
  form the paper uses for large systems (their §3).
- The matrix-free path accepts any finite GMRES iterate (inexact Newton), so it
  can take more Newton steps than the direct solve.

Sources: [FarrellBirkissonFunke2015](../references.md#numerical-continuation-predictor-corrector-jacobians-step-control).
API: [`DeflatedSolver`](../api/continuation.md#discrecontinual_equations.continuation.deflation.DeflatedSolver).

## Rings of identical cells

A ring of $N$ identical cells is equivariant under the cyclic shift $S$,
$(Sx)_i = x_{i+1}$: $f(Sx) = S f(x)$. At the fully symmetric equilibrium the
Jacobian is circulant, and the Fourier matrix diagonalises every circulant
(Gray 2006, Theorem 3.1). For nearest-neighbour coupling $J = aI + bS$ the mode-$k$
eigenvalue is $a + b\,e^{2\pi i k/N}$. `fourier_reduce` returns the per-mode
eigenvalues, checks that the matrix really was circulant, and names the critical
mode — the one with the largest real part, first to destabilise. A real critical
mode ($k = 0$ or $N/2$) signals a steady symmetry-breaking bifurcation, a complex
one a rotating wave.

```python
import numpy as np

from discrecontinual_equations.continuation.symmetry import (
    cyclic_action,
    fourier_reduce,
)

size = 6
jacobian = -1.0 * np.eye(size) - 0.8 * cyclic_action(size)
reduced = fourier_reduce(jacobian)
print(reduced.is_circulant, reduced.critical_mode, reduced.is_oscillatory)
print(np.round(reduced.eigenvalues.real, 3))
```

prints

```text
True 3 False
[-1.8 -1.4 -0.6 -0.2 -0.6 -1.4]
```

Mode $3 = N/2$, the alternating pattern, is closest to instability.

**Limits.** One scalar per cell: a ring of $d$-dimensional cells is
block-circulant, which this reduction does not handle (it reports
`is_circulant = False`). The module's mode $k$ is Gray's mode $N - k$ (the
Fourier matrix is built with $e^{+2\pi i jk/N}$); for a real matrix that swaps
the two members of a conjugate pair and changes nothing else. The steady/rotating
reading of the critical mode is the standard interpretation, not a check of
nondegeneracy. `equivariance_defect` tests $\lVert f(Ax) - Af(x)\rVert$ at one
point only.

Sources: [Gray2006](../references.md#numerical-continuation-predictor-corrector-jacobians-step-control).
API: [`fourier_reduce`](../api/continuation.md#discrecontinual_equations.continuation.symmetry.fourier_reduce),
[`equivariance_defect`](../api/continuation.md#discrecontinual_equations.continuation.symmetry.equivariance_defect).

## Focal values at a planar Hopf point

Whether a Hopf bifurcation is super- or subcritical is decided by the sign of the
first Lyapunov quantity, and at a Bautin point, where it vanishes, by the second.
`PlanarLyapunov` brings the linear part to the unit rotation
$(s, t) \mapsto (-t, s)$ and builds a Lyapunov function
$V = \tfrac12(s^2 + t^2) + V_3 + V_4 + \dots$ degree by degree so that
$$\dot V = \eta_4\,(s^2 + t^2)^2 + \eta_6\,(s^2 + t^2)^3 + \dots;$$
the focus is stable when the first non-zero focal value is negative
(Euzébio, Llibre and Tonon 2020, §2.1). Odd degrees are solved exactly; at each
even degree the radial term is the one direction the linear operator cannot
reach, and its coefficient is the focal value. The sign of $\eta_4$ is the sign
of the first Lyapunov coefficient $l_1$ (Kuznetsov 1998, eq. (3.20)), whose value
depends on normalisation but whose sign does not.

```python
from discrecontinual_equations.continuation.bivariate import Bivariate
from discrecontinual_equations.continuation.focal import recurse

s = Bivariate({(1, 0): 1.0}, 6)
t = Bivariate({(0, 1): 1.0}, 6)
r2 = s * s + t * t
a, b = -0.3, 0.7
field = [-1.0 * t + a * s * r2 + b * s * r2 * r2, s + a * t * r2 + b * t * r2 * r2]
print([round(value, 12) for value in recurse(field)])
```

prints `[-0.3, 0.7]`: on this normal form $\dot V = a r^4 + b r^6$ exactly.

**Erratum.** Euzébio, Llibre and Tonon print the general term as
$\eta_{2k}(x^2 + y^2)^{2k}$. Their own first terms, $\eta_2(x^2+y^2) +
\eta_4(x^2+y^2)^2$, show the exponent is $k$, which is what the code uses.

**Limits.** Planar systems only (reduce to the centre manifold first); polynomial
fields only (`+ - * / **`, with every denominator non-zero at the equilibrium); $\eta_6$ means something only where
$\eta_4 = 0$; the Jacobian must have a pair $\pm i\omega$, $\omega > 0$.

Sources: [EuzebioLlibreTonon2020, Kuznetsov1998](../references.md#numerical-continuation-predictor-corrector-jacobians-step-control).
API: [`PlanarLyapunov`](../api/continuation.md#discrecontinual_equations.continuation.focal.PlanarLyapunov),
[`recurse`](../api/continuation.md#discrecontinual_equations.continuation.focal.recurse).

## Families of diagrams and automatic exploration

### A diagram evolving in a second parameter

`FamilyDriver` traces one diagram per value of a second, *family* parameter. The
first point of each slice seeds the next, which is natural continuation of the
seed in the family parameter (Kuznetsov 1998, eq. (10.33)). For the cusp normal
form $\dot x = \mu + \nu x - x^3$ the diagram in $\mu$ has two folds when
$\nu > 0$ and none when $\nu < 0$:

```python
from discrecontinual_equations.continuation.family_builder import FamilyDriver
from discrecontinual_equations.continuation.family_config import FamilyConfig


class Nu(Parameter, name="nu", abbreviation="nu"):
    pass


class Cusp(DeterministicFunction):
    """x' = mu + nu x - x^3: two folds when nu > 0, none when nu < 0."""

    def eval(self, point, time=None):
        mu, nu = (parameter.value for parameter in self.parameters)
        return [mu + nu * point[0] - point[0] ** 3]


mu, nu = Mu(value=0.0), Nu(value=0.0)
cusp = DifferentialEquation(
    variables=[X()],
    time=T(),
    parameters=[mu, nu],
    derivative=Cusp(variables=[X()], parameters=[mu, nu], results=[X()], time=None),
)
family = FamilyDriver.run(
    FamilyConfig(
        family_parameter_index=1,
        family_values=[-0.5, 0.5, 1.0],
        initial_parameter=-2.0,
        detectors=["fold"],
        measure="component",
        parameter_lower_bound=-2.0,
        parameter_upper_bound=2.0,
    ),
    cusp,
    [-1.5],
)
for item in family.slices:
    folds = [p for p in item.branch.special_points if p.kind == "fold"]
    print(item.value, len(folds))
```

prints

```text
-0.5 0
0.5 2
1.0 2
```

**Limits.** If the seed equilibrium passes a fold in the family parameter between
two slices, the next slice's seed polish converges elsewhere and the family jumps
sheets; keep family values close there, or continue the fold curve in two
parameters.

### Exploring a bifurcation structure

`BifurcationExplorer` automates the continuation strategy of Kuznetsov (1998,
§10.4): continue an equilibrium in the first parameter and locate its folds and
Hopf points; activate the second parameter and continue each as a curve, locating
its codimension-two points; and, with a third parameter, continue those again.
Detectors are run only where they can occur (Hopf needs two dimensions, zero-Hopf
three, Hopf-Hopf four), and each curve is swept both ways from its seed.

```python
from discrecontinual_equations.continuation.exploration import (
    BifurcationExplorer,
    ExplorationConfig,
)

mu.value, nu.value = -1.0, 1.0
tree = BifurcationExplorer.explore(
    ExplorationConfig(
        parameters=[0, 1],
        ranges=[(-1.0, 1.0), (-1.0, 1.0)],
        initial_parameter=-1.0,
    ),
    cusp,
    [-1.3247],  # x^3 - x + 1 = 0 at mu = -1, nu = 1
)
for item in tree.flatten():
    print(item.codimension, item.kind, [round(abs(v), 3) for v in item.parameters])
```

prints

```text
1 fold [0.385]
1 fold [0.385]
2 cusp [0.0, 0.0]
2 cusp [0.0, 0.0]
```

The two folds at $\mu = \pm 2/(3\sqrt3) \approx \pm0.385$; each fold curve leads
to the cusp at the origin, which is found once from each.

**Limits.** Only what lies on the continued curves is found: disconnected
branches and global bifurcations (homoclinic orbits, cycle bifurcations) are
outside the tree. At most six points are escalated per level, a kind found more
than six times on one curve is dropped as noise, points are merged when their
parameters agree to three decimals, and a cusp within $0.1$ of a Bogdanov-Takens
point is dropped; all are absolute thresholds. The equation's parameter values
are left at whatever the last continuation set.

Sources: [Kuznetsov1998](../references.md#numerical-continuation-predictor-corrector-jacobians-step-control).
API: [`FamilyDriver`](../api/continuation.md#discrecontinual_equations.continuation.family_builder.FamilyDriver),
[`BifurcationExplorer`](../api/continuation.md#discrecontinual_equations.continuation.exploration.BifurcationExplorer).

## Labelling a region of parameter space

`describe_region` labels the dynamics at fixed parameters: it finds equilibria by
Newton from seeds, classifies each by its spectrum, and, in the plane, looks for a
stable limit cycle. A bounded forward trajectory whose limit set contains no
equilibrium is a periodic orbit (Poincaré–Bendixson; Teschl 2012, Lemma 7.13 and
Theorem 7.16), so the test integrates from `cycle_start` and checks that the
settled orbit stays away from every equilibrium and returns at a steady period.

```python
import numpy as np

from discrecontinual_equations.continuation.region_analysis import describe_region


class Hopf(DeterministicFunction):
    """Supercritical Hopf normal form past the bifurcation: a stable cycle r = 1."""

    def eval(self, point, time=None):
        x, y = point
        r2 = x * x + y * y
        return [x - y - x * r2, x + y - y * r2]


hopf = Hopf(variables=[X(), X()], parameters=[], results=[X(), X()], time=None)
print(describe_region(hopf, [np.array([0.1, 0.1])], cycle_start=np.array([0.1, 0.0])))
```

prints `unstable focus + limit cycle`.

**Limits.** Only attracting cycles are detected (forward integration cannot reach
a repelling one); the integration is fixed (6000 steps of $0.01$), so a slow
approach or a long period can be missed; and the thresholds are absolute. The
label is a description, not a proof.

Sources: [Teschl2012](../references.md#numerical-continuation-predictor-corrector-jacobians-step-control).
API: [`describe_region`](../api/continuation.md#discrecontinual_equations.continuation.region_analysis.describe_region).

## Diagrams

`BifurcationDiagram` splits a branch into runs of equal stability, sharing the
transition point so a renderer draws an unbroken curve, and lists the located
bifurcations as markers. The scalar plotted is the `measure`: `"norm"` uses
$\lVert u\rVert$ (which cannot tell $u$ from $-u$, so a pitchfork's two halves
overlap) and `"component"` one coordinate.

```python
from discrecontinual_equations.continuation.diagram import BifurcationDiagram

diagram = BifurcationDiagram(branch)  # the fold branch from the first example
print([segment.stability for segment in diagram.segments()])
print([point.kind for point in diagram.special_points()])
```

prints

```text
['stable', 'unstable']
['fold']
```

API: [`BifurcationDiagram`](../api/continuation.md#discrecontinual_equations.continuation.diagram.BifurcationDiagram),
[`Branch`](../api/continuation.md#discrecontinual_equations.continuation.branch.Branch).
