# Periodic orbits and Floquet multipliers

A limit cycle is a closed orbit, and most of what the library says about one comes
down to two questions: where it is, and whether it attracts. This page covers how a
cycle is computed as a boundary-value problem, how it is followed through a
parameter, how its Floquet multipliers are read and how far they can be trusted,
the normal forms of its bifurcations, and the curves those bifurcations trace in
two parameters. Every work cited is listed on the [References](../references.md#periodic-orbits-collocation-and-floquet-multipliers)
page.

## A cycle as a periodic boundary-value problem

A cycle of $\dot x = f(x, p)$ with unknown period $T$ is, on the rescaled interval
$s \in [0, 1]$,

$$
x'(s) = T\, f(x(s), p), \qquad x(1) = x(0),
$$

plus one **phase condition**, since any time shift of a cycle is a cycle too. The
default pins one coordinate, $x_i(0) = c$, which needs the orbit to cross $c$.
`CycleContinuation(..., phase="integral")` uses AUTO's integral condition against
the previous cycle $\hat x$ instead,

$$
\int_0^1 \langle x(s) - \hat x(s),\, \hat x'(s) \rangle \, ds = 0,
$$

which needs no crossing (Doedel, Keller and Kernévez 1991, part II).

The orbit is discretised on a mesh of $N$ intervals and the collocation equations,
the periodicity and the phase condition are solved together for the node states and
$T$ by Newton's method. Three schemes are offered (Ascher, Mattheij and Russell 1995,
chapter 5):

| `collocation` | per interval | order |
|---|---|---|
| `"trapezoidal"` | $x_{k+1} - x_k = \tfrac{h}{2} T (f_k + f_{k+1})$ | 2 |
| `"hermite_simpson"` (default) | $x_{k+1} - x_k = \tfrac{h}{6} T (f_k + 4 f_{k+1/2} + f_{k+1})$ | 4 |
| `"gauss_legendre"` | the equation imposed at $m$ Gauss points | $2m$ |

In Hermite–Simpson the midpoint state is the cubic Hermite interpolant's,
$x_{k+1/2} = \tfrac12 (x_k + x_{k+1}) + \tfrac{h}{8} T (f_k - f_{k+1})$. Substituting
the update for $x_{k+1}$ gives
$x_{k+1/2} = x_k + h T (\tfrac{5}{24} f_k + \tfrac13 f_{k+1/2} - \tfrac1{24} f_{k+1})$,
which is the middle stage of the three-stage Lobatto IIIA method: Hermite–Simpson
*is* collocation at the Lobatto points 0, ½, 1. Gauss–Legendre collocation with $m$
points is superconvergent: the polynomial between the mesh points is order $m + 1$,
but the mesh values, and with them the period, are order $2m$ (de Boor and Swartz
1973). The library measured orders 3.96–4.00, 5.79–5.99 and 7.86–7.99 for the
period at $m = 2, 3, 4$ on the Hopf branch.

Stability is read from the **Floquet multipliers**, the eigenvalues of the
monodromy matrix $Y(1)$ of the variational equation $Y' = T\, D_x f(x(s))\, Y$,
$Y(0) = I$. One multiplier is exactly $1$, along the orbit. A real multiplier
crossing $+1$ is a fold of cycles, one crossing $-1$ a period-doubling, and a
complex pair crossing the unit circle a Neimark–Sacker (torus) bifurcation.

```python
import math

import numpy as np

from discrecontinual_equations.continuation import HermiteSimpsonOrbit
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class Hopf(DeterministicFunction):
    """x' = mu x - y - x r^2, y' = x + mu y - y r^2: a cycle of radius sqrt(mu)."""

    def eval(self, point, time=None):
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        radius = x * x + y * y
        return [mu * x - y - radius * x, x + mu * y - radius * y]


def xy():
    return [Variable(name="x", abbreviation="x"), Variable(name="y", abbreviation="y")]


mu = Parameter(value=0.25, name="mu", abbreviation="mu")
field = Hopf(variables=xy(), parameters=[mu], results=xy(), time=None)

t = np.linspace(0.0, 2.0 * math.pi, 41)
seed = np.column_stack([0.4 * np.cos(t), 0.4 * np.sin(t)])  # the wrong radius
orbit = HermiteSimpsonOrbit(field, 40, phase_index=1, phase_value=0.0)
cycle = orbit.solve(seed, 2.0 * math.pi)
print(cycle.period, 2.0 * math.pi)  # 6.28319061, 6.28318531
print(sorted(abs(orbit.floquet_multipliers(cycle))))  # 0.0432137, 0.9999993
print(math.exp(-4.0 * math.pi * 0.25))  # the exact multiplier, 0.0432139
```

The monodromy is integrated along the cubic Hermite interpolant between the nodes
with the Jacobian at every Runge–Kutta stage, which makes it fourth order too; a
straight chord between the nodes would put an $O(h^2)$ floor under the multipliers
however well the orbit were solved.

The Jacobian $D_x f$ in the variational equation
$\Phi' = T\,D_x f(x(s))\,\Phi$, $\Phi(0) = I$ (Kuznetsov 2014, lecture 8, slide 4)
is exact wherever the field allows it: a field written in `+ - * /` and integer
powers is differentiated by dual numbers or Taylor jets (Griewank and Walther
2008). A field that calls `math`, branches on its state or takes a fractional
power is differenced centrally instead, each component moved by
$\varepsilon^{1/3}\max(|x_j|, 1)$ (Dennis and Schnabel 1983), the rule AUTO-07p
follows when no Jacobian is supplied; COLNEW asks the user for one outright. A
forward difference at a fixed step $h = 10^{-7}$ errs by $h f''/2$, the same on
every mesh, and that bias was a floor: on a forced Duffing cycle the multipliers'
error stayed at $5.9 \times 10^{-6}$ from 160 to 640 exact nodes, where with the
exact Jacobian it fell to $5.8 \times 10^{-10}$ (DEQ-128).

**Limits.** A solution means the *discrete* system was satisfied; it does not mean
the mesh resolves the orbit (see `estimate_period_error` on the
[API page](../api/continuation.md#trusting-a-computed-cycle)). The plain Newton
solver can slide into the spurious solution every collocation admits, all states
equal at period zero; `RobustPeriodicOrbit` keeps the period above a floor and
backtracks. A seed far from any cycle converges to nothing, or to another cycle.

API: [`PeriodicOrbit`](../api/continuation.md#discrecontinual_equations.continuation.periodic_orbit.PeriodicOrbit),
[`HermiteSimpsonOrbit`](../api/continuation.md#discrecontinual_equations.continuation.periodic_orbit.HermiteSimpsonOrbit),
[`RobustPeriodicOrbit`](../api/continuation.md#discrecontinual_equations.continuation.periodic_orbit.RobustPeriodicOrbit),
[`AdaptivePeriodicOrbit`](../api/continuation.md#discrecontinual_equations.continuation.periodic_orbit.AdaptivePeriodicOrbit).

## Following a cycle through a parameter

`CycleContinuation` follows a cycle by pseudo-arclength continuation (Doedel, Keller
and Kernévez 1991): the unknowns are the states, the period and the parameter, and
the extra equation fixes the step $\Delta s$ along the branch's tangent, so the branch
can turn round a fold. Between adjacent points `classify_transition` looks for a
multiplier crossing the unit circle and interpolates where it crossed.

The step is fixed unless `max_arclength` is given; then it adapts as AUTO-07p's
`ADPTDS` does with its recommended $\mathrm{ITNW} = 5$: doubled after a correction
that took at most one Newton iteration, times 1.5 after two, halved after five or
more, and never longer than `max_arclength`. A step that would jump over a
bifurcation is taken again at the base length.

```python
import math

import numpy as np

from discrecontinual_equations.continuation import (
    CycleContinuation,
    CycleSeed,
    resolved_branch,
)
from discrecontinual_equations.differential_equation import DifferentialEquation

# Hopf, xy and mu as in the example above.
equation = DifferentialEquation(
    variables=xy(),
    time=Variable(name="t", abbreviation="t"),
    parameters=[mu],
    derivative=Hopf(variables=xy(), parameters=[mu], results=xy(), time=None),
)


def circle(nodes: int) -> CycleSeed:
    """A seed with one state per mesh point, at mu = 0.25."""
    t = np.linspace(0.0, 2.0 * math.pi, nodes)
    states = np.column_stack([0.5 * np.cos(t), 0.5 * np.sin(t)])
    return CycleSeed(states, 2.0 * math.pi, 0.25)


for continuation, nodes in (
    (CycleContinuation(equation, 0, 40, phase_index=1), 41),
    (
        CycleContinuation(
            equation, 0, 12, phase_index=1,
            collocation="gauss_legendre", collocation_points=3,
        ),
        13,
    ),
):
    points, bifurcations = continuation.trace(circle(nodes), 0.05, 4)
    for point in points:
        exact = math.exp(-4.0 * math.pi * point.parameter)
        error = abs(min(abs(point.multipliers)) - exact)
        print(continuation.collocation, point.parameter, error, point.floquet_error)
    print(len(resolved_branch(points).points), "of", len(points), "resolved")
```

On this branch Hermite–Simpson on 40 intervals reads true errors of about
$1.5 \times 10^{-7}$ with `floquet_error` about $7 \times 10^{-7}$, and Gauss–Legendre
with $m = 3$ on 12 intervals true errors of about $5 \times 10^{-8}$ with
`floquet_error` about twice that.

**Limits.** The point phase condition loses a branch whose orbit stops crossing
`phase_value`; the integral one does not. With the default Euclidean arclength, a
step covers less of the parameter the more unknowns the scheme has, so
Gauss–Legendre wants about $\sqrt m$ times the step; `arclength="weighted"` (AUTO's
$L^2$ norm) removes that. Each Gauss–Legendre step costs about four times a
Hermite–Simpson one on 80 intervals, before the doubled-mesh check below.

API: [`CycleContinuation`](../api/continuation.md#discrecontinual_equations.continuation.CycleContinuation),
[`CyclePoint`](../api/continuation.md#discrecontinual_equations.continuation.CyclePoint),
[`resolved_branch`](../api/continuation.md#discrecontinual_equations.continuation.resolved_branch),
[`classify_transition`](../api/continuation.md#discrecontinual_equations.continuation.classify_transition);
the measurements behind the defaults are under
[Trusting a cycle's Floquet multipliers](../api/continuation.md#trusting-a-cycles-floquet-multipliers).

## How far the multipliers can be trusted

A converged cycle carries the discretisation's error into every multiplier, and
an under-resolved one can move a crossing of the unit circle a long way, which is
what the branch was traced to find. `CyclePoint.floquet_error` estimates that error
and `resolved()` gates on it (default $2 \times 10^{-2}$).

**The drift.** The trivial multiplier is exactly 1 for every cycle, so
$|\mu_{\text{triv}} - 1|$ is a free measurement. Under trapezoidal collocation on a
Bogdanov–Takens cycle hugging a saddle it read 8.9%, 2.5%, 0.6%, 0.2% from 80 to 640
nodes, second order in the mesh, where the period was only 0.2% off at 80 nodes.

**The two-mesh bound.** The drift is not always a bound. Gauss–Legendre reads its
multipliers off the collocation Jacobian, the derivative of a discrete period map
that is the flow of a nearby field to all orders, so its trivial multiplier stays at
1 far beyond the scheme's order. Its points therefore carry

$$
e \;\le\; \max\bigl(\,|\mu_{\text{triv}} - 1|,\; 2\,|\mu_N - \mu_{2N}|\,\bigr),
$$

with $\mu_{2N}$ the same cycle re-solved on twice the intervals. The factor 2 is the
whole argument: if doubling the mesh at least halves the error,
$e_N \le |\mu_N - \mu_{2N}| + e_{2N} \le |\mu_N - \mu_{2N}| + e_N/2$, so
$e_N \le 2 |\mu_N - \mu_{2N}|$. At order $2m$ the gap is $(1 - 4^{-m})\,e_N$, so
the bound asymptotically over-reads by $2(1 - 4^{-m})$, between 1.5 and 2. Against DOP853 references on 211 cycles (van der Pol and
the Brusselator, $m$ 1 to 4, 8 to 96 intervals) it never read below the true error,
with a median over-read of 2.0 (DEQ-55).

Under Hermite–Simpson the drift alone is not a bound either. The orbit is the exact
cycle of a field $O(h^4)$ away, the trivial multiplier is 1 for that field too, and
the drift sees only the monodromy's quadrature. The orbit's share is the
multipliers' sensitivity to the field, and on a period-doubled orbit that grows by
about Feigenbaum's $\delta \approx 4.669$ per doubling (Feigenbaum 1978) while the
drift grows only with the number of turns: on Rössler's cascade at 16 turns the
drift read 43 times low and passed cycles 6.2–6.7% wrong.

DEQ-105 gives Hermite–Simpson points the same
$\max(\text{drift}, 2\,\text{gap})$; on 45 cascade cycles it never read low, with a
median over-read of 1.87, the $2(1 - 2^{-4})$ that order four predicts.

**Limits.**

- *Trapezoidal (DEQ-125, pending).* A trapezoidal point's estimate is still the drift;
  the fix is not yet merged. On the
  DEQ-125 verifier's grid of 243 cases it passed 16 cycles whose true error was
  above the gate, 15 of them multi-turn Rössler cycles, reading as low as 0.007
  times the truth. Prefer the default scheme for multi-turn orbits.
- *The cycle actually computed (DEQ-127).* The bound is on the error of the cycle the
  solve converged to. In a forced system a seed can converge to a lower-period orbit
  traversed $k$ times, which the period cannot tell apart: seeded from a four-turn
  Duffing cycle or a twelve-turn Lorenz cycle, the solve landed on two-, six- or
  three-turn orbits traversed two or four times, and passed with errors of 1 to 68
  against the cycle seeded. Detecting the repetition is DEQ-127.
- *The Jacobian's own error.* A bias in $D_x f$ that both meshes share is seen by
  neither the gap nor the drift. Until DEQ-128 the monodromy differenced $f$
  forward at a fixed step, a bias of about $10^{-5}$ under which the estimate read
  up to four times low. With the exact Jacobian, re-run against the same Radau
  references (forced Duffing, Lorenz at $r = 350$, Langford), it read 1.87–1.88
  times the true error on all 34 cycles that converged to the one seeded, down to
  true errors of $9 \times 10^{-9}$. A field that can only be differenced carries
  the central difference's $\varepsilon^{2/3}$, about $4 \times 10^{-11}$
  relative, which a cycle of ordinary scale never reaches.
- The argument needs the error at least to halve per doubling. Before convergence is
  asymptotic two meshes can agree while both are wrong; a doubled solve that fails
  reads infinite and the point is refused.
- A point whose states collapsed onto one another reads infinite under every
  scheme, since its monodromy is near the identity and both readings above would
  call it resolved.

API: [`CyclePoint.floquet_error`](../api/continuation.md#discrecontinual_equations.continuation.CyclePoint.floquet_error),
[`ResolvedBranch`](../api/continuation.md#discrecontinual_equations.continuation.ResolvedBranch).

## Restarting a trace from one of its points

A traced branch is often wanted again from somewhere along it: further in the
parameter, with a finer step, or in the other direction. `CycleSeed.from_point`
makes a seed of a traced point that carries every one of its states, interior
collocation points included, and a `CycleContinuation` on the same scheme and
mesh starts from it as it is, as AUTO-07p restarts from a stored solution
(`IRS`). A converged point is taken without a Newton iteration; a seed that is
nearly a solution of the target scheme - a point's mesh states alone, or an
accurate integration - is corrected on that scheme by Newton's method, and kept
when every iteration at least halves the residual and the cycle lies within 5% of
the seed's size of the seed. Anything else goes through the staged solve
(trapezoidal first, then the target scheme), as before.

```python
import warnings

import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation import (
    CycleContinuation,
    CycleSeed,
    CycleSeedWarning,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class Rossler(DeterministicFunction):
    def eval(self, point, time=None):
        c = self.parameters[0].value
        x, y, z = point[0], point[1], point[2]
        return [-y - z, x + 0.2 * y, 0.2 + z * (x - c)]


def xyz():
    return [Variable(name=n, abbreviation=n) for n in "xyz"]


c = Parameter(value=2.85, name="c", abbreviation="c")
equation = DifferentialEquation(
    variables=xyz(),
    time=Variable(name="t", abbreviation="t"),
    parameters=[c],
    derivative=Rossler(variables=xyz(), parameters=[c], results=xyz(), time=None),
)


def flow(_t, u):
    return [-u[1] - u[2], u[0] + 0.2 * u[1], 0.2 + u[2] * (u[0] - 2.85)]


def section(_t, u):
    return u[1]


section.direction = 1.0
settled = solve_ivp(flow, (0, 400), [1, 1, 0], rtol=1e-10, atol=1e-12, events=section)
start, end = settled.t_events[0][-4], settled.t_events[0][-2]  # two turns
orbit = solve_ivp(flow, (0, end - start), settled.y_events[0][-4], rtol=1e-10,
                  atol=1e-12, dense_output=True)
seed = CycleSeed(orbit.sol(np.linspace(0, end - start, 41)).T, end - start, 2.85)

traced, _ = CycleContinuation(
    equation, 0, 40, phase_index=1, collocation="gauss_legendre"
).trace(seed, 0.01, 3)
point = traced[-1]
again, _ = CycleContinuation(
    equation, 0, 40, phase_index=1, collocation="gauss_legendre"
).trace(CycleSeed.from_point(point), 0.01, 3)
print(np.max(np.abs(again[0].solution.states - point.solution.states)))  # 0.0

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", CycleSeedWarning)
    hopeless = CycleSeed(0.1 * seed.states, 2.0 * seed.period, 2.85)
    print(CycleContinuation(equation, 0, 40, phase_index=1).trace(hopeless, 0.01, 3))
print(caught[0].message)  # no cycle found from the seed at parameter 2.85 ...
```

Rössler's period-2 cycle crosses the phase section $y = 0$ upward twice a period,
so the point phase condition has two solutions on it. Before DEQ-106 a point's
full states raised an `IndexError` under Gauss–Legendre, and from its mesh states
the trapezoidal stage could carry the cycle to the other crossing: on the cascade
page's four branches one restart in 45 came back 2.18 away, and at $c = 2.85$ on
40 intervals 1.44. Now all 45 restarts reproduce their point: exactly from its
full states, and to $4.4 \times 10^{-10}$ from its mesh states alone. A seed starts
1.3 to 2.3 times faster from 20 to 320 intervals. A seed from
which no cycle is found returns an empty trace in under a second at 320
intervals, with a `CycleSeedWarning` saying what was tried.

**Limits.** The restart is on the scheme and mesh the point was traced on; a seed
of any other length raises `ValueError`. A seed that is nearly a solution is
corrected on the target scheme rather than staged, so a trace started from an
accurate integration now begins up to about $10^{-14}$ from where it used to. The
5% window keeps the restart near the seed, but it cannot tell two cycles apart when
both lie inside it. `PeriodDoublingCascade` still retraces each branch from its
own seed, so that its levels keep the points they always had.

API: [`CycleSeed`](../api/continuation.md#discrecontinual_equations.continuation.CycleSeed),
[`CycleSeedWarning`](../api/continuation.md#discrecontinual_equations.continuation.CycleSeedWarning).

## Multipliers without forming the monodromy

The Gauss–Legendre multipliers are the eigenvalues of a product
$M = A_p \cdots A_2 A_1$ of per-interval transfer matrices, read off the collocation
Jacobian (Fairgrieve and Jepson 1991). Forming $M$ and calling an eigensolver gives
every eigenvalue to about $\varepsilon \lVert M \rVert$, so a strongly contracting
multiplier keeps no relative accuracy: on van der Pol at $\mu = 5$ an exact
multiplier of $7.7 \times 10^{-38}$ came out as $10^{-15}$ noise. The **periodic QR
algorithm** (Bojanczyk, Golub and Van Dooren 1992) never forms the product. Orthogonal
$Q_k$ bring every factor to triangular form, $Q_{k+1}^{H} A_k Q_k$, and each
eigenvalue of $M$ is then the product of the factors' diagonal entries in one
position, each carrying its own relative accuracy. Lust (2001) showed that this
moves the limit on the multipliers from the eigensolver to the discretisation.
The implementation follows Kressner (2006): Algorithm 2 for the periodic Hessenberg
reduction, Algorithm 4 with one complex Wilkinson shift for the QR step, and his
criterion (5) for deflation.

```python
import numpy as np

from discrecontinual_equations.continuation.periodic_schur import periodic_eigenvalues

# 100 factors stretching one direction by 1.3 and squeezing another by 0.5, in a
# basis that is not orthogonal: the product's eigenvalues are 1.3^100 and 0.5^100.
basis = np.array([[1.0, 0.5], [0.3, 1.0]])
factor = basis @ np.diag([1.3, 0.5]) @ np.linalg.inv(basis)
factors = np.array([factor] * 100)

print(1.3**100, 0.5**100)  # 2.479e11, 7.889e-31
print(abs(np.linalg.eigvals(np.linalg.multi_dot(factors))))  # 2.479e11, 0.0
print(abs(periodic_eigenvalues(factors)))  # 2.479e11, 7.889e-31
```

**Limits.** Only criterion (5) deflates; Kressner's criterion (6), a negligible
diagonal entry of a triangular factor, is not tested for, and his Algorithm 4
assumes the triangular factors nonsingular. Five random $4 \times 4$ factors with
one zero column overflowed and raised `PeriodicSchurError` after 240 steps. The
continuation's transfer matrices are nonsingular wherever their blocks solve, and on
that error it falls back to the explicit product with a `RuntimeWarning`
(`CyclePoint.multipliers_method` says which was used). No eigensolver removes the
discretisation's own error.

API: `periodic_schur.periodic_eigenvalues` is internal to the cycle continuation
and not rendered on the API page; its docstring is the reference.

## Normal forms of cycle bifurcations

At a fold, flip or torus bifurcation the cycle's centre manifold carries Iooss's
periodic normal form (Iooss 1988), in Kuznetsov's notation

$$
\begin{aligned}
\text{fold:}\quad & \dot\tau = 1 - \xi + a\xi^2, & \dot\xi &= b\,\xi^2,\\
\text{flip:}\quad & \dot\tau = 1 + a\xi^2, & \dot\xi &= c\,\xi^3,\\
\text{torus:}\quad & \dot\tau = 1 + a|\xi|^2, & \dot\xi &= \tfrac{i\theta}{T}\,\xi + d\,\xi|\xi|^2.
\end{aligned}
$$

$b$ says which side of the fold the two cycles lie on, $c < 0$ makes a flip
supercritical and $\operatorname{Re} d < 0$ a torus bifurcation; each vanishes at the
codimension-two point that ends its curve's nondegenerate stretch (cusp of cycles,
generalised period-doubling, Chenciner). Kuznetsov, Govaerts, Doedel and Dhooge
(2005) give each coefficient as a Fredholm solvability condition on linear periodic
boundary-value problems along the critical cycle, with the multilinear forms $B$
and $C$ of $f$; for the fold,

$$
b = \tfrac12 \int_0^T \bigl\langle \varphi^*(\tau),\, B(\tau; v, v) + 2A(\tau)v \bigr\rangle\, d\tau,
$$

with $v$ the generalised eigenfunction and $\varphi^*$ the adjoint null function
normalised $\int \langle \varphi^*, v \rangle = 1$ (Kuznetsov 2014, slides 9–11; the
flip's $c$ is slides 12–16, the torus's $d$ slides 17–21). `CycleNormalForm` solves
every one of those problems by Hermite–Simpson collocation on the cycle's own mesh,
where the notes use orthogonal collocation on $[0, 1]$; no Poincaré map is used.

```python
import math

import numpy as np

from discrecontinual_equations.continuation import CycleNormalForm
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

C, E = 1.0, 0.5


class Bautin(DeterministicFunction):
    """r' = r (m1 + m2 r^2 + C r^4 - r^6), theta' = 1 + E r^2, in x and y."""

    def eval(self, point, time=None):
        m1, m2 = self.parameters[0].value, self.parameters[1].value
        x, y = point[0], point[1]
        rho = x * x + y * y
        growth = m1 + m2 * rho + C * rho * rho - rho * rho * rho
        omega = 1.0 + E * rho
        return [growth * x - omega * y, growth * y + omega * x]


def xy():
    return [Variable(name="x", abbreviation="x"), Variable(name="y", abbreviation="y")]


def circle(rho: float, intervals: int) -> tuple[np.ndarray, float]:
    """The cycle of squared radius rho: nodes over one period, and the period."""
    period = 2.0 * math.pi / (1.0 + E * rho)
    angle = np.linspace(0.0, 2.0 * math.pi, intervals + 1)
    radius = math.sqrt(rho)
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle)]), period


# Cycles sit at the roots rho = r^2 of m1 + m2 rho + C rho^2 - rho^3; a double root
# is a fold, where b = -rho (2C - 6 rho)(1 + E rho) / E.
rho = 0.6
m1, m2 = -2.0 * rho**3 + C * rho**2, 3.0 * rho**2 - 2.0 * C * rho
parameters = [
    Parameter(value=m1, name="m1", abbreviation="m1"),
    Parameter(value=m2, name="m2", abbreviation="m2"),
]
field = Bautin(variables=xy(), parameters=parameters, results=xy(), time=None)
states, period = circle(rho, 40)
print(CycleNormalForm(field, states, period).fold())  # 2.495984
print(-rho * (2 * C - 6 * rho) * (1 + E * rho) / E)  # 2.496 exactly
```

**Limits.** The coefficients converge at fourth order, like the cycle: $b$ above
fell $1.7 \times 10^{-3}$, $1.05 \times 10^{-4}$, $6.5 \times 10^{-6}$,
$4.1 \times 10^{-7}$ (relative) from 10 to 80 intervals. The cycle must be *at* the
bifurcation, since the singular problems are bordered by near-null vectors that
exist only there; off it a number comes out and means nothing. $b$ needs the fold's
double multiplier to be a Jordan block: where the frequency does not vary with the
amplitude ($E = 0$ above) it is semisimple and $b$ is undefined. The torus formula
assumes no strong resonance, $e^{ik\theta} \ne 1$ for $k = 1, \dots, 4$. The field
must accept Taylor jets: no comparisons on the state and no `math` calls.

API: [`CycleNormalForm`](../api/continuation.md#discrecontinual_equations.continuation.CycleNormalForm).

## Bifurcation curves of cycles in two parameters

A fold, flip or torus found on a one-parameter branch persists along a curve in two
parameters. `CycleCurveContinuation` follows it on the cycle's own collocation
system, extended by the bifurcation condition in **minimally augmented** form
(Doedel, Govaerts and Kuznetsov 2003): a linear operator $L$ built from the
collocation blocks, singular exactly on the curve, is bordered to a nonsingular one,

$$
\begin{pmatrix} L & P \\ Q^{T} & 0 \end{pmatrix}
\begin{pmatrix} V \\ G \end{pmatrix}
=
\begin{pmatrix} 0 \\ I \end{pmatrix},
$$

and the corner $G$ is the test function. For a fold $L$ is the cycle system's
Jacobian; for a flip the interval blocks with the antiperiodic condition
$v_N + v_0 = 0$; for a torus, on the doubled mesh, the blocks with
$v_{2N} - 2\kappa v_N + v_0 = 0$, singular exactly when $e^{\pm i\theta}$ with
$\cos\theta = \kappa$ are multipliers, with $\kappa$ one more unknown. Along a curve
the codimension-two points of cycles (De Witte, Della Rossa, Govaerts and Kuznetsov
2013) are located where a test function changes sign: a second critical multiplier
(fold–flip, fold–Neimark–Sacker, flip–Neimark–Sacker, double Neimark–Sacker), the
strong resonances 1:1 to 1:4 as $\kappa$ passes $1, -1, -\tfrac12, 0$, and the
curve's own normal form coefficient vanishing (cusp, generalised period-doubling,
Chenciner).

Continuing from the normal-form example, trace a branch in $m_1$ to its fold, then
the fold curve in $(m_1, m_2)$ to its cusp, which is exactly at
$(C^3/27, -C^2/3)$:

```python
from discrecontinual_equations.continuation import (
    CycleContinuation,
    CycleCurveContinuation,
    CycleSeed,
)
from discrecontinual_equations.differential_equation import DifferentialEquation

m2, rho = -0.12, 0.8
m1 = -(m2 * rho + C * rho**2 - rho**3)
parameters = [
    Parameter(value=m1, name="m1", abbreviation="m1"),
    Parameter(value=m2, name="m2", abbreviation="m2"),
]
equation = DifferentialEquation(
    variables=xy(),
    time=Variable(name="t", abbreviation="t"),
    parameters=parameters,
    derivative=Bautin(variables=xy(), parameters=parameters, results=xy(), time=None),
)
states, period = circle(rho, 40)
branch = CycleContinuation(equation, 0, 40, phase_index=1)
points, found = branch.trace(CycleSeed(states, period, m1), 0.02, 80, direction=-1.0)
fold = next(b for b in found if b.kind == "fold_of_cycles")
start = min(points, key=lambda point: abs(point.parameter - fold.parameter))

curve = CycleCurveContinuation(equation, (0, 1), 40, phase_index=1).trace(
    start, "fold_of_cycles", 0.02, 24, direction=-1.0,
)
for special in curve.special:
    print(special.kind, special.parameters)  # cusp_of_cycles (0.0370370, -0.3333333)
print(C**3 / 27.0, -C * C / 3.0)
```

The located cusp agrees with the exact one to about $10^{-12}$. The example takes
some twenty seconds.

**Limits.** Only the one-point-per-interval schemes are supported; Gauss–Legendre is
refused. A codimension-two point is located by the secant method between the two
curve points that bracket the sign change; two sign changes within one step cancel
and are not seen, and the point's own normal form (De Witte et al.'s coefficients,
which decide the unfolding) is not computed. The corner's gradient is differenced
with relative steps of $10^{-7}$. Past a 1:1 or 1:2 resonance a torus curve
continues as a curve of neutral saddle cycles ($\kappa$ outside $[-1, 1]$), where
no $d$ and no Chenciner point is computed.

API: [`CycleCurveContinuation`](../api/continuation.md#discrecontinual_equations.continuation.CycleCurveContinuation),
[`CycleCurve`](../api/continuation.md#discrecontinual_equations.continuation.CycleCurve),
[`CycleCurvePoint`](../api/continuation.md#discrecontinual_equations.continuation.CycleCurvePoint),
[`CycleCodim2Point`](../api/continuation.md#discrecontinual_equations.continuation.CycleCodim2Point);
the measurements are under [Cycle bifurcations in two parameters](../api/continuation.md#cycle-bifurcations-in-two-parameters-deq-35).

## Evaluating a field at every node at once

Every Newton step of a cycle needs $f$ and $D_x f$ at every node and the monodromy
needs them at nine points of every interval. Called a state at a time, a field
written in Python costs one interpreter pass per state. Most fields are written in
`+ - * /` and integer powers, and those evaluate just as well with every coordinate
an array over all the states. `field_values` hands the field arrays;
`field_jacobians` hands it first-order dual numbers carrying every direction at once,
forward-mode automatic differentiation (Griewank and Walther 2008, chapter 3), so one
pass gives every column of every Jacobian. A field that cannot take arrays returns
`None`, and the caller evaluates a state at a time as before.

```python
import numpy as np

from discrecontinual_equations.continuation.batched_field import field_values
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class Cubic(DeterministicFunction):
    def eval(self, point, time=None):
        x, y = point[0], point[1]
        return [x - x**3 - y, x]


class Clipped(DeterministicFunction):
    """Branches on its state, so it cannot take a whole batch."""

    def eval(self, point, time=None):
        x, y = point[0], point[1]
        return [x if x > 0 else 0.0, -y]


def xy():
    return [Variable(name="x", abbreviation="x"), Variable(name="y", abbreviation="y")]


states = np.random.default_rng(1).standard_normal((1000, 2))
for kind in (Cubic, Clipped):
    field = kind(variables=xy(), parameters=[], results=xy(), time=None)

    def one(state, field=field):
        return np.array(field.eval(point=list(state)), dtype=float)

    values = field_values(field, states, one)
    if values is None:
        print(kind.__name__, "refused: evaluate a state at a time")
    else:
        worst = max(abs(values[k] - one(states[k])).max() for k in range(len(states)))
        print(kind.__name__, values.shape, "largest difference", worst)
```

**Limits.** The batch agrees with a state at a time *to rounding*, not to the last
bit (DEQ-75): NumPy's vectorised `pow` and libm's scalar `pow` differ by about one
ulp on some CPUs (2 of 200 entries, one ulp each, on CI's runners), and a platform
that fuses multiply and add can do the same to the duals' products. Through a
forward difference an ulp of $f$ becomes about $\mathrm{ulp}/h$ of $D_x f$. A field
that branches on its state, calls `math`, or takes a fractional power is refused. A
field that mis-evaluates arrays silently is caught by comparing the first and last
states with the caller's own evaluation, which is a spot check, not a proof.

API: `batched_field` is internal to the cycle continuation and not rendered on the
API page; its docstrings are the reference.

## References

[Ascher1995], [Bader1987], [Bojanczyk1992], [DeBoor1973], [DennisSchnabel1983],
[DeWitte2013], [Doedel1991a], [Doedel1991b], [Doedel2003], [DoedelAUTO07p],
[Fairgrieve1991], [Feigenbaum1978], [Griewank2008], [Iooss1988], [Kressner2006],
[Kuznetsov2005], [Kuznetsov2014], [Kuznetsov2014L8], [Lust2001]: full entries on the [References](../references.md#periodic-orbits-collocation-and-floquet-multipliers)
page.
