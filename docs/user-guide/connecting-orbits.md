# Connecting orbits and heteroclinic networks

A *connecting orbit* is a trajectory that leaves one equilibrium and arrives at
another, or at the same one: a **heteroclinic** orbit in the first case, a
**homoclinic** orbit in the second. Such an orbit takes infinite time in both
directions, so the library computes a long finite piece of it. Connecting orbits
matter because they organise global bifurcations. A homoclinic loop to a saddle
is where a periodic orbit's period diverges. A loop to a saddle-focus can create
chaos. A cycle of heteroclinic connections can attract, and the times a
trajectory spends near each saddle then grow without bound.

This page covers:

1. computing a single connecting orbit as a boundary-value problem, and checking how accurate it is;
2. finding the parameter value at which a homoclinic orbit exists, by shooting;
3. following a homoclinic orbit as a curve in two parameters;
4. the Shilnikov saddle index of a saddle-focus;
5. robust heteroclinic cycles and networks, their spectra and their asymptotic stability;
6. following a network in a parameter, and finding where a connection breaks;
7. stability indices, and which cycle of a network wins;
8. the systems used throughout: reflection-symmetric cubics and unidirectional sensor rings.

The examples on this page run as they stand. Sections 1 to 3 share one session:
the field defined in section 1 is used again in sections 2 and 3. Each later
section is self-contained. Every work cited is listed on the
[References](../references.md#connecting-orbits-homoclinic-curves-and-heteroclinic-networks)
page.

## 1. A connecting orbit as a boundary-value problem

**Problem.** Given a saddle $\xi$ (or two saddles $\xi_-$, $\xi_+$), find the orbit
$u(t)$ with $u(t) \to \xi_\mp$ as $t \to \mp\infty$.

**Idea.** Cut the time axis to $[-T, T]$. At each end, replace the condition "on
the manifold" with "in the manifold's tangent space". This is Beyn's (1990)
**projection boundary condition**. Let $L_s$ be the rows that project onto the
source's stable eigenspace, so that a displacement with $L_s v = 0$ lies in its
unstable eigenspace. Let $L_u$ be the rows that project onto the target's
unstable eigenspace. Then

$$
\begin{aligned}
u'(t) &= f(u(t)), \qquad t \in [-T, T],\\
L_s\,(u(-T) - \xi_-) &= 0, \\
L_u\,(u(T) - \xi_+) &= 0, \\
u_k(0) &= c .
\end{aligned}
$$

The last line is a **phase condition**. It pins the orbit's time translation by
fixing component $k$ at the centre node. Each row of $L_s$ and $L_u$ is a left
eigenvector. A real eigenvalue gives one real condition. A complex pair gives two,
the real and imaginary parts of one left eigenvector, so the spiral plane of a
saddle-focus is constrained in full. In HomCont's notation these are the conditions
$L_s (u(0) - x_0) = 0$ of the `FBHO` routine (Champneys, Kuznetsov and Sandstede
1996). Doedel et al. (2008) write them as $L^-(u(\tau_-) - \xi) = 0$, in their
eq. (5a).

The equations are discretised by the trapezoidal rule. They are solved by
Gauss–Newton, with a sparse bordered block-banded Jacobian, so each iteration
costs time and memory linear in the number of mesh nodes.

**Connections that come in a family (DEQ-175).** If the source's unstable
manifold meets the target's stable manifold in more than one orbit, the
connections form a family and none of them is isolated. The Newton matrix $J$
is then singular along the family. On the sheared Kirk–Silber edges leaving
$\xi_2$ the smallest singular value is $10^{-18}$ to $7 \times 10^{-17}$ of the
largest, and the next one is $8 \times 10^{-3}$. A plain Gauss–Newton step
divides rounding by rounding along the family: it came out 10 to 95 long,
different on every BLAS kernel, and the solve diverged.

Every step is therefore a Levenberg–Marquardt step with a fixed, tiny damping
$\lambda = (10^{-11}\sigma_{\max})^2$:

$$
\Delta x = -(J^\top J + \lambda I)^{-1} J^\top r .
$$

In the singular basis this multiplies each Gauss–Newton component by
$\sigma^2/(\sigma^2 + \lambda)$. Directions far above the floor keep their
Gauss–Newton step, changed by at most $(10^{-11}\operatorname{cond} J)^2$.
Directions far below it are dropped. As $\lambda \to 0^+$ the step tends to the
minimum-norm step $-J^+ r$ (Moré 1978, eq. (2.4)), which moves only across the
family. On the Kirk–Silber edges the residual then falls quadratically, from
0.59 to $10^{-15}$ in seven steps. The result matches Gauss–Newton with a dense
SVD pseudo-inverse to $10^{-12}$, and it is the same on every OpenBLAS kernel
tried. The member found lies 0.025 and 0.16 off the invariant plane: a genuine
connection, but not the one in the plane. `HeteroclinicNetwork.solve` finds that
one by solving inside the plane.

A step computed without full numerical rank ($\sigma_{\min} \le 10^{-8}\sigma_{\max}$,
or `lstsq` keeping fewer singular values than columns) is not the Gauss–Newton
step along the directions it dropped. Its size therefore never certifies
convergence (DEQ-180). Such a solve is converged only if its residual is below
$10^{-9}$.

```python
import numpy as np
from discrecontinual_equations.continuation import (
    HeteroclinicOrbit, MeshSpec, Terminus,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.variable import Variable


class Decoupled(DeterministicFunction):
    """A heteroclinic double well beside an uncoupled neutral direction z' = 0.

    Every plane z = c holds a connection, so they form a one-parameter family.
    """

    def eval(self, point, time=None):
        x, y, z = point
        return [y, -x + x**3, 0.0]


field = Decoupled(variables=[Variable()] * 3, parameters=[],
                  results=[Variable()] * 3, time=None)


def jacobian(x):
    return np.array([[0.0, 1.0, 0.0], [-1.0 + 3.0 * x * x, 0.0, 0.0], [0.0, 0.0, 0.0]])


orbit = HeteroclinicOrbit(
    field,
    Terminus(np.array([-1.0, 0.0, 0.0]), jacobian(-1.0)),
    Terminus(np.array([1.0, 0.0, 0.0]), jacobian(1.0)),
    MeshSpec(half_length=8.0, intervals=200),
)
times = np.linspace(-8.0, 8.0, 201)
seed = np.column_stack([np.tanh(times / 2), 0 * times, 0.3 + 0 * times])
# z' = 0 makes the Newton matrix exactly singular: a RuntimeWarning says so,
# and each step is the minimum-norm one, which never moves z.
solution = orbit.solve(seed)
assert solution.residual.is_trajectory
print("z of the member found:", solution.states[100, 2])
```

References: [More1978](../references.md).

**Two errors, measured separately.** The truncation to $[-T, T]$ costs
$O(e^{-2\lambda T})$, where $\lambda$ is the weakest eigenvalue magnitude involved
(Beyn 1990; Doedel et al. 2008 give $C e^{-2\min(\mu_-|\tau_-|, \mu_+|\tau_+|)}$).
The mesh costs $O(h^2)$. `estimate_orbit_error` re-solves at doubled intervals
and on a longer interval, and reports both shifts. `limited_by` says which one to
improve.

The example below uses the exact homoclinic orbit
$x = \tfrac32 \operatorname{sech}^2(t/2)$ of $x'' = x - x^2$. It sits inside the
two-parameter family $x' = y$, $y' = x - x^2 + \mu y + \nu x y$, which is used again
in sections 2 and 3.

```python
import numpy as np

from discrecontinual_equations.continuation.connecting_orbit import (
    HomoclinicOrbit,
    MeshSpec,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class State(Variable, name="state", abbreviation="x"):
    pass


class Mu(Parameter, name="mu", abbreviation="mu"):
    pass


class Nu(Parameter, name="nu", abbreviation="nu"):
    pass


class Well(DeterministicFunction):
    """x' = y, y' = x - x^2 + mu y + nu x y."""

    def eval(self, point, time=None):
        mu, nu = self.parameters[0].value, self.parameters[1].value
        x, y = point
        return [y, x - x * x + mu * y + nu * x * y]


field = Well(
    variables=[State(), State()],
    parameters=[Mu(value=0.0), Nu(value=0.0)],
    results=[State(), State()],
    time=None,
)
mesh = MeshSpec(half_length=15.0, intervals=120, phase_index=1, phase_value=0.0)
times = np.linspace(-15.0, 15.0, 121)
seed = np.column_stack([1.2 / np.cosh(0.45 * times) ** 2, np.zeros_like(times)])

orbit = HomoclinicOrbit(field, np.zeros(2), np.array([[0.0, 1.0], [1.0, 0.0]]), mesh)
solution = orbit.solve(seed)
exact = 1.5 / np.cosh(times / 2.0) ** 2
print(f"true error      {np.max(np.abs(solution.component(0) - exact)):.1e}")
resolution = orbit.estimate_orbit_error(solution)
print(f"estimate        {resolution.estimate:.1e}, limited by {resolution.limited_by}")
```

```text
true error      7.7e-03
estimate        5.8e-03, limited by spacing
```

The estimate runs at about 0.75 of the true error. That is the factor
$1 - \tfrac14$ of a second-order method. The calibration, and why both components
are needed, are on the API page under
[Trusting a computed connecting orbit](../api/continuation.md#trusting-a-computed-connecting-orbit).
`HeteroclinicOrbit` takes two `Terminus` endpoints, each with its own position and
Jacobian, and works the same way.

**Limits.**

- `solve` returns only a converged iterate and raises `NotConvergedError`
  otherwise (DEQ-120). Converged is not the same as a trajectory: the residual
  of the overdetermined least-squares system is not zero at the solution, so
  check `solution.residual.is_trajectory` and `estimate_orbit_error`.
- A connection that is one of a family is not isolated, and the Newton matrix
  is singular along the family. Each step is then the minimum-norm one, so
  the solve converges to *some* member of the family, the one Gauss–Newton
  with the pseudo-inverse reaches from your seed, not necessarily the one you
  meant (DEQ-175, below).
- `intervals` must be even. The phase condition pins the centre node.
- The seed must be close enough for Gauss–Newton to converge. Shooting (section 2)
  or a nearby solved orbit is the usual source.
- The eigenspaces are recomputed by `numpy.linalg.eig` and are not given
  HomCont's continuity normalisation (Beyn 1990, (4.4), as applied in AUTO's
  `PRJCTN`). For a fixed field this does not matter, because a zero condition is
  unchanged by rescaling its row.

API: [`connecting_orbit`](../api/continuation.md#discrecontinual_equations.continuation.connecting_orbit).

## 2. Locating a homoclinic orbit by shooting

**Problem.** A homoclinic orbit to a saddle is a codimension-one phenomenon. It
exists only on a curve in a two-parameter plane, so at fixed $\nu$ there is one
$\mu$ to find.

**Idea.** Leave the saddle along its one-dimensional unstable manifold, from
$\xi + \varepsilon v$ (the explicit start of Doedel et al. 2008, eq. 6). Integrate
the orbit, and measure where it comes back: take the closest return to $\xi$ and
project it on $v$. This **return gap** changes sign as $\mu$ crosses the
homoclinic value, so bisection finds that value. `trace` repeats the search along
a second parameter, centring each bracket on the previous result.

For this family Melnikov's method gives the curve to first order as
$\mu I_1 + \nu I_2 = 0$, with $I_1 = \int y^2\,dt = \tfrac65$ and
$I_2 = \int x y^2\,dt = \tfrac{36}{35}$. Then $\mu \approx -\tfrac67 \nu$.

```python
from discrecontinual_equations.continuation.homoclinic_shooting import (
    Departure,
    HomoclinicShooting,
    ReturnSettings,
)

field.parameters[1].value = 0.1  # nu
shooter = HomoclinicShooting(
    field,
    parameter_index=0,  # search over mu
    departure=Departure(np.zeros(2), direction=np.array([1.0, 1.0])),
    settings=ReturnSettings(dt=0.01, horizon=30.0),
)
print(f"gap at the ends {shooter.gap(-0.15):+.3f} {shooter.gap(-0.03):+.3f}")
mu = shooter.locate(-0.15, -0.03)
print(f"mu = {mu:.4f}; Melnikov, first order: {-6 / 7 * 0.1:.4f}")
field.parameters[1].value = 0.0
```

```text
gap at the ends +0.289 -0.284
mu = -0.0857; Melnikov, first order: -0.0857
```

**Limits.**

- The gap is `NaN` where there is no return. Here that means $\mu = -0.2$, where
  the manifold settles on the centre, and $\mu = 0$, where it escapes. `locate`
  needs finite gaps of opposite sign at both ends of its bracket, and stops with
  a `ValueError` at a midpoint whose gap is `NaN`: it cannot say which half holds
  the root. `trace` samples each window at nine points and hands `locate` the
  sign change of finite gaps nearest the previous result.
- A sign change of the gap need not be a root. The gap is the unstable component
  of the *closest* return, and where the closest return switches from one pass
  near the saddle to another the gap jumps across zero without vanishing. On the
  saddle-focus jerk system (section 3) at $b = 0.85$ it jumps from $-0.0123$ to
  $+0.0120$ within $10^{-5}$ of $a = 0.59113$, $7 \times 10^{-5}$ from the true
  homoclinic. So `locate` evaluates the gap once more where bisection converged
  and refuses a result whose $|\text{gap}|$ exceeds `ReturnSettings.jump_tolerance`
  (default $10^{-3}$): at a root the gap falls towards zero with the bracket
  (like $d^{0.46}$ at that saddle-focus), at a jump it does not. Near a
  saddle-focus, follow the curve with the boundary-value problem of section 3
  instead, and use shooting only for its seed.
- The saddle needs a one-dimensional unstable manifold with a real leading
  eigenvalue. The departure and return radii are absolute (0.5 and 0.4), and so
  is `jump_tolerance`, so a system far from unit scale needs its own
  `ReturnSettings`.
- The accuracy is that of fixed-step RK4 and of the closest sampled return.
  Refine the result with the boundary-value problem of section 1 or 3.
- `gap` and `trace` leave the parameters set to the last value they tried.

API: [`homoclinic_shooting`](../api/continuation.md#discrecontinual_equations.continuation.homoclinic_shooting).

## 3. Continuing a homoclinic curve in two parameters

**Idea.** Add the saddle and one parameter to the unknowns of section 1, and add
the equilibrium condition $f(\xi, \mu) = 0$ to the equations. The projection rows
are rebuilt at every iterate from the eigenvectors of $Df(\xi, \mu)$, keeping
the seed's split into stable and unstable ones (see the limits below). That gives
a square system: collocation, equilibrium, projection and phase, with one free
parameter. It is the defining system HomCont continues (Champneys, Kuznetsov and
Sandstede 1996). Sweep the second parameter and seed each point from the last.
The Newton matrix is again bordered block-banded and factored sparse.

```python
from discrecontinual_equations.continuation.homoclinic_curve import HomoclinicCurve
from discrecontinual_equations.differential_equation import DifferentialEquation


class Time(Variable, name="time", abbreviation="t"):
    pass


equation = DifferentialEquation(
    variables=[State(), State()],
    time=Time(),
    parameters=field.parameters,
    derivative=field,
)
loop = np.column_stack(
    [
        1.5 / np.cosh(times / 2) ** 2,
        -1.5 * np.tanh(times / 2) / np.cosh(times / 2) ** 2,
    ],
)
curve = HomoclinicCurve(equation, continuation_index=0, sweep_index=1, mesh=mesh)
for point in curve.trace([0.0, 0.05, 0.1], loop, np.zeros(2), 0.0):
    print(f"nu = {point.sweep:.2f}  mu = {point.continuation:+.4f}")
```

```text
nu = 0.00  mu = -0.0000
nu = 0.05  mu = -0.0427
nu = 0.10  mu = -0.0855
```

The located points follow the Melnikov line $\mu = -\tfrac67\nu$ to within the
mesh error at $h = 0.25$. The shooting value in section 2 agrees to $2 \times 10^{-4}$.

**A saddle-focus curve through a Belyakov point.** The jerk system
$x' = y,\ y' = z,\ z' = -a z - b y + x - x^2$ has a saddle-focus at the origin:
one unstable eigenvalue $\gamma$ and a stable pair $-\rho \pm i\omega$. A complex
pair gives one real and one imaginary projection row. Along its homoclinic curve
the saddle index $\rho / \gamma$ of section 4 crosses one between $b = 0.85$ and
$b = 0.80$: the Belyakov transition from Shilnikov chaos to a tame loop. Shooting
loses this curve near $b = 0.85$, where its gap jumps (section 2); the
boundary-value problem follows it. The seed is the unstable manifold at the
shooting value for $b = 0.9$, shifted so that the first peak of $x$, where
$y = 0$, sits at the mesh centre.

```python
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.shilnikov import classify_saddle_focus


class JerkA(Parameter, name="jerk_a", abbreviation="a"):
    pass


class JerkB(Parameter, name="jerk_b", abbreviation="b"):
    pass


class Jerk(DeterministicFunction):
    def eval(self, point, time=None):
        a, b = self.parameters[0].value, self.parameters[1].value
        x, y, z = point
        return [y, z, -a * z - b * y + x - x * x]


a0, b0 = 0.519, 0.9  # HomoclinicShooting.locate(0.519, 0.520) gives 0.51905
jerk_parameters = [JerkA(value=a0), JerkB(value=b0)]
jerk_variables = [State(), State(), State()]
jerk = Jerk(
    variables=jerk_variables,
    parameters=jerk_parameters,
    results=[State(), State(), State()],
)
jerk_equation = DifferentialEquation(
    variables=jerk_variables,
    time=Time(),
    parameters=jerk_parameters,
    derivative=jerk,
)

values, vectors = np.linalg.eig(
    np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -b0, -a0]]),
)
unstable = vectors[:, np.argmax(values.real)].real
unstable *= np.sign(unstable[0])
manifold = solve_ivp(
    lambda t, s: jerk.eval(list(s)),
    (0.0, 80.0),
    1e-9 * unstable,
    method="DOP853",
    rtol=1e-12,
    atol=1e-15,
    dense_output=True,
)
grid = np.linspace(0.0, 80.0, 80_001)
x = manifold.sol(grid)[0]
departed = np.argmax(np.abs(x) > 0.5)
peak = grid[departed + np.flatnonzero(np.diff(x[departed:]) < 0)[0]]
jerk_mesh = MeshSpec(half_length=12.0, intervals=480, phase_index=1, phase_value=0.0)
jerk_seed = manifold.sol(np.linspace(-12.0, 12.0, 481) + peak).T

jerk_curve = HomoclinicCurve(jerk_equation, continuation_index=0, sweep_index=1, mesh=jerk_mesh)
for point in jerk_curve.trace([0.9, 0.85, 0.8, 0.75, 0.7], jerk_seed, np.zeros(3), a0):
    focus = classify_saddle_focus(
        np.array([[0, 1, 0], [0, 0, 1], [1, -point.sweep, -point.continuation]]),
    )
    print(f"b = {point.sweep:.2f}  a = {point.continuation:.4f}  index {focus.saddle_index:.3f}")
```

```text
b = 0.90  a = 0.5188  index 0.917
b = 0.85  a = 0.5908  index 0.974
b = 0.80  a = 0.6624  index 1.031
b = 0.75  a = 0.7327  index 1.086
b = 0.70  a = 0.7995  index 1.137
```

An independent computation (SciPy's DOP853 and `brentq` on the first-return
gap projected on the *left* unstable eigenvector, sharing no code with the
library) puts the curve at $a = 0.519050, 0.591061, 0.662600, 0.732836, 0.799624$.
At $h = 0.05$ the boundary-value problem lies $1.5$ to $2.4 \times 10^{-4}$ below
it; the error falls as $h^2$ to a truncation floor of $2$ to $7 \times 10^{-5}$
at $T = 12$.

**Limits.**

- *Eigenvalue counts.* The boundary rows need the saddle's eigenvalues split
  into stable and unstable ones. They are counted once, at the seed, and then
  followed from Newton iterate to iterate by eigenvector, not re-counted: one
  near-singular step can move the saddle further than a small eigenvalue is from
  zero, and the next brings it back. An iterate whose saddle has the seed's
  counts takes its own split by sign: two eigenvalues that cross zero in
  opposite directions leave the counts unchanged, and the followed split would
  then converge on the other split's boundary rows. The counts are enforced at the accepted
  point: a point whose residual has converged but whose saddle has other counts
  is refused with an `EigenvalueCountError`, and so is a solve that runs out of
  iterations after an iterate with other counts. `trace` stops at such a
  refusal with a `RuntimeWarning` that names both counts and the parameters. A solve that diverges to
  non-finite values returns `None`.
- The equilibrium rows' derivative in the saddle is the exact (Taylor-jet)
  Jacobian; the boundary rows' dependence on the eigenvectors, and every column
  of a field that cannot be evaluated on jets, are forward differences with an
  absolute step of $10^{-7}$ (DEQ-171).
- `trace` sweeps a parameter in natural steps, with no pseudo-arclength, so it
  stops at a fold of the curve in the swept parameter. `trace` stops at the first
  point that fails to converge, and `solve_point` returns `None` there.

API: [`homoclinic_curve`](../api/continuation.md#discrecontinual_equations.continuation.homoclinic_curve).

## 4. Saddle-focus equilibria and the Shilnikov index

**Problem.** Will a homoclinic loop to this equilibrium create chaos?

**Idea.** In $\mathbb{R}^3$, let a saddle-focus have eigenvalues $\lambda \pm i\omega$
and $\gamma$, with $\lambda < 0 < \gamma$ and $\omega \neq 0$. Shilnikov (1965)
proved the following. If the saddle value $\lambda + \gamma$ is positive, a
homoclinic loop to the saddle-focus implies infinitely many saddle periodic orbits
near the loop, organised in horseshoes (as restated in Gonchenko et al. 2022,
section II). Equivalently, the **saddle index**

$$
\delta = \frac{|\operatorname{Re}\lambda|}{|\gamma|}
$$

is below one. Reversing time swaps the roles and leaves $\delta$ unchanged, so
`classify_saddle_focus` accepts either orientation.

```python
import numpy as np

from discrecontinual_equations.continuation.shilnikov import classify_saddle_focus

for a in (0.48, 0.70):
    # the jerk system x' = y, y' = z, z' = -a z - y + x - x^2, at the origin
    jacobian = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -1.0, -a]])
    focus = classify_saddle_focus(jacobian)
    print(a, f"{focus.saddle_index:.3f}", focus.satisfies_shilnikov_criterion)
```

```text
0.48 0.897 True
0.7 1.107 False
```

**Limits.** The classification reads the linearisation only. It does not show
that a loop exists. Locate the loop with section 2, then confirm the chaos with
the top Lyapunov exponent. Only a three-dimensional spectrum, with one real
eigenvalue and one pair, is classified; anything else returns `None`. At
$\delta = 1$ the criterion is degenerate and reports `False`.

API: [`shilnikov`](../api/continuation.md#discrecontinual_equations.continuation.shilnikov).

## 5. Robust heteroclinic cycles and networks

**Problem.** In a generic system, a saddle-to-saddle connection breaks under any
perturbation. Symmetry can make it robust. Suppose that for each $j$ there is a
flow-invariant subspace $P_j$ containing the unstable manifold of $\xi_j$, in
which $\xi_{j+1}$ is a sink. This is hypothesis (H1) of Krupa and Melbourne
(1995). Then every connection is a saddle-to-sink connection inside $P_j$, and the
cycle persists under every perturbation that keeps the $P_j$ invariant (their
Proposition 2.5). The fixed-point subspaces of a symmetry group are such
subspaces.

**Idea.** `HeteroclinicNetwork` holds:

- the saddles, refined by Newton's method;
- the connections, each with the basis of its $P_j$;
- the elementary cycles of the directed graph the connections form.

For each cycle, `spectrum` sorts the eigenvalues at every node by the role they
play *for that cycle*, as in Krupa and Melbourne (1995, section 2.3): radial $-r_j$
on $P_{j-1} \cap P_j$, contracting $-c_j$, expanding $e_j$, and transverse $t_j$. A
saddle shared by two cycles can play different roles in each. A cycle without
positive transverse eigenvalues is asymptotically stable if

$$
\prod_j \min(c_j,\, e_j - t_j) > \prod_j e_j
$$

(their Theorem 2.7, eq. 2.3). For a cycle in $\mathbb{R}^3$ this reduces to
$\rho = \prod c_j / \prod e_j > 1$, which is also necessary, generically
(Theorem 2.9, Remark 2.10(b)). Near an attracting cycle the time spent at
successive saddles grows by $c_j / e_{j+1}$ per passage (Gaunersdorfer 1992,
Lemma 1). The product of these ratios is $\rho$.

`solve` computes a connection with the projection boundary-value problem of
section 1 in the full space. The subspace is not imposed, so `subspace_defect`,
the orbit's distance from $P_j$, is a real check.

The classic example is the Guckenheimer–Holmes cycle (Guckenheimer and Holmes
1988). In Krupa's (1997) normalisation, eq. (2.1), it reads
$x_i' = x_i(\lambda + a_1 x_i^2 + a_2 x_{i+1}^2 + a_3 x_{i+2}^2)$. The cycle exists
if and only if $a_1 < 0$ and $a_2 < a_1 < a_3$ (Proposition 1). It is stable if
$2a_1 > a_2 + a_3$ (Theorem 1).

```python
import numpy as np

from discrecontinual_equations.continuation.heteroclinic_network import (
    HeteroclinicNetwork,
    coordinate_plane,
)
from discrecontinual_equations.continuation.network_continuation import (
    NetworkContinuation,
)
from discrecontinual_equations.systems.reflection_cubic import guckenheimer_holmes

field = guckenheimer_holmes(growth=1.0, self_coupling=-1.0, forward=-1.5, backward=-0.6)
network = HeteroclinicNetwork(field)
for k in range(3):
    network.add_saddle(f"xi{k + 1}", np.eye(3)[k])
for k in range(3):
    following = (k + 1) % 3
    network.add_connection(
        f"xi{k + 1}", f"xi{following + 1}", coordinate_plane(3, (k, following))
    )

cycle = network.cycles()[0]
spectrum = network.spectrum(cycle)
print(cycle, spectrum.contracting, spectrum.expanding)
print(f"rho = {spectrum.stability_ratio:.4f}, stable: {spectrum.is_asymptotically_stable}")

connection = network.connections[0]
orbit = network.solve(connection, network.mesh(connection, 24.0, 480))
print(f"defect {orbit.subspace_defect:.1e}, estimate {orbit.resolution.estimate:.1e}")
```

```text
('xi1', 'xi2', 'xi3') [0.5 0.5 0.5] [0.4 0.4 0.4]
rho = 1.9531, stable: True
defect 0.0e+00, estimate 4.7e-05
```

Krupa's eq. (2.2) gives $c = \lambda(a_2 - a_1)/a_1 = 0.5$ and
$e = \lambda(a_1 - a_3)/a_1 = 0.4$. Then $\rho = 1.25^3$, and $2a_1 = -2 > -2.1$.

**Limits.**

- `add_connection` refuses a subspace that is not invariant for the Jacobian at
  both ends. It does not check that the subspace is invariant for the flow.
- *Oblique coordinates.* On the current release, the full-space solve can return
  a path that is not a trajectory, with a near-zero error estimate. The verifier
  found this on the Kirk–Silber network in sheared coordinates: two of five
  connections returned their seed, with collocation residuals of 0.09. The fix
  solves inside the subspace and raises `NotATrajectoryError`. It is pending in
  PR #69 (DEQ-81). Until then, check `subspace_defect`, and solve connections in
  coordinates where the subspaces are coordinate planes.
- `solve` refuses a connection whose persistence margin (section 6) is not
  positive.
- The stability test `is_asymptotically_stable` is the sufficient condition of
  Theorem 2.7. A cycle in a network always has a positive transverse eigenvalue
  at a shared node, so it is never asymptotically stable. Section 7 says what
  it still attracts.

API: [`heteroclinic_network`](../api/continuation.md#discrecontinual_equations.continuation.heteroclinic_network).

## 6. Following a network, and where a connection breaks

**Idea.** Two conditions of (H1) can fail under a parameter change that keeps the
symmetry: the source must expand into $P$, and the target must attract within it.
`connection_margins` returns the source's leading eigenvalue on $P$ and minus the
target's. While both are positive, the connection persists. `NetworkContinuation`
refreshes every saddle at each parameter value and reports the margins and each
cycle's $\rho$. It can also re-solve every connection, seeded from the previous
one. `locate_break` marches until a margin is no longer positive, then bisects.

Continuing the session of section 5, vary the second growth rate $\lambda_2$. This
keeps the reflections but breaks the cyclic symmetry.

```python
found = NetworkContinuation(network, parameter_index=1).locate_break(
    1.0, 0.5, steps=20, tolerance=1e-10
)
print(found.connection.key, found.side, f"{found.parameter:.8f}")
```

```text
('xi1', 'xi2') target 0.66666667
```

At $\lambda_2 = 2/3$, $\xi_2$ stops attracting within the $x_1 x_2$-plane. The
cycle loses stability earlier: $\rho$ falls through one near $\lambda_2 = 0.71$.

**Limits.**

- Only the eigenvalue conditions are followed. A parameter that breaks the
  symmetry keeping $P$ invariant destroys every connection at once, and no
  margin will show it.
- A margin that dips below zero and recovers within one march step is missed.
- The $\rho$ reported per cycle decides stability only for cycles without
  positive transverse eigenvalues.

API: [`network_continuation`](../api/continuation.md#discrecontinual_equations.continuation.network_continuation).

## 7. Stability indices: which cycle of a network wins

**Problem.** When two cycles share a connection, neither is asymptotically stable.
Each node they share sends part of its unstable manifold along the other cycle.
Yet one cycle, or both, can still attract most nearby points. Kirk and Silber
(1994) found this competition in a $\mathbb{Z}_2^4$-symmetric system in
$\mathbb{R}^4$.

**Idea.** Podvigina and Ashwin's (2011) **local stability index** $\sigma$
measures how much of a small ball $B_\varepsilon$ about a point of a connection
is attracted to the cycle. Here $\sigma > 0$ when the unattracted part shrinks
like $\varepsilon^\sigma$. Also $\sigma < 0$ when the attracted part shrinks like
$\varepsilon^{|\sigma|}$, and $\pm\infty$ when all of the ball or none of it is
attracted. The index is constant along a connection (their Theorem 2.2). A cycle
attracts a set of positive measure if and only if some index exceeds $-\infty$.
It is predominantly asymptotically stable if and only if every index is positive
(Lohse 2015, Theorem 3.1; Castro and Lohse 2014, Theorem 2.6).

The indices come from Garrido-da-Silva and Castro (2019). In logarithmic
coordinates $\eta = \ln|x|$ the passage near node $j$ is linear. The passage
between nodes is a permutation (their Assumption A). So one trip from one
section to the next is a **basic transition matrix** (their eq. 7),

$$
M_j = A_j \begin{pmatrix} c_j/e_j & 0 \\ -t_{j,s}/e_j & I \end{pmatrix}.
$$

The index follows from the leading eigenvalue and eigenvector of the products of
these matrices (Lemma 3.2, Theorems 3.4 and 3.10), through the function
$F^{\text{index}}$ (Lemma 3.9). Here $A_j$ is found geometrically, by matching the
directions leaving one node with those arriving at the next.

```python
import numpy as np

from discrecontinual_equations.continuation.heteroclinic_network import (
    HeteroclinicNetwork,
    coordinate_plane,
)
from discrecontinual_equations.continuation.network_stability import NetworkCompetition
from discrecontinual_equations.systems.reflection_cubic import reflection_cubic

# eigenvalue[k, j]: the eigenvalue at the saddle on axis j along axis k
eigenvalue = np.array(
    [
        [-2.0, -1.0, 1.0, 1.0],
        [1.0, -2.0, -1.5, -1.5],
        [-1.0, 1.0, -2.0, -0.5],
        [-1.0, 0.5, -0.5, -2.0],
    ],
)
interaction = eigenvalue - 1.0
np.fill_diagonal(interaction, -1.0)
network = HeteroclinicNetwork(reflection_cubic(np.ones(4), interaction))
for j in range(4):
    network.add_saddle(f"xi{j + 1}", np.eye(4)[j])
for source, target in ((1, 2), (2, 3), (3, 1), (2, 4), (4, 1)):
    network.add_connection(
        f"xi{source}", f"xi{target}", coordinate_plane(4, (source - 1, target - 1))
    )

competition = NetworkCompetition(network)
for cycle, stability in competition.stabilities.items():
    print(cycle, stability.indices, f"delta = {stability.delta:+.3f}")
print("winner:", competition.winner)
```

```text
('xi1', 'xi2', 'xi3') [inf  1. inf] delta = +1.250
('xi1', 'xi2', 'xi4') [-inf -inf -inf] delta = -1.000
winner: ('xi1', 'xi2', 'xi3')
```

With unit growth rates and $A_{jj} = -1$, the saddle on axis $j$ sits at
$x_j = 1$, and its eigenvalue along axis $k$ is $1 + A_{kj}$. Every eigenvalue is
therefore set by one coupling coefficient. Entry $j$ of `indices` belongs to the
connection *into* node $j$. Castro and Lohse's $\delta$ (2014, eq. 3) has opposite
signs on the two cycles here. This is case (i) of their Proposition 5.3: the
$\xi_4$-cycle attracts nothing, and the $\xi_3$-cycle is predominantly
asymptotically stable.

**Erratum carried over.** For a type C4⁻ cycle with $b_1 < 0 < b_2, b_3, b_4$,
the closed forms printed by Podvigina and Ashwin (2011, pp. 908–909) keep only
the eigenvector terms, and they are wrong in most cases. Lohse (2015, Lemma B.5)
prints the corrected forms, agreed with Podvigina. The library uses the general
Theorem 3.10 of Garrido-da-Silva and Castro. It agrees with Lohse's corrected
forms in 1458 of 1458 sampled cases, and with the original branch in 77. A
simulation of the basin also sides with the corrected forms.

**Limits.**

- Only quasi-simple cycles are handled. Every connection must be
  one-dimensional, and all the $P_j$ must have the same dimension. Other cycles
  are refused, as are complex transverse eigenvalues.
- Directions are matched through orthogonal complements. This is exact for the
  fixed-point subspaces of an orthogonal symmetry group. In oblique coordinates,
  a type B cycle is still handled, but a type C cycle is refused.
- A repeated transverse eigenvalue leaves its eigenvectors undetermined, and is
  refused rather than guessed at.
- `delta` is defined only for a cycle with one transverse direction and exactly
  one node with a positive transverse eigenvalue.

API: [`network_stability`](../api/continuation.md#discrecontinual_equations.continuation.network_stability).

## 8. The systems

**Reflection-symmetric cubics.** `reflection_cubic` builds
$x_i' = x_i(\lambda_i + \sum_j A_{ij} x_j^2)$. Every coordinate reflection is a
symmetry, so every coordinate subspace is invariant, and saddle–sink connections
in coordinate planes are robust (Krupa and Melbourne 1995, Proposition 2.5). In
$u_i = x_i^2$ the system is Lotka–Volterra, Krupa (1997) eq. (7.1). Changing one
$\lambda_i$ keeps the reflections but breaks any cyclic symmetry.
`guckenheimer_holmes` is the circulant case of section 5.

**Unidirectional sensor rings.** `FluxgateRing` and `FerroelectricRing` couple $N$
identical bistable cells in a ring, with cell $i$ driven by cell $i+1$. For odd
$N$ the ring oscillates past a critical coupling. In the fluxgate ring the stable
oscillation is born out of a heteroclinic cycle between the ring's saddle states,
with infinite period (Bulsara et al. 2004, section III and Fig. 3). Near the
threshold the period diverges because the orbit lingers near those saddles, as in
the passage times of section 5. The ferroelectric ring does the same with
double-well cells ($a > 0$); with monostable cells ($a < 0$) its wave is born
instead at a Hopf bifurcation, as below. At zero field, the ferroelectric ring's trivial state has the
eigenvalues $a - \lambda$ and $a + \lambda/2 \pm i\lambda\sqrt3/2$:

```python
import numpy as np

from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.systems.coupled_ring import FerroelectricRing
from discrecontinual_equations.variable import Variable


class Cell(Variable, name="cell", abbreviation="x"):
    pass


class Value(Parameter, name="value", abbreviation="p"):
    pass


a, coupling = -0.3, 0.65  # monostable cells, past the Hopf at lambda = -2a = 0.6
ring = FerroelectricRing(
    variables=[Cell() for _ in range(3)],
    parameters=[Value(value=coupling), Value(value=a), Value(value=0.0)],
    results=[Cell() for _ in range(3)],
    time=None,
)
step = 1e-6
jacobian = np.column_stack(
    [
        (np.array(ring.eval(list(step * e))) - np.array(ring.eval(list(-step * e))))
        / (2 * step)
        for e in np.eye(3)
    ],
)
print(np.round(np.linalg.eigvals(jacobian), 4))
print(f"linear period 4 pi / (lambda sqrt 3) = {4 * np.pi / (coupling * np.sqrt(3)):.2f}")
```

```text
[-0.95 +0.j      0.025+0.5629j  0.025-0.5629j]
linear period 4 pi / (lambda sqrt 3) = 11.16
```

**Departures from the printed models**, both deliberate:

- Bulsara et al. (2004, eq. 2) write the fluxgate cell as
  $-x_i + \tanh(c(x_i + \lambda x_{i+1} + \varepsilon))$. `FluxgateRing` uses
  $-x_i + \tanh(c x_i + \lambda x_{i+1} + \varepsilon)$, so its $\lambda$ and
  $\varepsilon$ are the paper's multiplied by $c$.
- Their Duffing ring (eq. 1) couples by $\lambda(x_i - x_{i+1})$.
  `FerroelectricRing` couples by $-\lambda x_{i+1}$ alone. With $b = 1$ their
  cell's linear part is $(a + \lambda)x_i - \lambda x_{i+1}$, so the library's
  gain $a$ stands for their $a + \lambda$. Their Hopf value
  $a_c = -\tfrac32\lambda$ becomes $\lambda = -2a$.

The docstring of `FerroelectricRing` maps both signs of the gain $a$. That
includes the infinite-period birth of the rotating wave near $\lambda \approx 0.34a$
for double-well cells.

API: [`reflection_cubic`](../api/continuation.md#discrecontinual_equations.systems.reflection_cubic),
[`coupled_ring`](../api/continuation.md#discrecontinual_equations.systems.coupled_ring).

## Pending fixes in this area

These are open pull requests and running lanes that change the code documented
here. Their docstrings will be updated when they merge.

| Work | What it changes |
|---|---|
| PR #69 (DEQ-81) | network connections solved inside their subspace; `NotATrajectoryError` |
| DEQ-120 | refuse a diverged Gauss–Newton solve instead of returning the seed |
