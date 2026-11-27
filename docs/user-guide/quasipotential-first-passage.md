# Quasipotential, Fokker-Planck and first passage

A deterministic flow says where a system settles. A small noise added to it raises three further questions:

- where the mass sits in the long run, which is the stationary density;
- how hard it is to get from an attractor to a given state, and by which route, which are the quasipotential and the minimum action path;
- how long a trajectory lingers near each saddle of a heteroclinic cycle, and how that time grows.

This page covers the tools for each. Every module here takes the library's convention $dx = f\,dt + G\,dW$, with the diffusion matrix $D = G G^\top$ and no separate small parameter. Sources are on the [References](../references.md) page, under *Quasipotential, minimum action paths, Fokker-Planck and first passage*.

## The stationary density of a planar flow

**The problem.** In one dimension the stationary density of $dx = f\,dt + g\,dW$ has a closed form, $p(x) = \frac{C}{g^2} \exp\left( \int \frac{2f}{g^2}\, dx \right)$. That form is an antiderivative, and above one dimension an antiderivative of the drift exists only when the drift is a gradient. A planar flow with a rotation, such as a noisy Hopf oscillator, has no such formula. Its stationary Fokker-Planck equation must be solved.

**The idea.** `StationaryFokkerPlanck` in `continuation.fokker_planck` solves the stationary equation $\nabla \cdot J = 0$ on a uniform grid, with the Ito flux

$$
J_d = f_d\, p - \frac{1}{2} \sum_e \partial_e \left( D_{de}\, p \right)
$$

The discretisation is finite-volume, so mass is conserved by construction. The diagonal part of the flux uses the exponential fitting of Scharfetter and Gummel, with the weight $z / (e^z - 1)$. The fitting is applied to $w = D_{dd}\, p$, not to $p$. In $w$ the flux is an advection with velocity $f_d / D_{dd}$ and a constant diffusion. Fitting $p$ instead gives a density wrong by the factor $D_{dd}$ whenever the noise depends on the state. The off-diagonal part of $D$ is a central difference.

With reflecting boundaries the operator is singular by exactly one dimension, since its rows sum to zero. One row is replaced by the normalisation $\int p = 1$. A single sparse solve then returns the density, already normalised.

The error is deterministic discretisation error, so it can be checked against an analytic density to a tight tolerance. A path ensemble would give a sampling error of order $1/\sqrt{\text{paths}}$ instead.

**Example.** The rotating Ornstein-Uhlenbeck process $dx = A x\,dt + G\,dW$ has a drift with no potential and a correlated noise. Its density is Gaussian, with the covariance $S$ that solves $A S + S A^\top + G G^\top = 0$.

```python
import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    StationaryFokkerPlanck,
)
from discrecontinual_equations.continuation.noise import NoiseMatrix
from discrecontinual_equations.continuation.stochastic import Ito
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.variable import Variable

A = np.array([[-1.0, -2.0], [2.0, -1.0]])  # a stable focus: rotation, no potential
G = np.array([[1.0, 0.5], [0.0, 0.8]])  # correlated, additive noise


class State(Variable, name="State", abbreviation="s"):
    pass


class Drift(DeterministicFunction):
    def eval(self, point, time=None):
        return list(A @ np.asarray(point))


class Column(DeterministicFunction):
    def __init__(self, index):
        super().__init__([State(), State()], [], [State(), State()])
        self._column = list(G[:, index])

    def eval(self, point, time=None):
        return self._column


drift = Drift([State(), State()], [], [State(), State()])
noise = NoiseMatrix([Column(0), Column(1)])

grid = DensityGrid.box([-3.0, -3.0], [3.0, 3.0], [121, 121])
density = StationaryFokkerPlanck(drift, noise, Ito(), grid).solve()

exact = solve_continuous_lyapunov(A, -G @ G.T)  # A S + S A^T + G G^T = 0
print(f"mass {density.mass():.6f}, minimum {density.minimum():.2e}")
print(f"covariance error {np.max(np.abs(density.covariance() - exact)):.1e}")
```

This prints a mass of `1.000000`, a minimum at round-off level, and a covariance error of about `8e-4`.

**Limits.**

- $D$ must be non-singular at every grid point. Scalar noise (rank one) and noise on only some components are refused with a `ValueError`. A noisy oscillator forced only in its velocity has a smooth density, but this scheme cannot compute it.
- The off-diagonal differencing carries no positivity guarantee. Strongly correlated noise on a coarse grid can give small negative values. `GridDensity.minimum()` reports them rather than clipping them; a clearly negative minimum means the grid needs refining.
- The grid must be uniform on each axis, and its box should hold essentially all the mass, since the boundary reflects.
- The cost grows with the number of cells. Two dimensions is cheap, three is possible, and above three an ensemble of paths from the stochastic solvers is the better tool.
- A density whose peak has moved onto a ring has a whole circle of near-equal maxima. `GridDensity.interior_maxima()` then returns a chain of cells around the ring, not one point.

**Source.** [ScharfetterGummel1969](../references.md#quasipotential-minimum-action-paths-fokker-planck-and-first-passage) for the fitted flux. API: [`StationaryFokkerPlanck`](../api/continuation.md#discrecontinual_equations.continuation.fokker_planck.StationaryFokkerPlanck), [`DensityGrid`](../api/continuation.md#discrecontinual_equations.continuation.fokker_planck.DensityGrid), [`GridDensity`](../api/continuation.md#discrecontinual_equations.continuation.fokker_planck.GridDensity).

## The quasipotential

**The problem.** At small noise, the probability of a path $\psi$ is $e^{-S/\varepsilon}$, with the action

$$
S[\psi] = \frac{1}{2} \int \left( \dot\psi - f(\psi) \right)^\top D^{-1} \left( \dot\psi - f(\psi) \right) dt
$$

The quasipotential $U(x)$ of an attractor is the least action over all paths that leave the attractor and reach $x$, in any time (Freidlin and Wentzell). Escape rates scale like $e^{-U/\varepsilon}$, and the stationary density near the attractor is $e^{-U/\varepsilon}$ to leading order. $U$ measures how hard each state is to reach.

**Normalisation.** The library has no $\varepsilon$, so $U$ is defined at $\varepsilon = 1$. It solves the Hamilton-Jacobi equation

$$
\nabla U \cdot f + \frac{1}{2} \nabla U^\top D\, \nabla U = 0
$$

In this convention:

- a gradient drift $f = -\nabla V$ with $G = I$ has $U = 2V$;
- a linear system $dx = A x\,dt + G\,dW$ has $U = \tfrac{1}{2} x^\top S^{-1} x$, with $A S + S A^\top + D = 0$;
- the density from the grid solve above is $e^{-U}$ up to a constant.

!!! warning "Erratum: the linear quasipotential"
    The original ticket (DEQ-31) gave the linear case as $x^\top S^{-1} x$. That is twice too large. Substituting $U = x^\top M x$ into the Hamilton-Jacobi equation gives $M A + A^\top M + 2 M D M = 0$, which $M = S^{-1}/2$ solves.

**The method.** `OrderedLineIntegral` in `continuation.quasipotential` computes $U$ on a planar `DensityGrid` by the ordered line integral method of Dahiya and Cameron, in its midpoint variant, OLIM-MID. Grid points are accepted in increasing order of $U$, as in Dijkstra's method. When a point is accepted, every unaccepted point within `update_radius` cells of it is offered two kinds of update:

- a one-point update along the straight segment from the accepted point;
- a triangle update from the segment between the accepted point and each accepted neighbour, with $U$ interpolated linearly along that segment.

Each update integrates the geometric action of a straight segment, $\lVert x - a \rVert_A \lVert f \rVert_A - \langle x - a, f \rangle_A$ in the metric $A = D^{-1}$, by the midpoint rule. The paper tries triangles only around the best one-point predecessor. The library tries every triangle, because the paper's subset doubled the error on the rotating Ornstein-Uhlenbeck process at 161 points (1.06e-3 against 5.0e-4).

!!! warning "Departure: how the default triangle update finds $s$"
    Dahiya and Cameron minimise each triangle's action over the point $x_s$ on its base exactly. They solve the full derivative, their Eq. (23) of the Physica D paper, which includes the change of $f$ and $A$ along the base, by Wilkinson's hybrid secant and bisection method. The default straight base here does not. It holds $f$ and $A$ fixed at the current $s$, where the derivative is a quadratic with a closed-form root, and refines that root twice from the middle of the base. The exact root was built and measured (DEQ-116; maximum error at 41, 81 and 161 points, radius 8). On linear drifts it was better: on the rotating Ornstein-Uhlenbeck process, 1.90e-3, 7.53e-4 and 4.10e-4 against the frozen step's 2.89e-3, 1.23e-3 and 6.14e-4. On every nonlinear drift it was worse, by 1.1 to 1.7 times: on the double well, 0.210, 0.0524 and 0.0135 against 0.153, 0.0420 and 0.0113. The frozen step stops short of the least value, and that partly offsets the midpoint rule's underestimate of a convex integrand. So the default keeps the frozen step, and its results are unchanged. The opt-in parabolic update below solves Eq. (23).

Next to the attractor, $U$ is set exactly where it can be. Near a stable equilibrium (`EquilibriumAttractor`), the quadratic quasipotential of the linearisation is used. Near a limit cycle (`CycleAttractor`), the paper's Simpson-rule start from the nearest point of the cycle is used.

**Example.** This continues the example above, on the same process.

```python
from discrecontinual_equations.continuation.quasipotential import (
    EquilibriumAttractor,
    GeometricMinimumAction,
    OrderedLineIntegral,
)

grid = DensityGrid.box([-1.5, -1.5], [1.5, 1.5], [81, 81])
U = OrderedLineIntegral(drift, noise, grid).solve(EquilibriumAttractor([0.0, 0.0]))

inverse = np.linalg.inv(exact)
target = np.array([0.8, -0.3])
print(f"U {U.at(target):.4f}, exact {0.5 * target @ inverse @ target:.4f}")

path = OrderedLineIntegral(drift, noise, grid).minimum_action_path(U, target)
print("traced path arrived:", path.arrived, "nodes:", len(path.states))

straight = np.linspace([0.0, 0.0], target, 41)
relaxed = GeometricMinimumAction(drift, noise).path(straight)
print(f"gMAM action {relaxed.action:.4f}, converged {relaxed.converged}")
```

The marched $U$ at $(0.8, -0.3)$ is `0.9993` against the exact `0.9986`. The last two lines are explained in the next section.

**The update radius.** The default radius is 8 cells, and it is held fixed as the grid is refined. The table gives the maximum error of $U$ against two closed forms, doubling the grid: for the double well on the basin of $(-1, 0)$ within $\{U \le 2\}$, box $[-2, 1] \times [-1.5, 1.5]$; for the rotating Ornstein-Uhlenbeck process on $\{U \le 1.5\}$, box $[-1.5, 1.5]^2$.

| System | 41 points | 81 points | 161 points | Observed order |
|---|---|---|---|---|
| Double well $V = x^4/4 - x^2/2 + y^2/2$, $D = I$ | 0.153 | 0.0420 | 0.0113 | 1.87, 1.89 |
| Rotating Ornstein-Uhlenbeck | 2.27e-3 | 9.68e-4 | 5.00e-4 | 1.23, 0.95 |

The radius must be at least comparable to the drift's rotation-to-gradient ratio on the region in question. A minimum action path crosses the level sets of $U$ at a shallow angle when the rotation is strong, so its predecessor lies many cells back along the front. On the paper's SDE (21), where that ratio is 10, a radius of 2 or 4 does not converge at all: radius 2 gives errors of 0.381, 0.370 and 0.371 at 129, 257 and 513 points. The default of 8 converges there, slowly. **If $U$ does not settle as the grid is refined, the radius is too small.**

The paper's rule of thumb, its Eq. (28), $K = 10 + 4\,(\operatorname{round}(\log_2 N) - 7)$ for $2^7 \le N \le 2^{12}$, grows the radius with the grid. The library offers it as an opt-in, not as the default: [`default_radius(grid)`](../api/continuation.md#discrecontinual_equations.continuation.quasipotential.default_radius) returns it, held at 8 below 128 points, and you pass it as `update_radius`. A larger radius also lengthens the one-point updates, whose midpoint-rule error grows like the square of their length, and on the steep double well that term dominates: at 81 points the error is 0.0107, 0.0112 and 0.0420 at radii 2, 4 and 8. At 321 points, radius 14 against 8, Eq. (28) gains on the rotating systems (rotating Ornstein-Uhlenbeck 9.30e-5 against 3.34e-4, SDE (21) 2.55e-4 against 1.34e-3) and loses on the gradient-dominated ones (double well 9.05e-3 against 2.85e-3). Use it for a drift whose rotation-to-gradient ratio is near or above 8.

**Limits.**

- Planar only. A grid with other than two axes is refused.
- The drift and noise must be time-independent. A periodically forced system needs the quasipotential of its Poincare map instead.
- $D$ must be non-singular at every grid point. A point between grid points where $D$ vanishes, such as the origin of the conformal noise $\sigma z\, dB$, is treated as impassable. That matches the quasipotential's own behaviour there, where it diverges logarithmically, but the values in the surrounding cells are not claimed to the orders above.
- $U$ near the edge of the box is wrong wherever the true minimum action path would leave the box. Choose a box that contains the sublevel set $\{U \le U(x)\}$ you ask about.
- With the default straight base the error is first order in general (the linear interpolation along each triangle's base). Treat $U$ as accurate to the measured figures above, not to machine precision, or check the points that matter with `check`.
- `GridQuasipotential.values` is `inf` where no path from the attractor reached. `floor` is the level below which a state is as close to the attractor as the grid resolves.

**Two further opt-ins.** `OrderedLineIntegral(..., update_order="quadratic")` interpolates $U$ along each triangle's base by a parabola through three accepted points instead of a straight line. This is the library's own variant, not a published method. It minimises each triangle's action over $s$ exactly: it solves the full derivative, Dahiya and Cameron's Eq. (23) of the Physica D paper, by a hybrid secant and bisection method. On a linear drift with a constant $D$ both the parabola and the midpoint rule are exact, so the result is exact to rounding. On the rotating Ornstein-Uhlenbeck process the maximum error is 3.1e-15, 4.4e-15 and 8.0e-15 at 41, 81 and 161 points; on the tests' non-normal linear system it is 1.5e-15 and 1.1e-15 at 41 and 81 points. A linear drift that rotates faster than the radius reaches is still limited by the radius.

On a nonlinear drift the parabola does not help. It was 1.2 to 1.45 times worse than the parabola with the frozen step it replaced, on every nonlinear system measured: the double well, the anisotropic gradient, SDE (24), and the verification's rotating, multiplicative-noise and elliptic-cycle systems, at 41, 81 and 161 points. The exact root also costs 1.6 to 1.9 times more than that frozen step did, which was itself 2.3 to 2.8 times the straight base. At equal cost the default straight base was better on every nonlinear problem measured. Use the parabola for a drift that is linear, or nearly so on the region you ask about.

`OrderedLineIntegral.check(field, indices)` estimates the error point by point. At each chosen point it relaxes a `GeometricMinimumAction` chain from a small surface around the attractor, which needs no grid. It reports the gap to the grid value plus the reference's own estimated error. That estimate compares the value from the surface with values from a surface at twice the distance and from a chain of half the nodes. With a `tolerance`, points whose estimate exceeds it are replaced by the reference. Each point costs of the order of a second, against well under a millisecond for the march, so ask for it where it matters.

The reference's estimate is not a bound everywhere. Where the minimum action path winds round the attractor faster than a chain of nodes can follow, the value gets worse as the surface shrinks, not better. Inside the verification's fast elliptic cycle the reference at $(0.3, 0)$ was 1.66 off, with an estimate of only 1.47. So `check` also relaxes a chain from a third surface, at half the distance, and treats the estimate as no bound in two cases: where that third value moves by more than the estimate, or where the two outer values differ by more than half of $U$ on the outer surface. A single point never stops the check:

- a point with no bound gets status `UNBOUNDED` and an error of `inf`;
- a point where no reference could be computed, because the chain would not stay finite, gets status `FAILED` and an error of `nan`;
- neither is refined, whatever the `tolerance`, and both keep the grid's value;
- `field.unbounded` lists both kinds by flat index, and a `UserWarning` names them once the check is done, adding "and they were not refined" when a `tolerance` was given.

On 73 references at 81 points on ten systems, 16 of them beside or inside the elliptic cycle, every finite estimate bounded the reference's error. All 16 cycle points were labelled, along with 8 elsewhere whose estimate did in fact hold; that is the price of the test. The third relaxation costs up to a third more per point.

```python
from discrecontinual_equations.continuation.quasipotential import default_radius

print("Eq. (28) radius at 257 points:", default_radius(DensityGrid.box([-1.5, -1.5], [1.5, 1.5], [257, 257])))

parabolic = OrderedLineIntegral(drift, noise, grid, update_order="quadratic")
V = parabolic.solve(EquilibriumAttractor([0.0, 0.0]))
print(f"parabolic U {V.at(target):.4f}")

solver = OrderedLineIntegral(drift, noise, grid)
index = np.ravel_multi_index(grid.nearest(target), grid.shape)
checked = solver.check(U, indices=[index])
point = grid.points()[index]
print(f"at {point}: grid {U.values.ravel()[index]:.5f}, "
      f"parabolic {V.values.ravel()[index]:.5f}, "
      f"exact {0.5 * point @ inverse @ point:.5f}, "
      f"estimated error {checked.errors.ravel()[index]:.1e}")
print("points without a bound:", checked.unbounded.tolist())
```

This continues the example above. It prints a radius of `14` and a parabolic $U$ of `0.9990` against the exact `0.9986`. The parabola is exact at grid points; what is left at the target, which lies between them, is the bilinear interpolation of `at`. At the grid point `(0.7875, -0.3)` it prints a grid value of `0.97241`, a parabolic value of `0.97223`, and the exact `0.97223`, with an estimated error of `3.0e-04` that bounds the grid's true `1.8e-4`. No point lacks a bound here, so the last line prints an empty list.

**Sources.** [FreidlinWentzell2012](../references.md#quasipotential-minimum-action-paths-fokker-planck-and-first-passage) for the action and the quasipotential. [DahiyaCameron2018a](../references.md#quasipotential-minimum-action-paths-fokker-planck-and-first-passage) for OLIM-MID: the geometric action, eq. (2); the Hamilton-Jacobi equation, eq. (3); the cycle start, eqs. (9)-(12); the test systems, SDEs (21) and (24); and the radius rule, eq. (28). [DahiyaCameron2018b](../references.md#quasipotential-minimum-action-paths-fokker-planck-and-first-passage) for the midpoint metric $A = D^{-1}$ and its linear interpolation along a triangle's base with a variable $D$, eqs. (19), (21) and (22), and for the exact triangle minimisation, eq. (23), solved by a hybrid secant and bisection method. API: [`OrderedLineIntegral`](../api/continuation.md#discrecontinual_equations.continuation.quasipotential.OrderedLineIntegral), [`EquilibriumAttractor`](../api/continuation.md#discrecontinual_equations.continuation.quasipotential.EquilibriumAttractor), [`CycleAttractor`](../api/continuation.md#discrecontinual_equations.continuation.quasipotential.CycleAttractor), [`GridQuasipotential`](../api/continuation.md#discrecontinual_equations.continuation.quasipotential.GridQuasipotential).

## Minimum action paths

The minimiser of the action is the most probable route of a rare transition. The library offers two ways to compute it.

**Tracing back on the grid.** `OrderedLineIntegral.minimum_action_path(field, start)` follows the computed $U$ from `start` back to the attractor. Along a minimum action path $\dot\psi = f + D \nabla U$, so the trace runs backwards along $-(f + D \nabla U)$. It stops when a step would raise $U$. The result's `arrived` flag says whether it reached the attractor. A trace that starts in another basin stalls at the saddle between the basins, where $-(f + D \nabla U)$ vanishes. For an arrived path, `action` is an independent estimate of `U(start)`.

**Relaxing a whole path.** `GeometricMinimumAction(drift, noise).path(initial)` takes a chain of states with both ends fixed and relaxes it to the minimiser by the geometric minimum action method (gMAM) of Heymann and Vanden-Eijnden. The path is parametrised by arc length, so the method needs no grid and can cross saddles. The stiff $z''$ term is taken implicitly, which frees the step from the node spacing. The result's `converged` flag says whether the chain stopped moving within the iteration limit. A chain that is still moving is returned as it stands, not refused.

In the example above, the traced path arrives, and the relaxed chain of 41 nodes gives an action of `1.0020` against the exact `0.9986`. On this process the action is 0.34%, 0.16% and 0.11% too high with 41, 81 and 161 nodes. What remains is the infinitely many turns the exact path makes as it spirals out of the origin, which a finite chain cannot follow.

**Limits.**

- `GeometricMinimumAction` handles a state-dependent $D$ through the general form of the flow (Grafke, Schäfer and Vanden-Eijnden, eqs. (38)–(42)), which brings in the derivatives of $D$. On the tests' multiplicative system, where $D$ varies by a factor of two, the action is off by 1.36e-4, 3.11e-5 and 1.04e-5 with 41, 81 and 161 nodes. A $D$ that is the same along the chain is detected and the cheaper constant-$D$ flow is used.
- A traced path inherits the grid's error in $\nabla U$. It cannot cross a saddle; use the relaxed path for that.
- Check `arrived` and `converged`. Neither method raises when it falls short.

**Sources.** [HeymannVandenEijnden2008](../references.md#quasipotential-minimum-action-paths-fokker-planck-and-first-passage) for the gMAM; [VandenEijndenHeymann2008](../references.md#quasipotential-minimum-action-paths-fokker-planck-and-first-passage) for its semi-implicit form with additive noise; [GrafkeSchaferVandenEijnden2017](../references.md#quasipotential-minimum-action-paths-fokker-planck-and-first-passage), eq. (19) for the relaxation flow with additive noise and eqs. (38)–(42) for a state-dependent noise. API: [`OrderedLineIntegral.minimum_action_path`](../api/continuation.md#discrecontinual_equations.continuation.quasipotential.OrderedLineIntegral.minimum_action_path), [`GeometricMinimumAction`](../api/continuation.md#discrecontinual_equations.continuation.quasipotential.GeometricMinimumAction).

## Residence times near a heteroclinic cycle

**The problem.** Near an attracting heteroclinic cycle, a trajectory lingers near each saddle and moves quickly between them, and each linger is longer than the last. Gaunersdorfer's Lemma 1 gives the growth: with expanding eigenvalue $e_j$ and contracting eigenvalue $-c_j$ at saddle $j$, the time near the next saddle is asymptotically $c_j / e_{j+1}$ times the time near this one.

**The idea.** `Itinerary.measure` in `continuation.residence` finds every complete visit of a sampled trajectory to a ball around each saddle. Entry and exit times are interpolated linearly between samples. A visit cut off by either end of the record is dropped, since its length is unknown.

Each dwell time also carries an additive constant from the entry and exit geometry, $T_{k+1} = r\,T_k + K$. The plain ratio $T_{k+1} / T_k$ therefore converges only like $1 / T_k$. Differences remove the constant exactly, $T_{k+2} - T_{k+1} = r\,(T_{k+1} - T_k)$, and the ratios are built from them:

- `growth_ratios()` gives $(T_{k+2} - T_{k+1}) / (T_{k+1} - T_k)$, which converges to the common ratio of a symmetric cycle;
- `step_ratios(period)` gives the growth of each step of a cycle of `period` saddles separately, by comparing one revolution's growth at the same saddle. For saddles passed in cycle order it lines up with the predicted `CycleSpectrum.residence_ratios` of `continuation.heteroclinic_network`.

**Example.** The May-Leonard competition system has a heteroclinic cycle between the three single-species states. At each of them the expanding eigenvalue is $1 - \beta$ and the contracting one is $-(\alpha - 1)$, so the predicted ratio is $(\alpha - 1)/(1 - \beta)$.

```python
import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.residence import Itinerary

alpha, beta = 1.3, 0.8  # May-Leonard competition; the cycle attracts when alpha + beta > 2
interaction = -np.array([[1.0, alpha, beta], [beta, 1.0, alpha], [alpha, beta, 1.0]])


def log_lotka_volterra(_time, logs):
    return 1.0 + interaction @ np.exp(logs)  # x_i' = x_i (1 + (M x)_i) in log x


times = np.linspace(0.0, 600.0, 120001)
solution = solve_ivp(
    log_lotka_volterra, (0.0, 600.0), np.log([0.8, 0.1, 0.05]),
    method="DOP853", rtol=1e-11, atol=1e-12, t_eval=times,
)
states = np.exp(solution.y.T)

itinerary = Itinerary.measure(times, states, list(np.eye(3)), radius=0.1)
print("order", itinerary.order[:6])
print(f"growth ratio {itinerary.growth_ratios()[-1]:.4f}, "
      f"predicted {(alpha - 1) / (1 - beta):.4f}")
```

The measured growth ratio is `1.5000`, matching the prediction.

**Limits.**

- The lemma is asymptotic. The first few ratios of a record are transients; read the latest ones.
- Choose the radius small against the distance between saddles and large against the distance moved in one sample. Otherwise a visit is missed or split in two.
- The dwell times grow geometrically, so a long record is needed for several revolutions, and the integration must be accurate enough to follow a trajectory that passes extremely close to each saddle. Integrating the logarithms of the coordinates, as in the example, helps.

**Source.** [Gaunersdorfer1992](../references.md#quasipotential-minimum-action-paths-fokker-planck-and-first-passage), Lemma 1. API: [`Itinerary`](../api/continuation.md#discrecontinual_equations.continuation.residence.Itinerary), [`Visit`](../api/continuation.md#discrecontinual_equations.continuation.residence.Visit).
