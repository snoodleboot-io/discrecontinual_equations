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

The paper's rule of thumb, its Eq. (28), $K = 10 + 4\,(\operatorname{round}(\log_2 N) - 7)$ for $2^7 \le N \le 2^{12}$, grows the radius with the grid. The library does not apply it; to use it, pass `update_radius` yourself. A larger radius also lengthens the one-point updates, whose midpoint-rule error grows like the square of their length, and on the steep double well that term dominates: at 81 points the error is 0.0107, 0.0112 and 0.0420 at radii 2, 4 and 8. The radius is therefore left to the caller.

**Limits.**

- Planar only. A grid with other than two axes is refused.
- The drift and noise must be time-independent. A periodically forced system needs the quasipotential of its Poincare map instead.
- $D$ must be non-singular at every grid point. A point between grid points where $D$ vanishes, such as the origin of the conformal noise $\sigma z\, dB$, is treated as impassable. That matches the quasipotential's own behaviour there, where it diverges logarithmically, but the values in the surrounding cells are not claimed to the orders above.
- $U$ near the edge of the box is wrong wherever the true minimum action path would leave the box. Choose a box that contains the sublevel set $\{U \le U(x)\}$ you ask about.
- The error is first order in general (the linear interpolation along each triangle's base). Treat $U$ as accurate to the measured figures above, not to machine precision.
- `GridQuasipotential.values` is `inf` where no path from the attractor reached. `floor` is the level below which a state is as close to the attractor as the grid resolves.

**Work in progress.** Three open pull requests change `continuation.quasipotential` and are not yet on `main`: DEQ-57 (PR #58, a pointwise error estimate and refinement where it is large), DEQ-58 (PR #59, an opt-in parabolic update) and DEQ-59 (PR #60, minimum action paths through a state-dependent noise). This page describes `main`; the measured figures above, and the constant-$D$ limit of `GeometricMinimumAction` below, will be revisited as they merge.

**Sources.** [FreidlinWentzell2012](../references.md#quasipotential-minimum-action-paths-fokker-planck-and-first-passage) for the action and the quasipotential. [DahiyaCameron2018](../references.md#quasipotential-minimum-action-paths-fokker-planck-and-first-passage) for OLIM-MID: the geometric action, eq. (2); the Hamilton-Jacobi equation, eq. (3); the cycle start, eqs. (9)-(12); the test systems, SDEs (21) and (24); and the radius rule, eq. (28). API: [`OrderedLineIntegral`](../api/continuation.md#discrecontinual_equations.continuation.quasipotential.OrderedLineIntegral), [`EquilibriumAttractor`](../api/continuation.md#discrecontinual_equations.continuation.quasipotential.EquilibriumAttractor), [`CycleAttractor`](../api/continuation.md#discrecontinual_equations.continuation.quasipotential.CycleAttractor), [`GridQuasipotential`](../api/continuation.md#discrecontinual_equations.continuation.quasipotential.GridQuasipotential).

## Minimum action paths

The minimiser of the action is the most probable route of a rare transition. The library offers two ways to compute it.

**Tracing back on the grid.** `OrderedLineIntegral.minimum_action_path(field, start)` follows the computed $U$ from `start` back to the attractor. Along a minimum action path $\dot\psi = f + D \nabla U$, so the trace runs backwards along $-(f + D \nabla U)$. It stops when a step would raise $U$. The result's `arrived` flag says whether it reached the attractor. A trace that starts in another basin stalls at the saddle between the basins, where $-(f + D \nabla U)$ vanishes. For an arrived path, `action` is an independent estimate of `U(start)`.

**Relaxing a whole path.** `GeometricMinimumAction(drift, noise).path(initial)` takes a chain of states with both ends fixed and relaxes it to the minimiser by the geometric minimum action method (gMAM) of Heymann and Vanden-Eijnden. The path is parametrised by arc length, so the method needs no grid and can cross saddles. The stiff $z''$ term is taken implicitly, which frees the step from the node spacing. The result's `converged` flag says whether the chain stopped moving within the iteration limit. A chain that is still moving is returned as it stands, not refused.

In the example above, the traced path arrives, and the relaxed chain of 41 nodes gives an action of `1.0020` against the exact `0.9986`. On this process the action is 0.34%, 0.16% and 0.11% too high with 41, 81 and 161 nodes. What remains is the infinitely many turns the exact path makes as it spirals out of the origin, which a finite chain cannot follow.

**Limits.**

- `GeometricMinimumAction` assumes a constant $D$. A state-dependent $D$ needs the metric's own derivatives in the flow, so it is refused rather than silently ignored.
- A traced path inherits the grid's error in $\nabla U$. It cannot cross a saddle; use the relaxed path for that.
- Check `arrived` and `converged`. Neither method raises when it falls short.

**Sources.** [HeymannVandenEijnden2008](../references.md#quasipotential-minimum-action-paths-fokker-planck-and-first-passage) for the gMAM; [VandenEijndenHeymann2008](../references.md#quasipotential-minimum-action-paths-fokker-planck-and-first-passage) for its semi-implicit form with additive noise; [GrafkeSchaferVandenEijnden2017](../references.md#quasipotential-minimum-action-paths-fokker-planck-and-first-passage), eq. (19), for the relaxation flow as implemented. API: [`OrderedLineIntegral.minimum_action_path`](../api/continuation.md#discrecontinual_equations.continuation.quasipotential.OrderedLineIntegral.minimum_action_path), [`GeometricMinimumAction`](../api/continuation.md#discrecontinual_equations.continuation.quasipotential.GeometricMinimumAction).

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
