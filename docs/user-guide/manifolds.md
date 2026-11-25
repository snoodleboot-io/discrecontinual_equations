# Invariant manifolds

A saddle equilibrium has a stable manifold, the set of points whose orbits
converge to it, and an unstable manifold, the set of points whose orbits came
from it. These manifolds organise the phase space. They separate basins of
attraction, they carry the homoclinic and heteroclinic connections that
continuation tracks, and in the Lorenz system the stable manifold of the origin
decides which wing of the attractor an orbit visits next.

The library computes a manifold in two parts:

| Part | Class | Gives | Valid |
|---|---|---|---|
| Local | [`TaylorManifold`][tm] | a polynomial chart `P(θ)` of any dimension | near the equilibrium |
| Global | [`GeodesicLevelSets`][gls] | a triangulated two-dimensional mesh, ring by ring | as far as it can be grown |

The local chart seeds the global growth. A third module,
[`parameterization`](#naming-a-branchs-points-the-parameterization-module),
has a confusingly similar name. It holds the constraint that closes the
continuation corrector, and it is covered at the end of this page.

All examples below share this small amount of setup:

```python
import numpy as np

from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class State(Variable, name="state", abbreviation="u"):
    pass


class Unused(Parameter, name="unused", abbreviation="a"):
    pass


def field(kind, dimension):
    return kind(
        variables=[State() for _ in range(dimension)],
        parameters=[Unused(value=0.0)],
        results=[State() for _ in range(dimension)],
        time=None,
    )
```

## Local manifolds: the parameterization method

### The idea

Let `x0` be an equilibrium of `x' = f(x)` with Jacobian `A`. Pick `k` of its
eigenvalues `λ1, ..., λk`: the unstable ones for `W^u`, the stable ones for
`W^s`. Let `Λ = diag(λ1, ..., λk)`. A graph over the eigenspace fails once the
manifold folds back over it. The parameterization method avoids this. It looks
for a map `P` from `k` coordinates `θ` into state space whose image is the
manifold, and on which the flow is just the linear flow `θ ↦ e^{Λt} θ`. This
holds when `P` satisfies the invariance equation

$$
f\big(P(\theta)\big) = DP(\theta)\,\Lambda\theta,
\qquad P(0) = x_0,\quad DP(0) = [v_1 \;\cdots\; v_k],
$$

where `v_i` are the eigenvectors. Cabré, Fontich and de la Llave state the
method for maps as `F ∘ K = K ∘ R` (their eq. (1.4)), with `R` linear when
there are no internal resonances. Flows follow through time-`t` maps (their
p. 288). The equation above is the flow's infinitesimal form.

Write `P(θ) = Σ_m p_m θ^m` over multi-indices `m`. Matching the coefficients of
`θ^m` for `|m| ≥ 2` gives one linear solve per monomial:

$$
\big(A - (m\cdot\lambda)\, I\big)\, p_m = -\,[\,f(P_{<|m|})\,]_m ,
$$

Here the right-hand side is the `θ^m` coefficient of the field evaluated on the
series known so far. `TaylorManifold.compute` does exactly this. It evaluates
`f` on truncated multivariate series, one degree at a time, and solves each
system with `numpy.linalg.solve`. The solve is non-singular as long as no
`m · λ` is an eigenvalue of `A`. That is the non-resonance condition, condition
(4) of Cabré–Fontich–de la Llave's Theorem 1.1: no product of at most `L`
eigenvalues of the map lies in the rest of its spectrum.

### Example

The planar saddle `x' = x`, `y' = -y + x²` has unstable manifold
`y = x²/3` exactly. The order-5 chart reproduces it:

```python
from discrecontinual_equations.continuation.manifold import (
    TaylorManifold,
    UnstableManifold,
)


class Saddle(DeterministicFunction):
    """x' = x, y' = -y + x^2: the unstable manifold is y = x^2 / 3."""

    def eval(self, point, time=None):
        x, y = point[0], point[1]
        return [x, -y + x * x]


jacobian = np.array([[1.0, 0.0], [0.0, -1.0]])
chart = TaylorManifold(order=5).compute(
    field(Saddle, 2), np.zeros(2), jacobian, UnstableManifold()
)
for theta in (0.1, 0.5, 1.0):
    x, y = chart.point([theta])
    print(f"theta={theta}: y - x^2/3 = {y - x * x / 3:.1e}")  # 0.0e+00 each time

print(chart.flow_image([0.1], 1.0))  # [0.27182818 0.02463019]: P(0.1 e)
```

[`ManifoldChart.flow_image`][flow] returns `P(e^{Λt} θ)`. Comparing it with
an integration of the field from `P(θ)` is the natural invariance test.

### Limits

These were all checked against the code on `main`:

- **Real eigenvalues only.** The chart takes the real part of each selected
  eigenvector. For a complex pair (a focus) the two conjugate vectors have the
  same real part, so the chart collapses to a curve. For example, the stable
  manifold of `x' = -x + 2y`, `y' = -2x - y + z²`, `z' = z` comes out with rank
  1. No error is raised. `GeodesicLevelSets` checks for this case and seeds
  from the eigenspace instead (see below). Called directly, `TaylorManifold`
  does not.
- **Resonances.** An exact resonance, such as eigenvalues 1 and 2 with the
  `θ1²` term hitting 2, makes the solve singular, and
  `numpy.linalg.LinAlgError` is raised. A near-resonance is silent and
  dangerous: with 2 replaced by `2 + 1e-9`, `P(0.1, 0)` has a component of
  about `-1e7`.
- **Local and truncated.** The series is truncated at `order`, and its domain
  of validity is not estimated. The eigenvectors keep `numpy.linalg.eig`'s
  unit scaling, which sets how far in `θ` the series stays accurate. Use small
  `θ`, or check against `flow_image`.
- Only the real part of each coefficient is kept.

API: [`TaylorManifold`][tm], [`ManifoldChart`][chart],
[`StableManifold`][stable], [`UnstableManifold`][unstable]. References:
[CabreFontichDeLaLlave2003, HaroEtAl2016](../references.md#invariant-manifolds-and-geodesic-level-sets).

## Global two-dimensional manifolds: geodesic level sets

### The problem

Far from the equilibrium a manifold has to be grown, not expanded. The obvious
approach integrates a fan of trajectories from a small circle and joins them
up, so the manifold becomes the family of images `φ_t(S_δ)`. Krauskopf and
Osinga (2003, §2) explain why this fails: the circle does not grow uniformly
under the flow. The trajectories bunch together where the flow converges and
leave large gaps where it diverges.

### The idea

Krauskopf and Osinga grow the manifold instead as **level sets of geodesic
distance** from the equilibrium. Each new ring lies a distance `Δ` beyond the
previous one, measured along the manifold. For every point `r` of the current
ring `C_i`, take the hyperplane `F_r` through `r` perpendicular to the ring.
The new point is the end `b_r` of an orbit segment that starts on the ring and
lands in `F_r`:

$$
\varphi_0\big(q_r(\tau)\big) = q_r(\tau) \in C_i, \qquad
\varphi_\tau\big(q_r(\tau)\big) = b_r(\tau) \in F_r, \qquad
\lVert b_r - r\rVert = \Delta
$$

These are their eqs. (3.1)–(3.3), or (11) in the IJBC survey. The step `Δ` is
accepted if the radial line turns gently. With `α_r` the angle between
`p_r → r` and `r → b_r`, they require

$$
\alpha_r < \alpha_{\max}, \qquad \Delta\cdot\alpha_r < (\Delta\alpha)_{\max}
$$

(eqs. (3.4)–(3.5)). Otherwise `Δ` is halved and the ring is recomputed.
Points are inserted where neighbours on the new ring are further apart than
`Δ_F`, by solving the same problem from the ring's midpoint rather than by
interpolating, and removed where they are closer than `δ_F` (§3.3). Each pair
of consecutive rings is zipped into a band of triangles.

The mesh this produces can be crocheted. Each ring is a round, and inserted and
removed points are increases and decreases. Osinga and Krauskopf (2004)
crocheted the Lorenz manifold this way, to geodesic distance 110.75 in 25,511
stitches. [`GeodesicMesh.crochet_pattern`][crochet] reports the same rounds.

### How this implementation does it

`GeodesicLevelSets.compute` follows the scheme above, with these choices.
Some of them depart from the papers:

- **Seed on the chart.** The first ring lies at geodesic radius
  `initial_radius` on the `TaylorManifold` chart, measured along the chart's
  rays by Gauss–Legendre quadrature, so its curvature is included. The papers
  start from a circle in the flat eigenspace. If there is no Taylor chart, the
  tangent plane is used instead and `mesh.seed` is `"eigenspace"`. That happens
  for a complex pair or a resonance. For a saddle-focus the plane is spanned by
  the real and imaginary parts of **one** eigenvector of the pair. In this case
  the seed is only first-order accurate, so keep `initial_radius` small.
- **The boundary-value problem is solved from the landing end.** The unknowns
  are the landing point `q`, the flight time `τ` and the start's position `s`
  along the ring, and Newton's method solves for them. The first guess traces
  the point `Δ` straight out back to the ring. The papers instead continue in
  `τ` from `τ = 0` and stop at the first zero of `Δ - ‖b_r(τ) - r‖`, which
  makes `b_r` the *nearest* landing. Here a converged landing that falls inside
  the ring is rejected as lost. Flowing backwards contracts the manifold's
  directions, so this formulation stays well conditioned when one rate is many
  times the other.
- **Step control.** On `main`, only the angle bound (3.4) is applied, through
  `angles=(low, high)`. `Δ` is halved when the angle exceeds `high`, and grows
  by 1.5 (the papers double it) when the angle is below `low`, never beyond
  `step_range[1]`. The `Δ·α` bound (3.5) is not applied yet. Distances are
  chords in the normal plane, as in (3.3). On a curved manifold they are
  therefore second-order accurate in `Δ`; on a plane they are exact.
- **Stopping.** If more than a fraction `loss` of a ring's landings fail even
  at `step_range[0]`, the growth stops. It returns the mesh it has, with
  [`GeodesicMesh.stopped`][mesh] saying why and `lost` counting the dropped
  points.

### Example

Grow the Lorenz manifold `W^s(0)` to geodesic distance 20. The `z`-axis lies in
`W^s(0)` and, by the symmetry `(x, y, z) → (-x, -y, z)`, is a geodesic of it.
So every level set must cross the axis at `z = ±r`:

```python
from discrecontinual_equations.continuation.geodesic_level_sets import (
    GeodesicLevelSets,
    GeodesicSettings,
)
from discrecontinual_equations.continuation.manifold import StableManifold


class Lorenz(DeterministicFunction):
    def eval(self, point, time=None):
        x, y, z = point[0], point[1], point[2]
        return [10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z]


jacobian = np.array([[-10.0, 10.0, 0.0], [28.0, -1.0, 0.0], [0.0, 0.0, -8.0 / 3.0]])
settings = GeodesicSettings(initial_radius=1.0, step=1.5, spacing=(1.5, 4.5))
mesh = GeodesicLevelSets(settings).compute(
    field(Lorenz, 3), np.zeros(3), jacobian, StableManifold(), distance=20.0
)

print(mesh.seed, mesh.stopped, mesh.lost)  # taylor None 0
print(len(mesh.vertices), len(mesh.triangles))  # 297 560
ring = mesh.vertices[mesh.rings[-1]]
on_axis = ring[np.hypot(ring[:, 0], ring[:, 1]).argsort()[:2]]
print(np.sort(on_axis[:, 2]).round(6))  # [-20.  20.]
for round_ in mesh.crochet_pattern()[:3]:
    print(round_.number, round_.distance, round_.stitches, round_.increases)
```

This takes a few seconds. `mesh.ring_distances` runs 1.0, 2.5, 4.0, … , 19.0,
20.0.

### What the published computations reach

Krauskopf and Osinga use the Lorenz manifold `W^s(0)` (σ = 10, ρ = 28,
β = 8/3) as their test case:

| Source | Settings | Reached |
|---|---|---|
| SIADS 2003, §4.5 | 20 points at δ = 1, α ∈ (0.3, 0.4), Δα ∈ (0.1, 1.0), spacing 0.25–1.0 | 151.75, 72 circles, last of 2,300 points |
| IJBC 2005 survey, §2.3 | as above, Δ₁ = 0.25, spacing 0.67–2.0 | 154.75, 75 bands |
| Math. Intelligencer 2004 (crochet) | the crocheted model | 110.75, 25,511 stitches |

On `main`, with `spacing=(1.5, 4.5)`, the strict growth aimed at 170 stops at
157.16 with no point lost. Four of 1,119 landings fail at the smallest step,
where the flow runs along the level set. Because the step control differs, the
rings and point counts do not match the papers'. Only the distance can be
compared. An early target of "170" for this check was not taken from any of
the papers. The published figures are 151.75 and 154.75.

### Limits

- Two-dimensional manifolds only. Any other selected dimension raises
  `ValueError`. Use `TaylorManifold` for a curve.
- The seed comes from the Taylor chart, so the chart's limits apply near the
  equilibrium. A focus or a resonance falls back to the first-order tangent
  plane.
- Where the flow runs along a level set, landings fail and the growth stops.
  `loss > 0` lets it bridge a few holes, at the cost of a coarser ring there.
- Without the `Δ·α` bound, highly curved regions are resolved only by the angle
  bound and the spacing.
- `max_vertices` (250,000 by default) caps the mesh size.

**Work in progress.** These are open pull requests in the geodesic stack, not
yet on `main`: DEQ-61 (step control as Krauskopf and Osinga do it, with the
`Δ·α` bound), DEQ-71 (the saddle-focus basis), DEQ-62 (a complex-coordinate
Taylor chart for a saddle-focus) and DEQ-63. This page describes `main` and
will be updated as they merge.

API: [`GeodesicLevelSets`][gls], [`GeodesicSettings`][settings],
[`GeodesicMesh`][mesh], [`CrochetRound`][round]. References:
[KrauskopfOsinga2003, KrauskopfEtAl2005, OsingaKrauskopf2004](../references.md#invariant-manifolds-and-geodesic-level-sets).

## Manifolds of periodic orbits

Stable and unstable manifolds of a periodic orbit are computed as
Fourier–Taylor series in the Floquet normal form, after Castelli, Lessard and
Mireles James (SIADS 2015). That work (DEQ-38) is still an open pull request
and will be documented here once it reaches `main`.

## Naming a branch's points: the `parameterization` module

Despite its name, [`parameterization`][param] has nothing to do with the
parameterization method above. It decides how the continuation corrector names
a point of a solution branch. The equations `f(u, λ) = 0` are `n` equations in
`n + 1` unknowns `x = (u, λ)`. The corrector makes them square by appending one
scalar equation `g(x) = 0`, and appending its gradient as a border row to the
extended Jacobian `[f_u | f_λ]`:

- [`Natural`][natural] fixes `λ` at its predicted value, so `g = λ - λ_p`. The
  bordered matrix has determinant `det f_u`, which is zero at a fold, where the
  branch turns back in `λ`.
- [`PseudoArclength`][pal] is Keller's constraint. With `t` the unit tangent and
  `x_p = x_0 + Δs·t` the prediction, it is `g = t · (x - x_p)`, which is the
  same as Doedel's `(x - x_0) · t - Δs = 0`. The bordered matrix `[f_x; t]` is
  non-singular at every regular point, folds included (Doedel 2006, slide 27).

`ContinuationConfig(parameterization="pseudo_arclength")` is the default.
`"natural"` is cheaper, but it cannot pass a fold. At the fold of
`f(u, λ) = λ - u²`:

```python
from discrecontinual_equations.continuation.parameterization import (
    CorrectionStep,
    Natural,
    PseudoArclength,
)

extended = np.array([[0.0, 1.0]])  # [f_u | f_lambda] at (u, lambda) = (0, 0)
step = CorrectionStep(
    state=np.array([0.0]),
    parameter=0.0,
    predicted_state=np.array([0.0]),
    predicted_parameter=0.0,
    tangent=np.array([1.0, 0.0]),
)
for constraint in (Natural(), PseudoArclength()):
    bordered = np.vstack([extended, constraint.border(step)])
    print(type(constraint).__name__, np.linalg.det(bordered))
# Natural 0.0
# PseudoArclength -1.0
```

Neither constraint handles a branch point, where `f_x` itself loses rank;
branch switching does. Reference:
[Doedel2006](../references.md#invariant-manifolds-and-geodesic-level-sets).

[tm]: ../api/continuation.md#discrecontinual_equations.continuation.manifold.TaylorManifold
[chart]: ../api/continuation.md#discrecontinual_equations.continuation.manifold.ManifoldChart
[flow]: ../api/continuation.md#discrecontinual_equations.continuation.manifold.ManifoldChart.flow_image
[stable]: ../api/continuation.md#discrecontinual_equations.continuation.manifold.StableManifold
[unstable]: ../api/continuation.md#discrecontinual_equations.continuation.manifold.UnstableManifold
[gls]: ../api/continuation.md#discrecontinual_equations.continuation.geodesic_level_sets.GeodesicLevelSets
[settings]: ../api/continuation.md#discrecontinual_equations.continuation.geodesic_level_sets.GeodesicSettings
[mesh]: ../api/continuation.md#discrecontinual_equations.continuation.geodesic_level_sets.GeodesicMesh
[round]: ../api/continuation.md#discrecontinual_equations.continuation.geodesic_level_sets.CrochetRound
[crochet]: ../api/continuation.md#discrecontinual_equations.continuation.geodesic_level_sets.GeodesicMesh.crochet_pattern
[param]: ../api/continuation.md#discrecontinual_equations.continuation.parameterization
[natural]: ../api/continuation.md#discrecontinual_equations.continuation.parameterization.Natural
[pal]: ../api/continuation.md#discrecontinual_equations.continuation.parameterization.PseudoArclength
