# Continuation API

Numerical continuation, bifurcation detection, periodic and connecting orbits,
and stochastic analysis.

## Trusting a computed cycle

A solver returning a `PeriodicOrbitSolution` means the *discrete* collocation system
was satisfied to tolerance. It does **not** mean the mesh resolves the orbit. Those
are different claims, and on a stiff cycle they come apart: continuing van der Pol
to `mu = 40` on 200 nodes reaches the target, returns a solution, and reports a
period 12% wrong — with no error, no warning, and a residual well inside tolerance.

`estimate_period_error` is the check the residual gate cannot make:

```python
solution = orbit.solve(seed, period)
if solution is not None:
    estimate = orbit.estimate_period_error(solution.states, solution.period)
    if estimate is None:
        ...  # the cycle could not be refined at all - do not trust it
    elif estimate > 1.0e-3:
        ...  # resolved, but not to the accuracy you may need
```

It re-solves on a mesh with twice the nodes and reports the relative period shift.
`None` means the refined solve failed, which is a verdict rather than the absence of
one: a solution too coarse to seed a finer mesh is not resolved.

The estimate is one comparison between two meshes, so it is a screen, not a bound.
On van der Pol at `mu = 12` it measured 2.6x to 4.6x the true error, but before
convergence is asymptotic two meshes can agree while both are still wrong. It costs
one extra solve at double the nodes.

## How many nodes

Continuing van der Pol from `mu = 6`, relative period error against an integrated
oracle:

| `mu` | N = 100 | N = 200 | N = 400 | N = 800 | N = 1600 |
|---|---|---|---|---|---|
| 12 | 8.0e-3 | 1.4e-3 | 3.2e-4 | 9.0e-5 | |
| 16 | 1.4e-2 | 2.5e-3 | 5.0e-4 | | |
| 40 | stalls | 1.2e-1 | 3.0e-3 | 7.4e-4 | 2.2e-4 |

Error falls roughly 4-5x per doubling. As a rough guide, `N ~ 20 to 25 x mu` gives
~1e-3, and `N ~ 5 x mu` can arrive at the target badly wrong. Use the estimate
rather than the guide where it matters.

## Finding the resolution a problem needs

Detecting that a cycle is under-resolved leaves the obvious question unanswered:
*what resolution would work?* `continue_to_resolved` measures it, and returns the
measurements whether or not it can certify an answer.

```python
study = orbit.continue_to_resolved(0, target_mu, seed, seed_period)

for level in study.levels:
    level.intervals  # node count tried
    level.period  # None if the continuation stalled here
    level.agreement  # relative shift from the previous level that arrived

if study.resolved is not None:
    study.resolved.intervals  # certified node count
    study.resolved.estimate  # the agreement that certified it
else:
    study.best  # finest level that arrived, uncertified
```

It reruns the continuation at successively doubled meshes. A level is certified only
once **two consecutive** levels agree to the tolerance, and the larger agreement is
reported. Measured at `mu = 12`:

| N | period | agreement | true error |
|---|---|---|---|
| 100 | 22.336266 | — | 8.04e-3 |
| 200 | 22.189801 | 6.60e-3 | 1.43e-3 |
| 400 | 22.165394 | 1.10e-3 | 3.24e-4 |
| 800 | 22.160221 | 2.33e-4 | 9.02e-5 |

Here every agreement bounds the finer level's error, so one would have been enough.
It is not always: under an earlier mesh monitor the same problem gave a 200/400
agreement of 1.13e-3 against a 400-node error of 1.21e-3. The second confirmation
is what keeps a certified estimate honest when convergence is not yet asymptotic.

Tune the search with `ResolutionSettings(tolerance=..., doublings=...)`.

It doubles the *mesh* and re-continues rather than refining the answer in hand,
because neither shortcut works on a stiff cycle: an under-resolved solution fails
outright when interpolated onto a finer mesh, and solving cold at higher resolution
lands on a different cycle. Only continuation reliably reaches a stiff cycle.

That makes it expensive — one full continuation per resolution. Use
`estimate_period_error` to screen a cycle already in hand for gross error, and
`continue_to_resolved` when the node count is not known or the number matters.

## Trusting a cycle's Floquet multipliers

One multiplier of every cycle is exactly `1` — the direction along the orbit. The
computed one is not, and its distance from `1` is the error of the whole set:
`CyclePoint.floquet_error`. The monodromy is integrated along the *discrete* orbit,
so an under-resolved mesh shifts every multiplier, and by far more than it shifts
the period. Measured on a Bogdanov–Takens cycle hugging a saddle, true nontrivial
multiplier 6.163:

| nodes | period error | trivial multiplier | nontrivial |
|---|---|---|---|
| 80 | 0.22% | 0.911 (**8.9% off**) | 6.596 |
| 160 | 0.06% | 0.975 (2.5%) | 6.278 |
| 320 | 0.01% | 0.994 (0.6%) | 6.192 |
| 640 | 0.004% | 0.998 (0.2%) | 6.170 |

Second order in the mesh, forty times more sensitive than the period. On this
orbit the quadrature was not the limit — on the exact orbit sampled at 80 nodes it
returned the multipliers to four figures — so only nodes, or a higher-order scheme,
help. On a smooth cycle the quadrature *was* the limit, until the monodromy walked
the Hermite interpolant (below). Stage pages print the error beside the
multipliers and draw the trivial one grey.

### Refusing the part of a branch that cannot be believed

The error above is a property of the *discretisation*, so it varies enormously along
one branch. On the same Bogdanov–Takens branch at 80 nodes it reads 8.9% on the
representative cycle in that table and **59.3%** on the orbit nearest the
homoclinic — the one whose period the same mesh gets 3.8% wrong. A single sampled
point says nothing about the rest.

`resolved_branch` cuts a traced branch where its multipliers stop being trustworthy:

```python
points, _ = continuation.trace(seed, 0.02, 400, direction=-1.0)
branch = resolved_branch(points, tolerance=2.0e-2)
if branch.refused:
    ...  # the trace went further than the mesh can follow: raise the node count
for point in branch.points:
    ...  # stability here is worth reading
```

What comes back is an **unbroken run**, not a filter: `classify_transition` compares
*adjacent* points, so a branch with holes in it would have a crossing interpolated
across a gap that was never examined. `ResolvedBranch.bifurcations` is therefore
re-derived over the kept run.

It is the **longest** such run rather than the leading one, because a branch is not
always handed over in the order it was traced. Resolution degrades monotonically
outward from the seed, so for `trace`'s own output the leading run is the answer — but
a film that traces both ways from a seed and sorts by parameter, the usual shape here,
puts the *worst* point first, and a leading run would then refuse the whole branch
including the well-resolved middle. `refused` is reported as a count because "traced
further than it can be trusted" and "the cycle really ended" need telling apart.

The default tolerance is 2%: a crossing of the unit circle is located by linearly
interpolating `|mu| - 1` between adjacent points, so a couple of percent moves the
reported crossing by a comparable fraction of one step, while tens of percent can
invent a crossing, hide one, or place it a long way off.

### What does *not* fix it

Adapting the mesh. This was measured on the Bogdanov–Takens branch's worst orbit at
80 nodes and does not work:

| 80-node mesh | spacing ratio | period error | trivial multiplier |
|---|---|---|---|
| uniform | 1.0 | 3.83% | **59.3%** |
| equidistributed `curvature` | 61.4 | 9.13% | 61.0% |
| equidistributed `curvature^1/2` | 9.3 | 3.17% | 56.5% |
| equidistributed `curvature^1/3` (what `AdaptivePeriodicOrbit` uses) | 4.4 | 2.89% | 54.8% |
| equidistributed `norm(dx/ds)` | 32.7 | 3.69% | **46.5%** |
| equidistributed `norm(dx/ds)^1/2` | 6.2 | 2.55% | 49.2% |

The best of them is a factor of 1.3, and the most aggressive make the period *worse*.
Re-adapting along the branch instead of once — re-projecting the pseudo-arclength
tangent onto each new mesh and re-solving — took the worst case from 59.3% to 49.3%
for 7.5x the time (302 s against 40 s), left the median slightly worse (3.95%
against 3.71%), and stopped marginally sooner on the branch. There is no 80-node
mesh that resolves this orbit, so no monitor could have found one.

What fixes it is the **order of the collocation**, not the mesh. The same orbit, the
same 80 *uniform* nodes, under `HermiteSimpsonOrbit`'s fourth-order collocation:

| 80 uniform nodes | period error | trivial multiplier | nontrivial (true 226.45) |
|---|---|---|---|
| trapezoidal | 3.83% | 59.3% | 337.4 (49.0% off) |
| Hermite–Simpson | 0.00% | **0.12%** | 226.2 (0.1% off) |

A factor of 500 for no extra nodes, and 8.9% → under 0.01% on the representative
cycle. `CycleContinuation` now takes that scheme as its `collocation` argument,
`"hermite_simpson"` by default and `"trapezoidal"` on request, with the analytic
sparse Jacobian derived for both (each agrees with finite differences to about
5e-8, the difference scheme's own floor). The sparsity pattern is unchanged - an
interval still couples only its own two nodes - so the midpoint's chain-rule
terms simply widen the same four blocks, and the sparse solve stays the fast
path: on the Bogdanov–Takens branch a step costs 0.069 s at 80 nodes and 0.129 s
at 160, against 0.049 s and 0.113 s for trapezoidal (1.4x and 1.15x; the per-step
cost is shared with the monodromy and the tangent solve, which the scheme does
not touch).

With it the whole 80-node Bogdanov–Takens branch is under the 2% gate: the worst
Floquet error over its 605 cycles falls from 59.3% to 0.077%, and `resolved_branch`
keeps all 605 where it kept 212.

### The monodromy's own floor

Seven films did not move under the fourth-order scheme at all - hopf stayed at
0.950%, the ferroelectric ring at 15.3% - because their error was never the
orbit's. The variational equation `Y' = T Df(x(s)) Y` was integrated along a
straight chord between nodes with the Jacobian frozen over the four Runge–Kutta
stages, both second order, and on a cycle whose Jacobian trace varies with the
radius the chord lies to the same side all the way round, so its error never
cancels. `PeriodicOrbit.monodromy` now walks the cubic Hermite interpolant through
the node states and slopes `T f(x)` - the curve Hermite–Simpson converged - and
evaluates the Jacobian at each stage's own position. On the exact Hopf cycle the
trivial multiplier's drift falls 16x per mesh doubling where it fell 4.0x: 1.7e-7
at 80 nodes against 1.6e-3. Worst Floquet error over each film's branch at 80 nodes:

| film | chord, frozen Jacobian | Hermite, stage-wise Jacobian |
|---|---|---|
| hopf | 0.950% | 2.5e-5% |
| hopf_hopf | 0.761% | 2.0e-5% |
| zero_hopf | 0.313% | 8.3e-6% |
| fold_of_cycles | 1.178% | 3.1e-5% |
| extended_bautin | 3.904% | 1.0e-4% |
| snic | 1.759% | 4.4e-4% |
| van_der_pol | 1.126% | 0.053% |
| ferroelectric_ring | 15.30% | 0.088% |
| bogdanov_takens | 0.077% | 0.089% |

The ring's period-35 wave, which had looked like a mesh the orbit outran, was the
chord's error: at `T h = 0.43` per interval the straight line misses a curved orbit
by a lot. Bogdanov–Takens alone did not improve, because its residual was the
orbit's all along - on the integrated cycle at 80 nodes the old quadrature was
already at 1e-4 - and the chord's error had been cancelling a little of it. What is
left everywhere is the orbit: nodes, or a higher-order collocation scheme.

### Orthogonal collocation

`collocation="gauss_legendre"` is that higher-order scheme: AUTO's orthogonal
collocation, `collocation_points` Gauss–Legendre points per interval (default 4,
AUTO's NCOL), a degree-`m` polynomial through `m + 1` equally spaced points per
interval, order `2m` at the mesh points and for the period. By default only the
collocation is AUTO's: the phase condition is the point (Poincaré) condition
`x_0[phase_index] = phase_value` and the pseudo-arclength constraint the plain
Euclidean norm of the whole unknown vector. AUTO's integral phase condition and
weighted L2 arclength are options (see below). Period error on the continued Hopf branch, with the order per mesh
doubling:

| m | coarsest | doubled | doubled again | and again |
|---|---|---|---|---|
| 2 | 8: 3.2e-3 | 16: 2.1e-4 (3.96) | 32: 1.3e-5 (3.99) | 64: 8.1e-7 (4.00) |
| 3 | 3: 4.4e-3 | 6: 7.9e-5 (5.79) | 12: 1.3e-6 (5.95) | 24: 2.0e-8 (5.99) |
| 4 | 3: 8.0e-5 | 6: 3.5e-7 (7.86) | 12: 1.4e-9 (7.97) | 24: 5.4e-12 (7.99) |

Its Floquet multipliers are read off its own collocation Jacobian (Fairgrieve and
Jepson): each interval's block, solved for its far end, is a transfer matrix, and
their product round the mesh is the derivative of the discrete period map. Nothing
is integrated; the blocks are the ones the tangent solve just assembled. The
nontrivial multiplier on the Hopf cycle converges at 4.00, 5.96 and 7.95 for
`m = 2, 3, 4`. `CycleContinuation.hermite_multipliers(point)` is the Hermite
monodromy above, walked through every point of the solution, as a cross-check. It is
fourth order whatever `m` is, so the two agree only to that order (3.86-3.99 per
doubling), not to `2m`: the ticket's oracle of agreement "to the collocation order"
cannot be met by this cross-check.

**The trivial multiplier of that monodromy is not an error measure.** The discrete
map of a one-step method is the flow of a modified vector field to all orders, so
its trivial multiplier stays at one far beyond the scheme's order: on van der Pol
at `m = 2` it read 3.4e-4, 1.4e-9 and 4e-16 at 20, 40 and 80 intervals while the
nontrivial multiplier was off 2.0e-3, 9.8e-5 and 6.0e-6; on the rotation-symmetric
Hopf cycle it is one to rounding. A Gauss–Legendre `CyclePoint` therefore carries
its `floquet_error` explicitly, as below. The Hermite cross-check takes `df/dx` by
automatic differentiation, because the forward difference floors it at 3e-5 on the
Bogdanov–Takens orbit nearest the homoclinic however fine the mesh.

**`floquet_error` is the larger of the trivial drift and twice the disagreement
with the same cycle re-solved on twice the intervals** (DEQ-55). The factor two is
what bounds the error whenever doubling the mesh at least halves it:
`e_N <= |mu_N - mu_2N| + e_2N <= |mu_N - mu_2N| + e_N / 2`. Both readings converge
at the scheme's order, so the estimate is tight at every `m`. Against DOP853
references with variational multipliers on the verifier's 211-cycle grid (van der
Pol at `mu` 0.5, 2, 3, 4; the Brusselator at `b` 2.5, 3, 4; `m` 1 to 4 on 8 to 96
intervals):

| `m` | cycles | below the true error | median estimate / true | within 1-100x (before) |
|---|---|---|---|---|
| 1 | 53 | 0 | 2.0 | 44 (21) |
| 2 | 54 | 0 | 2.0 | 50 (46) |
| 3 | 52 | 0 | 2.0 | 51 (43) |
| 4 | 52 | 0 | 2.0 | 51 (39) |

The 15 outside 100x are cycles whose doubled-mesh solve failed (11, read infinite)
or landed far off (4); their true errors were 2.7e-3 to 1.1, and all but one are
above the default gate of 2e-2, so refusing them is right. On 1159 cycles of the fold-of-cycles normal form, against
exact multipliers, it read low only at `m = 1` on 3 and 4 intervals, by up to 11
times on true errors of 5e-7 to 6e-6. No cycle above 2e-2 passed `resolved()`
anywhere. Van der Pol at `mu = 6` with `m = 4` on 80 intervals, which the earlier
estimate refused (4.65e-2 against a true 6.25e-3), now passes.

The earlier estimate also took the disagreement with the Hermite cross-check.
Fourth order whatever `m` is, that term over-read by a median of 5 at `m = 3` and
140 at `m = 4` (up to 1.6e4 times on the grid). It is no longer computed per point;
`hermite_multipliers(point)` still gives it on request.

**The multipliers are eigenvalues of the transfer product, computed without
forming it** (periodic QR: Bojanczyk, Golub and Van Dooren 1992, in the form of
Kressner 2006). The explicit product gives every multiplier to an absolute accuracy
of about `eps` times the largest, so van der Pol's contracting multiplier at
`mu = 5`, exactly 7.7386e-38, read 1e-15 to 7e-15. Off the factors it reads
7.7395e-38, 7.73857e-38 and 7.738555e-38 on 80, 160 and 320 intervals with `m = 4`
(relative errors 1.2e-4, 1.6e-6, 7.4e-10), and at `mu = 10` on 320 intervals
3.700e-136 against 3.696e-136.

On the two films the fourth-order schemes could not take below 0.01%, worst
`floquet_error` over the branch - for Gauss–Legendre that is the estimate above, so the
true errors are smaller still (on the worst Bogdanov–Takens orbit, 1.3e-10 at 80
intervals and 1.3e-5 at 20) (Hermite–Simpson as shipped, against
Gauss–Legendre with `m = 4`; "80 intervals" has four times the unknowns, "20
intervals" the same 81 states as 80 nodes):

| | bogdanov_takens | ferroelectric_ring |
|---|---|---|
| Hermite–Simpson, 80 nodes | 0.089% (605 cycles) | 0.088% (761) |
| Gauss–Legendre, 80 intervals | 4.8e-5% (604) | 4.6e-3% (761) |
| Gauss–Legendre, 20 intervals | 8.0e-3% (605) | 0.24% (761) |

The Gauss–Legendre rows were recorded before the doubled-mesh reading joined the
estimate. On the worst Bogdanov–Takens orbit it leaves them unchanged (1.87e-7 on 80
intervals, 8.06e-5 on 20): at `m = 4` the Hermite term is the larger.

On 80 intervals both films are under the 0.01% bar by orders of magnitude, estimate
and all. On
Hermite–Simpson's own unknowns Bogdanov–Takens is under it and the ring is not: its
period-35 wave has `T h = 1.7` per interval at 20 intervals, and four points to an
interval do not make up for it.

With four times the unknowns, one pseudo-arclength step covers about half the
parameter, so the 80-interval runs doubled the step to cover the films' span; they
then go further toward each branch's end (Bogdanov–Takens to b1 = -0.4593 where
Hermite–Simpson stops at -0.4554, the ring to a period-62 wave at λ = 0.370 against
period 35 at 0.430), and the worst errors quoted include those harder orbits. The
branch costs 3.7x (Bogdanov–Takens, 86 s against 23 s) and 4.5x (ring, 287 s against
64 s) Hermite–Simpson's at 80 intervals, and 1.1x and 1.9x at 20, so the scheme is
offered beside the default rather than in place of it, and the films still ship
under Hermite–Simpson.

### Equidistributing the mesh, measured again

The adaptive mesh rejected above was rejected under trapezoidal collocation, where
the order was the limit. Under orthogonal collocation it was built again - the
Russell–Christiansen monitor `|x^(m+1)|^(1/(m+1))` from the jumps of the piecewise
constant `m`-th derivative, re-applied every step, with the polynomial and the
tangent carried onto the new mesh and the point re-solved there - and measured on
both films, `m = 4`, over the stretch of each branch both runs covered:

| worst error | uniform | equidistributed |
|---|---|---|
| bogdanov_takens, 80 intervals | 1.44e-7 | 1.43e-7 |
| ferroelectric_ring, 80 intervals | 3.7e-6 | 4.3e-6 |
| bogdanov_takens, 20 intervals | 5.7e-5 | 4.5e-5 (and ends sooner) |
| ferroelectric_ring, 20 intervals | 2.4e-3 | 1.6e-2 |

It grades the Bogdanov–Takens mesh by a factor of six and buys nothing the order has
not already bought, and on the coarse ring it is seven times worse, so it is left
out. The interval blocks take a graded mesh all the same.

### AUTO's phase condition and arclength (DEQ-56)

`CycleContinuation(..., phase="integral", arclength="weighted")` switches both to
AUTO's (Doedel, Keller and Kernévez, IJBC 1, 1991; Doedel's lecture notes give the
derivation). The integral phase condition `∫₀¹ ⟨x(t) − x̂(t), x̂′(t)⟩ dt = 0`, with
`x̂` the previous cycle, picks the phase that minimises the distance to it, so no
coordinate has to cross any value. The weighted arclength measures a step as the L2
norm of the orbit's change plus the parameter's (the period weighted zero, AUTO's
default `THL = {11: 0}` for cycles). Both integrals use the active scheme's
quadrature: the trapezoidal rule on the nodes for the one-wide schemes, the Gauss
rule on each interval's polynomial for Gauss–Legendre.

A Hopf circle of radius 1/2 sliding sideways along `x` (`phase_index = 0`,
`phase_value = 0`): the point condition follows it to `p = 1/2`, where `x = 0`
touches the circle, then rounds an artificial fold and comes back; the integral
condition carries it to `p = 1.41` in the same 40 steps, every cycle within 1e-3 of
the exact one under all three schemes. One step of 0.05 on the Hopf branch at 24
intervals spans:

| norm | trapezoidal | Hermite–Simpson | GL m=2 | GL m=3 | GL m=4 |
|---|---|---|---|---|---|
| Euclidean | 9.90e-3 | 9.90e-3 | 7.12e-3 | 5.85e-3 | 5.08e-3 |
| weighted | 3.5959e-2 | 3.5959e-2 | 3.5959e-2 | 3.5959e-2 | 3.5959e-2 |

The defaults are unchanged, and reproduce every film to the last bit.

### Batched fields, the corrector's tangent and AUTO's step (DEQ-75)

Three changes took cycle continuation from three quarters of a cycle film's
build to a few percent of it.

**One evaluation for every state.** The collocation blocks and the monodromy ask
for `f`, `df/dx` and `df/dp` at every node, midpoint and Runge–Kutta stage, and
used to ask one state at a time, the Jacobian through Taylor jets one direction
at a time. `batched_field` hands the field every coordinate as an array of all the
states, and for the Jacobian first-order dual numbers that carry every direction
at once. Each dual operation repeats the jet's arithmetic term for term, so the
Jacobians are the jets' to the last bit and every traced orbit is unchanged bit
for bit. A field that branches on its state (the SNIC film's `if radius <`),
calls `math`, or takes a fractional power raises on arrays and is evaluated a
state at a time as before; a field that silently mis-evaluates is caught by an
exact comparison at the first and last states. The monodromy builds all its RK4
substep propagators at once and multiplies them, which moves the multipliers by
rounding only (`6.7e-16` on the Hopf cycle).

**The tangent off the corrector's LU.** The tangent system is the corrector's
last matrix with the Jacobian one Newton update later. MATCONT's Moore–Penrose
corrector takes its tangent off the corrector's factorisation; AUTO-07p takes the
secant (`CONTBV`). Reusing the LU saves an assembly and a factorisation per step,
but tilts the tangent by the order of that last update, which moves the next
point along the branch by up to `1e-4` in the parameter: it changes the spacing,
not the branch. So only the adaptive step reuses it, and the fixed step stays bit
for bit.

**AUTO's step.** `trace(..., max_arclength=...)` adapts the step by AUTO-07p's
`ADPTDS` with `ITNW = 5`: doubled after one Newton iteration, half again after
two, halved at five, capped at `max_arclength`; a failed step is halved down to
`min_arclength` (by default the step asked for, so a branch ends where it always
ended). `steps * arclength` is the arclength budget. A step longer than the base
across which a multiplier crosses the circle is retaken at the base step, so the
crossing is bracketed as before.

The films opt in where their frames allow it. Measured, interleaved with the
pre-DEQ-75 build (CPU seconds of `trace`, then of the whole film build):

| film | traced points | trace CPU | film CPU |
|---|---|---|---|
| Bogdanov–Takens (step x4) | 802 → 206 | 54.6 → 1.8 | 74.5 → 22.2 |
| ferroelectric ring (x4) | 762 → 196 | 114.6 → 2.4 | 288.0 → 168.4 |
| van der Pol (x4) | 401 → 103 | 28.9 → 1.0 | 29.4 → 1.4 |
| fold of cycles (x2) | 602 → 305 | 42.0 → 2.1 | 42.4 → 2.5 |
| SNIC (x2; field branches) | 442 → 224 | 36.9 → 12.9 | 62.2 → 37.1 |
| Hopf (fixed) | 87 | 7.8 → 0.9 | 8.2 → 1.3 |
| extended Bautin (fixed) | 167 | 17.6 → 1.7 | 18.1 → 2.2 |
| Hopf–Hopf (fixed) | 102 | 21.7 → 2.3 | 243.3 → 222.0 |
| zero-Hopf (fixed) | 184 | 20.5 → 2.0 | 189.8 → 176.6 |

On the fixed-step films every traced orbit is bit for bit the old one and every
located bifurcation the same to `2e-14`. On the adaptive ones every point is a
converged cycle of the same discrete system, within `2e-4` (scaled) of the old
dense branch's polyline, with multipliers within `7e-14` of the old monodromy's
for that cycle; the fold of cycles moved by `1.1e-9` (both `6.1e-7` from the
exact `-1/4`). Hopf, extended Bautin, Hopf–Hopf and zero-Hopf keep the fixed step:
they already step at about their frame spacing, and a larger step would show a
frame's cycle up to four frames from it. The frames of the adaptive films show
the same number of cycles as before, each within half a frame spacing of its
frame (fold of cycles: within 0.78 of a spacing, against 0.38 before).

## Trusting a computed connecting orbit

A homoclinic or heteroclinic orbit has **two** independent resolutions, and either
can be the inadequate one:

- `MeshSpec.intervals` — the mesh spacing, which sets the collocation error, `O(h^2)`
- `MeshSpec.half_length` — where the infinite orbit is cut off, which sets the
  truncation error, `O(exp(-2 * lambda * T))`

`ConnectingOrbit.solve` cannot tell you about either. It returns its best iterate
whatever that is, and the least-squares minimum is nonzero by construction, so
neither a failure signal nor the residual size carries the information.

`estimate_orbit_error` measures both:

```python
resolution = orbit.estimate_orbit_error(solution)
if resolution.estimate < 1.0e-4:
    ...  # trustworthy to about that much
elif resolution.limited_by == "spacing":
    ...  # halve h: divides discretisation error by 4
else:
    ...  # extend T by ln(4) / (2 * lambda) for the same
```

It re-solves twice — once at doubled intervals, once on a slightly longer interval
at the same spacing — and reports the largest state shift from each.

### Why both, always

Each component alone will certify an orbit that is badly wrong. Measured against the
exact homoclinic `x = 1.5 sech^2(t/2)`:

| situation | discretisation | truncation | true error |
|---|---|---|---|
| `h = 0.25`, `T = 15` | 5.8e-3 | **5.4e-11** | 7.7e-3 |
| `T = 5`, `N = 640` | **2.3e-5** | 2.7e-4 | 2.8e-4 |

In the first row truncation is genuinely finished, and taken alone it claims ten
orders of magnitude more accuracy than the orbit has. In the second the mesh is
finer than the truncated interval can exploit, so the discretisation estimate
collapses while the orbit stops improving — 12x optimistic. `estimate` is the larger
of the two for exactly this reason, and `limited_by` names which it was.

### Calibration

Trapezoidal collocation is second order, so doubling the mesh moves the answer by
`E - E/4`. The estimate therefore runs consistently **0.75x the returned orbit's
error** and bounds the *refined* orbit's error by 3x — measured at 0.75 at every
resolution from 40 to 640 intervals, on homoclinic and heteroclinic alike. It is a
usable bound rather than the screen the cycle estimator turned out to be.

`intervals` must be even. The phase condition pins the centre node, and an odd mesh
has no matching centre when doubled, so the comparison would measure a translation
of `O(h)` rather than the error.

## Covariant Lyapunov vectors and finite-time exponents

`lyapunov_spectrum` gives rates; `covariant_lyapunov_vectors` gives the directions
that go with them. It runs the Benettin pass, keeps every triangular factor, and
iterates backward on them (Ginelli et al., PRL 99, 130601, 2007; J. Phys. A 46,
254005, 2013), so the vectors it returns are carried onto each other by the
tangent flow and depend on no inner product. `FlowTangent` drives a flow and
`MapTangent` an iterated map.

What it was checked against, none of it computed by the code under test:

| Check | Reference | Measured |
|---|---|---|
| `x' = J x`, Kuptsov–Parlitz Eq. 69 | eigenvectors, exponents `1, -1, -3` | vectors to `1e-10` in `1 - cos`, every window |
| non-normal `2 x 2`, eigenvectors 20° apart | `numpy.linalg.eig` | angle `0.35` rad to `1e-6` |
| Lorenz spectrum, `T = 1000` | `0.9056, 0, -14.5723` | `0.9058, -0.0011, -14.5713` |
| Lorenz sum | trace `-41/3` | to `1e-3` |
| covariance, Lorenz | SciPy DOP853 with the analytic Jacobian | `1 - cos < 1e-8` at lags 0.1 and 0.7 |
| second Lorenz vector | flow direction (Wolfe–Samelson: 0.02°) | median `3e-5`°, max `2.5e-4`° |
| stable vs centre-unstable plane, Lorenz | bounded away from 0 (Kuptsov 2012) | min `0.136` rad; unstable vs rest min `0.009` |
| Henon / Lozi `v1, v2` angle | tangencies / none (Ginelli 2007 Fig. 1a) | min `2e-4` / `0.59` rad over `1e5` iterates |

Both transients matter. The `j`-th vector is found inside the span of the first `j`
Gram-Schmidt vectors, so an unconverged forward frame spoils it as surely as a short
backward run; each converges like `exp(-gap * time)` with `gap` the smallest
neighbouring-exponent difference.

`CovariantLyapunov.finite_time_exponents(window)` returns the covariant growth rates
over tiling windows as a `FiniteTimeExponents` distribution, whose mean is the
exponent over the stretch. For noisy flows `stochastic_finite_time_exponents` calls
`matrix_noise_lyapunov` once per window with advancing seeds, so the same settings
give the same random numbers at every parameter value and the median is a usable
test function. For the noisy pitchfork `dx = (a x - x^3) dt + s x dW` at its origin
the window exponent is exactly `a - s^2/2 + s W(T)/T`: the measured spread is
`s / sqrt(T)` within sampling error, and the median's zero, located by
`DynamicalThreshold`, sits at the same `s^2 / 2` as the exponent's. The noisy Hopf
stage draws this distribution behind the exponent.

## Continuation

::: discrecontinual_equations.continuation

## Periodic orbits

::: discrecontinual_equations.continuation.periodic_orbit

## Connecting orbits

::: discrecontinual_equations.continuation.connecting_orbit

## Heteroclinic networks

::: discrecontinual_equations.continuation.heteroclinic_network

::: discrecontinual_equations.continuation.network_continuation

::: discrecontinual_equations.continuation.residence
