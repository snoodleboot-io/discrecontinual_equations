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

### Hermite–Simpson's Floquet error on multi-turn orbits (DEQ-105)

Under Hermite–Simpson the trivial drift alone is not a bound. A multiplier's
error has two sources: the monodromy's quadrature along the interpolant, and the
orbit, which is the exact cycle of a field `O(h^4)` away. The trivial multiplier is
one for every cycle of every field, so the drift sees only the quadrature: walked
along the *exact* Rössler orbit's nodes it equals the quadrature's whole error. The
orbit's share is the multipliers' sensitivity to the field times that `O(h^4)`, and
on a period-doubled orbit that sensitivity grows by about Feigenbaum's 4.67 per
doubling, while the drift grows only with the number of turns. Rössler
(`a = b = 0.2`) at 80 nodes a turn, against Poincaré-map shooting with DOP853 at
`rtol = 1e-12`:

| turns | c | `|dmu/dc|` | true error | drift (over-read) | estimate (over-read) |
|---|---|---|---|---|---|
| 1 | 2.8 | 0.67 | 1.42e-7 | 1.20e-7 (0.84) | 2.63e-7 (1.85) |
| 2 | 3.8 | 1.7 | 1.58e-5 | 8.44e-6 (0.53) | 2.96e-5 (1.87) |
| 4 | 4.1 | 7.3 | 1.66e-4 | 2.29e-5 (0.14) | 3.11e-4 (1.87) |
| 8 | 4.18 | 34 | 8.64e-4 | 4.85e-5 (0.056) | 1.62e-3 (1.87) |
| 16 | 4.198 | 155 | 4.24e-3 | 9.82e-5 (0.023) | 7.95e-3 (1.87) |

Both sources are fourth order (15.8-15.9 per doubling of the nodes), so a
Hermite–Simpson point now carries Gauss–Legendre's estimate: the larger of the
drift and twice the gap to the same cycle re-solved on twice the nodes, started
from the cubic Hermite interpolant. Each gap is scaled by the smaller of the pair's
sizes: scaled by the coarse reading alone, a monodromy that lost the trivial
multiplier and read 1540 (van der Pol at `mu = 5` on 12 nodes) was "off by one".

On the cascade at 1 to 16 turns, three parameters in each period's window and 40,
80 and 160 nodes a turn (45 cycles), it never read low, and read 1.11 to 2.3 times
the true error (median 1.87, the `2 (1 - 2^-4)` order four predicts). The drift had
read low on 41 of them, by up to 45 times, and passed three 16-turn cycles whose
true errors were 6.2-6.7%. On the DEQ-55 verifier's grid under Hermite–Simpson (van
der Pol at `mu` 0.5-6, the Brusselator at `b` 2.5, 3, 4, the fold-of-cycles normal
form at eight points, 3 to 96 nodes; 182 converged cycles) it read low on 6, every
one a cycle whose multipliers are wrong in the leading digit (true errors 0.79 to
773, estimates 0.031 to 523, all refused); six more had a failed doubled solve and
read infinite. No cycle with a true error above 2e-2 passed anywhere, where the
drift had passed one. Over all 227 converged cycles the over-read was 1.0 to 3 on
207 and the median 1.87.

The price is a re-solve on the doubled mesh per point, and a gate about twice as
strict: 13 cycles on the grid that the drift passed with true errors under 2% are
now refused, twelve of them between 1.1% and 2% and one, van der Pol at `mu = 2`
on 12 nodes, at 0.50% (read 16.7%, a 33-fold over-read). Every film keeps all its cycles: the worst error over each film's branch
roughly doubles (0.089% to 0.17% on Bogdanov–Takens, 0.088% to 0.165% on the ring)
and stays far under the gate.

The bound has two limits, both found by an independent verification against
Radau shooting on Lorenz, forced Duffing and Langford cycles:

- It bounds the error of the cycle *actually computed*. A seed can converge to
  another cycle, and in a forced system, where the period cannot tell them apart,
  to a lower-period orbit traversed `k` times. Seeded from a four-turn Duffing
  cycle, or a twelve-turn Lorenz cycle at `r = 99.6`, the solve landed on a two-,
  six- or three-turn orbit traversed two or four times, and passed with errors of
  1 to 68 against the cycle seeded. The estimate bounded the error against the
  orbit it found. Detecting the repetition is tracked as DEQ-127.
- The monodromy's forward-difference Jacobian biases both meshes alike, so
  neither the gap nor the drift sees it. Below a floor of about `1e-5` the
  estimate can read up to about four times low (a 5.9e-6 floor on Duffing),
  still orders of magnitude under the gate. Switching to an automatic-
  differentiation Jacobian is tracked as DEQ-128.

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
Jacobians are the jets', and the values a state at a time's, to rounding: on the
machine the films are built on they were identical over every row of five
films, but NumPy may take an array's `x**3` by a vectorised loop where a scalar
calls libm's `pow`, and on CI's runners the two differ by an ulp. A field that branches on its state (the SNIC film's `if radius <`),
calls `math`, or takes a fractional power raises on arrays and is evaluated a
state at a time as before; a field that silently mis-evaluates is caught by
comparing the first and last states to a state at a time, to within 16 ulps of
the row's scale. The monodromy builds all its RK4 substep propagators at once and
multiplies them, which moves the multipliers by rounding (`6.7e-16` on the Hopf
cycle); where the field values themselves differ by an ulp, its forward-difference
Jacobians (step `1e-7`) move by about `2e-9`.

**The tangent off the corrector's LU.** The tangent system is the corrector's
last matrix with the Jacobian one Newton update later. MATCONT's Moore–Penrose
corrector takes its tangent off the corrector's factorisation; AUTO-07p takes the
secant (`CONTBV`). Reusing the LU saves an assembly and a factorisation per step,
but tilts the tangent by the order of that last update, which moves the next
point along the branch by up to `1e-4` in the parameter: it changes the spacing,
not the branch. So only the adaptive step reuses it, and the fixed step keeps
its tangent as it was.

**AUTO's step.** `trace(..., max_arclength=...)` adapts the step by AUTO-07p's
`ADPTDS` with `ITNW = 5`: doubled after one Newton iteration, half again after
two, halved at five, capped at `max_arclength`; a failed step is halved down to
`min_arclength` (by default the step asked for, so a branch ends where it always
ended). `steps * arclength` is the arclength budget. A step longer than the base
across which a multiplier crosses the circle is retaken at the base step, so the
crossing is bracketed as before.

The films opt in where their frames allow it. Measured best of five builds,
interleaved with the pre-DEQ-75 build (CPU seconds of `trace`, then of the
whole film build):

| film | traced points | trace CPU | film CPU |
|---|---|---|---|
| Bogdanov–Takens (step x4) | 802 → 206 | 48.9 → 1.6 | 66.5 → 19.1 |
| ferroelectric ring (x4) | 762 → 196 | 111.3 → 2.2 | 279.7 → 168.3 |
| van der Pol (x4) | 401 → 103 | 25.2 → 0.7 | 25.7 → 1.1 |
| fold of cycles (x2) | 602 → 305 | 43.1 → 1.7 | 43.6 → 2.1 |
| SNIC (x2; field branches) | 442 → 224 | 32.7 → 12.8 | 53.1 → 33.0 |
| Hopf (fixed) | 87 | 7.8 → 0.8 | 8.1 → 1.1 |
| extended Bautin (fixed) | 167 | 16.9 → 1.7 | 17.3 → 2.2 |
| Hopf–Hopf (fixed) | 102 | 20.2 → 2.1 | 233.6 → 211.2 |
| zero-Hopf (fixed) | 184 | 18.2 → 1.9 | 181.5 → 166.4 |

On the fixed-step films, built on the development machine, every traced orbit was
bit for bit the old one and every located bifurcation the same to `2e-14`; on a
CPU whose vectorised powers round differently they agree to rounding. On the adaptive ones every point is a
converged cycle of the same discrete system, within `2e-4` (scaled) of the old
dense branch's polyline, with multipliers within `7e-14` of the old monodromy's
for that cycle; the fold of cycles moved by `1.1e-9` (both `6.1e-7` from the
exact `-1/4`). Hopf, extended Bautin, Hopf–Hopf and zero-Hopf keep the fixed step:
they already step at about their frame spacing, and a larger step would show a
frame's cycle up to four frames from it. The frames of the adaptive films show
the same number of cycles as before, each within half a frame spacing of its
frame (fold of cycles: within 0.78 of a spacing, against 0.38 before).

### Cycle bifurcations in two parameters (DEQ-35)

`CycleCurveContinuation(equation, (i, j), intervals, ...).trace(start, kind,
arclength, steps)` continues a fold (`"fold_of_cycles"`), flip
(`"period_doubling"`) or torus (`"neimark_sacker"`) bifurcation of a cycle in
parameters `i` and `j`, from a point `start` of a one-parameter `CycleContinuation`
branch near the crossing. The cycle keeps its collocation system; the
bifurcation is added in minimally augmented form (Doedel, Govaerts and
Kuznetsov, SIAM J. Numer. Anal. 41, 2003): a sparse operator built from the
collocation blocks, singular exactly on the curve, is bordered to a nonsingular
one and its corner is the test function. For the fold it is the cycle Jacobian
in the states and the period; for the flip the blocks with `v_N + v_0 = 0`; for
the torus the blocks on a doubled mesh with `v_2N - 2 kappa v_N + v_0 = 0`, two
test functions and `kappa = cos theta` one more unknown. Its gradient comes from
one more assembly of the blocks, displaced along the null vector.

Along a curve the codimension-two points are located, by a secant on the curve
itself: the cusp of cycles, the generalised flip and the Chenciner point (the
curve's own normal form coefficient vanishing), fold-flip, fold-torus,
flip-torus and double torus (a second critical multiplier), and the 1:1, 1:2,
1:3 and 1:4 resonances (`kappa = 1, -1, -1/2, 0`). `CycleNormalForm` computes the
periodic normal form coefficients `b`, `c` and `d` (Kuznetsov, Govaerts, Doedel
and Dhooge, SIAM J. Numer. Anal. 43, 2005) from Hermite-Simpson solutions of
the linear periodic boundary-value problems along the cycle, fourth order
(relative errors `1.7e-3, 1.05e-4, 6.5e-6, 4.1e-7` from 10 to 80 intervals).

| claim | reference | measured |
|---|---|---|
| fold `b` | closed form on `r' = r(m1 + m2 r^2 + r^4 - r^6)`, `theta' = 1 + r^2/2` | `6.5e-6` relative at 40 intervals |
| flip `c` | DOP853 doubled cycles of MATCONT's `adaptx`, `-beta / xi^2` extrapolated: `-0.30512` | `-0.30513` |
| torus `Re d` | closed form on a turned Bautin normal form; a DOP853 return map on a turned Takens unfolding (`1.0238`) | `8e-7` relative; `1.0236` |
| fold curve and cusp | closed form, cusp at `(1/27, -1/3)` | `1e-9`; `1e-12` |
| torus curve, `kappa` | `b1 = -b2^2`, `kappa = cos(omega T)` | `1e-8`, `1e-5` |
| 1:3, 1:4 resonances | closed form | `2e-7` in `b2` |
| 1:1 resonance | the origin | `(-1.7e-10, 1.2e-5)` |
| Chenciner, fold-torus | the origin | `3.5e-7`, `2e-11` |
| first flip of MATCONT's Rossler | tutorial III: `A = 0.27191773`, `T = 6.148967` | `0.27191774`, `6.148968`; DOP853 shooting reads `-1 +- 3e-6` along the curve |
| flip-torus of the forced predator-prey model | De Witte et al. (SIADS 12, 2013): `(b2, eps) = (0.277, 0.530)` | `(0.27672, 0.53044)` |

MATCONT prints a different normalisation of `c` and `d`: for `adaptx` and the
Rossler tutorial it reports `-4.27e-2` and `-4.28e-3` where the formula above,
confirmed by the doubled cycle, gives `-0.305` and `-9.12e-3`, a ratio that is
not a power of the period; and its manual and its tutorial print `-1.406e-6`
and `-4.912e-8` for the same Steinmetz-Larter torus. Signs agree throughout.
The atlas page `cycle_codim2.html` draws the fold and torus curves of the turned
Takens unfolding and their resonances.

## Following a period-doubling cascade

`PeriodDoublingCascade` follows a cascade of period-doublings: it traces a cycle
branch with `CycleContinuation` until the flip multiplier passes `-1`, locates that
doubling, switches onto the cycle born there, and traces that one on a mesh of twice
the intervals - the period has doubled, so the same resolution per turn needs twice
the mesh. Each doubling's parameter is the cubic in the flip multiplier through the
four traced points nearest `-1`, evaluated at `-1` (`locate_doubling`): fourth order
in the parameter step, where the linear estimate `classify_transition` reports is
second. The switch seeds the doubled branch with the flow from `x_0 + eps v` over two
periods, `v` the eigenvector of the Hermite monodromy for the multiplier nearest
`-1`, solved just past the doubling and accepted only if its two halves differ (the
old cycle traversed twice solves the same system). `feigenbaum_ratios` turns the
doublings into the ratios of successive gaps.

```python
cascade = PeriodDoublingCascade(equation, 0, 20, phase_index=1, phase_value=0.0)
levels = cascade.follow(seed, 0.5, 6)  # Gauss-Legendre, m = 4, by default
ratios = feigenbaum_ratios([level.doubling for level in levels])
```

### Rössler and Feigenbaum's delta

The Rössler system `x' = -y - z, y' = x + a y, z' = b + z (x - c)` at `a = b = 0.2`,
continued in `c` from the period-1 cycle, doubles at the values below. The
reference is independent of the collocation: Poincaré-map shooting on the section
`y = 0, y' > 0` with scipy's DOP853 at `rtol = 1e-12`, the period-`2^k` cycle a
fixed point of the `2^k`-th return map, each doubling the brentq root of its flip
eigenvalue plus one. Its first two lie inside Wilczak and Zgliczyński's
computer-assisted enclosures, `[2.83244, 2.832446]` and
`[3.83735812, 3.837358168411]` (arXiv:0712.1123, §6.3-6.4, whose `a` is this `c`).
The library column is `PeriodDoublingCascade` at twenty intervals per turn
(`m = 4`), arclength 1, eight points per predicted gap - the test's settings,
continued to six branches:

| doubling | period | intervals | points | reference `c` | library `c` | difference | worst bound |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 20 | 4 | 2.8324450278621 | 2.8324450911 | +6.3e-8 | 1.5e-6 |
| 2 | 2 | 40 | 19 | 3.8373581684065 | 3.8373579477 | -2.2e-7 | 6.8e-6 |
| 3 | 4 | 80 | 12 | 4.1242147320743 | 4.1242147771 | +4.5e-8 | 4.3e-5 |
| 4 | 8 | 160 | 10 | 4.1869969993951 | 4.1869971520 | +1.5e-7 | 1.0e-4 |
| 5 | 16 | 320 | 9 | 4.2005372929837 | 4.2005374443 | +1.5e-7 | 2.1e-4 |
| 6 | 32 | 640 | 9 | 4.2034406286571 | 4.2034407797 | +1.5e-7 | 4.3e-4 |

"Period" is in turns of the first cycle; "worst bound" is the largest
`floquet_error` on the branch, every point of which `resolved_branch` keeps. The
ratios of successive gaps:

| ratio | reference | library | library − delta |
|---|---|---|---|
| delta_1 | 3.503190 | 3.503186 | −1.16602 |
| delta_2 | 4.569070 | 4.569066 | −0.10014 |
| delta_3 | 4.636699 | 4.636708 | −0.03249 |
| delta_4 | 4.663702 | 4.663702 | −0.00550 |

Feigenbaum's `delta = 4.669201609102990671853...` (Briggs, Math. Comp. 57 (1991);
OEIS A006890 and Broadhurst's 1018 digits agree). The measured ratios approach it from below and
each is closer than the last - 1.17, 0.100, 0.032 and 0.0055 short, the error
falling by 11.6, 3.1 and 5.9 from one to the next - so the measured delta is
**4.6637** from the first six doublings, 0.12% short of Feigenbaum's. The library's
ratios agree with the reference's to 1e-5 at every depth, so that shortfall is the
cascade's own approach to its limit, not the discretisation's.

### What limits the depth

Not the accuracy of the multipliers, at Gauss–Legendre `m = 4`. At a fixed number of
intervals per turn the error in the flip multiplier against the shooting reference
grows by about four and a half per level, and falls by `2^8` per mesh doubling, as
the scheme's order says it should:

| turns (c) | 20 per turn, `m = 4` | 40 per turn, `m = 4` | 80 per turn, Hermite–Simpson |
|---|---|---|---|
| 1 (2.8) | 6.8e-8 (bound 1.5e-6) | 2.6e-10 (bound 1.0e-7) | 1.4e-7 (reads 1.2e-7) |
| 2 (3.8) | 4.0e-7 (1.5e-5) | 1.4e-9 (4.7e-7) | 1.6e-5 (reads 8.4e-6) |
| 4 (4.1) | 1.8e-6 (3.3e-5) | 6.1e-9 (1.3e-6) | 1.7e-4 (reads 2.3e-5) |
| 8 (4.18) | 3.3e-6 (2.2e-5) | 2.8e-8 (2.7e-6) | 8.6e-4 (reads 4.9e-5) |
| 16 (4.198) | 1.4e-5 (6.9e-5) | 1.3e-7 (5.4e-6) | 4.2e-3 (reads 9.8e-5) |

A doubling's parameter is off by the multiplier's error over the flip multiplier's
slope in `c`, and that slope grows by about `delta` per level as the gaps shrink, so
at twenty per turn every doubling is located to a few parts in ten million and the
ratios to five figures; the error bound (`CyclePoint.floquet_error`) stays
above the true error at every depth and two orders under the default tolerance at
sixteen turns. Hermite–Simpson on the same number of unknowns (eighty nodes per
turn) is a hundred times worse at two turns and three hundred at sixteen, and its
error measure - the trivial multiplier's drift - reads up to 43 times *low* there:
it is not a bound on a multi-turn orbit.

What limits the depth is the cost. The mesh doubles with the period, every point
under Gauss–Legendre re-solves itself on twice its mesh for its error bound, and
the doubled-mesh solve and the Hermite cross-check walk all `4N + 1` states, so a
point costs about four to five times the last level's: on this run, on a machine carrying a load of about thirty on twelve
cores, tracing the six branches took 9, 89, 72, 68, 491 and 1650 seconds, forty
minutes in all with the switches. At ten
intervals per turn the bound itself becomes the limit instead, growing about nine
times a level - 1.5e-4, 1.6e-3, 1.3e-2 on the first three branches, the third within a
factor of two of `resolved_branch`'s default 2e-2 - and the switch onto the fourth
branch found no seed.

## Trusting a computed connecting orbit

A homoclinic or heteroclinic orbit has **two** independent resolutions, and either
can be the inadequate one:

- `MeshSpec.intervals` — the mesh spacing, which sets the collocation error, `O(h^2)`
- `MeshSpec.half_length` — where the infinite orbit is cut off, which sets the
  truncation error, `O(exp(-2 * lambda * T))`

`ConnectingOrbit.solve` cannot tell you about either: a converged solve proves only
that the discrete system was satisfied on *this* mesh over *this* interval. What it
does tell you is whether it converged at all, and whether the path it returns is a
trajectory — see [Convergence and refusal](#convergence-and-refusal) below.

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

### Convergence and refusal

**The problem.** Gauss-Newton on the projection boundary-value problem can diverge.
When it did, `solve` used to return its best iterate anyway, and that iterate was
often the seed. On Kirk-Silber in sheared coordinates, solved in the full space on
the edges leaving `xi2`, the result was a path with trapezoidal residual 0.09 and
no warning. The connections leaving `xi2` form a family there, so the Newton matrix
is nearly singular. The refinement solves returned their seeds as well, so the
resolution estimate read 0.0.

**The concept.** A solve is converged only if its best iterate passes one of two
tests:

- its Newton residual is below `1e-9`, or
- its Gauss-Newton correction is below `1e-7 * (1 + |x_i|)` in every component.

This is the convergence test AUTO uses (`EPSU`) and COLNEW uses (`CONTRL`); neither
program returns an iterate that fails it. The second test certifies a stationary
point of the least squares. A system with no exact solution, such as a homoclinic
orbit with one boundary condition too many, stops at such a point. The correction
must also satisfy the normal equations `J^T (J dx + r) = 0`, so a broken linear
solve cannot certify an iterate by returning a zero step. Any other outcome raises
`NotConvergedError`, which carries the best iterate for diagnosis.

Converged is not the same as *a trajectory*. Every returned `OrbitSolution`
carries its `CollocationResidual` as `solution.residual`, and that residual is
what says whether the path satisfies the trapezoidal rule. A refinement solve that
does not converge makes its component of the estimate `inf`, not a shift, and
`OrbitResolution.valid` reports it. So `estimate == 0.0` means that both
refinements converged and moved nothing.

```python
import numpy as np
from discrecontinual_equations.continuation import (
    HeteroclinicOrbit, MeshSpec, NotConvergedError, Terminus,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.variable import Variable


class DoubleWell(DeterministicFunction):
    def eval(self, point, time=None):
        x, y = point
        return [y, -x + x**3]


field = DoubleWell(variables=[Variable(), Variable()], parameters=[],
                   results=[Variable(), Variable()], time=None)
def jacobian(x):
    return np.array([[0.0, 1.0], [-1.0 + 3.0 * x * x, 0.0]])
orbit = HeteroclinicOrbit(
    field,
    Terminus(np.array([-1.0, 0.0]), jacobian(-1.0)),
    Terminus(np.array([1.0, 0.0]), jacobian(1.0)),
    MeshSpec(half_length=8.0, intervals=200),
)
times = np.linspace(-8.0, 8.0, 201)
try:
    solution = orbit.solve(np.column_stack([np.tanh(times / 2), 0 * times]))
except NotConvergedError as refused:
    print("refused:", refused, refused.solution.residual.worst)
else:
    assert solution.residual.is_trajectory
    resolution = orbit.estimate_orbit_error(solution)
    assert resolution.valid
```

**Limits.**

- A connection that is not isolated, such as one in a family, is refused rather
  than solved. The full-space solve on those Kirk-Silber edges is refused in this
  way. `HeteroclinicNetwork.solve` solves the connection inside its invariant
  subspace instead, and records the full-space refusal on
  `ConnectionOrbit.full_space.converged`.
- Measured but not adopted: lsmr steps, and Levenberg-Marquardt, do converge on
  those edges. They converge to a *different* member of the family, 0.025 to 0.16
  off the plane. A damped Gauss-Newton step (Armijo backtracking, smallest step
  `1e-3` as in COLNEW) stalls without converging.
- A phase condition placed where the pinned coordinate is stationary fixes the time
  shift only to second order. An example is pinning `x` at its peak. Gauss-Newton
  then wanders along the shift and is refused, so pin a coordinate where it is
  changing.
- `HomoclinicCurve.solve_point` solves a square system and has always required its
  residual to be below `1e-10`. It returns `None` otherwise, including for an
  iterate that overflows, and `trace` stops at the first `None`. Its points carry
  their collocation residual as `point.residual`.

References: [Doedel2007](../references.md) (AUTO-07p manual, `EPSU`, `ITNW`);
[Bader1987](../references.md) (COLNEW, subroutine `CONTRL`).

### Continuing a homoclinic curve, and where it stops (DEQ-112)

A homoclinic orbit to a saddle exists only on a curve in a two-parameter plane.
`HomoclinicCurve` continues it: one parameter is swept, the other is solved for
together with the orbit and the saddle, using the defining system of
HomCont ([ChampneysKuznetsovSandstede1996](../references.md)) with
projection boundary conditions ([Beyn1990](../references.md)).

Those boundary conditions give one row per stable and one per unstable
eigenvalue of the saddle. With `N` nodes in `n` dimensions the Newton system has
`N n + s + u + 1` rows and `N n + n + 1` unknowns, so it is square only while
`s + u = n` (the saddle is hyperbolic), and a homoclinic orbit needs `s, u >= 1`.
The counts are read at the seed and held:

- a seed that is not a hyperbolic saddle is refused before Newton starts;
- an iterate at which the counts change is refused too.

Both raise `EigenvalueCountError`, a `ValueError` whose message names the
required and found counts and both parameter values, and which carries them as
`expected`, `found`, `continuation` and `sweep`. `trace` stops at such a point,
returns the points it solved, as the other curve tracers do, and issues a
`RuntimeWarning` with the same message.

```python
import math
import warnings

import numpy as np

from discrecontinual_equations.continuation.connecting_orbit import MeshSpec
from discrecontinual_equations.continuation.homoclinic_curve import (
    EigenvalueCountError,
    HomoclinicCurve,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class State(Variable, name="State", abbreviation="v"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


class BetaOne(Parameter, name="beta_one", abbreviation="b1"):
    pass


class BetaTwo(Parameter, name="beta_two", abbreviation="b2"):
    pass


class BogdanovTakens(DeterministicFunction):
    """x' = y, y' = b1 + b2 y + x^2 + x y."""

    def eval(self, point, time=None):
        b1, b2 = self.parameters[0].value, self.parameters[1].value
        x, y = point
        return [y, b1 + b2 * y + x * x + x * y]


parameters = [BetaOne(value=0.0), BetaTwo(value=0.0)]
equation = DifferentialEquation(
    variables=[State(), State()],
    time=Time(),
    parameters=parameters,
    derivative=BogdanovTakens(
        variables=[State(), State()],
        parameters=parameters,
        results=[State(), State()],
        time=None,
    ),
)

# The Bogdanov-Takens asymptotics: b1 ~ -eps^4 at b2 = 5 eps^2 / 7.
eps, intervals = 0.25, 30
half = 5.0 * math.sqrt(2.0) / eps
t = np.linspace(-half, half, intervals + 1)
profile = 1.0 / np.cosh(eps * t / math.sqrt(2.0)) ** 2
orbit = np.column_stack(
    [
        eps**2 * (1.0 - 3.0 * profile),
        eps**3 * 3.0 * math.sqrt(2.0) * profile * np.tanh(eps * t / math.sqrt(2.0)),
    ],
)
saddle = np.array([eps**2, 0.0])

# Free b1, sweep b2 towards the Bogdanov-Takens point.
curve = HomoclinicCurve(equation, 0, 1, MeshSpec(half, intervals, 1, 0.0))
points = curve.trace([5.0 / 7.0 * eps**2, 0.04], orbit, saddle, -(eps**4))
print([(round(p.sweep, 4), round(p.continuation, 6)) for p in points])

# Free b2 from a seed off the curve: b1 > 0 has no equilibrium at all.
free_b2 = HomoclinicCurve(equation, 1, 0, MeshSpec(half, intervals, 1, 0.0))
try:
    free_b2.solve_point(5.0 / 7.0 * eps**2, orbit, saddle, -(eps**4))
except EigenvalueCountError as refusal:
    print(refusal.expected, refusal.found, f"{refusal.continuation:.3g}")
```

This prints the two solved points, then `(1, 1) (2, 0) -0.46`: freeing `b2`
from a seed with `b1 > 0`, where the field has no equilibrium at all, Newton
moves `b2` until the "saddle" is a stable node.

#### Reading the counts exactly (DEQ-154)

The counts, and the eigenvectors the boundary rows project onto, come from the
saddle's exact Jacobian, computed by evaluating the field on Taylor jets
([Griewank2008](../references.md), ch. 13) - the same automatic differentiation
the node Jacobians use. A forward difference with step `dx = 1e-7` would read an
eigenvalue `lam` as about `lam + c dx`, where `c` is the field's curvature in
that direction, so a genuine saddle with `|lam| < c dx` would be refused as
non-hyperbolic or, worse, counted on the wrong side and solved with the wrong
rows. On `x' = x, y' = -y, z' = lam z + c z^2` (eigenvalues `1, -1, lam`) the
difference got 13 of 39 cases with `|lam|` from `2e-9` to `1e-6` and `c` up to 10
wrong; the exact Jacobian gets all 39 right.

The Newton matrix still differences the boundary rows in the end nodes, the
saddle and the parameter (as HomCont does). A shift of the saddle by `dx` moves
`lam` by about `2 c dx`, which can carry a small eigenvalue across zero although
the iterate is a perfectly good saddle. So a shifted saddle's eigenvalues are not
re-counted: each shifted eigenvector is matched to the iterate's eigenvector it
most nearly lies along - a maximum-overlap assignment on `|W_iterate V_shifted|`
([Bindel2008](../references.md), sec. 3.2; [Crouse2016](../references.md)) - and
keeps that eigenvector's side and row. Matching the eigenvalues by distance
instead would fail when a stable and an unstable eigenvalue are both closer to
zero than the shift moves them: each shifted one then lies nearer the other's
unshifted value, and their rows swap. Eigenvectors are only as well defined as
the eigenvalues are separated, so a repeated eigenvalue with no preferred basis
cannot be followed this way either. Only an iterate is ever refused, and an
iterate that strays far enough from the saddle to change its counts is refused
even though the converged point would be a saddle.

```python
import numpy as np

from discrecontinual_equations.continuation.connecting_orbit import MeshSpec
from discrecontinual_equations.continuation.homoclinic_curve import HomoclinicCurve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class Coordinate(Variable, name="Coordinate", abbreviation="v"):
    pass


class Clock(Variable, name="Clock", abbreviation="t"):
    pass


class Alpha(Parameter, name="alpha", abbreviation="a"):
    pass


class Beta(Parameter, name="beta", abbreviation="b"):
    pass


class NearlyCentral(DeterministicFunction):
    """x' = x, y' = -y, z' = -1e-7 z + z^2 (a and b unused)."""

    def eval(self, point, time=None):
        x, y, z = point
        return [x, -y, -1.0e-7 * z + z * z]


parameters = [Alpha(value=0.0), Beta(value=0.0)]
variables = [Coordinate(), Coordinate(), Coordinate()]
equation = DifferentialEquation(
    variables=variables,
    time=Clock(),
    parameters=parameters,
    derivative=NearlyCentral(
        variables=variables,
        parameters=parameters,
        results=[Coordinate(), Coordinate(), Coordinate()],
        time=None,
    ),
)
curve = HomoclinicCurve(equation, 0, 1, MeshSpec(10.0, 20, 0, 0.0))
point = curve.solve_point(0.0, np.zeros((21, 3)), np.zeros(3), 0.0)
print(point is not None)  # True: two stable, one unstable, eigenvalue -1e-7 read exactly
```

A forward difference reads the third eigenvalue here as `-1e-7 + 1e-7 = 0` and
refuses the seed as non-hyperbolic.

Limits:

- A field that cannot be evaluated on Taylor jets (one that calls `math.pow`,
  `float()` or another real-only function on the state) falls back to the
  forward difference, and then a saddle with `|lam|` below about `c * 1e-7` may
  still be read with the wrong counts.
- Real parts within `1e-9` of zero count as central, so a saddle with an
  eigenvalue that small is refused as non-hyperbolic.
- A refusal only says the counts changed. Whether the curve really ends there
  (for instance at a Bogdanov-Takens point, where the saddle merges with
  another equilibrium) or Newton just wandered off from a poor seed is for the
  caller to judge; a better seed or a smaller sweep step tells them apart.
- Natural-parameter continuation cannot pass a fold of the curve in the swept
  parameter.

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

## Counting modes from a grid or from samples

`TopologicalModes` counts the modes of a density as the components of its superlevel
sets whose persistence exceeds the density's own uncertainty (arXiv:2305.03118). On a
grid density that uncertainty is round-off, and the count is the local-maximum count
of `DensityModes`. From samples, `KernelDensity` estimates the density on a
`DensityGrid` together with its pointwise standard error, and a mode counts when its
persistence clears three standard errors. `ModeCountThreshold` locates where the
count reaches two, with any `ScalarRootFinder`; `PersistenceProbe` hands the same test
to `PhenomenologicalThreshold` for a solved Fokker-Planck grid.

### The route to recorded data

The sample path needs no model. Recorded states from a device or an experiment go
straight into `KernelDensity`, one array per parameter setting, and the count and the
threshold follow as they do for simulated paths. Three things are the caller's to get
right, because the estimate cannot check them:

- **Independence.** The standard error assumes independent samples. A time series
  sampled faster than it decorrelates is worth its length divided by twice its
  integrated autocorrelation time; pass that as `effective_samples`, or thin the
  record.
- **Bandwidth.** What is counted is the mode structure of the density smoothed by the
  kernel. For the multiplicative pitchfork the smoothing moves the P-threshold from
  `s^2/2` to `s^2/2 + h^2` exactly, and a peak narrower than `h` - such as the
  integrable spike a multiplicative noise leaves at an invariant point - is not seen.
  Locate at two bandwidths if the shift matters.
- **Detection lag.** A new mode is counted once its persistence clears the bar, which
  happens a little after it exists: for the pitchfork with 40,000 samples and
  `h = 0.08`, about 0.01 in the parameter past `s^2/2 + h^2`. The lag falls as the
  samples grow.

Across parameter settings, use common random numbers where you control the noise (a
simulation), so that the test function is a deterministic function of the parameter.

::: discrecontinual_equations.continuation.topological_modes

## Manifolds of cycles

`CycleManifold` computes the stable or unstable manifold of a hyperbolic cycle as a
Fourier-Taylor chart `P(theta, sigma)`, following Castelli, Lessard and Mireles
James, *Parameterization of invariant manifolds for periodic orbits I* (SIADS 14,
2015). The chart conjugates the flow to `(theta, sigma) -> (theta + t, e^(lambda t)
sigma)`, so `chart.flow_image(theta, sigma, t)` is where the flow carries
`chart.point(theta, sigma)`. Its input is any of the library's cycles (a
`CyclePoint`'s `solution` and `mesh`), refined first to a Fourier cycle by Newton's
method; the Floquet bundles are the real eigenpairs of `Df(gamma) - d/dtheta` on
`2T`-periodic grids, so a twisted (Mobius) bundle needs no special case; and the
homological equations are solved mode by mode in the Floquet basis, their
equation (21). Nothing integrates the field.

What was checked, and how:

| Check | Reference | Result |
|---|---|---|
| Chart against closed form | `r' = r(1 - r^2)`, `z' = mu z + z^3`, pulled through a rational diffeomorphism | `< 1e-11` at `|sigma| <= 0.1` (order 14, 48 modes) |
| Mobius band | normal plane turning half a revolution per period | multipliers `-e^(aT)`, chart `< 1e-12` |
| Conjugacy | DOP853 at `1e-13` | `< 1e-9` after 1.5 time units |
| Equation (21) | dense Fourier-Galerkin solve of (18), no Floquet basis | `< 1e-9` per coefficient |
| Published values | Lorenz `rho = 22` (paper II, Section 4.3) | period and stable exponent to `1e-9` |

`chart.invariance_error(radius)` is the paper's a-posteriori indicator (28): it
falls geometrically with the Taylor order at the oracle's measured rate, and
geometrically with the number of Fourier modes until the truncation in `sigma`
takes over. `chart.radius(tolerance)` is where the last term reaches
`tolerance`: where the chart should hand over to the flow.

One manifold coordinate per chart is supported (a two-dimensional manifold); complex
Floquet multipliers are refused. `webplot.cycle_tube.add_cycle_tubes` draws the
unstable manifold of every unstable cycle on a stage as the chart's fibres carried on
by the flow; the fold-of-cycles film shows the repelling cycle's band.

`CycleConnectingOrbit` solves the paper's equation (35),
`phi_L(P(theta_0, R_0)) = Q(theta_1, R_1)`, for an orbit from one cycle's unstable
chart to another's stable chart, by damped Gauss-Newton with DOP853, seeded by
scanning the flowed source ring for its nearest approach to the target ring. The
charts' conjugacies supply both infinite tails (`connection.orbit(tail)`).

## Invariant manifolds

A saddle's stable or unstable manifold is computed in two parts. Near the
equilibrium `TaylorManifold` solves the parameterization method's invariance
equation order by order and returns a polynomial chart. Further out,
`GeodesicLevelSets` grows a two-dimensional manifold ring by ring, each ring at
one geodesic distance from the equilibrium, and returns a triangulated mesh. The
[invariant manifolds guide](../user-guide/manifolds.md) explains both methods,
gives short examples and lists their limits. `parameterization` is a different
module despite its name: it holds the scalar constraint that closes the
continuation corrector's system, either pseudo-arclength or natural parameter.

::: discrecontinual_equations.continuation.manifold

::: discrecontinual_equations.continuation.geodesic_level_sets

::: discrecontinual_equations.continuation.parameterization

The Lorenz row's reference spectrum is Viswanath's (Ph.D. thesis, Cornell, 1998,
Table 5.1: `0.90566, 0, -14.57233`). The concepts, worked examples and limits are
on the [Lyapunov exponents and noise](../user-guide/lyapunov-noise.md) page, and
the sources on the [References](../references.md#lyapunov-exponents-covariant-vectors-and-noise)
page.

## Lyapunov exponents and stochastic bifurcation

::: discrecontinual_equations.continuation.lyapunov

::: discrecontinual_equations.continuation.covariant_lyapunov

::: discrecontinual_equations.continuation.noise

::: discrecontinual_equations.continuation.stochastic

::: discrecontinual_equations.continuation.stochastic_threshold

## One Jacobian per point

### The problem

An accepted continuation point needs the Jacobian at its converged state several
times: for the tangent, the branch-point and Hopf test functions and the point's
stability. Building it once per request would make the detectors cost more than
the corrector.

### The concept

`ContinuerBuilder` wraps every Jacobian provider in a `MemoizedJacobian`. Inside
`retained()`, which `Continuer.solve` opens around one continuation, the wrapper
keys each request by the exact bytes of the state and the parameter. A repeat
request gets a copy of the remembered array, so every value is bit-identical to
recomputing it.

How the wrapper remembers depends on the provider:

- A **finite-difference** provider builds the state columns and the parameter
  column separately. The wrapper remembers each part on its own, and builds the
  extended Jacobian `[df/du | df/dlambda]` from the two remembered parts. A request
  for the state part alone never pays for the parameter column.
- A provider that **overrides `extended_jacobian`** solves all the columns at once.
  `AnalyticJacobian` is one: it drives the minimally augmented cusp and
  generalized-Hopf curves of three-parameter continuation. For such a provider the
  wrapper remembers the extended Jacobian itself and slices the state and parameter
  parts out of it. Each point then costs one solve rather than one per part.

Outside a `retained()` scope the wrapper remembers nothing and passes each request
straight to the provider, one provider call per request.

```python
import numpy as np

from discrecontinual_equations.continuation.jacobian import (
    AnalyticJacobian,
    MemoizedJacobian,
)
from discrecontinual_equations.continuation.residual import DifferentiableResidual


class Fold(DifferentiableResidual):
    """f(u, lam) = lam - u^2, which supplies its own [df/du | df/dlam]."""

    solves = 0

    def evaluate(self, state, parameter):
        return np.array([parameter - state[0] ** 2])

    def extended_jacobian(self, state, parameter, epsilon):
        Fold.solves += 1
        return np.array([[-2.0 * state[0], 1.0]])


jacobian = MemoizedJacobian(AnalyticJacobian(Fold(), 1.0e-7))
state = np.array([0.5])
with jacobian.retained():
    extended = jacobian.extended_jacobian(state, 0.25)
    state_part = jacobian.state_jacobian(state, 0.25)
    parameter_part = jacobian.parameter_derivative(state, 0.25)

print(extended, state_part, parameter_part, Fold.solves)
# [[-1.  1.]] [[-1.]] [1.] 1
```

Before DEQ-122 the wrapper built the extended Jacobian from the two parts even for
an analytic provider, so each codim-3 curve point was solved twice. Continuing a
cusp curve for ten points now takes half the analytic solves, 21 instead of 42, and
about half the field evaluations. The curve is unchanged.

### Limits

- The memory lasts for one `Continuer.solve`. The residual reads the equation's
  other parameters live, so a Jacobian remembered across solves could be stale.
- A scope remembers the four most recently used points (`capacity`).
- Keys are exact bytes. A state that differs in the last bit is a new point.
- A joint provider asked for only its state part, inside a scope, still pays for
  the whole extended solve. For `AnalyticJacobian` that costs nothing extra,
  because its state part is a slice of the same solve.

### References

No published algorithm is involved: this is a cache. The sources of the minimally
augmented systems whose Jacobians it shares are cited in the
`discrecontinual_equations.continuation.cusp_curve` and
`discrecontinual_equations.continuation.generalized_hopf_curve` module docstrings.

## Telling bifurcation points apart

Detectors find a point by a sign change of a test function along a curve, and then
name it from normal-form coefficients. Two ways that goes wrong are guarded against
here: a test function that changes sign because a vector flipped, not because a
coefficient vanished, and a label that depends on the units of the model.

### Degenerate Bogdanov-Takens points (DEQ-124)

On a Bogdanov-Takens curve in three parameters, `degenerate_bogdanov_takens` marks
where the quadratic coefficient `a = (1/2) <p1, B(q0, q0)>` vanishes and
`degenerate_bogdanov_takens_b` where `b = <p1, B(q0, q1)> + <p0, B(q0, q0)>`
vanishes ([BosschaertKuznetsov2021](../references.md), section 2.2.2). The Jordan
chains `q0, q1, p1, p0` are fixed by their normalisation only "up to a plus or minus
sign" (section 2.2.1), and both coefficients are odd in that sign. The test functions
used to fix it by making the largest component of `q0` positive. On a curve where
`q0` turns, so that its largest component changes index, that rule flips `q0`, and
the scan reported a degenerate point that is not there.

`q0` is now oriented by continuity: by the BT curve's own kernel vector, whose
normalisation `c . q = 1` keeps it on one side all along the curve, or, when a decoded
point carries none, towards the previous point's `q0`. This is what the cusp test
does on a fold curve.

```python
from discrecontinual_equations.continuation.codim3_builder import Codim3Driver, Codim3Seed
from discrecontinual_equations.continuation.codim3_config import Codim3Config

config = Codim3Config(
    continuation_parameter_index=0,
    second_parameter_index=1,
    third_parameter_index=2,
    codim3_curve="bogdanov_takens",
    codim3_detectors=["degenerate_bogdanov_takens", "degenerate_bogdanov_takens_b"],
    initial_parameter=-0.5,
    parameter_lower_bound=-0.6,
    parameter_upper_bound=0.6,
)
# `equation`: three parameters, a Bogdanov-Takens point at x = 0, (b1, b2) = (0, 0),
# e.g. the turned system of tests/continuation/test_bogdanov_takens_orientation.py.
result = Codim3Driver.run(config, equation, Codim3Seed(state=[0.0, 0.0, 0.0], parameter_a=0.0, parameter_b=0.0))
print([(point.kind, point.parameter_c) for point in result.points])
```

Checked on a three-dimensional BT curve whose `q0` turns through a change of largest
component (`tests/continuation/test_bogdanov_takens_orientation.py`). In unmixed
coordinates its chains are unit vectors, so `a` and `b` are known in closed form.
Before the change, both tests reported a spurious point where the index changes, with
LAPACK's signs and under random SVD sign flips. After it, there are no spurious
points under either. A curve with `a = 0`, or with `b = 0`, at a known parameter
reports that one point.

Limits: with no curve vector, continuity holds only while `q0` turns by less than a
right angle between neighbouring points.

### Transcritical, pitchfork or neither (DEQ-126)

A branch point is labelled (`BranchPoint.refine_kind`) from the center-manifold coefficients `a = <p, B(q, q)>`
and `c = <p, C(q, q, q) + 3 B(q, h2)>`. It is `transcritical` when `a` is non-zero,
`pitchfork` when `a` vanishes and `c` does not, and `branch_point` when both vanish.
These coefficients were compared with fixed floors, `1e-3` and `1e-6`. Multiplying
the field by a constant `s`, which only changes the unit of time, scales `a` and `c`
by `s`, so the label depended on that unit. A transcritical point became a
`pitchfork` for `s <= 1e-4`. A point whose `c` cancels became a `pitchfork` for
`s >= 10`, because finite-difference noise in `c` passed `1e-6`.

A coefficient now counts as non-zero only when it passes two tests, and neither
depends on units:

- **It is not a cancellation remnant.** It must exceed `1e-5` times the size of the
  terms it is summed from: `|p| |B(q, q)|` for `a`, and
  `|p| (|C(q, q, q)| + 3 |B(q, h2)|)` for `c`.
- **It is not noise.** It must exceed `100` times its finite-difference error. The
  error is estimated by repeating the differences with half and with twice the step.

`a` must pass a third test before it names a transcritical point, because the point
is only located to within the localizer's tolerance. Off its vertex, a pitchfork has
a quadratic coefficient of its own. In the normal form
`w' = mu w + (a/2) w^2 + (c/6) w^3` with `a = 0`, the nontrivial branch at `w0` has
critical eigenvalue `lambda = (c/3) w0^2` and quadratic coefficient `c w0`, so
`a^2 = 3 |c lambda|` there. At a transcritical point the same ratio is
`2 |a| / |c w0|`, which grows as the location improves. So `a` must also satisfy
`a^2 > 30 |c lambda|`, with `lambda = <p, A q>`. Like the other two tests, this one
is unchanged by rescaling time or the state.

```python
from pathlib import Path

from sweet_tea.registry import Registry

import discrecontinual_equations
from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.continuation_config import ContinuationConfig
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

Registry.fill_registry(
    path=str(Path(discrecontinual_equations.__file__).parent),
    module="discrecontinual_equations",
    exclude=["*.tests", "*.examples", "*.plot"],
)


class Rate(Parameter, name="Rate", abbreviation="mu"):
    pass


class Population(Variable, name="Population", abbreviation="x"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


class SlowTranscritical(DeterministicFunction):
    """x' = s (mu x - x^2), with time measured in units a hundred thousand times too small."""

    def eval(self, point, time=None):
        mu = self.parameters[0].value
        x = point[0]
        return [1.0e-5 * (mu * x - x * x)]


parameters = [Rate(value=-0.5)]
variables = [Population()]
equation = DifferentialEquation(
    variables=variables,
    time=Time(),
    parameters=parameters,
    derivative=SlowTranscritical(
        variables=variables, parameters=parameters, results=[Population()], time=None
    ),
)
config = ContinuationConfig(
    detectors=["branch_point"],
    initial_parameter=-0.5,
    initial_step=0.04,
    maximum_step=0.12,
    parameter_lower_bound=-0.5,
    parameter_upper_bound=0.5,
)
branch = ContinuerBuilder.build(config, equation).solve(equation, initial_values=[0.0])
print([point.kind for point in branch.special_points])  # ['transcritical']; was ['branch_point']
```

Checked against the four three-dimensional reference cases of the DEQ-102 verifier
(`tests/continuation/test_branch_label_scale.py`): every label is now the same for
every `s` in `1e-6 ... 1e6`, with exact and with finite-difference Jacobians, 0 of
104 wrong against 17 before. In 2000 random mixings, offsets and time scales, plus a
reflection-symmetric case where `B(q, q) = 0`, every label is right. A pitchfork evaluated up to `1e-2` off its vertex, on its
nontrivial branch, stays a `pitchfork`, and a transcritical point stays
`transcritical`. Structurally
zero coefficients reach at most `2.6e-7` of their terms' size, and genuine ones at least `1.3e-2`. Noise in `c`
stays within `7.3` times its error estimate, and genuine `c` exceeds it at least
`5e4` times. Noise in `a` can exceed its estimate (by up to `1.8e4` with
finite-difference Jacobians far from the origin); the first test is what catches it.

Limits: the step `1e-4` is in the state's units, so rescaling the *state* still
changes how much noise the differences pick up. A genuine coefficient smaller than
`1e-5` of its terms is reported as zero. When the Jacobian is too noisy to resolve
`c`, the point is honestly `branch_point`, not a guessed `pitchfork`.

## Scanning a stochastic system without repeating work (DEQ-76)

**The problem.** A D-bifurcation is located by estimating the top exponent of an
invariant state at many parameter values, and a P-bifurcation by solving the
stationary Fokker-Planck equation at many parameter values. Both scans used to
evaluate, on every step or every solve, quantities that cannot have changed.

**At an invariant state the coefficients are computed once.** `stochastic_lyapunov`
and `matrix_noise_lyapunov` keep the fields and Jacobians of the last step and reuse
them while the state's bytes are unchanged. An equilibrium whose noise vanishes there
(`f(0) = G(0) = 0`) is left exactly where it was by every Euler-Maruyama step, so the
whole horizon costs one evaluation. The comparison is of bytes, so `-0.0` and `0.0`
are different states and the exponents are identical bit for bit to the estimate
without reuse.

**A shared diffusion.** When only the drift varies, `G G^T` on the grid is the same for
every solve. `GridDiffusion` evaluates it once, and a `StationaryFokkerPlanck` given
one skips those evaluations. Under Ito it evaluates no noise at all.

```python
from discrecontinual_equations.continuation import (
    DensityGrid,
    GridDiffusion,
    Ito,
    StationaryFokkerPlanck,
)
from examples.continuation.stage_noisy_hopf import _drift, _noise, _parameters

grid = DensityGrid.box([-2.5, -2.5], [2.5, 2.5], [40, 40])
diffusion = GridDiffusion(_noise(_parameters(0.0)), grid)  # sigma only
for mu in (0.3, 0.9, 1.5):
    parameters = _parameters(mu)
    density = StationaryFokkerPlanck(
        _drift(parameters), _noise(parameters), Ito(), grid, diffusion
    ).solve()
    print(mu, density.mass())
```

**Without noise, use the closed form.** For `dx = a x dt + s x dW` the exponent of
`x = 0` is `a - s^2 / 2` ([HigMaoYuan2007], Theorem 4.1), so at `s = 0` it is `a`.
The stochastic bifurcation page now draws that curve from `LyapunovExponent` instead
of integrating it. The Euler-Maruyama estimate converges to
`E log|1 + a dt + s sqrt(dt) Z| / dt` ([HigMaoYuan2007], eq. (4.4)), which at `s = 0`
is `log(1 + a dt) / dt`. That is below `a` by `a^2 dt / 2` to leading order, at most
`1.7e-4` on that page, and that is the only change to the page's numbers.

**Limits.** The reuse needs fields that depend on the state alone. Fields that read a
clock or keep their own state would be reused wrongly. A `GridDiffusion` is checked
against its grid but not against its noise: passing one is the caller's statement
that `D` does not depend on the parameter being scanned. Under Stratonovich the drift
shift still evaluates the columns and their Jacobians at the faces. See
[References](../references.md) for the full entries.

## Matching a detected branch point (DEQ-123)

**The problem.** A branch point is asked for under one name, `"branch_point"`, but
reported under the kind the detector classified. Code that filters
`branch.special_points` by the name it asked for finds nothing whenever the
classification succeeded. The atlas pages lost their symmetry-breaking line this way
until DEQ-14, and the ferroelectric ring film lost its "symmetry breaking (λ = a)"
label until DEQ-123.

**The concept.** Where two equilibrium branches cross, the Jacobian has a simple zero
eigenvalue, and the field on the center manifold reads
`w' = (a / 2) w^2 + (c / 6) w^3 + ...`. A non-zero quadratic coefficient `a` makes the
crossing *transcritical*: the branches exchange stability through each other
(normal form `x' = r x - x^2`). When `a` vanishes, as a reflection symmetry forces,
and the cubic `c` does not, one branch splits into three: a *pitchfork*
(`x' = r x -+ x^3`). With both zero the detector leaves the label as
`"branch_point"`. `BRANCH_POINT_KINDS` holds these three labels, and
`is_branch_point(kind)` tests a kind against them.

```python
from discrecontinual_equations.continuation import (
    ContinuationConfig,
    ContinuerBuilder,
    is_branch_point,
)
from examples.continuation.bifurcation_examples import (
    Pitchfork,
    State,
    build_equation,
    ensure_registry,
)

ensure_registry()
equation = build_equation(Pitchfork, [State()])  # x' = mu x - x^3
config = ContinuationConfig(
    detectors=["branch_point"],
    initial_parameter=-2.0,
    measure="component",
    parameter_lower_bound=-2.0,
    parameter_upper_bound=2.0,
)
branch = ContinuerBuilder.build(config, equation).solve(equation, [0.0])

print([p.kind for p in branch.special_points])  # ['pitchfork']
asked = [p for p in branch.special_points if p.kind == "branch_point"]
found = [p for p in branch.special_points if is_branch_point(p.kind)]
print(len(asked), len(found))  # 0 1
```

**Limits.** The set covers equilibrium branch points only. Cycle bifurcations
(`"fold_of_cycles"`, `"period_doubling"`, `"neimark_sacker"`) and folds are other
kinds. The match is exact and case-sensitive. A test reads the labels
`BranchPoint.refine_kind` returns and fails if they and `BRANCH_POINT_KINDS` part
ways. A second test fails if any file under `examples/` compares a `.kind` against
one of the three labels by name.

The normal forms are from Strogatz2001. The coefficients `a` and `c` are the branch-point
detector's, cited in its docstring. See the [references](../references.md).

## Continuation

::: discrecontinual_equations.continuation

## Numerical continuation

The predictor-corrector machinery, its Jacobians and the drivers built on it. The
[Numerical continuation guide](../user-guide/continuation-core.md) explains each
piece with a runnable example, its limits and its sources.

### Driver, configuration and assembly

::: discrecontinual_equations.continuation.continuer

::: discrecontinual_equations.continuation.continuation_config

::: discrecontinual_equations.continuation.builder

::: discrecontinual_equations.continuation.components

::: discrecontinual_equations.continuation.point_builder

### Predictor, corrector, tangent and step

::: discrecontinual_equations.continuation.predictor

::: discrecontinual_equations.continuation.corrector

::: discrecontinual_equations.continuation.tangent

::: discrecontinual_equations.continuation.step_controller

::: discrecontinual_equations.continuation.seed_refiner

::: discrecontinual_equations.continuation.residual

::: discrecontinual_equations.continuation.linear_solver

::: discrecontinual_equations.continuation.bordered

### Locating bifurcations and switching branches

::: discrecontinual_equations.continuation.localizer

::: discrecontinual_equations.continuation.root_finder

::: discrecontinual_equations.continuation.switching

### Jacobians and derivatives

::: discrecontinual_equations.continuation.jacobian

::: discrecontinual_equations.continuation.sparsity

::: discrecontinual_equations.continuation.derivative_provider

::: discrecontinual_equations.continuation.jet

::: discrecontinual_equations.continuation.bivariate

::: discrecontinual_equations.continuation.multivariate

### Deflation, symmetry, focal values and regions

::: discrecontinual_equations.continuation.deflation

::: discrecontinual_equations.continuation.symmetry

::: discrecontinual_equations.continuation.focal

::: discrecontinual_equations.continuation.region_analysis

### Branches, diagrams, families and exploration

::: discrecontinual_equations.continuation.branch

::: discrecontinual_equations.continuation.continuation_point

::: discrecontinual_equations.continuation.continuation_state

::: discrecontinual_equations.continuation.measure

::: discrecontinual_equations.continuation.diagram

::: discrecontinual_equations.continuation.family_config

::: discrecontinual_equations.continuation.family_builder

::: discrecontinual_equations.continuation.exploration

## Periodic orbits

::: discrecontinual_equations.continuation.periodic_orbit

## Connecting orbits

The concepts, worked examples and limits are on the
[connecting orbits concept page](../user-guide/connecting-orbits.md).

::: discrecontinual_equations.continuation.connecting_orbit

::: discrecontinual_equations.continuation.homoclinic_shooting

::: discrecontinual_equations.continuation.homoclinic_curve

::: discrecontinual_equations.continuation.shilnikov

## Heteroclinic networks

::: discrecontinual_equations.continuation.heteroclinic_network

::: discrecontinual_equations.continuation.network_continuation

::: discrecontinual_equations.continuation.network_stability

::: discrecontinual_equations.continuation.residence

### Systems with robust cycles

::: discrecontinual_equations.systems.reflection_cubic

::: discrecontinual_equations.systems.coupled_ring

## Slow manifolds and canards

::: discrecontinual_equations.continuation.slow_manifold

## Period-doubling cascades

::: discrecontinual_equations.continuation.period_doubling

## Cycle manifolds

::: discrecontinual_equations.continuation.cycle_manifold

::: discrecontinual_equations.continuation.cycle_connection

## Jacobians along an orbit

The Lyapunov estimators, the quasipotential's linearisation and the equilibrium
classifier read a field's Jacobian from `jacobian_at`. By default it forward-
differences the field with the step of Dennis and Schnabel (*Numerical Methods for
Unconstrained Optimization and Nonlinear Equations*, 1983, Sec. 5.4 and Algorithm
A5.4.1): `h_j = sqrt(eps) max(|x_j|, typical_j) sign(x_j)`, divided by the step the
machine actually took. `typical` (one by default) is the size a component has where
its value says nothing - near zero, or on a component that lives at `1e-8`.
`central=True` uses `eps^(1/3)` steps either side: twice the evaluations, error
`eps^(2/3)` instead of `sqrt(eps)`. `base`, the field at the state when the caller
has it, saves the forward difference one evaluation. `jacobians_along` takes the
same steps at every row of a batch of states, through `eval_batch`, and matches
`jacobian_at` row by row.

No step fixes everything. An entry `df_i/dx_j` is resolved only to about
`sqrt(eps) |f_i| / |x_j|` forward (`eps^(2/3)` central), so a coupling of one in a
row of size `1e8` comes back anywhere in `1 +- 3` forward and `1 +- 5e-3` central
(the worst of 2000 states). Wrap the field in
`JacobianField` and `jacobian_at` returns its exact Jacobian instead, either a
closed form you pass or, by default, Taylor jets.

Lorenz's top exponent over `T = 1000` (`dt = 0.01`, one BLAS thread, best CPU
time of three on a loaded machine):

| Jacobian | per call | `T = 1000` |
|---|---|---|
| forward difference | 4 field evaluations | 29 s |
| Taylor jets (`JacobianField(field)`) | 3 jet evaluations | 45 s |
| closed form (`JacobianField(field, jacobian)`) | about 1 field evaluation | 9 s |

All three scale linearly in `T` (fitted log-log slopes 1.07, 0.97, 1.01). Jets
buy exactness, not speed; the closed form buys both.

::: discrecontinual_equations.continuation.jacobian_field

## Stationary densities and the quasipotential

The [quasipotential guide](../user-guide/quasipotential-first-passage.md)
explains the grid density, the ordered line integral method and the minimum
action paths, with examples and limits.

::: discrecontinual_equations.continuation.fokker_planck

::: discrecontinual_equations.continuation.quasipotential

### The continuation Jacobian and the derivative fallback (DEQ-118)

`jacobian_at` is not the only finite difference. The default continuation
Jacobian, `FiniteDifferenceJacobian` (registry key `finite_difference_jacobian`),
and the derivative fallback, `FiniteDifferenceDerivatives`, used to step by an
absolute `epsilon`. On a state of size `2e12`, where an ulp is `2.4e-4`, the
default `finite_difference_epsilon = 1e-7` rounds away, the column comes back as
zero and Newton is handed a singular Jacobian.

Both now take the same per-component step as `jacobian_at`, centrally:
`h_j = epsilon * max(|u_j|, typical_j)`, and `FiniteDifferenceJacobian` divides by
the width the machine actually stepped; the parameter column steps
`epsilon * max(|lambda|, parameter_typical)`. `epsilon` is therefore a relative
step. **A configured `finite_difference_epsilon` keeps its meaning on every
component of size one or less** (with the default typical size of one, the step
there is `epsilon` exactly) and grows with a component above that.
`FiniteDifferenceDerivatives` steps a general direction `d` by

`t = epsilon min(1, max_j |d_j|) / max_j(|d_j| / s_j)`, with `s_j = max(|x_j|, typical_j)`,

so that no component moves further than its own step: `|t d_j| <= epsilon s_j`
(DEQ-163). Along a unit direction this is Dennis and Schnabel's
`epsilon max(|x_j|, typical_j)`; for any direction with `max |d| <= 1` that
moves only components of size one or less it is `epsilon` itself, as it was when the
step was absolute. Before DEQ-163 the spacing kept growing with `max |d|`: at
`x = d = 1e6` it moved the state by `1e6`, its own size, and the first Taylor
coefficient of `x^3` came out a third too large. Now that
coefficient is right to about `2e-11` at every size:

```python
import numpy as np

from discrecontinual_equations.continuation.derivative_provider import (
    FiniteDifferenceDerivatives,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.variable import Variable


class X(Variable, name="x", abbreviation="x"):
    pass


class Cube(DeterministicFunction):
    def eval(self, point, time=None):
        return [point[0] ** 3]


cube = Cube(variables=[X()], parameters=[], results=[X()], time=None)
x = np.array([1.0e6])
a1 = FiniteDifferenceDerivatives(1.0e-6).taylor(cube, x, x, 1, 0.0)[1].real[0]
print(abs(a1 - 3.0e18) / 3.0e18)  # about 2e-11; 0.33 when the step grew with |d|
```

A long direction at unit scale (`max |d| > 1`, components of size one or less)
now takes `epsilon / max |d|`, not `epsilon`. The library's detectors do pass
such directions, so their results moved when they run on this provider. The
DEQ-163 verifier measured, by instrumenting the calls:

| caller | largest `max |d|` | calls with `max |d| > 1` |
|---|---|---|
| Bogdanov-Takens / Hopf forms (`bt_hopf`) | 1.125 | 372 of 5440 |
| cusp | 2 | 127 of 954 |
| Bautin (generalised Hopf) | 2.12 | 126 of 922 |
| `MultilinearForms` polarization sums (van der Pol `l1`) | 2.74 | - |

Codimension-two points keep their kind and count, and their locations move by at
most `1.3e-6` relative. The error in the cusp coefficient `a` goes from `6.2e-7`
to `1.3e-6`. `FiniteDifferenceDerivatives` is used only when it is chosen through
the opt-in `derivative_provider` registry key; the default is automatic
differentiation, which these changes do not touch.

**Mixed-scale limit.** One spacing serves every component, so the components
that bind move `epsilon s_j` and the others move less, in proportion to
`|d_j| / s_j`. A component whose ratio is `R` times smaller than the binding
one moves `epsilon s_j / R`, and its share of the coefficient errs by about
`eps R / epsilon` relative (`eps` the machine epsilon): a few per cent once `R`
passes about `1e8` at `epsilon = 1e-6`, and near `R = epsilon / eps` the move
rounds away entirely and that share is lost. At
`x = (1e9, 1e-3)`, `d = (1, 1e6)` the spacing is `1e-12`, `x_0` does not move, and
the identity field's first coefficient comes back `(0, 1e6)`, not `(1, 1e6)`.
Difference such a direction component by component (the Jacobian) instead.

```python
import numpy as np

from discrecontinual_equations.continuation.jacobian import FiniteDifferenceJacobian
from discrecontinual_equations.continuation.residual import ResidualFunction


class Scaled(ResidualFunction):
    """f(u, p) = (u0^2 - p, u1^2 / 1e24 - p): u1 lives near 1e12."""

    def evaluate(self, state, parameter):
        return np.array([state[0] ** 2 - parameter, state[1] ** 2 / 1.0e24 - parameter])


jacobian = FiniteDifferenceJacobian(Scaled(), epsilon=1.0e-7)
print(jacobian.extended_jacobian(np.array([2.0, 2.0e12]), 4.0))
# [[4, 0, -1], [0, 4e-12, -1]], exact; an absolute step gave 0 for the 4e-12
```

**Limits.** A component that lives far below one needs `typical` (one per
component, or one for all) or its step is floored at one; the builder passes
only `epsilon`, so `typical` is reachable by constructing the provider yourself.
Continuation states and parameters of size above one now take larger steps than
before, so the Jacobian Newton is handed changes by the change in the difference
error, and a converged point can move within the Newton tolerance. The residuals
that difference their own columns follow the same rule since DEQ-159 (next section).

References: [Dennis and Schnabel 1983, SciPy `_numdiff` and `_nonlin`](../references.md#lyapunov-exponents-covariant-vectors-and-noise).

### The cusp and generalized-Hopf curves at scale (DEQ-159)

**The problem.** `Codim3Driver` continues the cusp and generalized-Hopf curves
with `AnalyticJacobian`, which asks the curve residual for its own Jacobian. The
state columns are analytic; the three parameter columns are differenced, and they
used to step `finite_difference_epsilon` (`1e-5`) absolutely. On a curve whose
parameters are of size `3e11`, where an ulp is `6e-5`, that step is no move at
all: the three columns came back as zero. At `3e7` they were off by `3.5e-4`
relative.

**The rule.** The columns now take the step every other finite difference in the
package takes, centrally: the unknown `p` a column moves steps
`epsilon * max(|p|, 1)`, and the quotient is divided by the width actually
stepped, `(p + h) - (p - h)`. `epsilon` is therefore relative, and on parameters of
size one or less the step is `epsilon` exactly, as before. The same rule sets the
step of the Jacobian the generalized-Hopf curve uses at each accepted point to
choose its pair of `G` entries. `finite_difference.central_column` is the shared
difference; a `DifferentiableResidual` of your own can use it for any column it
does not know in closed form:

```python
import numpy as np

from discrecontinual_equations.finite_difference import central_column


def rows(state, parameter):
    return np.array([state[0] * parameter, state[1] ** 2 / 1.0e24])


state = np.array([2.0, 3.0e11])
print(central_column(rows, state, 5.0, 2, 1.0e-5))  # d rows / d parameter
# [2. 0.]
print(central_column(rows, state, 5.0, 1, 1.0e-5))  # d rows / d state[1]
# [0.e+00 6.e-13]; an absolute 1e-5 returned [0. 0.]
```

At ordinary scales the four codim-3 curves of the test suite (two cusp curves, two
generalized-Hopf curves) are unchanged: the same number of points, every
coordinate within `5.4e-14`, and the same located swallowtails and degenerate
Bautin points.

**Limits.** The typical size is one for every unknown: a parameter that lives far
below one is stepped as if it were of size one. `Codim2Config.curve_epsilon` is not
read by any curve; the step is `finite_difference_epsilon`.

References: [Dennis and Schnabel 1983, SciPy `_numdiff`](../references.md#lyapunov-exponents-covariant-vectors-and-noise);
the curves themselves: [Govaerts, Kuznetsov and Sijnave 2000, MatCont manual 2019](../references.md#normal-forms-centre-manifolds-and-bifurcation-detection).

::: discrecontinual_equations.continuation.jacobian

::: discrecontinual_equations.continuation.derivative_provider
