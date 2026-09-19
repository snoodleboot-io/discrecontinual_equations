# Future work / deferred research

This note records work that was investigated but deliberately not shipped, so the
context and the reasoning are not lost. It is developer documentation, not a promise.

## 1. Adaptive periodic-orbit mesh for extreme relaxation oscillations (RESOLVED)

**Status:** resolved. `AdaptivePeriodicOrbit` (in `continuation/periodic_orbit.py`)
adapts the mesh and, on stiff relaxation oscillations, beats the uniform mesh by
more than two orders of magnitude at fixed node count, while leaving smooth cycles
unchanged.

**What made it work (after two earlier failures).** The earlier attempts failed for
two separate reasons, both now understood:

1. *Wrong regime.* Arc-length and curvature monitors were tested on smooth cycles,
   where the uniform time mesh is already near-optimal, so any redistribution hurt.
   Adaptation only helps genuinely stiff cycles (van der Pol at large `mu`), where a
   uniform mesh cannot resolve the fast jumps and its period error is first order and
   large (~10% at `mu = 10`, `N = 200`).
2. *Newton convergence, not the mesh.* On a concentrated mesh the interpolated
   uniform solution is a poor Newton seed, so the re-solve diverged - which looked
   like "adaptation does not help" but was really a solver-robustness failure.

The working recipe combines three ingredients: a curvature monitor (the cube root of
the true second derivative on the non-uniform mesh - see 1b for why the raw second
difference fails) equidistributed by the de Boor construction; *gradual* mesh
movement (each
sweep a partial step toward the equidistributed target, so the interpolated seed
stays valid); and a *damped, analytically-differentiated* Newton iteration
(backtracking line search plus a period floor) that converges from the mediocre
seed. The exact O(N) Jacobian from `AnalyticPeriodicOrbit` is what makes the many
re-solves affordable.

**Validation.** Against a tight-tolerance integrated oracle (SciPy Radau, period by
Poincare section), van der Pol at `mu = 10`, `N = 200`: uniform relative period error
~1.0e-1, adaptive ~6e-4 - a ~160x improvement. On a smooth Hopf cycle the adaptive
solver returns the same near-optimal result as the uniform mesh (no regression).

## 1b. Remaining adaptive-mesh headroom

- **Mesh monitor fixed (RESOLVED).** The two entries below were measured under a
  broken monitor, and both of their surprises came from it. The monitor was the raw
  second difference of neighbouring states - curvature times spacing squared - so it
  shrank wherever nodes were already dense, and its target was sharp enough that the
  re-solve on the moved mesh failed on most sweeps. A failed sweep left the mesh
  untouched while `solve` still reported success: continuing to `mu = 40`, 75% of
  solves failed at `N = 400` and 60% at `N = 800`, so the mesh stayed adapted to a
  smaller `mu` than the one being solved. Splitting every interval of the `N = 800`
  mesh without re-adapting gave 4.3e-5 at `N = 1600`; adapting gave 1.5e-3.

  The monitor is now the cube root of true curvature on the non-uniform mesh. No
  solves fail, results no longer depend on the seed's resolution, and error falls
  4-5x per doubling (Python 3.14.4 / NumPy 2.4.0 / SciPy 1.18.1):

  | `mu` | N = 100 | N = 200 | N = 400 | N = 800 | N = 1600 |
  |---|---|---|---|---|---|
  | 12 | 8.0e-3 | 1.4e-3 | 3.2e-4 | 9.0e-5 | |
  | 16 | 1.4e-2 | 2.5e-3 | 5.0e-4 | | |
  | 40 | stalls | 1.2e-1 | 3.0e-3 | 7.4e-4 | 2.2e-4 |

  So "needs 400 intervals, not 200" below was the monitor failing, not a resolution
  limit - 200 now reaches `mu = 16`. And the non-monotone convergence is gone. What
  remains true is the danger: `N = 200` at `mu = 40` still arrives 12% wrong, which
  is what `estimate_period_error` and `continue_to_resolved` exist to catch.

  Headroom left: a nested refinement of the `N = 800` mesh still beats the adapted
  `N = 1600` mesh (4.3e-5 against 2.2e-4), so the cube-root monitor is better, not
  optimal. Curvature itself and its square root were tried and fail to adapt at all.

- **Extreme stiffness (SUPERSEDED - the 200-node stall was the monitor).** Kept as
  measured at the time, because the diagnosis was wrong and the way it was wrong is
  the point. At `N = 200` the continuation stalled at `mu ~ 11.7` and could not reach
  `mu = 16` however the step size was controlled: lowering `_CONTINUATION_MIN_STEP`
  by 100x moved the stall only to `mu ~ 11.82`. That was read as a resolution limit -
  the jump layers narrow as `mu` grows, so a 200-node mesh was assumed unable to
  resolve them. It was not a resolution limit. The mesh was not adapting at all (see
  above), and with the monitor fixed `N = 200` reaches `mu = 16`. A stall whose cause
  is unproven should not be written down as a property of the discretisation.

  Measured on Python 3.14.4, NumPy 2.4.0, SciPy 1.18.1. The original note recorded no
  environment and no node count, which is why this took a bisection to re-establish;
  record both with any future measurement.

  A related defect was fixed while establishing this: `continue_to` used to return the
  cycle from wherever it stalled, reported as though it had reached the target. It now
  returns `None`, so a stall is visible instead of being read as an accuracy problem.

- **Node count against stiffness (SUPERSEDED - measured under the broken monitor).**
  Continuing to `mu = 40` from `mu = 6`, against the integrated oracle, on
  Python 3.14.4 / NumPy 2.4.0 / SciPy 1.18.1. This is the table DEQ-10 was opened on;
  the corrected one is above:

  | N | reached `mu` | relative period error |
  |---|---|---|
  | 200 | 13.7 (stalls) | 5.8e-2 |
  | 400 | 40.0 | **1.7e-1** |
  | 800 | 40.0 | 3.4e-4 |
  | 1600 | 40.0 | 1.5e-3 |

  Roughly `N ~ 20 to 25 x mu` for ~1e-3 accuracy. Two things in that table matter
  more than the rule of thumb.

  **Reaching the target is not the same as being right.** At `N = 400` the
  continuation arrives at `mu = 40` and reports success with a 17% period error. The
  residual gate proves the *discrete* system was solved; nothing checks that the mesh
  resolves the orbit, and an under-resolved mesh converges confidently to the wrong
  cycle. Treat a returned solution as trustworthy only where the node count is known
  to be adequate for the stiffness. This one survives the monitor fix: the node count
  moved (it is `N = 200` and 12% now) but the failure mode did not.

  **Convergence in N was not monotone - now explained (DEQ-10).** `N = 1600`
  (1.5e-3) was worse than `N = 800` (3.4e-4). The cause was the monitor, not the
  oracle: sweeps failed silently and left the mesh adapted to a smaller `mu` than the
  one being solved, so which mesh a run ended on depended on where its sweeps
  happened to fail rather than on how many nodes it had. With the monitor fixed the
  sequence is monotone - 3.0e-3 / 7.4e-4 / 2.2e-4 at `N` = 400 / 800 / 1600.

- **Extreme stiffness (original note).** Reaching very large `mu` from a cold uniform seed
  fails because the damped Newton diverges. `AdaptivePeriodicOrbit.continue_to`
  solves this with adaptive-step-size continuation in the stiffness parameter: the
  step shrinks on a failed solve and grows on success, warm-starting each step from
  the previous adapted mesh and solution. Validated against the integrated oracle,
  it continues van der Pol from `mu = 6` to `mu = 16` with relative period error
  ~3e-3, where a cold adaptive solve at `mu = 16` fails outright. (The earlier
  erratic, non-monotone cross-`mu` convergence is handled by the step-size control.)
- A monitor tuned to the trapezoidal truncation order (density proportional to
  `|x'''|^{1/3}`) is still the obvious next refinement. The cube-root exponent is now
  in place but applied to `|x''|`, not `|x'''|`; the headroom noted above - nested
  refinement still beating the adapted mesh at `N = 1600` - is where it would show up.

## 2. Smaller open frontier items

- **Two-parameter Shilnikov loop continuation (RESOLVED, with caveats).**
  `HomoclinicShooting.trace` continues a saddle-focus homoclinic in a second
  parameter. On the two-parameter jerk system `x'=y, y'=z, z'=-a z - b y + x - x^2`
  it traces a smooth, monotone homoclinic locus `a(b)` over `b` in [0.70, 0.90] along
  which the saddle index crosses one - a **Belyakov point** separating Shilnikov
  chaos (`delta < 1`, positive top Lyapunov exponent nearby) from a tame homoclinic
  (`delta > 1`). There is no closed-form oracle, so this is validated by convergent
  structural evidence (smooth monotone curve, saddle-focus along it, the `delta = 1`
  crossing). *Caveat:* near the primary homoclinic the saddle-focus return is
  oscillatory (the "wild" Shilnikov tangle of infinitely many nearby homoclinics),
  so the bracketing locator is locally erratic there (e.g. near `b ~ 1`); the trace
  is reliable only on the smoother segment away from the tangle. Robustly resolving
  the tangle would need a locator that tracks a specific homoclinic branch rather
  than bisecting the return gap.
- **Heteroclinic networks** among three or more saddles: still open. The
  connecting-orbit BVP and shooting machinery are in place, but a clean oracle for a
  three-saddle connection cycle is the missing piece.

## 3. Further refinements (optional)
- **Stochastic nD Lyapunov** is implemented and validated on a bounded chaotic
  attractor (noisy Lorenz: spectrum sum matches the exact phase-space divergence,
  top exponent stays positive). Extending to genuinely unbounded references or
  multi-channel (vector) noise remains open.
- **Higher-order collocation** is implemented (`HermiteSimpsonOrbit`, fourth order,
  verified by convergence rate). Gauss/Lobatto schemes would raise the order further
  and are the remaining option here.
- **Heteroclinic networks** among three or more saddles (see section 2) are the main
  substantial item still fully open, blocked on a clean oracle for a three-saddle
  connection cycle rather than on machinery.
