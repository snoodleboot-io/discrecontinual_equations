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

The working recipe combines three ingredients: a curvature (second-difference)
monitor equidistributed by the de Boor construction; *gradual* mesh movement (each
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

- **Extreme stiffness (RESOLVED - needs 400 intervals, not 200).** The claim holds,
  but it is resolution-bound and the node count is part of the result. At `N = 200`
  the continuation stalls at `mu ~ 11.7` and cannot reach `mu = 16` however the
  step size is controlled: lowering `_CONTINUATION_MIN_STEP` by 100x moved the stall
  only to `mu ~ 11.82`. The jump layers of the relaxation oscillation narrow as `mu`
  grows, and past `mu ~ 12` a 200-node mesh cannot resolve them however the nodes are
  redistributed. At `N = 400` the continuation reaches `mu = 16` with relative period
  error 1.8e-3, consistent with the 3e-3 recorded below.

  Measured on Python 3.14.4, NumPy 2.4.0, SciPy 1.18.1. The original note recorded no
  environment and no node count, which is why this took a bisection to re-establish;
  record both with any future measurement.

  A related defect was fixed while establishing this: `continue_to` used to return the
  cycle from wherever it stalled, reported as though it had reached the target. It now
  returns `None`, so a stall is visible instead of being read as an accuracy problem.

- **Extreme stiffness (original note).** Reaching very large `mu` from a cold uniform seed
  fails because the damped Newton diverges. `AdaptivePeriodicOrbit.continue_to`
  solves this with adaptive-step-size continuation in the stiffness parameter: the
  step shrinks on a failed solve and grows on success, warm-starting each step from
  the previous adapted mesh and solution. Validated against the integrated oracle,
  it continues van der Pol from `mu = 6` to `mu = 16` with relative period error
  ~3e-3, where a cold adaptive solve at `mu = 16` fails outright. (The earlier
  erratic, non-monotone cross-`mu` convergence is handled by the step-size control.)
- A monitor tuned to the trapezoidal truncation order (density proportional to
  `|x'''|^{1/3}`) rather than raw curvature could sharpen node placement further and
  is the last obvious refinement; not yet needed for the accuracies reached above.

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
