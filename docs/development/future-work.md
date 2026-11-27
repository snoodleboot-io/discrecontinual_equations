# Future work / deferred research

This note records work that was investigated but deliberately not shipped, so the
context and the reasoning are not lost. It is developer documentation, not a promise.

## 1. Relaxation oscillations as fast-slow orbits: the adaptive mesh, and the canards it cannot reach

**What a relaxation oscillation is, in the terms that explain the mesh.** Van der Pol
at large `mu` is a fast-slow system. In the Lienard plane, `x' = mu (y - x^3/3 + x)`,
`y' = -x / mu`, so with `eps = 1 / mu^2` the variable `x` is fast and `y` slow. The
critical manifold `y = x^3/3 - x` has two attracting outer branches (`|x| > 1`) and a
repelling middle one, separated by folds at `x = +-1`. Fenichel's theorem puts an
attracting slow manifold `S_a` within `O(eps)` of each outer branch; the cycle crawls
along `S_a`, loses it at a fold, and jumps along a fast fibre to the other branch.
Desroches et al. (SIAM Review 54, 2012, section 2.1.1) give the cycle as lying
`O(eps^(2/3))` from that singular candidate, the `eps^(2/3)` being the fold's own
scaling. Every difficulty below is one of these three places:

- **the fast fibres** are the "jump layers": they take `O(1/mu)` of a period of
  `O(mu)`, which is why a uniform time mesh's period error was first order and large
  (~10% at `mu = 10`), and why the curvature monitor - large exactly on the fibres -
  is the one that works;
- **the folds** are where normal hyperbolicity fails and the orbit turns from slow
  to fast over the `eps^(2/3)` window; the adapted mesh must resolve that turn as
  well as the jump, which is the headroom in 1b;
- **the slow branches** are where nothing happens and a uniform mesh wastes nodes.

The extreme case is the **canard explosion** (section 1d): just below the Hopf the
cycle *follows the repelling slow manifold*, and there no mesh refinement and no
forward integration helps, because the orbit is exponentially unstable in the
direction the integrator would have to go. That is now computed by boundary-value
problem rather than by mesh.

### The adaptive mesh (RESOLVED)

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

## 1c. Connecting-orbit resolution

- **Both resolution parameters are now measured (RESOLVED).**
  `ConnectingOrbit.estimate_orbit_error` reports discretisation and truncation
  separately. They fail independently and each alone certifies orbits that are badly
  wrong, measured against the exact homoclinic `x = 1.5 sech^2(t/2)`: at `h = 0.25`
  the truncation component reads 5.4e-11 against a true error of 7.7e-3, and at
  `T = 5, N = 640` the discretisation component reads 2.3e-5 against a true 2.8e-4.
  So `estimate` is the larger of the two and `limited_by` names which.

  Both components obey clean laws, which is what makes the estimate usable rather
  than a screen: discretisation is `O(h^2)` (exactly 4x per halving) and truncation
  is `O(exp(-2 lambda T))` (measured 7.7x per unit `T` at `lambda = 1`, against
  `e^2 = 7.39`). The estimate runs 0.75x the returned orbit's error at every
  resolution measured from 40 to 640 intervals, on homoclinic and heteroclinic
  alike - that factor is `E - E/4` for a second-order method, not a tuning constant.

  Headroom left:

  - **No iterative search.** There is no `solve_to_resolved` analogue of
    `continue_to_resolved`. The convergence laws are known, so a search could jump
    straight to the `h` and `T` that meet a tolerance rather than doubling blindly
    (`h -> h/2` per factor of four, `T -> T + ln(4) / (2 lambda)` for the same). It
    needs the leading eigenvalue, which the subclasses hold but the base class does
    not - that is the only reason it was not done here.
  - **Even meshes only.** The phase condition pins the centre node, so an odd mesh
    has no matching centre when doubled and the comparison would measure an `O(h)`
    translation instead of the error. Rejected with a message rather than worked
    around; pinning by time rather than by node index would remove the restriction.
  - **Truncation extension is a fixed fraction** (25% of the half-length). A
    problem whose leading eigenvalue is small would get a proportionally weaker
    reduction; scaling the extension by `1 / lambda` is the principled version and
    again needs the eigenvalue in the base class.

## 1d. Slow manifolds and canards by boundary-value problem (RESOLVED, DEQ-39)

**The gap.** The adaptive mesh resolves a relaxation oscillation because the cycle
stays on *attracting* slow manifolds between its jumps. The canard explosion is the
case where it does not: van der Pol with constant forcing,
`x' = y - x^3/3 + x`, `y' = eps (a - x)` (Desroches et al. 2012, eq. 2.7), has a
singular Hopf at `a = 1`, and just below it the cycle grows from Hopf size to
relaxation size inside a window of `a` about `1e-9` wide at `eps = 0.05`, by
following the *repelling* slow manifold `S_r`. Forward integration cannot follow
`S_r` at all, and the cycle's monodromy is a product of factors from `e^40` to
`e^-80`, so its Floquet multipliers off the collocation are meaningless there
(errors above one).

**What was built.** `continuation/slow_manifold.py`:

- `FenichelExpansion` - the slow manifold `x = h(y, eps)` as Fenichel's series,
  solved one order of `eps` at a time from the invariance equation (the homological
  operator is multiplication by the layer eigenvalue `f_x(h_0(y), y, 0)`, inverted
  as a series in `y`). A constant divisor, the first version, is *not* enough:
  `eps h_y` preserves total degree, so the `eps^3` coefficient came out wrong until
  the divisor was the series; the invariance residual now vanishes to 1e-17 through
  the order.
- `SlowManifold` - each manifold as a family of orbit segments of Desroches et al.'s
  problem (8.1), `u' = T F(u)`, `u(0)` on the Fenichel point, `u(1)` on a section,
  `T < 0` for `S_r`; the family is swept from `T = 0` to the target section and then
  continued in the parameter.
- `CanardSearch` - the maximal canard as the zero of the gap between `S_a` and `S_r`
  on the section `x = 1` through the fold (section 8.3).
- `MaximalCanard` - the concatenated problem (8.6)-(8.7), `u(0) = L_a`,
  `u(1) = L_r`, `T` and `a` free, continued in `eps`.

**Measured** (Python 3.14, NumPy 2.4, SciPy 1.18, `solve_bvp` at tol 1e-10):

- `a_c(0.05) = 0.9934909325`, the gap search and the concatenated problem agreeing
  to 1e-14; Desroches et al. report 0.993491 for the explosion at this `eps`.
- Against `a_c = 1 - eps/8 - 3 eps^2/32 - 173 eps^3/1024 - 7593 eps^4/16384 -
  436777 eps^5/262144 - ...` the error of the series truncated after `eps^k` falls
  with log-log slope `k + 1` for `k = 1 .. 4` over `eps` in [0.01, 0.04] (the
  table), and what is left after `eps^5` is the next term, `-15340159 eps^6 /
  2097152`, to 5% at `eps = 0.01` - a residual of 7.7e-12. The coefficients come
  from a symbolic expansion of the canard as a graph `y(x)` regular at the fold;
  the first is Krupa and Szmolyan's (SIAM J. Math. Anal. 2001, Thm 3.1, eq. 3.8)
  with `A = -1`. Their JDE paper (the canard explosion, 2001) could not be fetched
  (publisher captcha); through `eps^3` the series is the one Brons (arXiv
  1209.1109, eq. 9) quotes from Zvonkin and Shubin (Russian Math. Surveys 39,
  1984) for exactly this system. His own iterative scheme gets `-75/1024` at
  `eps^3`, which he notes agrees with (9) only to `eps^2`; the BVP sides with
  `-173/1024` (slope 4.09 for the `eps^3` truncation).

  | truncated after | `eps = 0.04` | `0.02` | `0.01` | slope (7 points) |
  |---|---|---|---|---|
  | `eps^1` | 1.6e-4 | 3.9e-5 | 9.6e-6 | 2.04 |
  | `eps^2` | 1.2e-5 | 1.4e-6 | 1.7e-7 | 3.07 |
  | `eps^3` | 1.4e-6 | 8.0e-8 | 4.8e-9 | 4.09 |
  | `eps^4` | 2.1e-7 | 5.9e-9 | 1.7e-10 | 5.11 |

  The located values move by 9e-13 between `solve_bvp` tolerances 1e-9 and 1e-10.
- The explosion itself, `CycleContinuation` on 160 Hermite-Simpson intervals,
  against a shooting continuation parametrised by the cycle's leftmost point:
  periods agree to 3e-5 relative, amplitudes to 3e-4, and `a` to 5e-10 through the
  whole explosion (1.7e-8 in the relaxation tail, where `a` steepens in `m` and the
  readout of `m` limits it); every cycle that follows `S_r` past `x = -0.3` sits within 3e-10
  of the slow-manifold canard.

**The film** (`examples/continuation/stage_canard.py`) scrubs `s` with
`a = a_c + 1e-10 sinh(s)`, because the explosion is invisible on a linear `a`
axis; at `eps = 0.05` the series through `eps^4` is 6.8e-7 from `a_c`, a thousand
times the explosion's width, so the axis is centred on the computed canard. Its
cycles carry Liouville's multiplier `exp(int (1 - x^2) dt)` instead of the
monodromy's, which is exact for a planar flow and shows every cycle stable.

Headroom left:

- **Planar only.** The expansion and the problems are for one fast and one slow
  variable. Folded nodes and mixed-mode oscillations need two slow variables, where
  `L` becomes a curve and the manifolds surfaces (problem 8.1 as written);
  Guckenheimer and Kuehn's saddle-type slow manifolds need two fast ones.
- **The library's first Lyapunov coefficient has the wrong sign here.**
  `normal_form.first_lyapunov` normalises the adjoint eigenvector by `<p, q>`
  rather than its conjugate, which leaves `<p, q> = c / conj(c)`; for this
  system's non-normal Jacobian `c` is imaginary and the sign flips (+2.13 against
  the hand-computed -2.13 at `eps = 0.05`). It is recorded as an expected failure
  in `tests/continuation/test_slow_manifold.py` and needs its own fix.

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
- **Heteroclinic networks (RESOLVED, starting from robust equivariant cycles).**
  `HeteroclinicNetwork` (`continuation/heteroclinic_network.py`) holds saddles,
  connections in flow-invariant subspaces, and the elementary cycles they form; each
  connection is solved by the projection BVP restricted to its subspace (where it
  is isolated even when the source's unstable manifold is 2-D, as at Kirk-Silber's
  `xi2`), checked by the full-space solve, and refused with `NotATrajectoryError`
  when its collocation residual under the full field exceeds, on any interval,
  twice that interval's trapezoidal truncation error `h^3 |x'''| / 12` (DEQ-81),
  and each cycle's
  eigenvalues are classified per node into radial, contracting, expanding and
  transverse (Krupa and Melbourne 1995, section 2.3). The oracles, all in
  `tests/continuation/test_heteroclinic_network.py`:

  - *Guckenheimer-Holmes cycle* (Krupa 1997, eq. 2.1, `lam = 1`, `a = (-1, -1.5,
    -0.6)`): seeded 1e-2 off its plane, each connection returns to it to 7e-14;
    saddles and spectra match eq. 2.2 to 1e-12; `prod c_j > prod e_j` agrees with
    `2 a1 > a2 + a3` (Theorem 1) on a nine-point grid; at the resonance
    `a2 + a3 = 2 a1` the connection is the exact arc `u = R^2 / (1 + e^{kt})`, and
    the BVP converges to it 4.00x per mesh doubling (9.07e-5, 2.27e-5, 5.67e-6).
  - *Residence times* (Gaunersdorfer 1992, Lemma 1): integrated with scipy DOP853
    in log coordinates, successive differences of the passage times grow by
    1.2500001 against `c/e = 1.25`; with the cyclic symmetry broken
    (`lam_2 = 0.85`) the three per-step ratios match `c_j / e_{j+1}` =
    (1.0204, 0.6875, 2.6) to 1e-8, and the May-Leonard system with
    `alpha = a2/a1`, `beta = a3/a1` grows by the same ratio.
  - *Symmetry-broken network*: lowering `lam_2` keeps every plane invariant but
    breaks `Z3`. `NetworkContinuation` follows the network, re-solving every
    connection from the last, with `rho` matching the closed form to 1e-9; `rho`
    falls through one between `lam_2 = 0.72` and `0.70`, and the continuation then
    locates the first break - `xi1 -> xi2`, its target ceasing to attract in the
    plane - at `lam_2 = 2/3` to 1.6e-11, ahead of the source-side break at 0.6.
  - *Kirk-Silber network* (two cycles sharing `xi1 -> xi2` in `R^4`): the roles at
    a shared saddle differ per cycle as they must, and the winner predicted by the
    signs of Castro and Lohse's `delta` (2014, Proposition 5.3) is the cycle
    simulation follows in both cases tested - including one where it has the
    smaller stability ratio.

  Headroom: connections are solved where the symmetry already guarantees them, so
  there is no search for connections not lying in a known subspace; and cycles
  whose nodes are periodic orbits or chaotic sets are not covered.
- **Stability indices and cycle competition (RESOLVED for quasi-simple cycles).**
  `network_stability.py` builds each node's basic transition matrix from the
  role-sorted eigenvalues (Garrido-da-Silva and Castro 2019, eq. 7), with the
  permutation found by matching directions normal to each connection, and returns
  the local stability index (Podvigina and Ashwin 2011) along every connection by
  their Theorems 3.4 and 3.10. `CycleStability.delta` is Kirk and Silber's
  `delta`; `NetworkCompetition` names the cycles that attract a set of positive
  measure. Oracles in `tests/continuation/test_network_stability.py` and
  `..._simulation.py`: both Kirk-Silber cycles against Podvigina and Ashwin's B3-
  closed forms on random eigenvalues (every case of their section 4.2.1), Castro
  and Lohse's Propositions 5.2/5.3 and eq. (3); the RSP cycle's published indices;
  Guckenheimer-Holmes `+inf` iff `2 a1 > a2 + a3`; `f^index` against the exact
  measure of its set; winners confirmed by simulation within 0.07 of `delta = 0`;
  and the basin complement of an index-1 connection scaling like `eps^1`.
  `..._independent.py` adds a five-node cycle whose permutation is a 3-cycle, and
  the type C4- closed forms as corrected by Lohse (2015, Lemma B.5): Podvigina and
  Ashwin's original `b1 < 0` branch drops the row terms of Theorem 3.10.

  Open: the matching compares directions through orthogonal complements, so a type
  C cycle in oblique coordinates (or a node with a repeated transverse eigenvalue)
  is refused rather than computed; a coordinate-free matching through the
  quotient by each `P_j` would lift that. Indices with respect to the whole
  network (Castro and Lohse's n-indices) are not computed.

## 3. Further refinements (optional)
- **Stochastic nD Lyapunov** is implemented and validated on a bounded chaotic
  attractor (noisy Lorenz: spectrum sum matches the exact phase-space divergence,
  top exponent stays positive). Extending to genuinely unbounded references or
  multi-channel (vector) noise remains open.
- **Higher-order collocation** is implemented (`HermiteSimpsonOrbit`, fourth order,
  verified by convergence rate) and is now the default scheme of
  `CycleContinuation` as well, with its own analytic sparse Jacobian. The
  monodromy walks the cubic Hermite interpolant between nodes with the Jacobian
  at every Runge-Kutta stage, fourth order itself (16x per doubling on the exact
  Hopf cycle); the smooth films' Floquet errors are now 1e-5 to 1e-3 percent.
  Orthogonal collocation at `m` Gauss-Legendre points (order `2m`, measured to
  7.99 for `m = 4`) is now a third scheme, with multipliers off its own Jacobian;
  on 80 intervals it takes Bogdanov-Takens from 0.089% to 4.8e-5% and the ring
  from 0.088% to 4.6e-3%. Mesh equidistribution was measured on top and left out.
  Open: whether the films should switch, and an arclength norm that means the
  same step under every scheme (an `m`-point scheme's step covers about
  `1/sqrt(m)` of the parameter).
- **Heteroclinic networks** among three or more saddles (see section 2) are
  resolved for robust cycles between equilibria, with stability indices for
  quasi-simple cycles inside them.
