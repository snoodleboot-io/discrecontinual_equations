# Complexity

This page records the time and space cost of every substantive algorithm in the
library, derived from the code as written rather than from what the textbook
algorithm would cost, and checked against measured scaling. Where the code is
asymptotically worse than its algorithm needs to be, that is called out under
[Findings](#findings) with the evidence. Most of those findings have since been
fixed: the table names the ticket that fixed each, and
[Before / after](#before-after) gives every ticket's measurements.

## How to read this page

**Size parameters.** `n` state dimension, `m` noise drivers (columns of a matrix
diffusion), `K` time steps, `P` ensemble paths, `N` mesh intervals, `M` grid points,
`I` Newton iterations, `k` Taylor order or tracked directions as stated, and `C_f`,
`C_g` the cost of one drift or diffusion evaluation. Every field in this library is a
Python method taking a Python list, so `C_f` is at least a few microseconds and is
`O(n)` for any field that touches every component. Many costs below are dominated by
the *number* of field calls rather than by the linear algebra, which is why the
measured slopes often sit below the asymptotic exponent.

**Measurements.** Each slope is a log-log least-squares fit of the best of 2-5
`time.perf_counter` runs over at least four sizes of one parameter, with the others
fixed; memory is the `tracemalloc` peak, which sees Python and NumPy allocations but
not SuperLU's or LAPACK's workspace. The machine was running other test suites, so
read the slopes, not the seconds. Where fixed per-call overhead pulls the full-range
fit down, the slope over the last pair of sizes ("local") is given too. Dense solves
were timed with `OPENBLAS_NUM_THREADS=1`: on a loaded machine threaded OpenBLAS made
a 128 x 128 solve cost 0.4-2 s instead of a millisecond.

The scaling scripts are not in the repository, which has no benchmark directory;
the numbers quoted here are their output. The survey's scripts and outputs are kept
outside the repository in `outputs/complexity/baseline/`, and each ticket's
before/after run in `outputs/complexity/before-after/<TICKET>/` (both under the main
checkout, git-ignored).

**Which code.** The survey was measured on `main` at `4ca6382` and on three branches
that have since merged (geodesic level sets, the quasipotential and Gauss-Legendre
collocation; the sections say which commit). Rows that a later ticket changed have
been rewritten for `main` at `8690798`: they give the cost the code has now and
quote that ticket's before/after slopes. Rows no ticket touched keep the survey's
numbers.

## ODE solvers

| Algorithm | Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|---|
| `EulerSolver`, `AdamsBashforth2Solver`, `AdamsBashforth3Solver` | `K`, `n` | `O(K (n + C_f))` | `O(K n)` (Python lists in a `Curve`) | per-step Python: list comprehensions and three `list.append` per component | Euler: time 1.02, memory 1.00 vs `K` (2000-32000) |
| `DormandPrinceSolver`, `CashKarpSolver`, `FehlbergSolver`, `DormandPrince89Solver` | accepted steps `K`, rejected `R`, stages `s` (7, 6, 6, 13) | `O((K + R)(s C_f + s^2 n))` | `O(K n)` | the `s` field calls per attempt | Dormand-Prince: time 0.98 (local), memory 0.98 vs horizon |

The fixed-step solvers iterate `for time in self.solver_config.times`, which builds
the time array once, so they are linear in `K`. The adaptive solvers form each stage
as `np.dot(a[i, :i+1], k[:i+1])`, `O(s^2 n)` per step, negligible beside the `s`
field calls. `AdamsBashforth3Solver` keeps every past derivative in a list although
it reads only the last three: `O(K n)` extra memory, the same order as the output.

Every solve builds a `Curve` of its own (DEQ-45), so solving the same equation again
costs the same and does not grow the first trajectory. Before that fix the solution
shared the equation's `Variable` objects and a second solve appended to the first
(`len(solution)` 1001, 2002, 3003 over three solves of a 1000-step problem). The fix
left the cost unchanged: Euler CPU-time slope 0.98 before and 0.97 after for `K`
2000-32000, 6.6 MB at `K = 32000` both ways.

## SDE solvers

### Per-path schemes

| Algorithm | Size parameters | Time per step | Space | Dominated by | Measured slope |
|---|---|---|---|---|---|
| `EulerMaruyamaSolver` | `K`, `n`, `m` | `O(C_f + C_g + n m)` | `O(K n)` | Python overhead | vs `K` (16000-128000, DEQ-47): CPU 1.04, memory 1.01 (1.76 before the fix) |
| `MilsteinSolver`, vector (diagonal) noise | `n`, `K` | undeclared: `2n C_g` (central differences of `b`) + `O(n^2)` for `P[j, k] = b_j d b_k / d x_j` and its check; declared `diagonal_noise`: `2 C_g + O(n)` | `O(n^2)` undeclared, `O(n)` declared | the `2n` diffusion calls | vs `n` (`K = 200`, DEQ-48): CPU 1.17 undeclared, 0.41 declared (2.12 before); memory 1.02 / 0.97 (1.98 before) |
| `MilsteinSolver`, matrix noise | `n`, `m` | `2n C_g + O(m^2 n^2)` | `O(m n^2)` | the products and commutativity check | not re-measured |
| `SRK2Solver` (strong 1.0) | `n`, `K` | `C_f + (n + 1) C_g`: one supporting diffusion evaluation per channel | `O(K n)` | the `n` supporting evaluations, each returning all `n` components | vs `n`: 1.39 overall, 2.22 local (`C_g = O(n)`) |
| `SRK3Solver`, `SRK4Solver` (scalar only) | `K` | a fixed number of drift and diffusion calls | `O(K)` | Python overhead | linear in `K` (same driver as SRK2) |
| `GottwaldMelbourneSolver` | `K`, `E = dt / epsilon` | `O(E (C_f + C_g + n))` | `O(K n)` | the inner fast-map loop | not measured; `E` multiplies everything |
| `ZhanDuanLiLiSolver` | `K`, `n` | `2 (C_f + C_g)` | `O(K n)` | Python overhead | not measured |

**The time grid.** `FixedStepConfig.times` is a property that rebuilds
`np.arange(start, end + h, h)` on every access (1.5 us at `K = 2000`, 55 us at
`K = 128000`). The stochastic solvers used to index it twice per step, which made a
path `O(K^2)`; since DEQ-47 every solver reads it once per solve, and every path is
linear in `K`.

**Milstein with vector noise.** Before DEQ-48 `column_jacobians` stored `(m, n, n)`
Jacobians and contracted them with `np.diag(b)` by `einsum`, an `O(n^4)` loop of
which only `n^2` entries could be non-zero. The step now forms the `n x n` matrix
`P[j, k] = b_j d b_k / d x_j` directly from the diffusion Jacobian and runs the same
commutativity check on it; an equation that declares `diagonal_noise` (checked once
per solve) gets `P`'s diagonal from two evaluations.

**SRK2 is quadratic in `n` by construction.** Platen's scheme needs one supporting
value per channel, and the `diffusion` interface returns every component, so the
`n` supporting evaluations cost `O(n C_g)`. That is the price of the interface, not of
the code.

### Wiener increments

| Algorithm | Size parameters | Time | Space | Measured slope |
|---|---|---|---|---|
| `GaussianWienerSource.increments` | `m` | `O(m)`, two normal draws per channel | `O(m)` | - |
| `BrownianPath.sample` | cells `C = (end - start) / resolution`, `m` | `O(C m)` (one `cumsum`) | `O(C m)` | - |
| `BrownianPath.increments` | cells per step `c` | `O(c m)` (slice and `trapezoid`) | `O(c m)` temporary | 0.75 local vs `c` (10-10000; fixed overhead below 1000 cells) |
| `BrownianPath` cell Levy areas (`_cell_areas`, drawn on the first area reading) | cells `C`, `m` | `O(C^(3/2))` at fixed `m` (about `C^(1/2)` series terms per cell) | `O(C m^2)` kept + one fixed-size chunk (DEQ-70) | the series terms per cell | time 1.41 vs `C` before and after; peak memory 1.41 before, 0.55 after (DEQ-70) |

A convergence study that reads a path at step `h` therefore costs `O(C m)` in
increments per path whatever `h`, which is the point of a fixed path.

### Ensembles (`StochasticEnsemble`)

| Algorithm | Size parameters | Time per step | Space | Dominated by | Measured slope |
|---|---|---|---|---|---|
| Euler-Maruyama | `P`, `n`, `m`, `K` | `O(P n m)` vectorised + the batch drift and diffusion | `O(P n)` + `O(S P n)` for `S` kept samples | NumPy calls (about 60 us per step at `P = 1000`) until `P` is about `10^4` | vs `P`: 0.85 local (100 to 100000); vs `K`: 0.98; memory 0.89 vs `P` |
| Milstein (and Stratonovich drift), vector noise | `P`, `n` | undeclared: `n` batched direction slopes, `O(P n^2)`; declared `diagonal_noise`: 2 batched calls, `O(P n)` | `O(P n)` | the direction slopes | vs `n` (`P = 200`, `K = 50`, DEQ-48): CPU 1.45 undeclared, 0.46 declared (2.70 before); memory 0.99 (2.94 before, 212 MB at `n = 32`) |
| Milstein, matrix noise | `P`, `n`, `m` | `2n` batched diffusion calls + `O(P m^2 n^2)` products | `O(P m n^2)` | the `(P, m, n, n)` Jacobian array | not re-measured |
| with a `WienerSource` | as above | `+ P` source calls per step (a Python list comprehension) | | the source | - |

`times` is read once into a local, so the ensemble is linear in `K`. The default
`eval_batch` and `diffusion_batch` loop over the paths in Python, so an equation that
does not override them pays `O(P (C_f + C_g))` Python calls per step and loses the
point of the ensemble. Before DEQ-48 the Milstein ensemble with vector noise built
the full `(P, n, n, n)` Jacobian and product tensors, and the Stratonovich
Euler-Maruyama drift built the same Jacobian tensor to read its `n` diagonal
entries; both now take the slopes direction by direction.

## Equilibrium continuation

`Continuer.solve` runs pseudo-arclength continuation with the collaborators the
builder injects. With the defaults (central-difference Jacobian wrapped in a
`MemoizedJacobian`, dense `LeastSquaresFallbackSolver`, detectors fold, branch point
and Hopf):

| Part | Size parameters | Time | Space | Notes |
|---|---|---|---|---|
| `NewtonCorrector.correct`, per iteration | `n` | `(2n + 2) C_f` (extended Jacobian; `(2g + 2) C_f` for `g` column groups when a sparsity pattern is given, DEQ-53) + dense `(n+1)^2` solve `O(n^3)`; `lstsq` `O(n^3)` on singular steps | `O(n^2)` | the Jacobian array is dense either way |
| `BorderedTangentComputer.compute` | `n` | `(2n + 2) C_f + O(n^3)` | `O(n^2)` | |
| `Fold.test_value` | | `O(1)` (the tangent's last entry) | | |
| `BranchPoint.test_value` | `n` | `(2n + 2) C_f + O(n^3)` (determinant of the bordered matrix) | `O(n^2)` | |
| `Hopf.test_value` | `n` | the shared Jacobian + one `eigvals` `O(n^3)` + `n(n-1)/2` pair sums (DEQ-46) | `O(n^2)` | see `bialternate_test_value` below |
| `PointBuilder.build` | `n` | `2n C_f + O(n^3)` (`eigvals`) | `O(n^2)` | |
| `RefiningLocalizer.localize` | root-finder iterations `L` | `L` corrector calls + `L` tangents + `L` test values | `O(n^2)` | only when a test function changes sign |
| **One accepted point** | `n`, `I` | `(I + 1)(2n + 2) C_f + (I + 3) O(n^3)` | `O(n^2)` + `O(n)` stored | the tangent, detectors and stability share the converged point's Jacobian (DEQ-53; `I + 4` before) |

Survey (`4ca6382`), on a ring of `n` fold-carrying cells (`C_f = O(n)`), 20 points:
with only the fold detector the run grew with slope 0.98 (1.28 local) for `n` 4 to
32; with the default three detectors the slope was 1.84 overall and 3.68 local,
almost all of it the Hopf test, whose bialternate determinant also overflowed.

The Hopf test used to build the `s x s` bialternate product, `s = n(n-1)/2`, entry by
entry (`O(n^4)` Python calls and memory) and take its dense `O(n^6)` determinant,
which overflowed at `n = 32` on a random matrix. Since DEQ-46
`bialternate_test_value` reads the same sign from one eigenvalue solve, `O(n^3)`.
Per call on a random Jacobian the time slope went from 4.14 (`n` 8-64) to 1.56
(`n` 8-128), and a 20-point continuation with all three detectors from 3.23 to 1.26;
see [DEQ-46](#deq-46-hopf-test-from-one-eigenvalue-solve).

Since DEQ-53 the builder wraps the Jacobian provider in a `MemoizedJacobian`, so one
accepted point builds `I + 1` Jacobians instead of `I + 4`, every value bit-identical.
With only the fold detector that made a 20-point run 2.4x faster at `n = 64` (slope
1.10 to 0.85); with all three detectors the run is dominated by the remaining work
and did not change (7.8 s to 7.7 s at `n = 32`, wall clock under load 20-50).

## Limit cycles

### Periodic orbits (`periodic_orbit.py`)

| Algorithm | Size parameters | Time per Newton iteration | Space | Dominated by | Measured slope |
|---|---|---|---|---|---|
| `PeriodicOrbit.solve` (trapezoidal, column-grouped forward differences, SuperLU) | `N`, `n` | a few residuals per state component plus one for the period, independent of `N`, of `(N + 1) C_f` each: `O(N n C_f)`; sparse LU on `O(N n^2)` non-zeros; dense `lstsq` only where the LU is singular | `O(N n^2)` | the residuals | vs `N` (40-640, `n = 2`, DEQ-49): time 1.05, memory 0.93 (2.35 and 1.81 before); 0.38 s against 62 s at `N = 640` |
| `HermiteSimpsonOrbit.solve`, `RobustPeriodicOrbit.solve` | `N`, `n` | as above with `2N + 1` field calls per residual; Robust adds up to 20 residuals per line search | `O(N n^2)` | as above | DEQ-49: Hermite-Simpson 1.99 to 0.96, Robust 2.18 to 0.73 |
| `AnalyticPeriodicOrbit.solve` (autodiff blocks, COO assembly, `spsolve`) | `N`, `n` | `N + 1` autodiff Jacobians `O(N n C_f)` + vectorised assembly of `O(N n^2)` triplets + SuperLU | `O(N n^2)` | the Python autodiff passes | vs `N`: time 1.14 (100-3200), memory 0.97; vs `n` (`N = 100`): time 1.67, memory 1.82 |
| `AdaptivePeriodicOrbit.solve` | `N`, sweeps `W <= 15`, line-search steps `B <= 20` | `(1 + W)` damped solves, each `I` iterations of the above plus up to `B` residuals | `O(N n^2)` | the re-solves | linear in `N` per solve |
| `AdaptivePeriodicOrbit.continue_to_resolved` | doublings `D` | `D + 1` full continuations at `N, 2N, ..., 2^D N`, so about twice the finest one | finest level | the finest continuation | - |
| `PeriodicOrbit.monodromy` | `N`, `n` | `9N` forward-difference Jacobians (`9N (n + 1) C_f`, batched over the stages where the field takes arrays) and `4N` RK4 propagators, then their ordered product: `O(N (n C_f + n^3))` | `O(N n^2)` (the stage Jacobians, batched) | the Jacobians at small `n`, the products at large `n` | survey (before batching): vs `N` 1.01; vs `n` (`N = 40`) 1.29 overall, 2.04 local |
| `PeriodicOrbit.floquet_multipliers` | | monodromy + `eigvals` `O(n^3)` | `O(n^2)` | | |

**What SuperLU actually factors.** The analytic Jacobian is bordered
almost-block-diagonal: `N` interval blocks `[-I - w Df(x_i), I - w Df(x_{i+1})]`,
the periodicity rows coupling the first and last node, one dense column for the
period and the phase row. `spsolve` hands it to SuperLU with the default COLAMD
column ordering. The fill stays proportional to the matrix: `nnz(L + U) / nnz(A)` was
1.70 at `n = 2` for `N` = 100, 400 and 1600, and 0.34 at `n = 8` (the autodiff
blocks store structural zeros that SuperLU drops). So the factorisation is
`O(N n^3)` time and `O(N n^2)` memory, linear in `N`, and the factor took 1.2 ms at
`N = 1600`, `n = 2`. The periodic wrap-around and the dense column do not cause
cyclic fill under this ordering.

### Cycle continuation (`cycle_continuation.py`)

`CycleContinuation.trace` continues `(states, period, parameter)` by pseudo-arclength
on the trapezoidal or Hermite-Simpson collocation system and reads the Floquet
multipliers of every point.

| Part | Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|---|
| `_solve_seed` | `N`, `n` | one staged `PeriodicOrbit.solve` (now sparse, `O(N n C_f)` per iteration), then a sparse correction; a direct damped Newton on the target scheme when the staged answer is not a cycle | `O(N n^2)` | the staged solve | vs `N` (40-640, DEQ-49): 0.66, 0.23 s at `N = 640` (1.97 and 26 s before) |
| corrector iteration (`_correct`) | `N`, `n` | `(N + 1)` autodiff Jacobians + `(N + 1)` parameter differences (+ `N` midpoint ones for Hermite-Simpson) + one `spsolve` on the bordered matrix | `O(N n^2)` | the per-node Python calls | (in the per-step figure) |
| tangent (`_tangent`) | `N`, `n` | one more assembly and `spsolve` | `O(N n^2)` | | |
| point (`_point`): multipliers | `N`, `n` | one `PeriodicOrbit.monodromy`, `O(N (n C_f + n^3))` + `eigvals` | `O(N n^2)` | the monodromy's `9N` forward-difference Jacobians | |
| **One continuation step** | `N`, `n`, `I` | `(I + 1)` sparse solves + monodromy: `O(I N n C_f + N n^3)` | `O(N n^2)` | | vs `N` (`n = 2`): trapezoidal 1.05, Hermite-Simpson 1.00; vs `n` (`N = 40`): 1.71 |

Measured parts of one Hermite-Simpson step at `n = 2`, `N` 40 to 640: assembly
0.009 s to 0.13 s (linear), `spsolve` 0.7 ms to 1.7 ms with `nnz(L + U) / nnz(A)`
constant at 1.35, and the multipliers 0.021 s to 0.34 s (linear). The monodromy
costs about 2.5 times one assembly, so with the typical two or three corrector
iterations it is roughly half of a step.

The bordered matrix adds a dense parameter column and a dense border row (the
previous tangent) to the periodic-orbit pattern. The border row is placed last and
COLAMD keeps it out of the elimination until the end, so the step stays linear in
`N`. The dense fallbacks (`_numerical_jacobian`, `_dense_tangent_system`) are
`O(N^2 n C_f)` and `O((N n)^3)` and run only when the sparse LU returns a
non-finite update.

The seed used to be the exception: it was solved by the dense forward-difference
`PeriodicOrbit`, so a short trace at large `N` was dominated by it (at `N = 160`,
trapezoidal: seed 1.8 s against 0.15 s per step). Since DEQ-49 the seed is as linear
as the step. DEQ-75 then evaluates the cycle fields a batch at a time and adapts the
arclength step as AUTO does; in the films that cut the continuation's own CPU time
from 7.75-111.3 s to 0.70-12.85 s per film (see
[DEQ-75](#deq-75-cycle-continuation-in-the-films)).

### Gauss-Legendre collocation

The survey measured this section on the DEQ-34 branch at `4e6b463`, before it
merged, with `m = 4` points per interval unless stated. DEQ-55 has since changed the
point: see the paragraph after the table.

Orthogonal collocation at `m` Gauss points per interval carries `N m + 1` states.
Each interval contributes a dense `(m n) x ((m + 1) n)` block, so the bordered matrix
has `O(N m^2 n^2)` non-zeros.

| Part | Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|---|
| `_solve_seed` | `N`, `n` | as on `main` (dense trapezoidal on `N + 1` nodes) + Hermite fill of the interior points + sparse correction | `O(N^2 n^2)` | the dense trapezoidal solve | 1.48 vs `N` (10-160), 1.29 vs `n` |
| `interval_blocks` (assembly) | `N`, `m`, `n` | `N m` autodiff Jacobians, field and parameter calls + `einsum` chain `O(N m^2 n^2)` | `O(N m^2 n^2)` | the `N m` Python calls | 0.94 vs `N`, 0.90 vs `m`, 1.39 vs `n` |
| `spsolve` | `N`, `m`, `n` | banded LU, block bandwidth `(m + 1) n`: `O(N m^3 n^3)` worst case | `O(N m^2 n^2)` | | 0.67 vs `N` (fill 1.24, constant), 0.55 vs `m`, 1.69 vs `n` |
| `_jacobian_monodromy` | `N`, `m`, `n` | `N` batched dense solves of size `m n` with `n` right-hand sides, `O(N m^3 n^3)`, + `N` products `O(N n^3)` | `O(N m^2 n^2)` (the stored blocks) | NumPy batched `solve` | 0.45 vs `N`, 1.50 vs `n`; under 1 ms at `N = 160` |
| `_point` at `4e6b463` (Jacobian multipliers + Hermite cross-check + matching) | `N`, `m`, `n` | the cross-check is a full `monodromy` over all `N m` sub-intervals with autodiff Jacobians, `O(N m (n C_f + n^3))`, + `linear_sum_assignment` `O(n^3)` | `O(n^2)` | the cross-check monodromy | 0.96 vs `N`, 0.98 vs `m`, 1.42 vs `n` |
| `_point` on `main` (Jacobian multipliers + doubled-mesh bound) | `N`, `m`, `n` | the `N` transfer matrices of the assembled collocation Jacobian, multiplied out or read by periodic QR (`periodic_schur`), `O(N m^3 n^3)`; then the cycle re-solved on `2N` intervals at fixed parameter and its multipliers read the same way, for the error bound; the Hermite cross-check only when asked for | `O(N m^2 n^2)` | the doubled-mesh re-solve | vs `N` (10-160, `n = 2`, DEQ-55): 1.05 (1.25 before); vs `n` (2-16, `N = 20`): 0.98 (1.73 before) |
| **One continuation step** at `4e6b463` | `N`, `m`, `n` | `O(I N m n C_f + N m^3 n^3)` | `O(N m^2 n^2)` | the cross-check: about 60% of a step | 0.99 vs `N`, 0.97 vs `m`, 1.44 vs `n` (`n` 2-16) |

At `4e6b463` the Jacobian monodromy (Fairgrieve-Jepson) was essentially free, and a
Gauss-Legendre step paid for the independent Hermite cross-check, evaluated eagerly at
every point to fill `floquet_error` (0.58 s of a 1.0 s step at `N = 160`). DEQ-55
reads the multipliers by periodic QR where needed, which costs more than the plain
product (`_jacobian_multipliers` 0.99 ms before, 28.6 ms after at `N = 160`), and
bounds their error by the disagreement with a doubled-mesh re-solve rather than the
cross-check; the point as a whole
went from 2.91 s to 0.49 s at `N = 160` (see
[DEQ-55](#deq-55-gauss-legendre-multipliers-by-periodic-qr)).

## Bifurcation curves, connecting orbits and Taylor manifolds

Size parameters used in this section: `n` state dimension, `N` mesh intervals of a
connecting orbit, `K` Taylor truncation order (fixed at 6 in the center-manifold and
focal-value code), `d` manifold dimension, `C_f` the cost of one field evaluation
(`O(n)` for a sparse coupling, `O(n^2)` for a dense one), `B = (K+1)(K+2)/2` the
number of monomials in a two-variable series (28 at `K = 6`), and `I` Newton
iterations.

### Taylor jets (`jet.py`)

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `k` order | add/sub `O(k)`; mul, reciprocal, divide `O(k^2)`; `**e` `O(k^2 log e)` | `O(k)` per jet | Python list loops | mul 1.67 (8-128; 1.95 between 64 and 128), reciprocal 1.62, add 0.70 |
| `n`, `C_f` | `AutomaticDifferentiation.jacobian`: `n` jet passes, `O(n * C_f)` | `O(n)` jets live | one Python jet object per state entry | 1.94 (ring field, `C_f = O(n)`) |
| `k` | `AutomaticDifferentiation.taylor`, polynomial field of degree `p`: `O(p k)` per product | `O(n k)` | zero-skip in `__mul__` | 0.77 vs `k` |

The product skips zero left coefficients, so a seed jet (two nonzeros) times
anything is `O(k)`, and a degree-`p` polynomial in the seed stays `O(p k)`. That is
why a directional Taylor expansion of a polynomial field grows nearly linearly in
`k` (slope 0.77) even though a dense product is quadratic. A rational field fills the
jet and pays the full `O(k^2)`. The Jacobian takes `n` separate order-1 passes;
for order 1 a jet is a two-element Python list, so the constant is about 25x that of
a float evaluation.

### Equilibrium-adjacent helpers (`seed_refiner.py`, `switching.py`, `shilnikov.py`, `snic.py`)

| Algorithm | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `NewtonSeedRefiner.refine` | `I * (2n C_f + n^3)` | `O(n^2)` | FD Jacobian then dense solve | not measured (same engine as the corrector) |
| `NullSpaceBranchSwitcher.secondary_seed` | `(2n+2) C_f + O(n^3)` (SVD of `n x (n+1)`) | `O(n^2)` | SVD | not measured |
| `classify_saddle_focus` | `O(n^3)` (`eigvals`) | `O(n^2)` | LAPACK | not measured |
| `characterize_snic` (`S` samples) | `1200` least-squares fits of `S x 2`: `O(S)` | `O(S)` | 1200 Python iterations | not measured; constant 1200 dominates for any realistic `S` |

### Deflation (`deflation.py`)

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `n`, `R` known roots, `I` iterations | dense: `I ((n+1)(C_f + R n) + n^3)` per root | `O(n^2)` | `n+1` residuals per Jacobian | 1.27 (`n` 16-256, single-threaded BLAS) |
| `n`, `R`, `g` GMRES iterations | matrix-free: `I g (C_f + R n)` | `O(n * restart)` | GMRES matvecs | 0.39 (`n` 16-256) |
| `S` seeds | `find`: up to `S (R + 1)` Newton runs | `O(R n)` | | |

On the test field (`f_i = x_i^3 - x_i + 0.1 x_{i+1}`) GMRES converges in a handful
of iterations whatever `n`, which is why the matrix-free path is almost flat; a
field whose Jacobian conditioning grows with `n` will raise `g`.

### Homoclinic shooting (`homoclinic_shooting.py`)

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `S = horizon/dt` RK4 steps | `gap`: `4 S C_f` + one equilibrium solve | `O(n)` | Python RK4 step, about 49 us per step at `n = 2` | 0.96 vs `S` |
| bisections `b = 32` | `locate`: `(b + 1)` gaps | `O(n)` | | linear in `b` by construction |
| `C` control values | `trace`: `C (b + 1)` gaps | `O(C)` | | |

At the defaults (horizon 60, `dt = 2e-3`) one gap is 30,000 steps, about 1.5 s, so
`locate` costs about 50 s and every point of `trace` the same again. The integration
always runs to the horizon (or divergence) even after the closest return has been
passed; stopping once the distance has grown past the return band after the minimum
would cut most gaps roughly in half, a constant factor only.

### Connecting orbits (`connecting_orbit.py`, on `main`)

`HomoclinicOrbit` / `HeteroclinicOrbit` `.solve`: Gauss-Newton on the trapezoidal
projection boundary-value problem, `(N n + n_b + 1) x (N+1) n` with the exact
(autodiff) Jacobian. Since DEQ-50 the Jacobian is assembled sparse and each step
solves the scaled augmented system `[[alpha I, J], [J^T, 0]]` by two sparse LU
factorisations whose fill is linear in `N`; `lsmr` and then dense least squares
remain as fallbacks.

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `N`, `n`, `I` (on `main`) | per iteration: `2N C_f` (residual) + `(N+1) n C_f` (autodiff Jacobians) + two sparse LU of `O(N n^2)` non-zeros | `O(N n^2)` | the per-node Python calls | vs `N` (200-3200, `n = 2`, DEQ-50): solve 1.13, Newton iteration 0.99; memory 0.98 |
| as above, before DEQ-50 | `+ O(N^2 n^2)` dense zero-fill and `csr_matrix(dense)` + `L ~ N` `lsmr` iterations of `O(N n^2)` | `O(N^2 n^2)` | `lsmr` iterations and the dense array | survey: time 1.13 (`N` 100-1600), memory 1.98; DEQ-50 before: solve 1.59, memory 1.99, 657 MB at `N = 3200` |

The survey's derivation of the old cost (saddle `x'' = x - x^2`, half-length 10, 4
Gauss-Newton iterations at every `N`):

* `lsmr` took 100, 200, 390, 765, 1500 iterations at `N` = 100 ... 1600: the
  iteration count is proportional to `N`, as the condition number of a first-order
  collocation operator grows like `N`. Each iteration is a sparse product over the
  `O(N n^2)` nonzeros, so the solve is `O(N^2 n^2)`. At these sizes `lsmr`'s fixed
  per-iteration Python overhead hides the second factor of `N`, which is why the
  overall time slope reads 1.13 rather than 2.
* `_analytic_jacobian` allocates `np.zeros((rows, unknowns))` and fills it, then
  `_least_squares` converts it with `csr_matrix(jacobian)`: `O(N^2 n^2)` memory and a
  full scan. Peak memory 0.7, 2.7, 10, 41, 165 MB for `N` = 100 ... 1600 at `n = 2`
  (slope 1.98); at `n = 10`, `N = 1600` the dense array alone would be 2 GB.
* `_residual` evaluates the field twice per interior node (`2N` evaluations, against
  the `N + 1` the periodic-orbit residual already uses).

### Homoclinic curve continuation (`homoclinic_curve.py`, on `main`)

`HomoclinicCurve.solve_point` / `.trace`: Newton on the orbit, the saddle and one
parameter. Since DEQ-50 the bordered block-banded Newton matrix is assembled in
`O(N)` field work and factored by SuperLU (dense `lstsq` only as a fallback); before,
it was a forward-difference Jacobian of the whole augmented residual and a dense
`lstsq`.

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `N`, `n`, `I` (on `main`) | per iteration: `O(N n C_f)` assembly + sparse LU, fill linear in `N` | `O(N n^2)` | the per-node calls | vs `N` (DEQ-50): Newton iteration 0.87 (100-3200), point 0.82 (101-3201 nodes); memory 0.97 |
| as above, before DEQ-50 | `(N+1) n + n + 2` full residuals of `2N C_f + (n+1) C_f + O(n^3)` = `O(N^2 n C_f)`, plus dense `lstsq` `O((N n)^3)` | `O(N^2 n^2)` | `O(N^2)` field evaluations | survey: 1.80 per point (`N` 100-300); DEQ-50 before: 2.37 per iteration (`N` 100-400), 24 s per iteration at `N = 400` |
| `P` sweep values | `trace`: `P` point solves | `O(N n^2)` | | |

### Taylor invariant manifolds (`manifold.py`, `multivariate.py`)

`TaylorManifold.compute`: for each degree `2..K`, evaluate the field on the current
`d`-variable series, then one dense `n x n` solve per monomial of that degree.

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `K`, `d`, `n` | `(K-1)` field evaluations on series of `M = C(d+K, d)` terms; one series product visits `M^2` pairs (pairs above degree `K` are skipped but still visited), so `O(K M^2)` per product: `O(K^3)` for `d = 1`, `O(K^5)` for `d = 2`; plus `C(d+K, K)` dense solves `O(n^3)` each | `O(n M)` coefficients | Python dictionary products | 1-D: 2.16 (`K` 5-80; 2.8 between 40 and 80); 2-D: 2.44 (`K` 6-25; 2.9 between 13 and 25), memory 1.53 (`M ~ K^2`); vs `n` at `K = 5`: 1.12 (`n` 16-256, field evaluation still dominant) |
| `Multivariate.__mul__` | `O(M_a M_b d)` (a tuple of length `d` built per pair) | `O(M)` | | via the rows above |
| `Multivariate.compose` (into `Bivariate`) | `O(terms * deg * B^2)` | `O(B)` | `Bivariate` products | via the center manifold below |

The 2-D measured slope sits below the derived 5 because the coefficient dictionaries
only hold degrees below the current one during the sweep and this field's manifold
leaves many of them zero; the local slope is still rising at `K = 25`.

### Center manifold reduction and focal values (`center_manifold.py`, `focal.py`)

`TaylorCenterManifold.lyapunov_quantities`, order fixed at `K = 6`: expand the field
in all `n` split coordinates, solve the homological equation degree by degree
(degrees 2-6), substitute back, and run the planar recursion.

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `n`, field degree `p` | expansion `O(n^2 C(n+p, p))` dictionary ops (the `inverse @ raw` recombination); manifold sweep `5 * n * C(n+p, p) * p * B^2` (compose of every row at every degree, `B^2 = 784`); homological solves `O((n K)^3)` per degree | `O(n C(n+p, p))` | `Multivariate.compose` into `Bivariate`, Python | 4.14 (`n` 3-8, cubic field rotated by a random orthogonal matrix, so every series is dense); memory 2.19 |
| planar `recurse` | constant (`K = 6`, `B = 28`) | constant | | |
| `PlanarLyapunov.quantities` | `O(B^2)` constant per call | constant | | |

Phase split at `n = 8`: expansion 0.03 s, manifold sweep 0.48 s, reduction 0.13 s.
The `n^4` comes from composing each of `n` rows, each with `C(n+3, 3) ~ n^3/6` terms,
at five degrees. `arguments[axis] ** power` is recomputed per term, per row, per
degree; caching the powers per call would cut the constant several-fold without
changing the order.

For comparison, `normal_form.first_lyapunov` (Kuznetsov's formula: one Jacobian,
two `eig`, two solves, ten jet passes of order 2-3) measured slope 2.14 on the same
field for `n` 3-24, and 0.06 s against 0.79 s at `n = 8`.

### Multilinear forms and normal-form coefficients (`normal_form.py`)

| Algorithm | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `MultilinearForms.bilinear` / `trilinear` | 3 / 7 jet passes of order 2 / 3: `O(C_f)` each | `O(n)` | jet arithmetic | constant in `k` |
| `cusp_coefficient`, `cubic_fold_coefficient` | `n` jet passes (Jacobian) + 2 SVD `O(n^3)` + 3 or 7 jet passes | `O(n^2)` | SVD for large `n`, Jacobian for small | not measured separately |
| `bogdanov_takens_coefficient`, `..._second_coefficient` | Jacobian + 2 SVD + 1-2 `lstsq` `O(n^3)` + 3-6 jet passes | `O(n^2)` | | |
| `first_lyapunov` | Jacobian + 2 `eig` + 2 solves `O(n^3)` + about 25 jet passes | `O(n^2)` | Jacobian (`n C_f`) | 2.14 (`n` 3-24, `C_f = O(n^2)`) |

### Codimension-two and -three curves

Files: `two_parameter_problem.py`, `fold_curve.py`, `hopf_curve.py`,
`bogdanov_takens_curve.py`, `codim2.py`, `codim3.py`, `cusp_curve.py`,
`generalized_hopf_curve.py`.

All curves run through the ordinary equilibrium engine (`ContinuerBuilder.assemble`
with detectors switched off): a central-difference Jacobian of the augmented residual
(`2(m+1)` residual evaluations for `m` augmented unknowns) and a dense bordered solve
`O(m^3)`.

| Curve (augmented size `m`) | One residual | One Newton iteration | Space | Dominated by | Measured slope (residual / Newton iteration) |
|---|---|---|---|---|---|
| Fold, Moore-Spence (`2n+1`) | `C_f` + one jet pass `O(C_f)` | `(4n+4)` residuals + `O(n^3)` solve: `O(n C_f + n^3)` | `O(n^2)` | FD Jacobian columns | 2.02 / 2.67 (`n` 4-32, `C_f = O(n^2)`; 3.3 between 16 and 32) |
| Hopf (`3n+2`) | `C_f` + two jet passes | `(6n+6)` residuals + `O(27 n^3)`: `O(n C_f + n^3)` | `O(n^2)` | FD Jacobian columns | 2.12 / 2.44 (2.8 between 16 and 32) |
| Bogdanov-Takens (`3n+2`) | `C_f` + two jet passes | as Hopf | `O(n^2)` | | not measured (same shape as Hopf) |
| Cusp, codim-3, minimally augmented (`n+2`, DEQ-51) | `C_f` + one LU of the bordered `(n+1)` matrix + a bilinear form | analytic state columns (`n` bilinear forms, solves against the same factors) + 3 differenced parameter columns: `O(n C_f + n^3)` | `O(n^2)` | the state columns | Newton iteration 2.58 (`n` 4-32, DEQ-51), 10.6 s at `n = 32` |
| Generalized Hopf, codim-3, minimally augmented (`n+3`, DEQ-51) | `C_f` + LU of the bordered `f_u^2 + kappa I` + `first_lyapunov` | analytic state columns, `O(n^2)` each after one factorisation plus a fixed number of multilinear forms, + 3 differenced parameter columns | `O(n^2)` | the state columns | Newton iteration 2.86 (`n` 4-32, DEQ-51), 35 s at `n = 32` |
| Cusp before DEQ-51 (Moore-Spence, `2n+2`) | `C_f` + jet pass + `cusp_coefficient` (Jacobian `n C_f`, 2 SVD `O(n^3)`) | `(4n+6)` residuals: `O(n^2 C_f + n^4)` | `O(n^2)` | the cusp coefficient re-derived inside every FD column | survey 2.11 / 3.28 (53 s per iteration at `n = 32`); DEQ-51 before: 3.42, 99 s at `n = 32` |
| Generalized Hopf before DEQ-51 (`3n+3`) | `C_f` + two jet passes + `lyapunov_quantities` (order-6 center-manifold reduction) | `(6n+8)` residuals: `O(n^5)` | `O(n^4)` | center-manifold reduction inside every FD column | survey 3.44 / 3.93 (`n` 3-6); DEQ-51 before: 57-136 s per iteration at `n` 4-6 |

A continuation step costs `(I + 1)` such Jacobians: `I` corrector iterations and
one tangent solve, none shared.

**Fold and Hopf curves.** Both the measured slopes and the derivation give
`O(n C_f + n^3)` per Newton iteration, which is what a dense Moore-Spence system
needs. The outer Jacobian is a central difference of an inner autodiff directional
derivative; it costs `4n+4` (fold) or `6n+6` (Hopf) residuals where an assembled
Jacobian (`f_u`, the Hessian contraction `(f_u q)_u`, `f_p`) would cost about `n`
jet passes. That is a constant factor, not an order.

The two paragraphs below describe the cusp and generalized-Hopf residuals as the
survey found them; DEQ-51 replaced both with the minimally augmented systems in the
table.

**Cusp curve (before DEQ-51).** `CuspCurveResidual.evaluate` called `cusp_coefficient`, which
rebuilds the Jacobian and takes two SVDs to find the null vectors `q` and `p` -
although `q` is already one of the unknowns (`state[n:2n]`). The finite-difference
Jacobian then repeats that `4n+6` times, so one Newton iteration is `O(n^2 C_f + n^4)`
against the `O(n C_f + n^3)` the fold curve pays. (The single-residual slope, 2.11,
is under the derived 3 because at `n <= 32` the per-jet Python overhead of the
Jacobian's `n` passes outweighs its `O(n^2)` field cost and the SVDs are
microseconds; the Newton-iteration slope, 3.3, shows the extra factor of `n`.)

**Generalized-Hopf curve (before DEQ-51).** `GeneralizedHopfCurveResidual.evaluate` called
`TwoParameterProblem.lyapunov_quantities`, the full order-6 center-manifold
reduction, and keeps only the first quantity. Inside the finite-difference Jacobian
that is `6n+8` reductions per Newton iteration. Kuznetsov's `first_lyapunov` (already
in `normal_form.py`, `O(n C_f + n^3)`, 12x cheaper at `n = 8` and growing as `n^2`
slower) gives the same zero set from the eigenvectors already in the unknowns.

`Codim2Analyzer.analyze` / `Codim3Analyzer.analyze` scan each of `P` branch points
once per test function: `BogdanovTakens` free; `ZeroHopf` Jacobian + `det`;
`HopfHopf` and `FoldBogdanovTakens` Jacobian + `eigvals`; `Cusp` and
`GeneralizedHopf` the normal-form coefficients above, all `O(n C_f + n^3)` per point.
`DegenerateBautin` (codim-3) calls `lyapunov_quantities`, the center-manifold
reduction, at every point.


## Stochastic analysis

Size parameters used in this section: `n` state dimension, `K` noise drivers
(columns of `G`), `k` tracked tangent directions (`count`, default `n`), `H` horizon
steps and `T` transient steps of a Benettin estimate, `d` grid dimension (`= n`),
`M` grid cells (the product of the per-axis counts), `S` seeds, `C_f` and `C_g` the
cost of one evaluation of the drift and of one noise column. Every field evaluation
in these modules is a Python call on a Python list, so `C_f` is at least a few
microseconds and is `O(n)` for any field that touches every component.


### Stationary Fokker-Planck on a grid (`StationaryFokkerPlanck.solve`)

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `M`, `d`, `K` | build `O(M K C_g + d M (C_f + K C_g))` field work, one batched call per face family since DEQ-53, and `O(d^2 M)` vectorised assembly; solve: SuperLU (COLAMD) on a 5-point (2-D) / 7-point (3-D) operator plus one dense row | operator `O(d^2 M)`; LU factor `O(M^1.23)` (2-D), `O(M^1.62)` (3-D) measured | 2-D: the field work (before DEQ-53 a Python face loop, 87-93% of runtime); 3-D: the sparse LU from about `M = 10^4` | survey: 2-D total 1.00, solve 1.18, fill 1.23; 3-D total 1.25 (rising), solve 1.92, fill 1.62; `tracemalloc` peak 1.00 / 1.02. DEQ-53: 2-D 8.5 s to 4.2 s at `M = 58 081`, 3-D unchanged (8.9 s to 9.1 s at `M = 15 625`), both under load 20-50 |

The assembly is two things of different cost. The coefficients are gathered first:
`_diffusion` evaluates all `K` columns at each of the `M` cells, and
`_face_coefficients` evaluates the whole drift and all `K` columns at each of the
roughly `d M` face centres (plus `K (d + 1)` column evaluations per face for the
Stratonovich shift). Since DEQ-53 each of these is one `eval_batch` call per face
family where the field offers it, so no Python runs per face beyond the field's own.
Everything after that - the
exponential-fitting weights, the cross-diffusion stencil, the COO triplets - is
vectorised NumPy and costs `O(d^2 M)`. Each row of the operator has `2d + 1` entries
from the diagonal diffusion and up to `8(d - 1)` more where `D` has off-diagonal
entries.

The solve is `spsolve` on a CSR matrix, which SciPy hands to SuperLU as the
transpose: the unit-mass row becomes a dense *column* of the factored matrix. That
ordering matters. With the library's own row scaling (`cell_volume`) the dense row
adds about 1% to the fill, and through the CSR route it costs 1.0x to 1.9x a
one-entry pin row; the same matrix passed as CSC factors 4.6x to 8x slower in 3-D.
The fill itself follows the grid: `M^1.23` in 2-D is close
to the `M log M` a nested-dissection ordering would give, while `M^1.62` in 3-D is
above the `M^(4/3)` bound for nested dissection, and the factor time grows as
`M^1.92`. `tracemalloc` does not see SuperLU's allocations, so the measured peak
(linear) is the Python side only; at `M = 58 081` the factor holds 5.6 million
nonzeros, about 67 MB, against a 38 MB traced peak.

### Density descriptors (`GridDensity`)

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `M`, `d` | `mass`, `marginal`, `dominant_mode` `O(M)`; `mean` `O(M d)`; `covariance` `O(M d^2)`; `interior_maxima` `O(3^d M)` | `covariance` builds an `(M, d, d)` temporary; others `O(M d)` | vectorised NumPy | not measured (negligible beside the solve) |

`interior_maxima` compares the interior against each of its `3^d - 1` shifted copies,
so it is the one descriptor whose constant grows exponentially in `d`; at `d <= 3`
that is 26 comparisons.

### One-dimensional closed-form density (`StationaryDensity.evaluate`)

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `M` grid points | `O(M (C_f + C_g))`: five diffusion and one drift evaluation per point | `O(M)` | the per-point Python calls | 0.86 (Ito), 1.00 (Stratonovich); memory 0.99 |

The integrand evaluates the diffusion, then its central-difference slope (two more
evaluations) for the convention's correction, and `_diffusion_squared` evaluates the
diffusion once more. The slope is computed in the Ito case too, where the
correction it feeds is identically zero.

### Benettin spectrum of a flow (`lyapunov_spectrum`)

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `n`, `k`, `H`, `T` | `O(T C_f + H (n C_f + n^2 k))` plus `O(H n k^2 / 10)` for the QR every tenth step | `O(n k)` frame, `O(n^2)` Jacobian | the `4(n + 2)` field evaluations per horizon step | survey: `H` 0.95; `n` (ring, `C_f ~ n`) 1.31 overall, 1.71 local; memory 1.45. DEQ-52 with `T = 50`, `H = 10`: 1.29 to 1.17, 2.9x at `n = 64` |

Each horizon RK4 step evaluates the field at the four stage points and a
forward-difference Jacobian at the same four points; `jacobian_at` re-evaluates its
base point, so the step costs `4(n + 2)` field evaluations and four
`(n x n) @ (n x k)` products. Since DEQ-52 a transient step advances the state alone,
four field evaluations, and lands on the same state bit for bit; before, it ran the
full step and threw the frame away.

### Benettin spectrum with one shared driver (`stochastic_lyapunov`)

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `n`, `k`, `H`, `T` | horizon step `O(n (C_f + C_g) + n^2 k)`, plus `O(n^3)` for `g_x @ g_x` under Stratonovich only; transient step `O(C_f + C_g)` (Ito) or `O(C_f + n C_g)` (Stratonovich) | `O(n^2)` | the `2(n + 2)` evaluations per horizon step | survey: `n` 1.17 overall, 1.53 local. DEQ-52, `T = 100`, `H = 10`: Ito 1.26 to 1.11 (4.9x at `n = 64`), Stratonovich 1.45 to 1.06 (7.7x) |

Before DEQ-52 `_Coefficients` took both forward-difference Jacobians at every step,
transient included, and formed `g_x @ g_x` (`O(n^3)`) before multiplying it by the
convention factor, which is zero under Ito. Now the transient advances the state
alone and the Ito shift is skipped rather than multiplied by zero.

### Benettin spectrum with `K` drivers (`matrix_noise_lyapunov`)

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `n`, `K`, `k`, `H`, `T` | horizon step `O(K n C_g + n C_f + K n^2 + n^2 k)`, plus the `O(K n^3)` variational shift under Stratonovich only; the transient advances the state alone | `O(K n^2)` column Jacobians | the `K (n + 1)` column evaluations | survey: `K` (`n = 8`) 0.78; `n` with `K = n` 1.97 overall, 2.42 local; memory 2.36. DEQ-52, `n = K`, `T = 50`, `H = 5`: Ito 2.19 to 1.83 (11.5x at `n = 64`), Stratonovich 2.39 to 2.11 (1.8x) |

Per horizon step `jacobians_at` takes `K (n + 1)` column evaluations and stores a
`(K, n, n)` array. Before DEQ-52 two contractions followed on every step:
`variational_drift_shift`, `sum_k (dG_k/dx)^2`, `O(K n^3)` and computed under Ito as
well, where its factor is zero; and the tangent noise term
`einsum("k,kij,jl->il", dW, J, Y)`, a three-operand `einsum` without `optimize` that
NumPy ran as one nested loop, `O(K n^2 k)` with no BLAS (18.5 ms per step at
`n = K = 64`). Now the Ito shift is skipped and the tangent noise is
`tensordot(dW, J, 1) @ Y`, `O(K n^2 + n^2 k)` (0.21 ms at the same size).

### Locating a stochastic threshold (`stochastic_threshold`)

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| bracket width `W`, tolerance `tol`, `S` seeds, `C` control values, `V` scan values | `PhenomenologicalThreshold.locate`: `2 + O(log2(W / tol))` Fokker-Planck solves; `DynamicalThreshold` with `MeanTopExponent`: the same count times `S` Benettin runs; `ThresholdCurve.trace`: `C` locates (two when the window misses); `StochasticScan.run`: `V` solves plus `V` exponents | one solve's worth at a time, plus the memo of test values | the test function | bisection, `W = 1.2`: 7, 10, 14, 17 solves at `tol = 1e-1 ... 1e-4`, about `log2(10) = 3.3` per decade |

`locate` memoises the endpoint values, so the bracket check and the bisection share
their first two evaluations. The cost of a threshold is therefore a logarithmic
number of the expensive calls above, and a threshold curve is `C` times that.

### Region analysis (`region_analysis`)

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `n`, `S` seeds, `I <= 80` Newton iterations, `E` equilibria found | `find_equilibria`: `O(S I ((n + 1) C_f + n^3) + S E n)`; `classify_equilibrium`: `O(n^3)` (`eigvals`); `has_limit_cycle`: `O(6000 * 4 C_f)` fixed horizon | `O(n^2)` for Newton; the 6001 x `n` trajectory | forward-difference Jacobian columns at small `n`; the fixed 24 000 field evaluations of the cycle test | `find_equilibria` vs `S`: 1.00, vs `n`: 1.40 overall, 1.83 local; `has_limit_cycle` vs `n`: 0.62 overall (fixed overhead), memory 0.96 |

The cycle test integrates a fixed 6000 RK4 steps whatever the system, so its cost is
`24 000 C_f` and its memory the stored trajectory; the return-map scan over the last
40% of it is a Python comprehension over 2400 points.

### Recursive exploration (`BifurcationExplorer.explore`)

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `P` = `maximum_points` per curve, branching capped at 6 | at most `2 + 2*6 + 2*6*6 = 86` continuations of up to `P` points each, every one priced as in the continuation sections | the tree plus each continuation's branch, `O(86 P n)` | the codimension-2 and -3 curve continuations | not measured: the count depends on the bifurcations the system has |

The explorer adds nothing but bookkeeping (`_dedup` is a set lookup per point); its
cost is the number of continuations it launches, which is bounded by the two sweep
directions and the cap of six escalated points per level.

### Noise helpers (`noise`)

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `n`, `K` | `jacobian_at`: `(n + 1) C_f`; `columns_at`: `K C_g`; `jacobians_at`: `K (n + 1) C_g`; `diffusion_matrix`: `O(K n^2)`; `state_drift_shift`: `O(K n^2)`; `variational_drift_shift`: `O(K n^3)` | `O(K n^2)` | field evaluations | measured through the estimators above |

### Coupled rings (`systems.coupled_ring`)

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| ring length `n` | `eval`: `O(n)` | `O(n)` | a Python comprehension over the cells | used as the `C_f ~ n` field in the scripts above |

The ring's Jacobian has two nonzeros per row. Since DEQ-53 a `SparsityPattern`
(given, or detected at the first point) lets the finite-difference and autodiff
Jacobians difference each group of structurally orthogonal columns at once, a
constant number of groups for the ring whatever `n`: at `n = 512` the
finite-difference Jacobian went from 0.267 s to 2.3 ms (time slope 1.80 to 0.85),
and the autodiff one from 1.25 s to 9.7 ms at `n = 256` (1.84 to 1.05). The array
handed to the solve is still dense, `O(n^2)` memory.


## Algorithms measured before they merged

These two sections were measured on their branches before merge; both have since
merged (DEQ-31 as #21, DEQ-30 as #23). Open pull requests change both again: see
DEQ-57, DEQ-58 and DEQ-59 (quasipotential) and DEQ-61 (geodesic level sets) under
[Before / after](#before-after).

### Quasipotential (ordered line integral method and minimum action paths)

Measured at `eea3fab` on the DEQ-31 branch.

`M` is the number of grid points (`M = N_x N_y`, planar only), `K` the update
radius in cells (`update_radius`, default 8), `S = |{offsets}| ~ pi K^2` the
stencil size, `P` the number of nodes on a gMAM chain, `I` its iterations,
`L/h` the length of a traced path in cells.

| Algorithm | Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|---|
| `_HalfLattice` (built once per `solve`) | `M` | `O(4M C_f)` | `O(M)` (`(2N_x-1)(2N_y-1) x 5` table) | Python loop: one `field_at` and one `diffusion_matrix` per half-lattice point | 0.70 vs `M` (noisy; 0.75 s at `M = 25921`) |
| `OrderedLineIntegral.solve` (the march) | `M`, `K` | `O(M K^2 + H log H)` with `H` heap pushes; `O(M log M)` at fixed `K` | `O(M)` arrays + heap `O(front x K)` | per-accept fixed cost (~200-300 us of numpy calls), then the `8 S` triangle updates vectorised | 1.04 vs `M` (whole solve), 1.12 march only; 0.88 vs `K` (4-24) |
| `minimum_action_path` (descent on `U`) | `M`, `L/h` | `O(M)` once (`np.gradient`) + `O(L/h)` steps | `O(M)` | four `RegularGridInterpolator` calls per step, ~200 us each step | 1.05 vs grid side |
| `GeometricMinimumAction.path` (simplified gMAM) | `P`, `I` | `O(I P (C_f + n C_f))`; tridiagonal solve `O(P)` per iteration | `O(P)` | Python loop over nodes for the drift and its forward-difference Jacobian, ~18 us per node per iteration | 0.93 vs `P` per iteration (`P >= 50`); `I` ~ 355 independent of `P` (25-200) |

**The march.** Every accepted point offers a one-point update to each
unaccepted point within `K` cells, and a triangle update through each of its
up to eight accepted neighbours: `S + 8S` candidate values, evaluated as one
batch of array operations (`_action`, `_triangle`). A target enters the heap only
when its value improves, so the heap holds stale entries rather than supporting
a decrease-key. On the rotating Ornstein-Uhlenbeck box at `K = 8` that is about
11 pushes per grid point (not `S ~ 200`), and the largest the heap grew was
34,334 entries at `M = 25,921`. It grows like the accepted front times `K`
(slope 0.61 against `M`), not like `M K^2`. The heap's `log` factor is invisible
at these sizes. The measured slope near 1 against `M` comes from the
per-accept Python and numpy-call overhead, which is a constant ~280 us. Against
`K` the vectorised `O(K^2)` work only starts to show above `K ~ 8`: from 4 to 24
the march went 1.4 s to 5.7 s at `M = 6561`, slope 0.88 rather than 2. Below that
the fixed overhead dominates, and at large `K` the stencil is clipped by the
81-point box and half of it is already accepted. Every triangle round the
accepted point is tried, not the paper's hierarchical subset. The module
docstring records why (the subset doubled the rotating-case error), so the
`8 S` constant is deliberate.

**gMAM.** Each iteration evaluates the drift and its Jacobian (forward
differences, `n + 1` field calls) node by node in Python, solves one
tridiagonal system per coordinate with `solve_banded`, and reparametrises to
equal arc length by `np.interp`. All of it is linear in `P`. The semi-implicit
step keeps the iteration count flat as the chain is refined (364, 352, 355, 357
iterations to `1e-10` at `P = 25, 50, 100, 200`), so the whole run is `O(P)`.

### Geodesic level sets (two-dimensional invariant manifolds)

Measured at `e363836` on the DEQ-30 branch.

`n` is the state dimension (3 in every example), `R` the points on a ring
(about `2 pi r / spacing` on a plane at geodesic radius `r`), `L` the number of
rings (`distance / Delta` when the step is not shrunk), `V = sum R` the mesh
vertices, `I <= 25` Newton iterations per landing, and `S` the RK4 steps per
orbit segment (`integration_steps = 24`, raised to `ceil(max_i tau_i rate_i /
0.5)` up to 4000).

| Algorithm | Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|---|
| `GeodesicLevelSets.compute` | `R`, `L`, `V`, `I`, `S`, `n` | `O(L I S c) + O(V I S (n+1) C_f)`, where `c` is the fixed cost of one batched numpy call | `O(V)` mesh (Python lists of vertices and triangles) + `O(n^2 R)` per ring | small rings: number of rings, through numpy-call overhead; large rings: batched RK4 in `_flow` | ~1.0 against `L` (Lorenz: 0.44 s to 1.69 s for 10 to 40 rings); 0.83 against `V` from 393 to 165k at fixed `L`, but 1.39 over the last three points (retries, see below) |
| one ring (`_advance`) | `R`, `I`, `S` | `O(I S (n+1) R C_f + T R log R + R n^3)` with `T <= 2000` trace-back steps | `O(n R)` states, `O((n+2)^2 R)` Newton matrices | `_flow` on `(n+1) R` stacked rows; `_trace_back` (Python loop, one KD-tree query per step) | 0.95 against `V` (peak memory) |

**Derivation.** Each ring solves `R` two-point boundary-value problems at
once. Every Newton iteration integrates the landing point and its `n`
forward-difference shifts back together, `(n+1) R` rows for `S` RK4 steps, then
solves `R` batched `(n+2) x (n+2)` systems. The field is evaluated on whole
columns when it allows that (`_batched`), so the per-row cost is numpy's.
Insertion passes (at most 6) re-solve only the inserted midpoints. Thinning
and zipping are `O(R)` Python loops. Up to a few thousand points per ring,
the cost tracks the number of rings, not the number of vertices: on the
linear saddle `us/vertex` fell from 783 to 69 as rings grew from 64 to 4,024
points at a fixed 10 rings.

**Retries.** With the default `loss = 0`, one failed landing anywhere on the ring
discards the whole ring and recomputes it at half the step. The share of
failed landings at the first attempt grew with ring size on the linear saddle
(3% at `R = 2011` and 4022, 13-14% at 8044 and 16088), so the number of
whole-ring recomputations grows with `R`. At `spacing = 3.1e-4` (rings up to
16,088 points), 6 of 22 ring solves were discarded and the step was cut,
giving 16 rings instead of 10. That run took 257 us/vertex against 69 at
`spacing = 1.25e-3`. The Newton step along the ring is capped at one ring
*segment* per iteration (`_RING_JUMP`), so the same geometric error in the
first guess needs more iterations as the spacing shrinks. That is the likely
cause of the growing loss, but it was not verified separately.

## Findings

Places where the implementation cost more, in order, than its algorithm needs, as
the survey found them at `4ca6382`, and what has happened to each since.

| # | Where | Cost then | Cost needed | Evidence (survey) | Status on `main` at `8690798` |
|---|---|---|---|---|---|
| 1 | `FixedStepConfig.times`, read twice per step by `EulerMaruyamaSolver`, `MilsteinSolver`, `GottwaldMelbourneSolver`, `ZhanDuanLiLiSolver` and the `srk*` driver | `O(K^2)` per path | `O(K)` | Euler-Maruyama local slope 1.85 in `K`; one `times` access 55 us at `K = 128000` | fixed by DEQ-47 (#25): every solver reads the grid once; Euler-Maruyama CPU slope 1.76 to 1.04 |
| 2 | `bialternate_determinant`, the default Hopf detector | `O(n^4)` Python calls and memory, `O(n^6)` determinant, per point | `O(n^3)` | slope 4.0 in time and memory; overflow at `n = 32` | fixed by DEQ-46 (#31): `bialternate_test_value`, one `eigvals`; slope 4.14 to 1.56 |
| 3 | `MilsteinSolver` and the Milstein `StochasticEnsemble` with diagonal noise; the Stratonovich ensemble drift | `O(n^4)` time and `O(n^3)` memory per step (times `P` in the ensemble) | `O(n C_g)` | per-path local slope 3.44; ensemble 12.5 s and 212 MB at `n = 32` | fixed by DEQ-48 (#26): `O(n^2)` undeclared, `O(n)` with `diagonal_noise` |
| 4 | `CycleContinuation._solve_seed` | dense forward-difference `PeriodicOrbit.solve` | `O(N n C_f)` | seed 1.8 s against 0.15 s per step at `N = 160` | fixed by DEQ-49 (#43): seed slope 1.97 to 0.66 |
| 5 | `PeriodicOrbit`, `HermiteSimpsonOrbit`, `RobustPeriodicOrbit` | as in 4 | `O(N)` per iteration | slope 2.01 in time, 1.61 in memory | fixed by DEQ-49 (#43): column-grouped differences and SuperLU |
| 6 | `ConnectingOrbit._analytic_jacobian` | dense Jacobian, `O(N^2 n^2)` memory; `lsmr` with about `N` iterations | `O(N n^2)` memory, sparse direct solve | memory slope 1.98, 165 MB at `N = 1600` | fixed by DEQ-50 (#30): memory slope 1.99 to 0.98 |
| 7 | `HomoclinicCurve` | dense forward-difference Newton, `O(N^2 n C_f)` + `O((N n)^3)` per iteration | `O(N n C_f)` | slope 1.92 per iteration | fixed by DEQ-50 (#30): slope 2.37 to 0.87 |
| 8 | `CuspCurveResidual` | both null vectors re-derived inside every finite-difference column, `O(n^2 C_f + n^4)` | `O(n C_f + n^3)` | slope 3.28; 53 s per iteration at `n = 32` | fixed by DEQ-51 (#37): minimally augmented, analytic state columns; 99 s to 10.6 s at `n = 32` |
| 9 | `GeneralizedHopfCurveResidual` | an order-6 center-manifold reduction in every finite-difference column, `O(n^5)` | `O(n C_f + n^3)` | slope 3.93; 9 s per iteration at `n = 6` | fixed by DEQ-51 (#37): `first_lyapunov` on the minimally augmented Hopf system |
| 10 | `matrix_noise_lyapunov` tangent-noise term | `einsum` without `optimize`, `O(K n^2 k)`, no BLAS | `O(K n^2 + n^2 k)` | 18.5 ms against 0.21 ms per step at `n = K = 64` | fixed by DEQ-52 (#32): `tensordot` |
| 11 | `stochastic_lyapunov`, `matrix_noise_lyapunov` under Ito | an `O(K n^3)` drift shift multiplied by zero every step | nothing | 5.9 ms per step at `n = K = 64` | fixed by DEQ-52 (#32): skipped under Ito |

Constant factors, not orders:

* No Jacobian was shared within an equilibrium-continuation step (`I + 4` per point
  where `I + 1` would do). Fixed by DEQ-53 (#48): `MemoizedJacobian`.
* Every Jacobian in the library was dense, including for fields with `O(n)`
  non-zeros. DEQ-53 (#48) adds column grouping to the continuation's
  finite-difference and autodiff Jacobians, and DEQ-49 to the periodic orbits; the
  arrays handed to the dense solves are still `n x n`.
* The Benettin transients did the full variational work and discarded it: fixed by
  DEQ-52 (#32). The Fokker-Planck assembly was a Python loop per face: batched by
  DEQ-53 (#48). Taylor series products still visit every term pair, including those
  above the order cap (open).
* Gauss-Legendre evaluated the Hermite cross-check monodromy eagerly at every point:
  since DEQ-55 (#45) it runs only when asked for, and the point's error bound comes
  from a doubled-mesh re-solve instead.
* Geodesic level sets discard a whole ring when one landing fails. DEQ-61 (open,
  PR #62) retries only the failing landings; see
  [DEQ-61](#deq-61-geodesic-level-sets-step-control).
* `scene_builder._resample_by_arclength` and `_resample_by_measure` scanned the branch
  for every sample, `O(samples x points)`: fixed by DEQ-53 (#48), one walk, slope
  1.82 to 0.91 and 1.95 to 1.18.

Found on the way, not a cost item: a solver appended its trajectory to the equation's
own variables, so solving the same equation twice returned a curve holding both runs.
Fixed by DEQ-45 (#24).

## Before / after

Every performance ticket measured its change before merging, and the runs are kept in
the main checkout's git-ignored `outputs/complexity/before-after/<TICKET>/`: a CSV
of every run (or the best of each size), the scripts and, for most tickets, a README
naming the commits compared. The survey that found the costs is in
`outputs/complexity/baseline/`. Nothing in those folders is committed; the paths
below are relative to `outputs/complexity/before-after/`.

**How the numbers here were made.** Every slope below was refitted for this page
from the raw CSV by `scripts/complexity_before_after.py`: the best (smallest) time of
each size, the largest traced peak, and an ordinary least-squares line through
`log(cost)` against `log(size)`. That agrees with the tickets' own summaries to two
decimals everywhere it was checked except DEQ-50, whose `summarize.py` fitted both of
its passes as ten points; see [Checks](#checks). "Ratio" is after over before at the
largest size both variants reached, so below 1 is faster. Times are wall clock
unless marked CPU. Most runs shared the machine with other test suites; a dagger (†)
marks wall-clock numbers taken at a recorded load average of 12 or more, or that the
folder itself calls noisy, and for those the slopes and CPU times are the figures to
trust. "Not recorded" means the folder does not hold the number; nothing here is
estimated.

### Summary

| Ticket | Status | Measured | Sizes | Time slope before → after | Ratio at largest common size | Peak memory before → after | Results unchanged? |
|---|---|---|---|---|---|---|---|
| DEQ-35 | merged (#52) | no cost measurement: verification scripts and mutants only | - | - | - | - | not applicable |
| DEQ-45 | merged (#24) | `EulerSolver.solve` vs steps `K` | `K` 2000-32000 | CPU 0.98 → 0.97 | CPU 1.02 | 6.6 → 6.6 MB | not recorded |
| DEQ-46 | merged (#31) | Hopf test per call vs `n`; 20-point continuation with three detectors | `n` 8-64 (before), 8-128 (after) | 4.14 → 1.56 (call); 3.23 → 1.26 (run) | 0.00074 (call), 0.010 (run) at `n = 64` | 65.2 → 0.13 MB (call, `n = 64`) | not recorded |
| DEQ-47 | merged (#25) | Euler-Maruyama and Milstein solve vs `K` | `K` 16000-128000 | CPU 1.76 → 1.04 (EM), 1.56 → 1.05 (Milstein) | CPU 0.056 (EM), 0.24 (Milstein) | 16 → 16 MB | not recorded |
| DEQ-48 | merged (#26) | Milstein, per path and ensemble, vector noise vs `n`, undeclared and declared diagonal | `n` 4-128 (ensemble before: 4-32) | CPU 2.12 → 1.17 / 0.41 (path); 2.68 → 1.45, 2.70 → 0.46 (ensemble) | CPU 0.026 / 0.00072 (path, `n = 128`); 0.013 / 0.0020 (ensemble, `n = 32`) | 51.9 → 1.9 / 1.5 MB (path); 212 → 0.77 / 0.46 MB (ensemble) | not recorded |
| DEQ-49 | merged (#43) | periodic-orbit solves, cycle seed and step vs `N`; `PeriodicOrbit` vs `n` | `N` 40-640, `n` 2-16 | 2.35 → 1.05 (`PeriodicOrbit`); 1.97 → 0.66 (seed); step 0.98 → 0.99 | 0.0062 (`PeriodicOrbit`), 0.0087 (seed), 0.96 (step) | 27.9 → 0.27 MB (`PeriodicOrbit`, `N = 640`) | not recorded |
| DEQ-50 | merged (#30) | connecting-orbit and homoclinic-curve Newton iteration and solve vs `N` | `N` 200-3200 (orbit); 100-400 before, 100-3200 after (curve) | 1.59 → 1.13 (orbit solve); 2.37 → 0.87 (curve iteration) † | 0.052 (orbit solve, 3200); 0.0047 (curve iteration, 400) † | 657 → 2.6 MB (orbit solve, 3200) | not recorded |
| DEQ-51 | merged (#37) | one Newton iteration of the cusp and generalized-Hopf curves vs `n` | `n` 4-32 (GH before: 4-6) | 3.42 → 2.58 (cusp) †; GH before 3 points only | 0.11 (cusp, `n = 32`); 0.00073 (GH, `n = 6`) † | 0.13 → 0.34 MB (cusp); 2.1 → 0.14 MB (GH, `n = 6`) | not recorded |
| DEQ-52 | merged (#32) | `lyapunov_spectrum`, `stochastic_lyapunov`, `matrix_noise_lyapunov`, with and without transients | `n` 4-64, `K` 2-32, `H` 250-2000 | e.g. 2.19 → 1.83 (matrix, Ito, transient) †; controls 0.85 → 0.69, 1.30 → 1.21 † | 0.087 (matrix, Ito, transient); 0.13-0.55 (other transients); controls 1.0, 1.5 † | 6.5 → 6.5 MB (matrix, `n = 64`) | not recorded |
| DEQ-53 | merged (#48) | shared Jacobian, grouped sparse Jacobians, Fokker-Planck assembly, scene resampling | `n` 4-512, `M` 729-58081, samples 40-2560 | 1.80 → 0.85 (FD ring Jacobian); 1.82 → 0.91 (resample) † | 0.0088 (FD Jacobian, 512); 0.41 (fold run); 0.50 (2-D FP); 1.03 (3-D FP) † | 38.4 → 40.3 MB (2-D FP) | not recorded |
| DEQ-55 | merged (#45) | Gauss-Legendre `_point` and multipliers vs `N` and `n` | `N` 10-160, `n` 2-16 | 1.25 → 1.05 (point, `N`); 0.46 → 1.34 (multipliers, `N`) | 0.17 (point), 29 (multipliers) at `N = 160` | 2.17 → 2.17 MB | not recorded |
| DEQ-70 | merged (#53) | `BrownianPath._cell_areas` vs cells | `2^11`-`2^18` before, to `2^20` after | 1.41 → 1.41 †; peak 1.41 → 0.55 | 1.51 at `2^18` † | 1149 → 21.1 MB at `2^18`; before fails at `2^20` | bit-identical, `e` = 11-18 |
| DEQ-73 | merged (#40) | manifold fan flow vs branches per frame | 4-64 | 0.82 → 0.19 | 0.053 at 64 | 5.4 → 11.2 MB | not recorded |
| DEQ-75 | merged (#51) | nine stage films, best of 5 | one build per film | no size sweep | continuation CPU 0.020-0.39; film CPU 0.04-0.92 | not recorded | not recorded (adaptive steps move the points) |
| DEQ-57 | PR #58 | OLIM vs grid side and radius; gMAM per iteration; new `check()` | side 41-161, radius 2-16, nodes 25-400 | 1.80 → 1.96 vs side (unchanged) | 1.00 (OLIM, 161); 0.65 (gMAM, 400) | 28.9 → 28.9 MB | not recorded |
| DEQ-58 | PR #59 | half lattice vs side; OLIM with the parabolic update and Eq. 28 radius (now opt-in) | side 41-321, radius 2-16 | 2.41 → 2.30 (lattice); 2.45 → 2.43 (OLIM) | 0.026 (lattice, 321); 1.97-2.52 (OLIM, 161) | 115 → 79 MB (lattice); 28.9 → 19.9 MB (OLIM) | not recorded |
| DEQ-59 | PR #60 | gMAM per iteration, constant and state-dependent `D` | nodes 25-400 | 1.33 → 1.12 (constant `D`); before refuses varying `D` | 0.88 (constant `D`, 400) | 0.20 → 0.30 MB | not recorded |
| DEQ-61 | PR #62 | `GeodesicLevelSets.compute` vs `1/h`; four film-scale runs | `1/h` 200-3200 | CPU 1.30 → 1.01 † | CPU 0.30 at 3200 †; films 1.76-16.1x costlier, deliberately | 103.8 → 66.2 MB | memo mesh bit-identical to un-memoised escalation |
| DEQ-65 | PR #56 | ensemble Milstein step vs paths `P`, channels `m` | `P` 256-16384, `m` 2-16 | 0.73-0.96 → 0.80-1.11 vs `P` (commuting) | 0.35-0.89 (commuting, largest `P`) | not recorded | commuting: final-state hashes equal at all 15 sizes |
| DEQ-72 | PR #57 | full atlas build and a 12-unit subset vs workers | workers 1-12 | no fitted slope (worker count) | 0.29 wall at 12 workers vs before serial †; CPU 1.09 | 638 → 636 MB per process (RSS) | byte-identical, 62 files |
| DEQ-94 | PR #68 | Lorenz `lambda_1` vs horizon `T` by Jacobian kind; Jacobian entry errors | `T` 125-1000 | CPU 1.13 → 1.07 (FD), 1.01 (analytic), 0.97 (jets) | CPU 1.02 (FD), 0.30 (analytic), 1.55 (jets) at `T = 1000` | not recorded | no: `lambda_1` within 2.1e-9; 129 exponents within 2.5e-7 |
| DEQ-105 | PR #74 | Hermite-Simpson `_point` and one step vs `N` | `N` 20-320 | 1.27 → 0.90 (point); 1.05 → 1.05 (step) † | 1.74 (point), 2.10 (step) at 320; 5.47 at `N = 20` †, deliberately | not recorded | not applicable: the error estimate changes by design |

### By status

**Merged:** DEQ-35 (#52), DEQ-45 (#24), DEQ-46 (#31), DEQ-47 (#25), DEQ-48 (#26),
DEQ-49 (#43), DEQ-50 (#30), DEQ-51 (#37), DEQ-52 (#32), DEQ-53 (#48), DEQ-55 (#45),
DEQ-70 (#53), DEQ-73 (#40), DEQ-75 (#51).

**In open pull requests:** DEQ-57 (#58), DEQ-58 (#59), DEQ-59 (#60), DEQ-61 (#62),
DEQ-65 (#56), DEQ-72 (#57), DEQ-94 (#68), DEQ-105 (#74). The rest of this page
describes `main`, so none of these is reflected in the sections above.

**Costlier on purpose, for correctness:**

* DEQ-105 (#74): the Hermite-Simpson Floquet error becomes a bound, the larger of the
  drift and twice the doubled-mesh gap, and the point costs 1.74x to 5.47x what it did
  (ratio 5.47, 4.87, 5.28, 3.68, 1.74 at `N` = 20 ... 320; the folder's README rounds
  these to 5.5 ... 1.7). Film builds took 1.8x to 3.2x as long under concurrent load.
* DEQ-61 (#62): escalating only the failing landings reaches further. The strict
  film reached 165.25 instead of 157.375 and cost 16.1x the 200-step baseline (1853 s
  against 115 s); the IJBC and SIADS spacings reached the same distance at 2.21x and
  1.76x, and the lossy run lost 1 point instead of 57 at 2.56x
  (`DEQ-61/memo_numbers_2026-11-23.csv`).
* DEQ-58 (#59): the parabolic update and the Eq. (28) radius, opt-in since the PR
  but the defaults in the measured commit `d893783`, cost 1.78x to 2.79x the
  midpoint march across all three OLIM sweeps (1.97x, 2.52x and 2.01x at the largest
  size). The PR body says 2.3-2.8x; the CSV's range starts lower.
* DEQ-70 (#53) also trades time for memory: the chunked draw walks the generator
  twice to stay bit-identical and is 1.22x to 1.67x slower per size, for a peak 54x
  smaller at `2^18` cells.

### Checks

Eleven slopes were refitted from the raw rows by a least-squares fit written for this
page (no NumPy), and compared with what the folders claim:

| Ticket | Series | Claimed | Refit | Agrees? |
|---|---|---|---|---|
| DEQ-45 | `euler.solve` CPU, before / after | 0.98 / 0.97 (`summary.txt`) | 0.98 / 0.97 | yes |
| DEQ-46 | Hopf test per call, before / after | 4.14 / 1.56 (`summarize.py` output) | 4.14 / 1.56 | yes |
| DEQ-47 | Euler-Maruyama CPU, before / after | 1.76 / 1.04 (`summary.txt`) | 1.76 / 1.04 | yes |
| DEQ-48 | `milstein.solve_diagonal` CPU, before / after | 2.12 / 1.17 (`summary.txt`) | 2.12 / 1.17 | yes |
| DEQ-50 | `connecting_orbit.newton_iteration`, before / after | 1.06 / 1.17, 3.819 s / 0.774 s at 3200 (`summarize.py` output) | 1.02 / 0.99, 2.061 s / 0.219 s | **no**: the script fits both passes' ten rows and reports the slower pass's time; the refit takes the best of the two passes at each size |
| DEQ-51 | cusp Newton iteration, before / after | 3.42 / 2.58 (`summarize.py` output) | 3.42 / 2.58 | yes |
| DEQ-52 | `matrix_noise_lyapunov` transient Ito, before / after | 2.19 / 1.83 (`summary.txt`) | 2.19 / 1.83 | yes |
| DEQ-53 | `fd_jacobian_ring`, before / after | 1.80 / 0.85 (`summary.md`) | 1.80 / 0.85 | yes |
| DEQ-65 | commuting `m = 2` vs `P`, before / after | 0.92 / 0.80 (`slopes.csv`) | 0.92 / 0.80 | yes |
| DEQ-70 | peak bytes, before / after | 1.41 / 0.55 (`summary.txt`) | 1.41 / 0.55 | yes |
| DEQ-105 | `_point`, before / after | 1.27 / 0.90 (README) | 1.27 / 0.90 | yes |

Two other differences between a folder and its own files: DEQ-75's README says 19
mutants, all killed, but `DEQ-75/mutants.log` records 17 killed and 2 survived
(mutant 16, `np.array_equal(factored[0], border)`, among them); and DEQ-70's README
gives the slowdown as 1.2-1.6x where its `summary.txt` times give 1.22x to 1.67x.

### Per ticket

#### DEQ-35: cycle codimension two

`DEQ-35/` holds the verification scripts for the periodic normal forms and the fold,
flip and torus curves of cycles (DOP853 shooting, closed forms, MATCONT and De Witte
et al. references) and `mutants.log`. It records no timing or memory, so there is
nothing to fit.

#### DEQ-45: a curve per solve

`EulerSolver.solve` against `K` = 2000-32000, best of 5 interleaved, CPU recorded
(`DEQ-45/deq_45_scaling.csv`, `raw.csv`). The fix was not meant to change the cost and
did not: CPU slope 0.98 before, 0.97 after, 0.1816 s against 0.1845 s at
`K = 32000`, memory slope 1.00 and 6.6 MB both ways. The wall slopes (1.08, 0.71)
scatter more than the CPU ones. Commits `7f8bc92` → `7209310`.

#### DEQ-46: Hopf test from one eigenvalue solve

Per-call cost of the Hopf test on a random `n x n` Jacobian, and a 20-point
continuation with the fold, branch-point and Hopf detectors, best of 5 interleaved
(`DEQ-46/measurements.csv`). The old test was skipped at `n = 128` (minutes per run).
Per call: time slope 4.14 → 1.56, memory 4.08 → 1.59; 2.87 s and 65 MB against 2.1 ms
and 0.13 MB at `n = 64`. Continuation: 3.23 → 1.26, 110 s against 1.09 s at
`n = 64`, peak 67.8 → 2.75 MB. Commits `4ca6382` → `ebe13b6`. Load not recorded.

#### DEQ-47: the time grid read once

Euler-Maruyama and Milstein solves of a two-state OU against `K` = 16000-128000,
best of 5 with CPU (`DEQ-47/deq_47_scaling.csv`). CPU slope 1.76 → 1.04
(Euler-Maruyama) and 1.56 → 1.05 (Milstein); at `K = 128000` 36.8 s → 2.07 s and
56.4 s → 13.3 s CPU. Wall times (91 s → 6.4 s, 139 s → 24.6 s) are the same story
with more scatter; Milstein's last wall pair has slope 0.10. Memory 16 MB both ways,
slope 1.0. Commits `7209310` → `c31efdd`.

#### DEQ-48: vector-noise Milstein

Per-path Milstein (`K = 200`) and the ensemble's Milstein and Stratonovich drift
(`P = 200`, `K = 50`) against `n`, for an OU whose noise is found commuting
(undeclared) or declared diagonal, best of 5 with CPU (`DEQ-48/deq_48_scaling.csv`).
The ensemble's before runs stop at `n = 32` (64 for Stratonovich), where they needed
212 MB (421 MB).

| Series | CPU slope before → after | CPU at largest common `n` | Peak memory |
|---|---|---|---|
| path, undeclared | 2.12 → 1.17 | 49.3 s → 1.27 s (`n = 128`) | 51.9 → 1.9 MB |
| path, declared | 2.12 → 0.41 | 48.6 s → 0.035 s (`n = 128`) | 51.9 → 1.5 MB |
| ensemble Milstein, undeclared | 2.68 → 1.45 | 13.2 s → 0.17 s (`n = 32`) | 212 → 0.77 MB |
| ensemble Milstein, declared | 2.70 → 0.46 | 13.7 s → 0.027 s (`n = 32`) | 212 → 0.46 MB |
| ensemble Stratonovich drift, undeclared | 1.98 → 1.58 | 4.19 s → 0.83 s (`n = 64`) | 421 → 1.2 MB |
| ensemble Stratonovich drift, declared | 1.92 → 0.47 | 3.72 s → 0.043 s (`n = 64`) | 421 → 0.93 MB |

The declaration is ignored before the change, so before-declared and before-undeclared
are the same code. Commits `c31efdd` → `365394c`.

#### DEQ-49: sparse periodic orbits

`PeriodicOrbit`, `HermiteSimpsonOrbit` and `RobustPeriodicOrbit.solve`,
`CycleContinuation._solve_seed` and one continuation step against `N` = 40-640
(`n = 2`), and `PeriodicOrbit.solve` against `n` = 2-16, 5 interleaved rounds
(`DEQ-49/deq49.csv`).

| Series | Slope before → after | At largest size | Memory slope, peak |
|---|---|---|---|
| `PeriodicOrbit.solve` vs `N` | 2.35 → 1.05 | 61.7 s → 0.38 s | 1.81 → 0.93; 27.9 → 0.27 MB |
| `HermiteSimpsonOrbit.solve` | 1.99 → 0.96 | 36.2 s → 0.18 s | 1.81 → 0.93 |
| `RobustPeriodicOrbit.solve` | 2.18 → 0.73 | 53.9 s → 0.13 s | 1.81 → 0.93 |
| `_solve_seed` | 1.97 → 0.66 | 26.2 s → 0.23 s | 1.81 → 0.94; 27.9 → 0.58 MB |
| one continuation step (control) | 0.98 → 0.99 | 0.249 s → 0.237 s | 0.96 → 0.96 |
| `PeriodicOrbit.solve` vs `n` | 1.63 → 1.09 | 0.86 s → 0.042 s (`n = 16`) | 1.78 → 1.75 |

Commits `d24b753` → `12b69c8`. Load not recorded.

#### DEQ-50: sparse connecting orbits and homoclinic curves

Connecting-orbit Newton iteration and solve against mesh intervals `N` = 200-3200,
homoclinic-curve Newton iteration and `solve_point` against `N` (`DEQ-50/deq50_scaling.csv`,
best of 5). The curve's before runs stop at `N = 400` (about 25 s per solve); its after
runs go to 3200. Load average 12-27, so the time slopes are noisy (†) and the memory
slopes are not. Peaks are `tracemalloc`'s, which misses SuperLU's factor (1.8 MB for
the orbit, 0.68 MB for the curve at `N = 3200`, per the README).

| Series | Time slope † | At largest common size † | Memory slope, peak |
|---|---|---|---|
| orbit Newton iteration | 1.02 → 0.99 | 2.06 s → 0.22 s (3200) | 1.98 → 0.98; 329 → 2.5 MB |
| orbit solve | 1.59 → 1.13 | 20.7 s → 1.07 s (3200) | 1.99 → 0.98; 657 → 2.6 MB |
| curve Newton iteration | 2.37 → 0.87 | 23.6 s → 0.11 s (400) | 1.85 → 0.97 |
| curve `solve_point` | 1.54 → 0.82 | 24.9 s → 0.13 s (301 nodes) | 1.84 → 0.97 |

The orbit iteration was run twice, on `ed8171e` → `a045604` and again on
`8f514cb` → `3f94bb7` with the same orbit code; the refit takes the faster of the two
at each size. Its own `summarize.py` gives 1.06 → 1.17 because it fits all ten rows.

#### DEQ-51: minimally augmented cusp and Bautin curves

One Newton iteration (residual, extended Jacobian, bordered solve) of each curve on
dense fields against `n` (`DEQ-51/deq51_scaling.csv`, best of 5). Load 30-40
throughout (†). Cusp: 3.42 → 2.58, 99.0 s → 10.6 s at `n = 32`; memory slope
1.27 → 1.36, peak 0.13 → 0.34 MB, so the new system holds more but stays small.
Generalized Hopf: before stops at `n = 6` (57 s, 136 s, 85 s at `n` = 4, 5, 6, not
monotone under the load), after runs to 32 with slope 2.86 and 35.5 s at `n = 32`;
at `n = 6` 84.6 s → 0.062 s. Commits `a984a5c` → `78d22e1`.

#### DEQ-52: Lyapunov estimator costs

13 series of the three Benettin estimators, with and without transients and under
both conventions, against `n` = 4-64, `K` = 2-32 or `H` = 250-2000, best of 5
(`DEQ-52/deq52.csv`, `summary.txt`). Load 30-50; the README notes the unchanged-path
controls scatter 0.5-1.5x, and one of them, `stochastic_lyapunov` Ito with no
transient, reads 2.2x slower (†). The transient series carry the change:

| Series (`n` 4-64) | Time slope † | Ratio at `n = 64` † |
|---|---|---|
| `lyapunov_spectrum`, `T = 50`, `H = 10` | 1.29 → 1.17 | 0.34 |
| `stochastic_lyapunov`, `T = 100`, `H = 10`, Ito | 1.26 → 1.11 | 0.20 |
| `stochastic_lyapunov`, `T = 100`, `H = 10`, Stratonovich | 1.45 → 1.06 | 0.13 |
| `matrix_noise_lyapunov`, `n = K`, `T = 50`, `H = 5`, Ito | 2.19 → 1.83 | 0.087 |
| `matrix_noise_lyapunov`, `n = K`, `T = 50`, `H = 5`, Stratonovich | 2.39 → 2.11 | 0.55 |

Memory slopes are unchanged (1.34-2.52 before and after; 6.5 MB at `n = K = 64`).
`per_step_analytic_noise.txt` isolates the tangent-noise step with the field cost
removed: Ito 37.8 → 3.3 ms per step at `n = K = 64`. Commits `4ca6382` → `67a68f2`.

#### DEQ-53: constant factors

Nine series (`DEQ-53/deq53.csv`, `summary.md`); load 20-50 (†), single cold runs for
the continuations and Fokker-Planck, means over 0.2 s for the sub-millisecond kernels.

| Series | Sizes | Time slope † | At largest size † | Peak |
|---|---|---|---|---|
| continuation, fold detector | `n` 4-64 | 1.10 → 0.85 | 0.97 s → 0.40 s | 2.7 → 2.9 MB |
| continuation, all detectors | `n` 4-32 | 2.04 → 2.51 | 7.85 s → 7.67 s | 6.4 → 6.5 MB |
| FD Jacobian, ring | `n` 32-512 | 1.80 → 0.85 | 0.267 s → 2.3 ms | 2.1 → 2.2 MB |
| autodiff Jacobian, ring | `n` 32-256 | 1.84 → 1.05 | 1.25 s → 9.7 ms | 1.7 → 0.55 MB |
| continuation, ring, declared sparsity | `n` 16-128 | 1.77 → 1.54 | 3.44 s → 1.35 s | 3.4 → 4.0 MB |
| Fokker-Planck 2-D | `M` 1681-58081 | 1.14 → 1.28 | 8.51 s → 4.22 s | 38.4 → 40.3 MB |
| Fokker-Planck 3-D | `M` 729-15625 | 1.46 → 1.55 | 8.85 s → 9.10 s | 15.0 → 16.2 MB |
| resample by arclength | 40-2560 | 1.82 → 0.91 | 0.270 s → 3.7 ms | 0.47 → 0.39 MB |
| resample by measure | 40-2560 | 1.95 → 1.18 | 0.224 s → 3.9 ms | 0.33 → 0.33 MB |

The all-detector continuation's slope rose while its time at `n = 32` did not move:
this base still had the `O(n^6)` Hopf determinant (DEQ-46 merged later), which
dominates that run, so the shared Jacobian's constant saving is lost in it and in the
load.
Commits `4ca6382` → `dde829b`.

#### DEQ-55: Gauss-Legendre multipliers by periodic QR

`CycleContinuation._point` (multipliers and the Floquet error estimate) and the
Jacobian multipliers alone, Gauss-Legendre with `m = 4`, against `N` = 10-160
(`n = 2`) and `n` = 2-16 (`N = 20`), 5 interleaved rounds (`DEQ-55/deq55.csv`).
Point: 1.25 → 1.05 against `N`, 2.91 s → 0.49 s at `N = 160`; 1.73 → 0.98 against
`n`, 4.83 s → 1.19 s at `n = 16`. The multipliers alone got dearer, 0.46 → 1.34
against `N` (0.99 ms → 28.6 ms at 160) and 1.62 → 1.82 against `n`, and are a small
part of the point. Peaks unchanged (2.17 MB, 13.6 MB). Commits `efdc0dd` → `7d4a170`.
Load not recorded.

#### DEQ-70: Brownian-path areas in chunks

`BrownianPath._cell_areas` against cells `2^11`-`2^20` with `m = 2`, best of 3
(`DEQ-70/deq70_scaling.csv`, `summary.txt`, README). Load 19-23 (†). Peak memory
slope 1.41 → 0.55: 1149 MB → 21.1 MB at `2^18`, and at `2^20` the old code hit the
5 GB limit while the new one peaked at 58.8 MB; beyond the increments the working
set stays at 8.5-9.0 MB from `2^14` on. Time slope 1.41 both ways, the new code
1.22x-1.67x slower per size (4.24 s → 6.38 s at `2^18`). The readings are
bit-identical (`np.array_equal` and byte equality) for `e` = 11-18, stored as
`before_<e>.npy` and `after_<e>.npy`. Commits `8a76b6a` → `0f43b4c`.

#### DEQ-73: manifold branches in one Dormand-Prince pass

`manifold_fan_flow` against branches per frame 4-64, with an intermediate `step1`
(`DEQ-73/ba.csv`; no README, so neither the rounds nor the load are recorded).
Slope 0.82 → 0.83 (`step1`) → 0.19; 6.21 s → 3.05 s → 0.33 s at 64 branches. Peak
memory rose, 5.4 → 4.7 → 11.2 MB, as the branches are carried together. Commits
`0153340` → `3219a6c` → `5722f8c`.

#### DEQ-75: cycle continuation in the films

Nine stage films built 5 times each from the old and new trees, interleaved, with the
continuation's own CPU time recorded (`DEQ-75/best_of_5.csv`, `builds.csv`). There is
no size sweep, so no slopes. Continuation CPU went from 7.75-111.3 s to 0.70-12.85 s
per film (ratio 0.020 for ferroelectric to 0.39 for SNIC); whole-film CPU from 0.04x
(van der Pol, 25.7 s → 1.06 s) to 0.92x (zero-Hopf, where other work dominates).
Memory was not recorded. The adaptive step moves the continuation points, so results
are compared by `compare_adapt.py` and `mult_check.py`, whose outputs are not in the
folder. Base `d2d71b4`.

#### DEQ-57: quasipotential error estimate

OLIM against grid side 41-161 (radius 8) and radius 2-16 (81 points), gMAM per
iteration against nodes 25-400, and the new `check()` per point (after only), best of
5 (`DEQ-57/ba.csv`). The march is unchanged, as intended: slope 1.80 → 1.96 against
the side (about 0.9-1.0 against `M = side^2`), ratio 0.68-1.10 per size, 28.9 MB both.
gMAM 1.08 → 1.05. `check()` takes 5.0-14.0 s per point (slope 0.72 against nodes
41-121). Commits `9df11d9` → `2419505`. Load not recorded.

#### DEQ-58: quasipotential order and batching

The half lattice against side 81-321, and OLIM against side 41-161 at radius 8, at
the default radius, and against radius 2-16 (`DEQ-58/ba.csv`, best of 5). The
batched half lattice is 0.026x the time at side 321 (22.7 s → 0.58 s, slope 2.41 →
2.30 against the side) and peaks lower (115 → 79 MB). The OLIM runs measure the
parabolic update and Eq. (28) radius, the defaults at `d893783` and opt-in in the PR:
1.78x-2.79x the midpoint march, peak 28.9 → 19.9 MB at side 161. Commits
`026c7e8` → `d893783`.

#### DEQ-59: gMAM with state-dependent noise

gMAM per iteration against nodes 25-400 with constant `D` (should be unchanged) and
25-200 with a state-dependent `D`, which the old code refused (`nan` rows)
(`DEQ-59/ba.csv`). Constant `D`: 1.33 → 1.12, ratio 0.64-1.61 per size, 0.88 at 400.
State-dependent: slope 1.12, 0.096 s per iteration at 200 nodes. Commits `7202a32` →
`4b2c42c`.

#### DEQ-61: geodesic level sets, step control

`GeodesicLevelSets.compute` on the linear saddle, step 0.1 to distance 1.0, against
`1/h` = 200-3200, best of 5 with CPU (`DEQ-61/deq_61_scaling.csv`); load 40-58 (†).
CPU slope 1.30 → 1.01, 61.9 s → 18.5 s CPU at 3200; wall 1.50 → 1.04; memory slope
1.14 → 1.01, peak 103.8 → 66.2 MB. The film-scale runs in
`memo_numbers_2026-11-23.csv` (all eight started together, load rising to about 47)
show the price of reaching further: 16.1x on the strict film, 2.21x, 1.76x and 2.56x
on the others; `memo_numbers.csv` notes the memoised mesh is bit-identical to the
un-memoised escalation. Commits `3a5ccae` → `581e50d`.

#### DEQ-65: non-commuting noise in the Milstein ensemble

Ensemble Milstein step time against paths `P` = 256-16384 (4096 at `m = 16`) for
`m` = 2, 4, 8, 16 channels, commuting and non-commuting noise, best of the steps
(`DEQ-65/before.csv`, `after.csv`, `slopes.csv`). Before, non-commuting noise was
refused. Commuting slopes against `P` 0.73-0.96 before, 0.80-1.11 after; at the
largest `P` the step is 0.35x-0.89x the old time. Non-commuting: 0.81-1.17 against
`P`, and 2.44, 2.77, 2.91 against `m` at `P` = 256, 1024, 4096, as the `m^2` Levy
areas grow. The commuting final states hash equal before and after at all 15 sizes
(a 16-hex-digit SHA-256 prefix). Memory and load not recorded.

#### DEQ-72: parallel atlas build

The full atlas and a 12-unit subset, wall and CPU (user + system with children),
peak RSS per process (`DEQ-72/atlas_build.csv`, `README.txt`); load 33-47 (†).

| Build | Wall | CPU |
|---|---|---|
| before, in-process serial, unpinned BLAS | 3453.6 s | 1994.7 s |
| before, serial, BLAS threads 1 | 5399.7 s | 1835.3 s |
| after, 1 worker | 5564.8 s | 2004.6 s |
| after, 12 workers | 987.5 s (repeat 1142.6 s) | 2173.8 s (2086.7 s) |

The subset went 361.0, 168.7, 92.0, 69.3, 72.3 s (repeat 62.7 s) at 1, 2, 4, 8 and 12
workers. Peak RSS stays 636-644 MB per process. All 62 output files are
byte-identical (SHA-256) across after 12 workers, its repeat, after 1 worker and
before with BLAS pinned to one thread; the unpinned before differs in 4 files at
rounding level.

#### DEQ-94: per-component finite-difference steps

The top Lorenz exponent against horizon `T` = 125-1000 by Jacobian kind, CPU best of
interleaved runs (`DEQ-94/lorenz_lambda1_cpu_interleaved.csv`; the wall-only
`lorenz_lambda1_timing.csv` is noisier), plus the error of every Jacobian entry and
the exponents of 17 tests (no README; load not recorded). CPU slope 1.13 before, 1.07
(finite differences), 1.01 (analytic), 0.97 (jets) after; at `T = 1000` 29.3 s →
29.8 s, 8.7 s and 45.4 s. On the small-scale system the default forward difference
kept 6 of 9 entries within four times its error allowance before and 7 of 9 after,
and 9 of 9 with `typical=scale` (`jacobian_entry_errors.csv`). Results are not
bit-identical: `lambda_1` moves by at most 2.1e-9, and the 129 exponents of the 17
tests by at most 2.5e-7, with 1 test bit-identical
(`lyapunov_exponents_before_after.csv`). Commits `8a76b6a` → `b248cf9`.

#### DEQ-105: a Hermite-Simpson Floquet error bound

`CycleContinuation._point` and a one-step trace on van der Pol (`mu = 1`), Hermite-
Simpson, against `N` = 20-320, two interleaved rounds run beside the full suite (†)
(`DEQ-105/bench.csv`, which has no header: label, case, `N`, point seconds, step
seconds). Point: 1.27 → 0.90, 3.4 → 18.7 ms at `N = 20` and 139 → 242 ms at 320.
Step: 1.05 both ways, 1.9x-2.4x the old time. The README adds film builds 1.8x-3.2x
longer. The point now carries a bound, the larger of the drift and twice the
doubled-mesh gap, so its value differs from before by design; the accuracy evidence
is in `outputs/verification/verify-deq105/`. Commits `e1912a1` → `b1f777b`.
