# Complexity

This page records the time and space cost of every substantive algorithm in the
library, derived from the code as written rather than from what the textbook
algorithm would cost, and checked against measured scaling. Where the code is
asymptotically worse than its algorithm needs to be, that is called out under
[Findings](#findings) with the evidence; none of those findings has been fixed yet.

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
the numbers quoted here are their output.

**Unmerged code.** Three sections describe algorithms on branches that had not been
merged when this page was written. They are marked with a note naming the branch and
the commit measured.

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

Every solver writes its solution into a `Curve` that shares the equation's
`Variable` objects, so the trajectory is appended to the equation itself. A second
`solve` on the same equation appends a second trajectory to the first: measured
`len(solution)` 1001, 2002, 3003 over three solves of a 1000-step problem. Memory
therefore grows with the number of solves, not just with `K`.

## SDE solvers

### Per-path schemes

| Algorithm | Size parameters | Time per step | Space | Dominated by | Measured slope |
|---|---|---|---|---|---|
| `EulerMaruyamaSolver` | `K`, `n`, `m` | `O(C_f + C_g + n m)` **plus `O(K)`** (see below) | `O(K n)` | Python overhead; the `O(K)` term from about `K = 10^4` | vs `K`: 1.20 overall, **1.85 local** (`K` 2000-64000); vs `n`: 1.20 local; memory 1.00 |
| `MilsteinSolver`, diagonal noise (`m = n`) | `n`, `K` | `2n C_g` (central differences) + `O(n^4)` (`einsum` over `(m, n, n)` Jacobians against an `n x n` diagonal) **plus `O(K)`** | `O(n^3)` | the products and commutativity check | vs `n` (`K = 200`): 1.36 overall, 3.44 local (`n` 32 to 64); memory 1.95 |
| `MilsteinSolver`, matrix noise | `n`, `m` | `2n C_g + O(m^2 n^2)` **plus `O(K)`** | `O(m n^2)` | as above | (as above) |
| `SRK2Solver` (strong 1.0) | `n`, `K` | `C_f + (n + 1) C_g`: one supporting diffusion evaluation per channel | `O(K n)` | the `n` supporting evaluations, each returning all `n` components | vs `n`: 1.39 overall, 2.22 local (`C_g = O(n)`) |
| `SRK3Solver`, `SRK4Solver` (scalar only) | `K` | a fixed number of drift and diffusion calls | `O(K)` | Python overhead | linear in `K` (same driver as SRK2) |
| `GottwaldMelbourneSolver` | `K`, `E = dt / epsilon` | `O(E (C_f + C_g + n))` | `O(K n)` | the inner fast-map loop | not measured; `E` multiplies everything |
| `ZhanDuanLiLiSolver` | `K`, `n` | `2 (C_f + C_g)` | `O(K n)` | Python overhead | not measured |

**The `O(K)` per step.** `FixedStepConfig.times` is a property that rebuilds
`np.arange(start, end + h, h)` on every access. The Euler-Maruyama, Milstein,
Gottwald-Melbourne and Zhan-Duan-Li-Li solvers and the `srk*` driver
(`runge_kutta.py`) index it twice per step (`times[i]`, `times[i + 1]`), so each step
pays two `O(K)` allocations and a whole path is `O(K^2)`. One access costs 1.5 us at
`K = 2000` and 55 us at `K = 128000`. Euler-Maruyama at `n = 2` costs 27 us per step
at `K = 2000` and 63 us per step at `K = 64000`, and the difference is the two
`arange` calls. The ODE solvers and the ensemble read the array once and are linear.

**Milstein with diagonal noise.** `column_jacobians` stores `(m, n, n)` Jacobians and
`_commuting_products` contracts them with `np.diag(b)` by
`einsum("kil,jl->jki")`, an `O(m^2 n^2) = O(n^4)` loop for diagonal noise, where
only the `n` entries `(d b_i / d x_i) b_i` are non-zero and the commutativity check
is vacuous.

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

A convergence study that reads a path at step `h` therefore costs `O(C m)` in
increments per path whatever `h`, which is the point of a fixed path.

### Ensembles (`StochasticEnsemble`)

| Algorithm | Size parameters | Time per step | Space | Dominated by | Measured slope |
|---|---|---|---|---|---|
| Euler-Maruyama | `P`, `n`, `m`, `K` | `O(P n m)` vectorised + the batch drift and diffusion | `O(P n)` + `O(S P n)` for `S` kept samples | NumPy calls (about 60 us per step at `P = 1000`) until `P` is about `10^4` | vs `P`: 0.85 local (100 to 100000); vs `K`: 0.98; memory 0.89 vs `P` |
| Milstein (and Stratonovich drift) | `P`, `n`, `m` | `2n` batched diffusion calls + `O(P m^2 n^2)` products | `O(P m n^2)` Jacobians + `O(P m^2 n)` products | the `(P, m, n, n)` Jacobian array | diagonal noise vs `n` (`P = 200`, `K = 50`): time 3.17 (3.68 local), memory 2.94; 12.5 s and 212 MB at `n = 32` |
| with a `WienerSource` | as above | `+ P` source calls per step (a Python list comprehension) | | the source | - |

`times` is read once into a local, so the ensemble is linear in `K`. The default
`eval_batch` and `diffusion_batch` loop over the paths in Python, so an equation that
does not override them pays `O(P (C_f + C_g))` Python calls per step and loses the
point of the ensemble. With diagonal noise the Milstein ensemble builds the full
`(P, n, n, n)` Jacobian tensor and an `(P, n, n, n)` product tensor; the Stratonovich
Euler-Maruyama drift builds the same Jacobian tensor to read its `n` diagonal entries.

## Equilibrium continuation

`Continuer.solve` runs pseudo-arclength continuation with the collaborators the
builder injects. With the defaults (central-difference Jacobian, dense
`LeastSquaresFallbackSolver`, detectors fold, branch point and Hopf):

| Part | Size parameters | Time | Space | Notes |
|---|---|---|---|---|
| `NewtonCorrector.correct`, per iteration | `n` | `(2n + 2) C_f` (extended Jacobian) + dense `(n+1)^2` solve `O(n^3)`; `lstsq` `O(n^3)` on singular steps | `O(n^2)` | the Jacobian is dense whatever the field's sparsity |
| `BorderedTangentComputer.compute` | `n` | `(2n + 2) C_f + O(n^3)` | `O(n^2)` | |
| `Fold.test_value` | | `O(1)` (the tangent's last entry) | | |
| `BranchPoint.test_value` | `n` | `(2n + 2) C_f + O(n^3)` (determinant of the bordered matrix) | `O(n^2)` | |
| `Hopf.test_value` | `n` | `2n C_f` + `O(n^4)` Python build + `O(n^6)` determinant | `O(n^4)` | see `bialternate_determinant` below |
| `PointBuilder.build` | `n` | `2n C_f + O(n^3)` (`eigvals`) | `O(n^2)` | |
| `RefiningLocalizer.localize` | root-finder iterations `L` | `L` corrector calls + `L` tangents + `L` test values | `O(n^2)` | only when a test function changes sign |
| **One accepted point** | `n`, `I` | `(I + 4)(2n + 2) C_f + (I + 3) O(n^3)` + Hopf | `O(n^2)` + `O(n)` stored | none of the `I + 4` Jacobians is shared |

Measured on a ring of `n` fold-carrying cells (`C_f = O(n)`), 20 points: with only
the fold detector the run grows with slope 0.98 (1.28 local) for `n` 4 to 32, as the
`O(n C_f) = O(n^2)` Jacobians take over from fixed overhead; with the default three
detectors the slope is 1.84 overall and **3.68 local**, and at `n = 32` the run costs
1.19 s against 0.12 s, almost all of it the Hopf test. The bialternate determinant
also overflowed during that run.

`bialternate_determinant` builds the `s x s` bialternate product, `s = n(n-1)/2`, by
calling `_offdiagonal` for every one of the `s^2` entries: `O(n^4)` Python calls and
an `O(n^4)` dense array, of which only `O(n^3)` entries are non-zero, then a dense
`O(s^3) = O(n^6)` determinant. Measured slope 4.01 in time and 4.10 in memory for
`n` 8 to 48 (0.48 s and 20 MB at `n = 48`, against 0.7 ms for `eigvals` at the same
size); the `n^6` determinant takes over beyond that. The determinant of a product
with entries of order one also overflows for moderate `n` (it overflowed at
`n = 32` on a random matrix), and an infinite test value cannot change sign.

## Limit cycles

### Periodic orbits (`periodic_orbit.py`)

| Algorithm | Size parameters | Time per Newton iteration | Space | Dominated by | Measured slope |
|---|---|---|---|---|---|
| `PeriodicOrbit.solve` (trapezoidal, forward-difference Jacobian, `lstsq`) | `N`, `n` | `((N+1) n + 1)` full residuals of `(N + 1) C_f` = `O(N^2 n C_f)`, plus an SVD least squares `O((N n)^3)` | `O(N^2 n^2)` dense Jacobian | the `O(N^2)` field calls up to `N n` in the low thousands; the cubic solve beyond | time 2.01 (2.24 local), memory 1.61 vs `N` (20-160, `n = 2`); 0.80 s at `N = 160` |
| `HermiteSimpsonOrbit.solve`, `RobustPeriodicOrbit.solve` | `N`, `n` | as above with `2N + 1` field calls per residual; Robust adds up to 20 residuals per line search | `O(N^2 n^2)` | as above | same driver |
| `AnalyticPeriodicOrbit.solve` (autodiff blocks, COO assembly, `spsolve`) | `N`, `n` | `N + 1` autodiff Jacobians `O(N n C_f)` + vectorised assembly of `O(N n^2)` triplets + SuperLU | `O(N n^2)` | the Python autodiff passes | vs `N`: time 1.14 (100-3200), memory 0.97; vs `n` (`N = 100`): time 1.67, memory 1.82 |
| `AdaptivePeriodicOrbit.solve` | `N`, sweeps `W <= 15`, line-search steps `B <= 20` | `(1 + W)` damped solves, each `I` iterations of the above plus up to `B` residuals | `O(N n^2)` | the re-solves | linear in `N` per solve |
| `AdaptivePeriodicOrbit.continue_to_resolved` | doublings `D` | `D + 1` full continuations at `N, 2N, ..., 2^D N`, so about twice the finest one | finest level | the finest continuation | - |
| `PeriodicOrbit.monodromy` | `N`, `n` | per interval 9 forward-difference Jacobians (`9 (n + 1) C_f`) and 4 RK4 substeps of four `n x n` products: `O(N (n C_f + n^3))` | `O(n^2)` (plus the `9 n x n` Jacobians of one interval) | the Jacobians at small `n`, the products at large `n` | vs `N`: 1.01; vs `n` (`N = 40`): 1.29 overall, 2.04 local; 1.2 s at `n = 64` |
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
| `_solve_seed` | `N`, `n` | one `PeriodicOrbit.solve` (dense, `O(N^2 n C_f)` per iteration and `O((N n)^3)` least squares), then a sparse correction | `O(N^2 n^2)` | the dense trapezoidal solve | trapezoidal 1.81, Hermite-Simpson 1.63 vs `N` (20-160); 1.8 s at `N = 160` |
| corrector iteration (`_correct`) | `N`, `n` | `(N + 1)` autodiff Jacobians + `(N + 1)` parameter differences (+ `N` midpoint ones for Hermite-Simpson) + one `spsolve` on the bordered matrix | `O(N n^2)` | the per-node Python calls | (in the per-step figure) |
| tangent (`_tangent`) | `N`, `n` | one more assembly and `spsolve` | `O(N n^2)` | | |
| point (`_point`): multipliers | `N`, `n` | one `PeriodicOrbit.monodromy`, `O(N (n C_f + n^3))` + `eigvals` | `O(n^2)` | the monodromy's 9N forward-difference Jacobians | |
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

The step is linear in `N`, but the *seed* is not: the first point is solved by the
dense forward-difference `PeriodicOrbit`, so a short trace at large `N` is dominated
by its seed (at `N = 160`, trapezoidal: seed 1.8 s against 0.15 s per step).

### Gauss-Legendre collocation

!!! note "On branch feature/deq-34-orthogonal-collocation, not yet merged"
    `continuation/cycle_continuation.py` at `4e6b463`. Measured on that commit with
    `m = 4` points per interval unless stated.

Orthogonal collocation at `m` Gauss points per interval carries `N m + 1` states.
Each interval contributes a dense `(m n) x ((m + 1) n)` block, so the bordered matrix
has `O(N m^2 n^2)` non-zeros.

| Part | Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|---|
| `_solve_seed` | `N`, `n` | as on `main` (dense trapezoidal on `N + 1` nodes) + Hermite fill of the interior points + sparse correction | `O(N^2 n^2)` | the dense trapezoidal solve | 1.48 vs `N` (10-160), 1.29 vs `n` |
| `interval_blocks` (assembly) | `N`, `m`, `n` | `N m` autodiff Jacobians, field and parameter calls + `einsum` chain `O(N m^2 n^2)` | `O(N m^2 n^2)` | the `N m` Python calls | 0.94 vs `N`, 0.90 vs `m`, 1.39 vs `n` |
| `spsolve` | `N`, `m`, `n` | banded LU, block bandwidth `(m + 1) n`: `O(N m^3 n^3)` worst case | `O(N m^2 n^2)` | | 0.67 vs `N` (fill 1.24, constant), 0.55 vs `m`, 1.69 vs `n` |
| `_jacobian_monodromy` | `N`, `m`, `n` | `N` batched dense solves of size `m n` with `n` right-hand sides, `O(N m^3 n^3)`, + `N` products `O(N n^3)` | `O(N m^2 n^2)` (the stored blocks) | NumPy batched `solve` | 0.45 vs `N`, 1.50 vs `n`; under 1 ms at `N = 160` |
| `_point` (Jacobian multipliers + Hermite cross-check + matching) | `N`, `m`, `n` | the cross-check is a full `monodromy` over all `N m` sub-intervals with autodiff Jacobians, `O(N m (n C_f + n^3))`, + `linear_sum_assignment` `O(n^3)` | `O(n^2)` | the cross-check monodromy | 0.96 vs `N`, 0.98 vs `m`, 1.42 vs `n` |
| **One continuation step** | `N`, `m`, `n` | `O(I N m n C_f + N m^3 n^3)` | `O(N m^2 n^2)` | the cross-check: about 60% of a step | 0.99 vs `N`, 0.97 vs `m`, 1.44 vs `n` (`n` 2-16) |

The Jacobian monodromy (Fairgrieve-Jepson) is essentially free; what a
Gauss-Legendre step pays for is the independent Hermite cross-check, which the code
evaluates eagerly at every point to fill `floquet_error`. At `N = 160`, `m = 4` the
cross-check took 0.58 s of a 1.0 s step, against 1 ms for the Jacobian monodromy.

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
(autodiff) Jacobian and `lsmr` on it.

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `N`, `n`, `I` | per iteration: `2N C_f` (residual) + `(N+1) n C_f` (autodiff Jacobians) + `O(N^2 n^2)` (dense zero-fill and `csr_matrix(dense)`) + `L * O(N n^2)` for `L` `lsmr` iterations, with `L ~ N` measured, so `O(N^2 n^2)` overall | `O(N^2 n^2)` (dense Jacobian) | `lsmr` iterations and the dense array | time 1.13 (`N` 100-1600, `n = 2`); memory 1.98 |

Derivation and measurements (saddle `x'' = x - x^2`, half-length 10, 4 Gauss-Newton
iterations at every `N`):

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
parameter, with a forward-difference Jacobian of the whole augmented residual and a
dense `lstsq`.

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `N`, `n`, `I` | per iteration: `(N+1) n + n + 2` full residuals, each `2N C_f + (n+1) C_f + O(n^3)` (FD saddle Jacobian, `eig`, `inv`) = `O(N^2 n C_f)`, plus dense SVD `lstsq` `O((N n)^3)` | `O(N^2 n^2)` | `O(N^2)` field evaluations in Python; the cubic `lstsq` takes over only at large `N n` | 1.80 per point (`N` 100-300, `n = 2`; 1.92 per Newton iteration, since the iteration count drops from 4 to 3); memory 1.83 |
| `P` sweep values | `trace`: `P` point solves | `O(N^2 n^2)` | | |

At `N = 300` one Newton iteration is about 600 residuals, 360,000 field evaluations,
about 4 s. This is the dense finite-difference construction that
`connecting_orbit.py` replaced with the banded autodiff Jacobian; the curve solver
never got the same treatment.

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
| Cusp, codim-3 (`2n+2`) | `C_f` + jet pass + `cusp_coefficient` (Jacobian `n C_f`, 2 SVD `O(n^3)`): `O(n C_f + n^3)` | `(4n+6)` residuals: `O(n^2 C_f + n^4)` | `O(n^2)` | the cusp coefficient re-derived inside every FD column | 2.11 / 3.28 (53 s per iteration at `n = 32`) |
| Generalized Hopf, codim-3 (`3n+3`) | `C_f` + two jet passes + `lyapunov_quantities` (the order-6 center-manifold reduction above, `O(n^4)` for a cubic field) | `(6n+8)` residuals: `O(n^5)` | `O(n^4)` | center-manifold reduction inside every FD column | 3.44 / 3.93 (`n` 3-6; 9 s per iteration at `n = 6`) |

A continuation step costs `(I + 1)` such Jacobians: `I` corrector iterations and
one tangent solve, none shared.

**Fold and Hopf curves.** Both the measured slopes and the derivation give
`O(n C_f + n^3)` per Newton iteration, which is what a dense Moore-Spence system
needs. The outer Jacobian is a central difference of an inner autodiff directional
derivative; it costs `4n+4` (fold) or `6n+6` (Hopf) residuals where an assembled
Jacobian (`f_u`, the Hessian contraction `(f_u q)_u`, `f_p`) would cost about `n`
jet passes. That is a constant factor, not an order.

**Cusp curve.** `CuspCurveResidual.evaluate` calls `cusp_coefficient`, which
rebuilds the Jacobian and takes two SVDs to find the null vectors `q` and `p` -
although `q` is already one of the unknowns (`state[n:2n]`). The finite-difference
Jacobian then repeats that `4n+6` times, so one Newton iteration is `O(n^2 C_f + n^4)`
against the `O(n C_f + n^3)` the fold curve pays. (The single-residual slope, 2.11,
is under the derived 3 because at `n <= 32` the per-jet Python overhead of the
Jacobian's `n` passes outweighs its `O(n^2)` field cost and the SVDs are
microseconds; the Newton-iteration slope, 3.3, shows the extra factor of `n`.)

**Generalized-Hopf curve.** `GeneralizedHopfCurveResidual.evaluate` calls
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
| `M`, `d`, `K` | build `O(M K C_g + d M (C_f + K C_g))` Python calls, `O(d^2 M)` vectorised assembly; solve: SuperLU (COLAMD) on a 5-point (2-D) / 7-point (3-D) operator plus one dense row | operator `O(d^2 M)`; LU factor `O(M^1.23)` (2-D), `O(M^1.62)` (3-D) measured | 2-D: the Python face loop (87-93% of runtime up to `M = 58 081`); 3-D: the sparse LU from about `M = 10^4` | 2-D total 1.00, solve 1.18, fill 1.23; 3-D total 1.25 (rising), solve 1.92, fill 1.62; `tracemalloc` peak 1.00 / 1.02 |

The assembly is two things of different cost. The coefficients are gathered by a
Python loop: `_diffusion` evaluates all `K` columns at each of the `M` cells, and
`_face_coefficients` evaluates the whole drift and all `K` columns at each of the
roughly `d M` face centres (plus `K (d + 1)` column evaluations per face for the
Stratonovich shift, through `jacobians_at`). Everything after that - the
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
| `n`, `k`, `H`, `T` | `O((T + H) (n C_f + n^2 k))` plus `O(H n k^2 / 10)` for the QR every tenth step | `O(n k)` frame, `O(n^2)` Jacobian | the `4(n + 2)` field evaluations per RK4 step | `H`: 0.95; `n` (ring, `C_f ~ n`): 1.31 overall, 1.71 local, toward 2; memory 1.45 |

Each RK4 step evaluates the field at the four stage points and a forward-difference
Jacobian at the same four points; `jacobian_at` re-evaluates its base point, so the
step costs `4(n + 2)` field evaluations and four `(n x n) @ (n x k)` products. The
transient loop runs the same step, Jacobians and frame included, and throws the frame
away, so the `T` transient steps cost `n + 2` times what advancing the state alone
would.

### Benettin spectrum with one shared driver (`stochastic_lyapunov`)

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `n`, `k`, `H`, `T` | `O((T + H) (n (C_f + C_g) + n^3 + n^2 k))` | `O(n^2)` | `2(n + 2)` evaluations per Euler-Maruyama step; the `n^3` term from `g_x @ g_x` once `n` is large | `n`: 1.17 overall, 1.53 local |

`_Coefficients` takes both forward-difference Jacobians at every step, transient
included, and `variational_drift` forms `g_x @ g_x` (`O(n^3)`) before multiplying it
by the convention factor, which is zero under Ito.

### Benettin spectrum with `K` drivers (`matrix_noise_lyapunov`)

| Size parameters | Time | Space | Dominated by | Measured slope |
|---|---|---|---|---|
| `n`, `K`, `k`, `H`, `T` | `O((T + H) (K n C_g + n C_f + K n^3 + K n^2 k))` | `O(K n^2)` column Jacobians | the `K (n + 1)` column evaluations; then the two `K n^3`-class contractions | `K` (`n = 8`): 0.78; `n` with `K = n`: 1.97 overall, 2.42 local (toward 3 for `C_g ~ n`); memory 2.36 |

Per step `jacobians_at` takes `K (n + 1)` column evaluations and stores a
`(K, n, n)` array. Two contractions follow. `variational_drift_shift`,
`sum_k (dG_k/dx)^2`, is `O(K n^3)` and is computed under Ito as well, where its
factor is zero. The tangent noise term `einsum("k,kij,jl->il", dW, J, Y)` is a
three-operand `einsum` without `optimize`, which NumPy evaluates as one nested loop
over all four indices: `O(K n^2 k)` with no BLAS. Contracting the increments first,
`tensordot(dW, J, 1) @ Y`, is `O(K n^2 + n^2 k)`; at `n = K = 64` that is 0.21 ms
against 18.5 ms per step, and the `einsum` is about a fifth of the whole run.

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

The ring's Jacobian has two nonzeros per row, but every consumer in the library
takes it as a dense forward-difference or autodiff Jacobian, `(n + 1)` evaluations
and `O(n^2)` memory, and solves with it densely.


## Unmerged algorithms

### Quasipotential (ordered line integral method and minimum action paths)

!!! note "On branch feature/deq-31-quasipotential, not yet merged"
    `continuation/quasipotential.py` at `eea3fab`. The numbers below were
    measured on that commit and will move if the branch changes before merge.

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

!!! note "On branch feature/deq-30-geodesic-level-sets, not yet merged"
    `continuation/geodesic_level_sets.py` at `e363836`.

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

Places where the implementation costs more, in order, than its algorithm needs.
Each is a candidate follow-up ticket; none is fixed here.

| # | Where | Cost now | Cost needed | Evidence | Suggested fix |
|---|---|---|---|---|---|
| 1 | `FixedStepConfig.times`, read twice per step by `EulerMaruyamaSolver`, `MilsteinSolver`, `GottwaldMelbourneSolver`, `ZhanDuanLiLiSolver` and the `srk*` driver | `O(K^2)` per path | `O(K)` | Euler-Maruyama local slope 1.85 in `K`; one `times` access 55 us at `K = 128000`; 4.0 s for 64000 steps | read `times` once before the loop, as `StochasticEnsemble.solve` already does, or cache it on the config |
| 2 | `bialternate_determinant`, the default Hopf detector | `O(n^4)` Python calls and memory, `O(n^6)` determinant, per point | `O(n^3)` | slope 4.0 in time and memory; 0.48 s and 20 MB at `n = 48` against 0.7 ms for `eigvals`; overflow at `n = 32` | build the `O(n^3)` non-zeros with vectorised indices, or test the signed product of `lambda_i + lambda_j` from one `eigvals` |
| 3 | `MilsteinSolver` and the Milstein `StochasticEnsemble` with diagonal noise; the Stratonovich ensemble drift | `O(n^4)` time and `O(n^3)` memory per step (times `P` in the ensemble) | `O(n C_g)`: only `(d b_i / d x_i) b_i` is non-zero | per-path local slope 3.44; ensemble 3.17 time, 2.94 memory, 12.5 s and 212 MB at `n = 32`, `P = 200`, 50 steps | a diagonal-noise path that differentiates `b_i` along `x_i` only and skips the commutativity check, which diagonal noise passes trivially |
| 4 | `CycleContinuation._solve_seed` (also on `deq-34`) | dense forward-difference `PeriodicOrbit.solve`: `O(N^2 n C_f)` per iteration, `O((N n)^3)` least squares, `O(N^2 n^2)` memory | `O(N n C_f)` with the sparse trapezoidal Jacobian | seed slope 1.81 against 1.05 per step; seed 1.8 s against 0.15 s per step at `N = 160` | seed with `AnalyticPeriodicOrbit`, keeping the dense solver as the fallback for fields autodiff rejects |
| 5 | `PeriodicOrbit`, `HermiteSimpsonOrbit`, `RobustPeriodicOrbit` | as in 4 | `O(N)` per iteration | slope 2.01 in time, 1.61 in memory for `N` 20-160 | `AnalyticPeriodicOrbit` already does this; give the Hermite-Simpson and robust variants the same sparse assembly |
| 6 | `ConnectingOrbit._analytic_jacobian` | dense `zeros((rows, N n))` then `csr_matrix(dense)`: `O(N^2 n^2)` memory; `lsmr` needs about `N` iterations, `O(N^2 n^2)` time | `O(N n^2)` memory, about `O(N n^3)` with a sparse direct solve | memory slope 1.98, 165 MB at `N = 1600`, `n = 2` (2 GB at `n = 10`) | build COO triplets as `AnalyticPeriodicOrbit` does and solve with SuperLU (augmented system or sparse QR) |
| 7 | `HomoclinicCurve` | dense forward-difference Newton: `O(N^2 n C_f)` field calls + `O((N n)^3)` per iteration | `O(N n C_f)` | slope 1.92 per iteration; 4 s per iteration at `N = 300` | the banded autodiff assembly of `connecting_orbit` |
| 8 | `CuspCurveResidual` | both null vectors re-derived (Jacobian + 2 SVD) inside every finite-difference column: `O(n^2 C_f + n^4)` per Newton iteration | `O(n C_f + n^3)` | slope 3.28; 53 s per iteration at `n = 32` against 2.9 s for the fold curve | use `q` from the unknowns and carry `p` as unknowns, or one bordered solve |
| 9 | `GeneralizedHopfCurveResidual` | a full order-6 center-manifold reduction in every finite-difference column: `O(n^5)` per Newton iteration | `O(n C_f + n^3)` | slope 3.93; 9 s per iteration at `n = 6` | `normal_form.first_lyapunov` with the eigenvectors in the unknowns (0.06 s against 0.79 s at `n = 8`) |
| 10 | `matrix_noise_lyapunov` tangent-noise term | three-operand `einsum` without `optimize`: `O(K n^2 k)` with no BLAS | `O(K n^2 + n^2 k)` | 18.5 ms against 0.21 ms per step at `n = K = 64` | `np.tensordot(dW, J, 1) @ Y` (agrees to 4e-12) |
| 11 | `stochastic_lyapunov`, `matrix_noise_lyapunov` under Ito | an `O(K n^3)` drift shift computed and multiplied by zero every step | nothing | 5.9 ms per step at `n = K = 64` | skip it when the convention factor is zero |

Constant factors, not orders:

* No Jacobian is shared within an equilibrium-continuation step: the corrector's last
  iteration, the tangent, the branch-point and Hopf detectors and the point's
  stability each rebuild the same dense finite-difference Jacobian, `I + 4` per
  point where `I + 1` would do.
* Every Jacobian in the library is dense, including for fields with `O(n)` non-zeros
  (the coupled rings have two per row).
* The Benettin transients do the full variational work and discard it; the
  Fokker-Planck assembly is a Python loop per face (87-93% of 2-D runtime); Taylor
  series products visit every term pair, including those above the order cap.
* Gauss-Legendre (`deq-34`) evaluates the Hermite cross-check monodromy eagerly at
  every point, about 60% of a step; computing it on first use of `floquet_error`
  would roughly halve a trace that never asks.
* Geodesic level sets (`deq-30`) discard a whole ring when one landing fails, which
  makes the cost superlinear in the ring size once the spacing is fine.
* `scene_builder._resample_by_arclength` and `_resample_by_measure` scan the branch
  for every sample, `O(samples x points)`; `np.interp` is linear. Both sizes are in
  the hundreds, so a build does not notice.

Found on the way, not a cost item: a solver appends its trajectory to the equation's
own variables, so solving the same equation twice returns a curve holding both runs
(see [ODE solvers](#ode-solvers)).
