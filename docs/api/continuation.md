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
    level.intervals   # node count tried
    level.period      # None if the continuation stalled here
    level.agreement   # relative shift from the previous level that arrived

if study.resolved is not None:
    study.resolved.intervals   # certified node count
    study.resolved.estimate    # the agreement that certified it
else:
    study.best                 # finest level that arrived, uncertified
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
    ...                             # trustworthy to about that much
elif resolution.limited_by == "spacing":
    ...                             # halve h: divides discretisation error by 4
else:
    ...                             # extend T by ln(4) / (2 * lambda) for the same
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

## Continuation

::: discrecontinual_equations.continuation

## Periodic orbits

::: discrecontinual_equations.continuation.periodic_orbit

## Connecting orbits

::: discrecontinual_equations.continuation.connecting_orbit
