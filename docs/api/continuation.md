# Continuation API

Numerical continuation, bifurcation detection, periodic and connecting orbits,
and stochastic analysis.

## Trusting a computed cycle

A solver returning a `PeriodicOrbitSolution` means the *discrete* collocation system
was satisfied to tolerance. It does **not** mean the mesh resolves the orbit. Those
are different claims, and on a stiff cycle they come apart: continuing van der Pol
to `mu = 40` on 400 nodes reaches the target, returns a solution, and reports a
period 17% wrong — with no error, no warning, and a residual well inside tolerance.

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

The estimate is one comparison between two meshes, so it is a screen, not a bound:
2.5x conservative at `mu = 40`, but 7% optimistic at `mu = 12`. It costs one extra solve at double the nodes: roughly a fifth of a
continuation on a resolved cycle, and far less on an unresolved one, which fails to
refine almost immediately.

As a rough guide, a relaxation oscillation needs `N ~ 20 to 25 x mu` for ~1e-3
accuracy. Use the estimate rather than the guide where it matters.

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
once **two consecutive** levels agree to the tolerance. One agreement is not enough:
before convergence is asymptotic, two meshes can agree while both are still wrong.
Measured on van der Pol at `mu = 12`:

| N | period | agreement | true error |
|---|---|---|---|
| 200 | 22.210033 | — | 2.34e-3 |
| 400 | 22.185012 | 1.13e-3 | 1.21e-3 |
| 800 | 22.160267 | 1.12e-3 | 9.23e-5 |

The 200/400 agreement (1.13e-3) is *smaller* than the 400-node error (1.21e-3), so
certifying 400 on that alone would have reported an estimate the answer did not meet.
Note also that agreement stays flat while the true error falls 25x: in this regime
agreement bounds the coarser mesh, not the finer one.

Tune the search with `ResolutionSettings(tolerance=..., doublings=...)`.

It doubles the *mesh* and re-continues rather than refining the answer in hand,
because neither shortcut works on a stiff cycle: an under-resolved solution fails
outright when interpolated onto a finer mesh, and solving cold at higher resolution
lands on a different cycle (period 91.6 against a true 66.5).

That makes it expensive — one full continuation per resolution. Use
`estimate_period_error` to screen a cycle already in hand for gross error, and
`continue_to_resolved` when the node count is not known or the number matters.

## Continuation

::: discrecontinual_equations.continuation

## Periodic orbits

::: discrecontinual_equations.continuation.periodic_orbit

## Connecting orbits

::: discrecontinual_equations.continuation.connecting_orbit
