# SRK5

## Overview

SRK5 refuses to run. `SRK5Solver.solve` raises `NotImplementedError` with a message
naming the solvers that do have a measured order.

The previous version of this file described a five-stage scheme of "strong order 2.5
and weak order 5.0". The solver drew no random number: its noise terms were weighted
by `sqrt(dt)` in place of a Wiener increment, so it integrated the ordinary
differential equation `dX = (a + b / sqrt(h)) dt`, a drift with a deterministic bias
that grows relative to the drift as the step shrinks (DEQ-25). The other three
`srk*` solvers were rebuilt from derivations in Kloeden and Platen and measured; for
this one there was nothing defensible to build.

## Why there is nothing to build

- For a scalar SDE with multiplicative noise the strong order is set by which
  iterated Ito integrals a scheme carries with their exact joint law. Order 1.0
  needs `dW` and `I_(1,1)` (SRK2); order 1.5 adds `I_(1,0)` and `I_(1,1,1)` (SRK3);
  order 2.0 adds `I_(1,1,0)`, `I_(1,0,1)`, `I_(0,1,1)` and `I_(1,1,1,1)`, whose mixed
  members have no closed-form sampler; order 2.5 adds another layer. Strong orders
  of 2.0 and above are written down only for additive noise, and this library's
  interface cannot tell additive from multiplicative noise before the fact.
- Weak order 5.0 is not a scheme for a general SDE. Kloeden and Platen reach high
  weak orders only by Richardson extrapolation of expectations across step sizes
  (Section 15.3), which yields a number rather than a path and does not fit a solver
  that returns a trajectory.
- A stage count says nothing about stochastic order: the deterministic order of a
  Runge-Kutta tableau does not survive the addition of noise, since the noise terms
  carry their own order conditions (Burrage and Burrage 1996; Roessler 2010).

## What would be needed

Either a strong order 2.0 scheme for additive noise with `(dW, I_(1,0), I_(1,0,0))`
drawn jointly and a check that the diffusion is constant, measured against the
Ornstein-Uhlenbeck process; or a weak order 3.0 scheme for additive noise. Each is a
derivation with its own convergence test. Until one is done, the solver raises,
because an `srk5` that silently ran at an unknown order would be the present defect
in a harder-to-notice form.

## Architecture

```
SRK5Solver
├── Config: start_time, end_time, step_size, random_seed, calculus (constructible)
└── solve(): raises NotImplementedError pointing at SRK3Solver and SRK4Solver
```

## Folder Structure

```
srk5/
├── __init__.py
├── srk5_config.py
├── srk5_solver.py
└── SRK5.md
```

## References

- Kloeden, P. E. and Platen, E. (1992). *Numerical Solution of Stochastic
  Differential Equations*. Springer. Chapters 10, 11, 14 and 15.
- Burrage, K. and Burrage, P. M. (1996). High strong order explicit Runge-Kutta
  methods for stochastic ordinary differential equations. *Applied Numerical
  Mathematics* 22.
- Roessler, A. (2010). Runge-Kutta methods for the strong approximation of
  solutions of stochastic differential equations. *SIAM Journal on Numerical
  Analysis* 48.

---

**Parent Module:** [STOCHASTIC](../STOCHASTIC.md)

**Related Modules:**
- [SRK3](../srk3/SRK3.md) - highest measured strong order (1.5, scalar noise)
- [SRK4](../srk4/SRK4.md) - highest measured weak order (2.0, scalar noise)
