# SRK4

## Overview

SRK4 is Platen's explicit weak scheme of order 2.0 for scalar Ito SDEs, Kloeden and
Platen (1992) equation (15.1.4): the derivative-free form of the order 2.0 weak
Taylor scheme. It approximates expectations of the solution to second order in the
step; path by path it is the Milstein scheme, strong order 1.0.

The previous version of this file claimed strong order 2.0 and weak order 4.0; the
solver it described drew no random number (DEQ-25). Strong order 2.0 for
multiplicative noise needs iterated integrals with no exact sampler, and weak order
4.0 exists only as an extrapolation of expectations (Kloeden-Platen Section 15.3),
not as a path scheme. The orders below are measured.

## Architecture

```
SRK4Solver
├── Config: start_time, end_time, step_size, random_seed, calculus
├── Base: StochasticRungeKuttaSolver (shared stepping loop, owns the Wiener source)
├── Method: Kloeden-Platen (15.1.4), scalar noise, increment dW
├── Order: Weak 2.0 (measured 1.96), Strong 1.0 (measured 1.05)
├── Systems: refused with a ValueError pointing at SRK2Solver
└── Calculus: Ito, or Stratonovich via the Ito drift a + (1/2) b b'
```

## Core Classes

```python
class SRK4Config(StochasticConfig): ...

class SRK4Solver(StochasticRungeKuttaSolver):
    def __init__(self, solver_config: SRK4Config, wiener: WienerSource | None = None): ...
    def _check_dimension(self, dimension: int) -> None: ...  # raises for dimension != 1

    @staticmethod
    def _step(y, t, h, coefficients, increments) -> np.ndarray:
        """One step of Kloeden-Platen (15.1.4)."""
```

## Folder Structure

```
srk4/
├── __init__.py
├── srk4_config.py
├── srk4_solver.py
└── SRK4.md
```

## Example

```python
from discrecontinual_equations.solver.stochastic.srk4.srk4_config import SRK4Config
from discrecontinual_equations.solver.stochastic.srk4.srk4_solver import SRK4Solver

# Choose SRK4 when the quantity of interest is an expectation E[g(X_T)].
config = SRK4Config(start_time=0.0, end_time=1.0, step_size=0.01, random_seed=42)
solver = SRK4Solver(config)
solver.solve(scalar_equation, [1.0])
```

## Mathematical Foundation

```
Y_bar  = Y + a h + b dW
Y+-    = Y + a h +- b sqrt(h)
Y_next = Y + (a(Y_bar) + a) h / 2
         + (b(Y+) + b(Y-) + 2 b) dW / 4
         + (b(Y+) - b(Y-)) (dW^2 - h) / (4 sqrt(h))
```

Expanded, the first line is `a h + (1/2) a a' h^2 + (1/2) a' b h dW + (1/4) a'' b^2 h
dW^2`, the second is `b dW + (1/2)(a b' + (1/2) b^2 b'') h dW`, and the third is the
Milstein term. Against the order 2.0 weak Taylor scheme (Kloeden-Platen (14.2.1)),
`(1/2) h dW` stands in for `I_(1,0)` and `I_(0,1)`: same mean, same covariance with
`dW`, but variance `h^3 / 4` rather than `h^3 / 3`. No expectation at order `h^2`
sees that difference, which is all weak order 2.0 promises; path by path it is an
error of order `h^{3/2}`, so the strong order stays at 1.0. Two drift and four
diffusion evaluations per step.

## Convergence, as measured

Geometric Brownian motion `dX = X dt + 0.5 X dW`, `X(0) = 1`, on `[0, 1]`, 100000
paths, the same Brownian path at every step size; weak error estimated
path-coupled, standard error in brackets.

| h | strong error | weak error |
|---|---|---|
| 1/8 | 1.679e-02 | 6.388e-03 (1e-04) |
| 1/16 | 7.584e-03 | 1.675e-03 (4e-05) |
| 1/32 | 3.582e-03 | 4.119e-04 (2e-05) |
| 1/64 | 1.751e-03 | 9.584e-05 (9e-06) |
| 1/128 | 8.660e-04 | 2.420e-05 (4e-06) |
| 1/256 | 4.320e-04 | 8.141e-06 (2e-06) |
| **slope** | **1.05** | **1.96** |

## References

- Kloeden, P. E. and Platen, E. (1992). *Numerical Solution of Stochastic
  Differential Equations*. Springer. Section 15.1, equation (15.1.4); Section 14.2,
  equation (14.2.1).
- Talay, D. and Tubaro, L. (1990). Expansion of the global error for numerical
  schemes solving stochastic differential equations. *Stochastic Analysis and
  Applications* 8.

---

**Parent Module:** [STOCHASTIC](../STOCHASTIC.md)

**Related Modules:**
- [SRK3](../srk3/SRK3.md) - strong order 1.5 for paths
- [SRK2](../srk2/SRK2.md) - strong order 1.0, handles systems with commuting noise
