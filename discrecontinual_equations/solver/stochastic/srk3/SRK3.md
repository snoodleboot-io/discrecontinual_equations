# SRK3

## Overview

SRK3 is Platen's explicit strong scheme of order 1.5 for scalar Ito SDEs, Kloeden
and Platen (1992) equation (11.2.1): the order 1.5 strong Ito-Taylor scheme with
every derivative of the drift and diffusion replaced by a difference of supporting
values. It is driven by two Wiener functionals per step, the increment `dW` and its
time integral `dZ`, drawn jointly.

The previous version of this file claimed "order 3 convergence" from a "simplified
Butcher tableau" and strong 1.5 / weak 3.0 further down; the solver it described
drew no random number (DEQ-25). The order below is measured. Systems are refused:
the scheme is derived for one Wiener process, and its several-channel form needs
Levy areas that cannot be sampled.

## Architecture

```
SRK3Solver
├── Config: start_time, end_time, step_size, random_seed, calculus
├── Base: StochasticRungeKuttaSolver (shared stepping loop, owns the Wiener source)
├── Method: Kloeden-Platen (11.2.1), scalar noise, increments (dW, dZ)
├── Order: Strong 1.5 (measured 1.48); weak slope measured 1.98
├── Systems: refused with a ValueError pointing at SRK2Solver
└── Calculus: Ito, or Stratonovich via the Ito drift a + (1/2) b b'
```

## Core Classes

```python
class SRK3Config(StochasticConfig): ...

class SRK3Solver(StochasticRungeKuttaSolver):
    def __init__(self, solver_config: SRK3Config, wiener: WienerSource | None = None): ...
    def _check_dimension(self, dimension: int) -> None: ...  # raises for dimension != 1

    @staticmethod
    def _step(y, t, h, coefficients, increments) -> np.ndarray:
        """One step of Kloeden-Platen (11.2.1)."""
```

## Folder Structure

```
srk3/
├── __init__.py
├── srk3_config.py
├── srk3_solver.py
└── SRK3.md
```

## Example

```python
from discrecontinual_equations.solver.stochastic.srk3.srk3_config import SRK3Config
from discrecontinual_equations.solver.stochastic.srk3.srk3_solver import SRK3Solver

config = SRK3Config(start_time=0.0, end_time=1.0, step_size=0.01, random_seed=42)
solver = SRK3Solver(config)
solver.solve(scalar_equation, [1.0])  # one variable; systems raise
```

## Mathematical Foundation

With `dW = W(t+h) - W(t)` and `dZ = integral_t^{t+h} (W(s) - W(t)) ds`, jointly
Gaussian with `Var dZ = h^3 / 3` and `Cov(dW, dZ) = h^2 / 2`:

```
Y+-    = Y + a h +- b sqrt(h)
Phi+-  = Y+ +- b(Y+) sqrt(h)
Y_next = Y + b dW
         + (a(Y+) - a(Y-)) dZ / (2 sqrt(h))
         + (a(Y+) + 2 a + a(Y-)) h / 4
         + (b(Y+) - b(Y-)) (dW^2 - h) / (4 sqrt(h))
         + (b(Y+) - 2 b + b(Y-)) (dW h - dZ) / (2 h)
         + (b(Phi+) - b(Phi-) - b(Y+) + b(Y-)) (dW^2 / 3 - h) dW / (4 h)
```

Line by line these are `a' b dZ`; `a h + (1/2)(a a' + (1/2) b^2 a'') h^2`; the
Milstein term `(1/2) b b' (dW^2 - h)`; `(a b' + (1/2) b^2 b'')(dW h - dZ)`; and
`(1/2) b (b b'' + b'^2)((1/3) dW^2 - h) dW`, the `I_(1,1,1)` term - the terms of
the order 1.5 strong Taylor scheme (Kloeden-Platen (10.4.1)), each reproduced to a
remainder of strong order 2.0. Three drift and six diffusion evaluations per step.

## Convergence, as measured

Geometric Brownian motion `dX = X dt + 0.5 X dW`, `X(0) = 1`, on `[0, 1]`, 100000
paths, the same Brownian path at every step size; weak error estimated
path-coupled, standard error in brackets.

| h | strong error | weak error |
|---|---|---|
| 1/8 | 1.818e-02 | 6.366e-03 (9e-05) |
| 1/16 | 6.609e-03 | 1.664e-03 (3e-05) |
| 1/32 | 2.397e-03 | 4.091e-04 (1e-05) |
| 1/64 | 8.665e-04 | 1.036e-04 (4e-06) |
| 1/128 | 3.088e-04 | 2.618e-05 (1e-06) |
| 1/256 | 1.090e-04 | 6.691e-06 (5e-07) |
| **slope** | **1.48** | **1.98** |

The weak slope is reported as measured; the derivation is a strong one and the
docstring makes no weak-order claim beyond this number.

## References

- Kloeden, P. E. and Platen, E. (1992). *Numerical Solution of Stochastic
  Differential Equations*. Springer. Section 11.2, equation (11.2.1); Section 10.4,
  equation (10.4.1).

---

**Parent Module:** [STOCHASTIC](../STOCHASTIC.md)

**Related Modules:**
- [SRK2](../srk2/SRK2.md) - strong order 1.0, handles systems with commuting noise
- [SRK4](../srk4/SRK4.md) - weak order 2.0
