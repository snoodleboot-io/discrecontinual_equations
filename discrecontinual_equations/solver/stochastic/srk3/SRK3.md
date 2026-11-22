# SRK3

## Overview

SRK3 is Platen's explicit strong scheme of order 1.5 for scalar Ito SDEs, Kloeden
and Platen (1992) equation (11.2.1): the order 1.5 strong Ito-Taylor scheme with
every derivative of the drift and diffusion replaced by a difference of supporting
values. It is driven by two Wiener functionals per step, the increment `dW` and its
time integral `dZ`, drawn jointly.

The previous version of this file claimed "order 3 convergence" from a "simplified
Butcher tableau" and strong 1.5 / weak 3.0 further down; the solver it described
drew no random number (DEQ-25). The orders below are measured.

Since DEQ-66 it takes several noise channels - an `n x m` matrix diffusion, or a
vector of `n > 1` coefficients (diagonal noise) - with the several-channel form of
the same scheme: one pair of supporting values per channel, one pair per ordered
pair of channels, and the twofold and threefold integrals `I_(j,k)`, `I_(j,k,l)`
drawn jointly by Kuznetsov's Legendre series (`iterated_integrals.py`). A single
variable with vector noise still runs (11.2.1)'s own code, number for number.

## Architecture

```
SRK3Solver
├── Config: start_time, end_time, step_size, random_seed, calculus
├── Base: StochasticRungeKuttaSolver (shared stepping loop, owns the Wiener source)
├── Method: Kloeden-Platen (11.2.1), scalar noise, increments (dW, dZ)
├── Several channels: the derivative-free order 1.5 Taylor scheme, with I_(j,k), I_(j,k,l)
├── Order: Strong 1.5 (measured 1.48 scalar; 1.42 and 1.46 on two channels)
├── Sources: GaussianIteratedIntegralSource (default), IteratedIntegralPath
└── Calculus: Ito, or Stratonovich via the Ito drift a + (1/2) b b'
```

## Core Classes

```python
class SRK3Config(StochasticConfig): ...

class SRK3Solver(StochasticRungeKuttaSolver):
    def __init__(self, solver_config: SRK3Config, wiener: WienerSource | None = None): ...
    def solve(self, equation, initial_values) -> Curve: ...  # one variable: (11.2.1)

    @staticmethod
    def _step(y, t, h, coefficients, increments) -> np.ndarray:
        """One step of Kloeden-Platen (11.2.1)."""
```

## Folder Structure

The integrals live in `../iterated_integrals.py`.


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
solver.solve(scalar_equation, [1.0])  # (11.2.1)
solver.solve(matrix_noise_equation, [0.3, -0.2])  # n x m noise: several channels
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

## Several channels

With `G` the `n x m` diffusion, columns `b^j`, and `I_(0,j) = h dW_j - I_(j,0)`:

```
Ups+-^j       = Y + a h / m +- b^j sqrt(h)          (at time t + h / m)
Phi+-^(j1,j2) = Ups+^j1 +- b^j2(Ups+^j1) sqrt(h)
Y_next = Y + a h + sum_j b^j dW_j
         + sum_j (a(Ups+^j) - a(Ups-^j)) I_(j,0) / (2 sqrt(h))
         + sum_j (a(Ups+^j) - 2 a + a(Ups-^j)) h / 4
         + sum_j sum_k (b^j(Ups+^k) - 2 b^j + b^j(Ups-^k)) I_(0,j) / (2 h)
         + sum_(j1,j2) (b^j2(Ups+^j1) - b^j2(Ups-^j1)) I_(j1,j2) / (2 sqrt(h))
         + sum_(j1,j2,j3) (b^j3(Phi+^(j1,j2)) - b^j3(Phi-^(j1,j2))
                           - b^j3(Ups+^j2) + b^j3(Ups-^j2)) I_(j1,j2,j3) / (2 h)
```

Each line reproduces a term of the order 1.5 Ito-Taylor scheme (Kloeden-Platen
(10.4.6); Kuznetsov, arXiv:1802.00888, (5)) to a remainder of L2 size `O(h^2)` and
mean zero; the deterministic lines to `O(h^3)`. The `1 / m` makes the `m` first
differences of `a` add up to one `a' a` (and one `da/dt`); subtracting
`b^j3(Ups+-^j2)`, not `b^j3(Ups+-^j1)`, in the last line is what cancels the
`O(sqrt(h))` leftover when `j1 != j2`. With `m = 1` the lines are (11.2.1)'s. The
tests compare one step with the Taylor step written out with analytic derivatives:
the difference falls like `h^2` (`h^3` with no noise drawn).

Rossler's order 1.5 SRK methods (SIAM J. Numer. Anal. 48, 2010) cover scalar,
diagonal and additive noise only. Kloeden and Platen give a derivative-free
several-channel form in Section 11.2; the book was not to hand, so this one was
derived from the Taylor scheme and checked against it as above.

**Cost.** Per step `1 + 2m` drift and `1 + 2m + 2m^2` diffusion evaluations, and
the integrals: `q_1 ~ 1 / (8 h^2)` Legendre coefficients per channel for the
twofold ones and `q_3 ~ 0.13 / h` of them in the `m^3` threefold ones, `O(q_3^3 m
+ q_3^2 m^2 + q_3 m^3)` operations - the `m^3` term leads only once `m > q_3`.

**As measured.** `dx = dW_1`, `dz = z/2 dt + z x dW_2` (`x(0) = 1/2`, `z(0) = 1`)
against its exact solution `z = exp(t/2 + int x dW_2 - (1/2) int x^2 dt)`, and the
cosine system `dx = -x/2 dt + cos(y) dW_1`, `dy = -y/2 dt + cos(x) dW_2` against
srk3 at `2^-11`, 60 paths of an `IteratedIntegralPath` on `2^-12`:

| h | z, srk3 | cos, srk3 | z, Milstein |
|---|---|---|---|
| 1/8 | 6.21e-02 | 8.80e-03 | 1.47e-01 |
| 1/16 | 2.74e-02 | 3.52e-03 | 9.70e-02 |
| 1/32 | 8.74e-03 | 1.25e-03 | 3.51e-02 |
| 1/64 | 6.46e-03 | 4.76e-04 | 3.53e-02 |
| 1/128 | 1.48e-03 | 1.75e-04 | 1.58e-02 |
| 1/256 | 3.95e-04 | 5.39e-05 | 7.35e-03 |
| **slope** | **1.42** | **1.46** | **0.84** |

`z` is lognormal and its mean error is carried by a few paths (the 1/64 row).

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
  equations (10.4.1) and (10.4.6).
- Kuznetsov, D. F. (2022). To numerical modeling with strong orders 1.0, 1.5, and
  2.0 of convergence for multidimensional dynamical systems with random
  disturbances. arXiv:1802.00888v6.

---

**Parent Module:** [STOCHASTIC](../STOCHASTIC.md)

**Related Modules:**
- [SRK2](../srk2/SRK2.md) - strong order 1.0, handles systems with commuting noise
- [SRK4](../srk4/SRK4.md) - weak order 2.0
