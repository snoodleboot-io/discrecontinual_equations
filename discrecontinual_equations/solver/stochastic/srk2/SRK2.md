# SRK2

## Overview

SRK2 is Platen's two-stage explicit strong scheme of order 1.0 for Ito SDEs,
Kloeden and Platen (1992) equation (11.1.7). It is the Milstein scheme with the
derivative of the diffusion replaced by a difference between two stages, so it
needs no derivative from the user and reaches the same strong order 1.0 Milstein
does. The "2" counts stages; the scheme is not second order in any sense.

Until DEQ-25 this file described, and the solver implemented, a scheme that drew no
random number: `sqrt(dt)` stood where the Wiener increment belonged. The solver now
draws its increments from a source it owns and the orders below are measured.

## Architecture

```
SRK2Solver
├── Config: start_time, end_time, step_size, random_seed, calculus
├── Base: StochasticRungeKuttaSolver (shared stepping loop, owns the Wiener source)
├── Method: Kloeden-Platen (11.1.7), one supporting value per noise channel
├── Order: Strong 1.0, Weak 1.0 (measured 0.99 / 0.97)
└── Calculus: Ito, or Stratonovich via the Ito drift a + (1/2) b b'
```

## Core Classes

### SRK2Config

```python
class SRK2Config(StochasticConfig):
    start_time: float = 0.0
    end_time: float = 1.0
    step_size: float = 0.01
    random_seed: int | None = None
    calculus: Literal["ito", "stratonovich"] = "ito"
```

### SRK2Solver

```python
class SRK2Solver(StochasticRungeKuttaSolver):
    def __init__(self, solver_config: SRK2Config, wiener: WienerSource | None = None): ...

    @staticmethod
    def _step(y, t, h, coefficients, increments) -> np.ndarray:
        """One step of Kloeden-Platen (11.1.7)."""
```

`wiener` defaults to a `GaussianWienerSource` seeded from `random_seed`; a
`BrownianPath` fixed in advance may be passed instead, which is how the
convergence tests drive every step size with the same path.

## Folder Structure

```
srk2/
├── __init__.py
├── srk2_config.py
├── srk2_solver.py
└── SRK2.md
```

Shared with the other `srk*` packages:

```
stochastic/
├── wiener.py        - WienerIncrements, WienerSource, GaussianWienerSource, BrownianPath
├── coefficients.py  - ItoCoefficients (drift in the Ito sense, diffusion)
└── runge_kutta.py   - StochasticRungeKuttaSolver (the stepping loop)
```

## Example

```python
from discrecontinual_equations.solver.stochastic.srk2.srk2_config import SRK2Config
from discrecontinual_equations.solver.stochastic.srk2.srk2_solver import SRK2Solver

# dX = mu X dt + sigma X dW
config = SRK2Config(start_time=0.0, end_time=1.0, step_size=0.01, random_seed=42)
solver = SRK2Solver(config)
solver.solve(equation, [100.0])
```

## Mathematical Foundation

With `a` the Ito drift, `b` the diffusion, `h` the step and `dW = W(t+h) - W(t)`:

```
Y_bar  = Y + a h + b sqrt(h)
Y_next = Y + a h + b dW + (b(Y_bar) - b) (dW^2 - h) / (2 sqrt(h))
```

Expanding `b(Y_bar)` gives `(b(Y_bar) - b) / (2 sqrt(h)) = (1/2) b b' + O(sqrt(h))`,
so the last term is the Milstein correction `(1/2) b b' (dW^2 - h)` to a remainder
of strong order 1.5. The `sqrt(h)` in the supporting value is a probe distance for
that difference quotient, not an increment.

For a system the diffusion interface returns one coefficient per component, each
with its own Wiener process. The scheme uses one supporting value per channel,
`Y_bar_j = Y + a h + b_j e_j sqrt(h)`, reproducing the diagonal Milstein terms
`(1/2) b_j (d b_j / d x_j) (dW_j^2 - h)`. The cross terms `b_k (d b_j / d x_k)
I_(k,j)` need Levy areas, which have no exact sampler, and are omitted: order 1.0
therefore holds when each `b_j` depends only on `x_j`, and falls to 0.5 otherwise.

A Stratonovich equation `dX = a dt + b o dW` is integrated as the Ito equation with
drift `a + (1/2) b b'`; see `coefficients.py` for the sign.

## Convergence, as measured

Geometric Brownian motion `dX = X dt + 0.5 X dW`, `X(0) = 1`, on `[0, 1]`, exact
solution `exp((mu - sigma^2 / 2) t + sigma W(t))`, 100000 paths, the same Brownian
path at every step size. Weak error estimated path-coupled, standard error in
brackets.

| h | strong error | weak error |
|---|---|---|
| 1/8 | 2.130e-01 | 1.521e-01 (8e-04) |
| 1/16 | 1.109e-01 | 8.015e-02 (5e-04) |
| 1/32 | 5.616e-02 | 4.139e-02 (2e-04) |
| 1/64 | 2.809e-02 | 2.096e-02 (1e-04) |
| 1/128 | 1.397e-02 | 1.054e-02 (6e-05) |
| 1/256 | 6.936e-03 | 5.271e-03 (3e-05) |
| **slope** | **0.99** | **0.97** |

Euler-Maruyama and Milstein in the same run: 0.59 / 0.97 and 0.97 / 0.97.

Cost per step: one drift evaluation and `1 + d` diffusion evaluations for `d`
components (two for a scalar equation).

## References

- Kloeden, P. E. and Platen, E. (1992). *Numerical Solution of Stochastic
  Differential Equations*. Springer. Section 11.1, equation (11.1.7).
- Platen, E. (1984). *Zur zeitdiskreten Approximation von Itoprozessen*. Diss. B,
  Akademie der Wissenschaften der DDR.

---

**Parent Module:** [STOCHASTIC](../STOCHASTIC.md)

**Related Modules:**
- [MILSTEIN](../milstein/MILSTEIN.md) - the scheme this is a derivative-free form of
- [SRK3](../srk3/SRK3.md) - strong order 1.5, scalar noise
