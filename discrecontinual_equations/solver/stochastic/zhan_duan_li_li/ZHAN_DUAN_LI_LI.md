# ZHAN_DUAN_LI_LI

## Overview

`ZhanDuanLiLiSolver` takes an explicit, theta-weighted stochastic Heun step. It
converges to the solution in the calculus the config names, `"ito"` (the default)
or `"stratonovich"`, for every theta (DEQ-133). The name is historical. The paper
it came from, arXiv:2010.07691, describes a different, symplectic Marcus scheme.

## The step

With `theta = method_parameter`, `a` the drift as written, and `G` the diffusion:

```
F       = a + (kappa - theta) c,   c = sum_k (dG_k/dx) G_k   (b_i b_i' for diagonal noise)
Y~      = Y + F(Y, t) h + G(Y, t) dW
Y_{n+1} = Y + [(1 - theta) F(Y, t) + theta F(Y~, t + h)] h
            + [(1 - theta) G(Y, t) + theta G(Y~, t + h)] dW
```

`kappa` is 0 for an Ito equation and 1/2 for a Stratonovich one. Without the
`(kappa - theta) c` term the step converges to the Ito equation with drift
`a + theta c` (Rumelin's theorem, Saito and Mitsui 1991, Theorem 1). That is the
Stratonovich solution at theta = 1/2, and matches neither calculus for other
theta. The term moves the limit to `a + kappa c`. For a Stratonovich equation at
theta = 1/2 the term is zero, so no derivative is taken. Additive noise has
`c = 0`.

## Order

| theta | strong | weak |
|-------|--------|------|
| 1/2 (stochastic Heun) | 1.0 for scalar or diagonal noise, 0.5 for non-commuting matrix noise | 1.0 |
| other | 0.5 | 1.0 |

The measurements are in the solver's docstring.

## References

- Saito, Mitsui. "Discrete approximations for stochastic differential equations", RIMS Kokyuroku 746 (1991), Theorem 1, (15), (16).
- Rumelin. "Numerical treatment of stochastic differential equations", SIAM J. Numer. Anal. 19 (1982), 604-613 (via Saito-Mitsui).
- Higham. "Mean-square and asymptotic stability of the stochastic theta method", SIAM J. Numer. Anal. 38 (2000), eq. (1.2): a different, drift-implicit method.

---

**Parent Module:** [STOCHASTIC](../STOCHASTIC.md)

**Related Modules:**
- [EULER_MARUYAMA](../euler_maruyama/EULER_MARUYAMA.md) - Basic explicit method
- [MILSTEIN](../milstein/MILSTEIN.md) - Strong order 1.0 method
