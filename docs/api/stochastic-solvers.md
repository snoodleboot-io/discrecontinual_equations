# Stochastic Solvers API

The concepts, examples and limits are in the
[stochastic solvers guide](../user-guide/stochastic-solvers.md).

::: discrecontinual_equations.solver.stochastic

## Solvers

::: discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_solver

::: discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config

::: discrecontinual_equations.solver.stochastic.milstein.milstein_solver

::: discrecontinual_equations.solver.stochastic.milstein.milstein_config

::: discrecontinual_equations.solver.stochastic.runge_kutta

::: discrecontinual_equations.solver.stochastic.srk2.srk2_solver

::: discrecontinual_equations.solver.stochastic.srk3.srk3_solver

::: discrecontinual_equations.solver.stochastic.srk4.srk4_solver

::: discrecontinual_equations.solver.stochastic.srk5.srk5_solver

::: discrecontinual_equations.solver.stochastic.gottwald_melbourne.gottwald_melbourne_solver

::: discrecontinual_equations.solver.stochastic.gottwald_melbourne.gottwald_melbourne_config

::: discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_solver

::: discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_config

## Wiener increments, Levy areas and iterated integrals

::: discrecontinual_equations.solver.stochastic.wiener

::: discrecontinual_equations.solver.stochastic.iterated_integrals

## Coefficients and ensembles

::: discrecontinual_equations.solver.stochastic.coefficients

::: discrecontinual_equations.solver.stochastic.ensemble

::: discrecontinual_equations.function.stochastic

## The step of the noise slopes

Milstein's correction, the check that decides whether matrix noise commutes, the
check of a `diagonal_noise` declaration and the Stratonovich drift correction all
read slopes of the diffusion, `d b_i / d x_j`. The solvers take them by central
differences rather than ask for one more function, in `ItoCoefficients` (per path)
and `BatchedCoefficients` (the ensemble).

**The problem.** A fixed absolute step rounds away on a large state. One millionth
either side of `3e12`, where an ulp is about `5e-4`, is no move at all: the
difference is exactly zero, the slope comes back as zero, Milstein silently drops
its correction and noise that does not commute is judged to commute. Nothing warns.

**The step.** Component `j` is moved by `h_j = slope_step * max(|x_j|, typical_j)`,
the per-component step of Dennis and Schnabel (1983, Sec. 5.4, Algorithm A5.4.1),
and each quotient is divided by the width the machine actually stepped,
`(x_j + h_j) - (x_j - h_j)`, as SciPy's `approx_derivative` does. `slope_step` is a
relative step, `eps^(1/3)` (about `6.06e-6`) by default, the value that balances a
central difference's truncation, `h^2 |b'''| / 6`, against its rounding,
`eps |b| / h`. `typical` (one per component, one for all, or one by default) is the
size of a component whose value says nothing: near zero, or a component that lives
at `1e-8`, which a step floored at one would swamp. Where every component is no
larger than its typical size the step is `slope_step` itself, which is what the
absolute step used to be.

```python
import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.solver.stochastic.coefficients import ItoCoefficients
from discrecontinual_equations.variable import Variable


class X(Variable, name="x", abbreviation="x"):
    pass


class T(Variable, name="t", abbreviation="t"):
    pass


class GrowingNoise(StochasticFunction):
    """b(x) = 0.1 x^2 / 1e12: geometric-like noise on a state of size 1e12."""

    def eval(self, point, time=None):
        return [0.0]

    def diffusion(self, point, time=None):
        return [0.1 * point[0] ** 2 / 1.0e12]


variables = [X()]
equation = DifferentialEquation(
    variables=variables,
    time=T(),
    parameters=[],
    derivative=GrowingNoise(variables, [], [X()], T()),
)
coefficients = ItoCoefficients(equation, "ito")
print(coefficients.diagonal_slopes(np.array([3.0e12]), 0.0))  # [0.6], exact
```

An absolute step of `1e-6` returned `[0.]` here.

**Limits.** The solvers build their coefficients with the defaults, so `typical`
is reachable only by constructing `ItoCoefficients` or `BatchedCoefficients`
directly; a state whose components live far below one is differenced at a step
floored at one. No step fixes the rounding of the diffusion itself: a slope is
resolved only to about `eps^(2/3) |b_i| / max(|x_j|, typical_j)`, so a weak
coupling in a large row comes back with that error. Seeded paths at ordinary
scales move by the difference between the old and new slopes, about `1e-10`
relative a step (DEQ-118 lists the recorded values that moved).

References: [Dennis and Schnabel 1983](../references.md#stochastic-solvers-and-iterated-integrals),
[SciPy 1.14.1 `_numdiff`](../references.md#stochastic-solvers-and-iterated-integrals).

::: discrecontinual_equations.finite_difference
