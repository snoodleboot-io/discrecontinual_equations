# Lyapunov exponents and noise

A Lyapunov exponent is the mean exponential rate at which nearby trajectories
separate. Its sign decides whether a reference motion is stable, so it is the
natural test function for a bifurcation in a noisy system, where eigenvalues of a
fixed point no longer tell the whole story. This page covers:

- the Lyapunov spectrum of a flow, by Benettin's QR method;
- covariant Lyapunov vectors, the directions that go with the exponents, and the
  angles between them that decide hyperbolicity;
- finite-time exponents, deterministic and noisy;
- the two notions of stochastic bifurcation, phenomenological (P) and dynamical
  (D), in closed form in one dimension;
- locating P and D thresholds in a parameter.

Every work cited is listed on the [References](../references.md#lyapunov-exponents-covariant-vectors-and-noise)
page.

All examples define their fields the same way, as a `DeterministicFunction`
subclass whose `eval` returns the vector field:

```python
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class X(Variable, name="x", abbreviation="x"):
    pass


class A(Parameter, name="a", abbreviation="a"):
    pass
```

Each example below repeats what it needs, so you can copy and run it on its own.

## The Lyapunov spectrum

**Problem.** Find the rates $\lambda_1 \ge \dots \ge \lambda_n$ at which a flow
$\dot x = f(x)$ stretches and contracts volumes along an orbit.

**Idea** (Benettin et al. 1980). Carry a frame $Y$ of tangent vectors along the
orbit under the variational equation $\dot Y = J(x(t))\,Y$, and re-orthonormalise
it every few steps by a QR factorisation $Y = QR$. The logarithms of the diagonal
of $R$, summed and divided by the elapsed time, converge to the exponents:

$$
\lambda_i = \lim_{T \to \infty} \frac{1}{T} \sum_{k} \log |R^{(k)}_{ii}| .
$$

`lyapunov_spectrum` integrates the state and the frame together with RK4, takes
$J$ by finite differences of the field, and renormalises every ten steps. For a
linear system $\dot x = Ax$ the spectrum is the real parts of the eigenvalues of
$A$, which is how it is tested.

```python
import numpy as np

from discrecontinual_equations.continuation import lyapunov_spectrum
from discrecontinual_equations.continuation.lyapunov import LyapunovSettings
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.variable import Variable


class X(Variable, name="x", abbreviation="x"):
    pass


class Lorenz(DeterministicFunction):
    def eval(self, point, time=None):
        x, y, z = point
        return [10.0 * (y - x), x * (28.0 - z) - y, x * y - 8.0 / 3.0 * z]


lorenz = Lorenz(variables=[X(), X(), X()], parameters=[], results=[X(), X(), X()])
spectrum = lyapunov_spectrum(
    lorenz,
    np.array([1.0, 1.0, 1.0]),
    settings=LyapunovSettings(dt=0.01, horizon=10000, transient=500),
)
print(np.round(spectrum.exponents, 2).tolist())  # [0.86, -0.01, -14.52]
print(round(spectrum.exponents.sum(), 2))  # -13.67, the trace -41/3
```

Viswanath (1998, Table 5.1) gives $0.90566$, $0$, $-14.57233$ for these
parameters. The example runs only to $T = 100$, so its top exponent is still a few
percent low; at $T = 200$ it reads $0.89$. The library's tests check it against
$0.906$. The *sum* is exact much sooner, because it is the
time-average of the trace of $J$, here the constant $-41/3$.

**Limits.**

- The estimate converges like $O(1/T)$ at best, and more slowly on attractors
  with intermittency; there is no error estimate. Run it twice with different
  horizons.
- The orbit must stay bounded. The transient must carry it onto the attractor
  first.
- The Jacobian is a forward difference. Its step is being changed to the
  per-component step of Dennis and Schnabel (Algorithm A5.4.1) in DEQ-94, which
  is not yet merged; this page will describe it when it is.
- The cost is $4(n+2)$ field evaluations per RK4 step for $n$ variables, all in
  Python.

API: [`lyapunov_spectrum`](../api/continuation.md#lyapunov-exponents-and-stochastic-bifurcation).

## Covariant Lyapunov vectors

**Problem.** The QR frame of Benettin's method is orthogonal by construction. It
depends on the inner product and on the ordering of the vectors, and the tangent
flow does not carry it onto itself. So it cannot say whether the stable and
unstable directions of a chaotic attractor ever become tangent, which is what
hyperbolicity means.

**Idea** (Ginelli et al. 2007; 2013, Sec. 4). The intrinsic directions are the
*covariant* Lyapunov vectors $v_j$. The tangent flow carries $v_j(t)$ onto a
multiple of $v_j(t+\tau)$, and $v_j$ grows on average at rate $\lambda_j$. The
$j$-th covariant vector lies in the span of the first $j$ Gram-Schmidt vectors, so
it can be written as $v_n = Q_n c_n$ with an upper-triangular coefficient matrix
$C_n$. Since $J_n Q_n = Q_{n+1} R_n$, the coefficients obey the backward
recursion (Ginelli et al. 2013, eqs. (28), (32))

$$
C_n = R_n^{-1}\, C_{n+1}\, D_n ,
$$

where $D_n$ is the diagonal matrix of growth factors. Run backward from any
non-singular triangular start, the columns converge to the true coefficients
(eqs. (33)-(40)). `covariant_lyapunov_vectors` runs the four stages of Sec. 4.2:
forward transient, forward pass recording $Q_n$ and $R_n$, backward transient, and
backward pass. It solves $R_n c_n = c_{n+1}$ column by column rather than forming
$R^{-1}$, as Kuptsov and Parlitz (2012, eq. (66)) recommend. Column $j$ then
depends only on the leading $j \times j$ block, so a strongly contracting tail
cannot spoil the leading vectors.

The angle between two vectors is $\varphi = \arccos |v_j \cdot v_k| \in [0, \pi/2]$
(Ginelli et al. 2007, eq. (4)). A distribution of angles that reaches zero means
tangencies; one bounded away from zero is the numerical signature of
hyperbolicity. `splitting_angles(k)` gives the smallest principal angle between
the span of the first $k$ vectors and the rest (Ginelli et al. 2013, Sec. 7).

The example reproduces the finding of Fig. 1a of the 2007 letter: the Hénon map
has tangencies and the Lozi map does not. The figure shows the two angle
distributions, not their minima; the minima printed below are the library's own,
over 2000 samples:

```python
import numpy as np

from discrecontinual_equations.continuation import (
    CovariantLyapunovSettings,
    MapTangent,
    covariant_lyapunov_vectors,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.variable import Variable


class X(Variable, name="x", abbreviation="x"):
    pass


class Henon(DeterministicFunction):
    def eval(self, point, time=None):
        x, y = point
        return [1.0 - 1.4 * x * x + y, 0.3 * x]


class Lozi(DeterministicFunction):
    def eval(self, point, time=None):
        x, y = point
        return [1.0 - 1.4 * abs(x) + y, 0.3 * x]


settings = CovariantLyapunovSettings(transient=100, samples=2000, backward_transient=100)
for name, field in (("Henon", Henon), ("Lozi", Lozi)):
    vectors = covariant_lyapunov_vectors(
        MapTangent(field(variables=[X(), X()], parameters=[], results=[X(), X()])),
        np.array([0.1, 0.1]),
        settings=settings,
    )
    angles = vectors.angles(0, 1)
    print(name, np.round(vectors.exponents, 3), round(angles.minimum, 4))
# Henon [ 0.425 -1.629] 0.0014   the stable and unstable directions nearly touch
# Lozi  [ 0.192 -1.396] 0.5907   bounded away from tangency: hyperbolic
```

For a flow use `FlowTangent(field, dt, steps)` in place of `MapTangent`. Its forward
pass is the Benettin pass of `lyapunov_spectrum`. On Lorenz the library's tests
check the second vector against the flow direction, and covariance against an
independent SciPy integration. The [API page](../api/continuation.md#covariant-lyapunov-vectors-and-finite-time-exponents)
tabulates those checks.

**Limits.**

- The spectrum must be non-degenerate. Inside a covariant subspace of dimension
  above one the algorithm returns an arbitrary basis that depends on the starting
  matrix (Ginelli et al. 2013, Sec. 4.2).
- Both transients matter. Each converges like $e^{-\Delta\lambda\, t}$, with
  $\Delta\lambda$ the smallest gap between neighbouring exponents. Vectors good to
  $10^{-9}$ need $\Delta\lambda\, t \gtrsim 20$ in each stage.
- Every $Q_n$ and $R_n$ of the sampled stretch and the backward transient is
  kept in memory (ibid., eq. (41)).
- Angles depend on the coordinates. Only "zero" versus "bounded away from zero"
  is invariant (ibid., Sec. 7).

API: [`covariant_lyapunov_vectors`, `CovariantLyapunov`, `FlowTangent`, `MapTangent`](../api/continuation.md#discrecontinual_equations.continuation.covariant_lyapunov).

## Finite-time exponents

The logarithms of the growth factors over a window of length $T$ are the
finite-time exponents (Ginelli et al. 2013, eq. (29)). Their mean over tiling
windows is the exponent. `CovariantLyapunov.finite_time_exponents(window)` returns
them as a `FiniteTimeExponents` distribution, with `mean`, `median`, `spread`,
`quantile` and `fraction_positive`.

For a noisy flow, `stochastic_finite_time_exponents` runs `matrix_noise_lyapunov`
once per window with the seed advanced by one each time. The same settings
therefore give the same random numbers at every parameter value. That is what
makes the median a smooth test function in the parameter. For the noisy pitchfork
$dx = (ax - x^3)\,dt + s x\,dW$ (Itô) at its origin, the window exponent is exactly

$$
\lambda_T = a - \tfrac{s^2}{2} + \frac{s\, W(T)}{T},
$$

so the spread is $s/\sqrt{T}$. The second half of the [threshold
example](#locating-a-threshold-in-the-parameter) below measures it.

**Limit.** The window must be a whole number of renormalisation intervals (ten
steps). A partial interval would be timed but not measured, so other windows are
refused.

API: [`stochastic_finite_time_exponents`, `FiniteTimeExponents`](../api/continuation.md#discrecontinual_equations.continuation.covariant_lyapunov).

## Two notions of stochastic bifurcation

For $dx = f(x)\,dt + g(x)\,dW$, two different events deserve the name
*bifurcation* (Arnold 1998, Ch. 9):

- **Phenomenological (P).** The stationary density $p_s$ changes shape, for
  example when its peak leaves the origin.
- **Dynamical (D).** The top Lyapunov exponent of the invariant reference state
  changes sign, so paths that fell into it start to escape.

They can happen at different parameters, so the library keeps them as separate
objects.

**In one dimension both have closed forms.** The zero-flux stationary density is

$$
p_s(x) = \frac{C}{g(x)^2}\,\exp\!\left( \int^x \frac{2 f_{\text{Itô}}(y)}{g(y)^2}\, dy \right),
$$

and about an equilibrium $x_0$ with $g(x_0) = 0$ the linearisation is
geometric Brownian motion, with exponent

$$
\lambda = f_{\text{Itô}}'(x_0) - \tfrac12\, g'(x_0)^2 .
$$

Here $f_{\text{Itô}}$ is the drift read in Itô's convention. A Stratonovich
equation has the Itô drift $f + \tfrac12 g g'$ (Kloeden and Platen 1992, Ch. 4).
`NoiseConvention` (`Ito()` or `Stratonovich()`) applies that shift, and it moves
both thresholds. `StationaryDensity` evaluates $p_s$ on a grid, `DensityModes`
reports its shape, and `LyapunovExponent` returns $\lambda$.

For the pitchfork with multiplicative noise, $f = ax - x^3$ and $g = sx$:

| | D: $\lambda = 0$ | P: peak leaves 0 |
|---|---|---|
| Stratonovich | $a = 0$ | $a = s^2/2$ |
| Itô | $a = s^2/2$ | $a = s^2$ |

```python
import numpy as np

from discrecontinual_equations.continuation import (
    DensityModes,
    Ito,
    LyapunovExponent,
    StationaryDensity,
    Stratonovich,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class X(Variable, name="x", abbreviation="x"):
    pass


class A(Parameter, name="a", abbreviation="a"):
    pass


class Pitchfork(DeterministicFunction):  # f(x) = a x - x^3
    def eval(self, point, time=None):
        return [self.parameters[0].value * point[0] - point[0] ** 3]


class Multiplicative(DeterministicFunction):  # g(x) = s x, s = 0.6
    def eval(self, point, time=None):
        return [0.6 * point[0]]


drift = Pitchfork(variables=[X()], parameters=[A(value=0.0)], results=[X()])
noise = Multiplicative(variables=[X()], parameters=[], results=[X()])
grid = np.linspace(1.0e-3, 3.0, 3000)  # g vanishes at 0, so start just off it
for convention in (Stratonovich(), Ito()):
    exponent = LyapunovExponent(drift, noise, convention)
    for a in (0.1, 0.25, 0.45):
        drift.parameters[0].value = a
        xs, density = StationaryDensity(drift, noise, convention, grid).evaluate()
        mode = DensityModes(xs, density).dominant_mode()
        print(type(convention).__name__, a, round(exponent.at(0.0), 3), round(mode, 2))
# Stratonovich 0.1  0.1   0.0
# Stratonovich 0.25 0.25  0.27   peak at sqrt(a - s^2/2)
# Stratonovich 0.45 0.45  0.52
# Ito          0.1  -0.08 0.0
# Ito          0.25 0.07  0.0
# Ito          0.45 0.27  0.3    peak at sqrt(a - s^2)
```

**Limits.**

- One dimension only. Above one dimension the density has no antiderivative
  formula unless the drift is a gradient. `StationaryFokkerPlanck` solves for it
  on a grid instead (see the quasipotential and Fokker-Planck page). The exponent
  has to be accumulated along an orbit by `stochastic_lyapunov` or
  `matrix_noise_lyapunov`.
- Derivatives are central differences with a fixed step of $10^{-6}$.
- The density is normalised over the grid only. A grid that reaches a zero of
  $g$ makes it singular.

API: [`stochastic`](../api/continuation.md#discrecontinual_equations.continuation.stochastic).

## Stochastic Lyapunov exponents in any dimension

`stochastic_lyapunov` (one shared Brownian motion) and `matrix_noise_lyapunov`
($K$ independent drivers, through a `NoiseMatrix`) carry Benettin's frame along a
noisy orbit. The state and the frame both step by Euler-Maruyama with the same
increments, and the frame is re-orthonormalised every ten steps. Under
Stratonovich they add the Itô correction to the state's drift and to the
variational drift. Two exact checks hold:

- With additive noise the tangent equation carries no noise, so a linear system's
  spectrum is the real parts of its eigenvalues, with no Monte-Carlo error.
- With $G = \operatorname{diag}(s_i x_i)$ the components are independent geometric
  Brownian motions, with exponents $a_i - s_i^2/2$ (Itô).

**Limits.** Euler-Maruyama is strong order ½. The estimate carries a Monte-Carlo
error of order $s/\sqrt{T}$ on top of the $O(1/T)$ bias of the deterministic
method. Use a fixed seed whenever you compare parameter values.

API: [`stochastic_lyapunov`, `matrix_noise_lyapunov`](../api/continuation.md#lyapunov-exponents-and-stochastic-bifurcation).

## Locating a threshold in the parameter

`stochastic_threshold` gives P and D the same shape as the deterministic
detectors: a scalar test function of the parameter, plus a root finder.
`ParameterThreshold.locate(low, high)` first checks that the test changes sign on
the bracket. If it does not, it returns `None`; it never returns a root it has not
bracketed.

- **D:** `DynamicalThreshold` with the top exponent as the test, usually
  `MeanTopExponent.at`, which averages `matrix_noise_lyapunov` over a fixed set of
  seeds.
- **P:** `PhenomenologicalThreshold` with a `DensityProbe` applied to the solved
  `StationaryFokkerPlanck`. `RadialCrater` takes the difference of $\log p$ one
  cell out from the centre. Its sign is the sign of the radial curvature of
  $\log p$, and it changes smoothly in the parameter, unlike the location of the
  discrete argmax.

**Common random numbers.** Because every evaluation reuses the same seeds, the
exponent estimate is a smooth, deterministic function of the parameter, and
secant or bisection converge. The Monte-Carlo error does not go away. It becomes
one shared offset, $s W(T)/T$ for the pitchfork, so *differences* between located
thresholds are far more accurate than the thresholds themselves. The example
locates D for the pitchfork in both conventions. Each threshold is off by about
$0.09$, but their difference is $s^2/2$ to six digits:

```python
import numpy as np

from discrecontinual_equations.continuation import (
    DynamicalThreshold,
    Ito,
    MatrixNoiseSystem,
    MeanTopExponent,
    NoiseMatrix,
    StochasticLyapunovSettings,
    Stratonovich,
    stochastic_finite_time_exponents,
)
from discrecontinual_equations.continuation.root_finder import RootFinderSettings, Secant
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class X(Variable, name="x", abbreviation="x"):
    pass


class A(Parameter, name="a", abbreviation="a"):
    pass


class Pitchfork(DeterministicFunction):  # f(x) = a x - x^3
    def eval(self, point, time=None):
        return [self.parameters[0].value * point[0] - point[0] ** 3]


class Multiplicative(DeterministicFunction):  # g(x) = s x, s = 0.6
    def eval(self, point, time=None):
        return [0.6 * point[0]]


def family(a, convention=Ito()):
    drift = Pitchfork(variables=[X()], parameters=[A(value=a)], results=[X()])
    noise = NoiseMatrix([Multiplicative(variables=[X()], parameters=[], results=[X()])])
    return MatrixNoiseSystem(drift, noise, convention)


settings = StochasticLyapunovSettings(dt=0.01, horizon=4000, transient=0)
secant = Secant(RootFinderSettings(1.0e-6, 1.0e-4, 10))
located = {}
for convention in (Ito(), Stratonovich()):
    exponent = MeanTopExponent(
        lambda a, c=convention: family(a, c), np.array([0.0]), settings, seeds=(0,)
    )
    located[type(convention).__name__] = DynamicalThreshold(
        exponent.at, secant
    ).locate(-0.3, 0.5)
print({name: round(value, 3) for name, value in located.items()})
# {'Ito': 0.27, 'Stratonovich': 0.09}: exact 0.18 and 0, sharing one offset
print(round(located["Ito"] - located["Stratonovich"], 6))  # 0.179999, s^2/2

windows = stochastic_finite_time_exponents(family(0.18), np.array([0.0]), 500, 100)
print(np.round(windows.median, 3), np.round(windows.spread, 3))
# [-0.006] [0.249]: centred on the exponent 0, spread near s/sqrt(T) = 0.27
```

This example takes about half a minute, because every exponent evaluation
integrates 4000 noisy steps in Python.

The P threshold of the noisy Hopf normal form with conformal noise
$\sigma z\, dB$ has the closed form $\mu = \sigma^2$. For $\mu > 0$ the density is
$C\, r^{2\mu/\sigma^2 - 2} e^{-r^2/\sigma^2}$, so its peak leaves the origin
there. The D threshold of the same system is at $\mu = 0$: its origin's exponent
is exactly $\mu$.

```python
from discrecontinual_equations.continuation import (
    DensityGrid,
    Ito,
    NoiseMatrix,
    PhenomenologicalThreshold,
    RadialCrater,
    StationaryFokkerPlanck,
)
from discrecontinual_equations.continuation.root_finder import RootFinderSettings, Secant
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

SIGMA = 0.7


class X(Variable, name="x", abbreviation="x"):
    pass


class Mu(Parameter, name="mu", abbreviation="mu"):
    pass


class Hopf(DeterministicFunction):  # supercritical Hopf normal form
    def eval(self, point, time=None):
        mu, (x, y) = self.parameters[0].value, point
        r2 = x * x + y * y
        return [mu * x - y - x * r2, x + mu * y - y * r2]


class Conformal(DeterministicFunction):  # columns of sigma [[x, -y], [y, x]]
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def eval(self, point, time=None):
        x, y = point
        return [SIGMA * x, SIGMA * y] if self.index == 0 else [-SIGMA * y, SIGMA * x]


def plane():
    return {"variables": [X(), X()], "results": [X(), X()]}


grid = DensityGrid.box([-2.5, -2.5], [2.5, 2.5], [120, 120])  # straddles the origin
noise = NoiseMatrix([Conformal(k, parameters=[], **plane()) for k in range(2)])


def density_at(mu):
    return StationaryFokkerPlanck(Hopf(parameters=[Mu(value=mu)], **plane()), noise, Ito(), grid)


threshold = PhenomenologicalThreshold(
    density_at,
    RadialCrater([0.0, 0.0], offset=4),
    Secant(RootFinderSettings(1.0e-8, 1.0e-4, 20)),
)
print(round(threshold.locate(0.2, 0.9), 3))  # 0.526; the closed form is 0.49
```

With additive noise the noisy Hopf has a P-bifurcation at $\mu = 0$ (Baxendale
2024, Sec. 1). Without shear, the library measures a top exponent that stays
negative across $\mu \in [-0.5, 1.5]$ at $\sigma = 0.7$ (from $-1.23$ to $-0.30$),
so there is no D event to locate. That measurement is the library's own; it agrees
with Baxendale's result that the exponent is negative whenever the shear ratio
$b/a$ is small enough, whatever the other parameters. The conformal noise above is the isotropic noise that vanishes at the
origin and so keeps it invariant, and with it P and D fall at different
parameters. `StochasticScan` records both diagnostics along a range, and
`ThresholdCurve` follows a located threshold as a second parameter varies.

**Limits.**

- P is only as good as the grid. The example locates $0.562$, $0.526$ and
  $0.513$ on 80, 120 and 160 cells per axis. A second-order extrapolation of
  these gives $0.497$, against the exact $0.49$. A probe within
  a few cells of a singular centre carries a finite-volume overshoot that
  refinement does not remove. Step out with `offset`.
- D carries the shared Monte-Carlo offset, of order $s/\sqrt{\text{paths}\cdot T}$.
  Average over more seeds or a longer horizon to shrink it.
- `ThresholdCurve` stops at the first control value where even the full bracket
  shows no sign change. It does not extrapolate.

API: [`stochastic_threshold`](../api/continuation.md#discrecontinual_equations.continuation.stochastic_threshold).
