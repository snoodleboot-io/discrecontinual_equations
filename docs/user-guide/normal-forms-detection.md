# Bifurcations, normal forms and centre manifolds

Continuing an equilibrium in one parameter finds where it loses stability. Continuing
that loss of stability in two parameters finds the points where two things go wrong at
once, and in three parameters the points where three do. This page covers the pieces
that find those points and then say what happens near them:

| Question | Tool | Module |
|---|---|---|
| Is this equilibrium stable? | `EigenvalueStabilityAnalyzer` | `stability` |
| Has a pair of eigenvalues crossed the imaginary axis? | `bialternate_test_value` | `bialternate` |
| Where does a fold or Hopf point move in two parameters, and what codim-2 points lie on the way? | `Codim2Driver` | `codim2_builder`, `hopf_curve` |
| Which kind of Bogdanov-Takens, zero-Hopf, double-Hopf or cusp point is it? | `bogdanov_takens_normal_form`, ... , `classify` | `codim2_normal_form`, `critical_forms` |
| Is an $n$-dimensional Hopf point sub- or supercritical, and is a Bautin point degenerate? | `TaylorCenterManifold` | `center_manifold` |
| Where does a codim-2 point move in three parameters? | `Codim3Driver` | `codim3_builder`, `generalized_hopf_curve` |
| Is this period divergence a SNIC? | `characterize_snic` | `snic` |

All of them work in any state dimension. The examples below run as written; the
continuation drivers take a few seconds each.

Every system in the examples is written the same way. The drivers find their parts
(residuals, test functions, derivative providers) through a registry, which has to be
filled once:

```python
from pathlib import Path

from sweet_tea.registry import Registry

import discrecontinual_equations
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

# The drivers resolve their parts through the registry; fill it once.
Registry.fill_registry(
    path=str(Path(discrecontinual_equations.__file__).parent),
    module="discrecontinual_equations",
    exclude=["*.tests", "*.examples", "*.plot"],
)


class Beta1(Parameter, name="beta1", abbreviation="b1"):
    pass


class Beta2(Parameter, name="beta2", abbreviation="b2"):
    pass


class State(Variable, name="State", abbreviation="s"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


class BogdanovTakens(DeterministicFunction):
    """x' = y, y' = b1 + b2 y + x^2 - x y."""

    def eval(self, point, time=None):
        b1, b2 = self.parameters[0].value, self.parameters[1].value
        x, y = point
        return [y, b1 + b2 * y + x * x - x * y]


parameters = [Beta1(value=-0.25), Beta2(value=-0.5)]
equation = DifferentialEquation(
    variables=[State(), State()],
    time=Time(),
    parameters=parameters,
    derivative=BogdanovTakens(
        variables=[State(), State()],
        parameters=parameters,
        results=[State(), State()],
        time=None,
    ),
)
```

This is the Bogdanov-Takens normal form with $a = 1$, $b = -1$. Its folds lie on
$\beta_1 = 0$ and its Hopf points on $\beta_1 = -\beta_2^2$, $\beta_2 < 0$; the two curves
meet at the Bogdanov-Takens point at the origin.

## Stability of an equilibrium

At a hyperbolic equilibrium (no eigenvalue of the Jacobian $f_x$ on the imaginary axis)
the eigenvalues decide the local picture. `EigenvalueStabilityAnalyzer` counts those
with real part above a tolerance, which is the dimension of the unstable manifold, and
names the point `stable`, `unstable` or `saddle`:

```python
import numpy as np

from discrecontinual_equations.continuation.stability import (
    EigenvalueStabilityAnalyzer,
)

analyzer = EigenvalueStabilityAnalyzer(tolerance=1e-9)
result = analyzer.analyze(np.array([[0.1, -1.0], [1.0, 0.1]]))
print(result.label, result.unstable_dimension, result.eigenvalues)
# unstable 2 [(0.1, 1.0), (0.1, -1.0)]
```

**Limits.** The label means something only away from bifurcations. An eigenvalue
within the tolerance of the axis counts as stable, so a centre is labelled `stable` and
the label at a fold or Hopf point is the label of one side. Whether a non-hyperbolic
point attracts is decided by its normal form, below.

API: [`stability`](../api/continuation.md).

## Detecting a Hopf point: the bialternate test

**Problem.** Along a branch of equilibria, a Hopf bifurcation is where a complex pair
crosses the imaginary axis. Looking for it directly means tracking eigenvalues, which
swap order and collide. A scalar function that changes sign at the crossing is easier
to locate.

**Idea.** The bialternate product $2A \odot I$ of an $n \times n$ matrix $A$ is an
$\tfrac{n(n-1)}{2}$-square matrix whose eigenvalues are the pairwise sums
$\lambda_i + \lambda_j$, $i < j$ (Guckenheimer, Myers and Sturmfels 1997, Theorem 3.1).
So

$$
\det(2A \odot I) = \prod_{i<j} (\lambda_i + \lambda_j)
$$

vanishes exactly when two eigenvalues sum to zero: a pair $\pm i\omega$, which is a
Hopf point, or a real pair $\pm r$, a *neutral saddle*. MatCont uses this determinant,
bordered, as its Hopf test (MatCont manual, eq. (36)).

Forming that matrix costs $O(n^6)$ and its determinant overflows once $n$ reaches a
few dozen. `bialternate_test_value` reads the same product from one eigenvalue solve:
the sign of $\det(2A \odot I)$ is the parity of the factors with negative real part
(conjugate factors pair up with a positive product), and the magnitude is replaced by
the smallest factor, $\min_{i<j} |\lambda_i + \lambda_j|$. The result changes sign where
the determinant does, is continuous, and is measured in units of the eigenvalues, so a
tolerance on it means the same thing at every dimension.

```python
import numpy as np

from discrecontinual_equations.continuation.bialternate import (
    bialternate_test_value,
)


def jacobian(mu):
    # A focus whose pair crosses the axis at mu = 0, plus a stable direction.
    return np.array([[mu, -1.0, 0.0], [1.0, mu, 0.0], [0.0, 0.0, -2.0]])


for mu in (-0.1, 0.0, 0.1):
    print(f"mu = {mu:+.1f}: test = {bialternate_test_value(jacobian(mu)):+.3f}")
# mu = -0.1: test = -0.200
# mu = +0.0: test = +0.000
# mu = +0.1: test = +0.200

saddle = np.diag([1.0, -1.0, -3.0])  # lambda1 + lambda2 = 0, but no Hopf
print(f"neutral saddle: test = {bialternate_test_value(saddle):+.3f}")
# neutral saddle: test = +0.000
```

**Limits.**

- A zero is a *candidate*: the last line shows a neutral saddle giving the same zero
  as a Hopf point. Confirm a purely imaginary pair before calling it Hopf.
- Two pairs crossing at the same step cancel in the sign and are missed.
- A $1 \times 1$ system returns the constant $1$; there is no pair to cross.

An earlier version formed the matrix entry by entry with one sign wrong against
Guckenheimer, Myers and Sturmfels, eq. (9), which moved or lost Hopf points of coupled
systems with $n \ge 3$ (DEQ-46). Reading the eigenvalues has no table to get wrong, and
for $n = 64$ it is about a thousand times faster.

References: [GuckenheimerMyersSturmfels1997], [GovaertsEtAl2019] on the
[References](../references.md) page. API: [`bialternate`](../api/continuation.md).

## Fold and Hopf curves, and the codim-2 points on them

**Problem.** A fold or Hopf point found by one-parameter continuation moves when a
second parameter changes. The curve it traces in the parameter plane is a bifurcation
curve, and on it lie the codimension-2 points where the picture changes again: a
Bogdanov-Takens point (double zero eigenvalue), a zero-Hopf point (zero plus an
imaginary pair), a double Hopf point (two imaginary pairs), a cusp (a fold whose
quadratic coefficient vanishes) and a generalized Hopf, or Bautin, point (a Hopf whose
first Lyapunov coefficient vanishes).

**Idea.** Append to $f(u, p_a, p_b) = 0$ the condition that makes the point singular,
and treat $p_a$ as an unknown. For a Hopf curve that condition is the eigenproblem
$f_u q = i\omega q$ written in real form,

$$
\begin{aligned}
f(u; p_a, p_b) &= 0, \\
f_u q_R + \omega q_I &= 0, \\
f_u q_I - \omega q_R &= 0, \\
c^\top q_R - 1 &= 0, \qquad c^\top q_I = 0,
\end{aligned}
$$

in the unknowns $(u, q_R, q_I, \omega, p_a)$ with $p_b$ the curve coordinate. This is the
eigenproblem written out with a normalisation row; the first rows of Govaerts,
Kuznetsov and Sijnave's maximally extended generalized-Hopf system (2000, section 7.2,
eq. (7.6)) have the same form. The fold curve appends $f_u q = 0$, $c^\top q = 1$ instead.
Pseudo-arclength continuation follows either. Test functions evaluated at every
accepted point change sign at the codim-2 points, which are then located and
classified by their normal form (next section). The detectors are those MatCont runs on
the same curves (MatCont manual, sections 8.1.2 and 8.2.2).

```python
from discrecontinual_equations.continuation.codim2_builder import (
    Codim2Driver,
    CurveSeed,
)
from discrecontinual_equations.continuation.codim2_config import Codim2Config

# A Hopf point of the field: x = -sqrt(-b1), b2 = x, omega = sqrt(-2x).
seed = CurveSeed(state=[-0.5, 0.0], parameter_a=-0.25, frequency=1.0)
config = Codim2Config(
    continuation_parameter_index=0,  # b1 is solved for along the curve
    second_parameter_index=1,  # b2 is the curve coordinate
    curve="hopf",
    codim2_detectors=["bogdanov_takens"],
    initial_parameter=-0.5,
    direction=1,
    parameter_lower_bound=-0.6,
    parameter_upper_bound=0.05,
)
result = Codim2Driver.run(config, equation, seed)
for point in result.points:
    print(point.kind, point.classification, point.coefficients)
    print(f"  at (b1, b2) = ({point.parameter_a:.1e}, {point.parameter_b:.1e})")
# bogdanov_takens s=-1 {'a': 1.0, 'b': -1.0, 's': -1.0}
#   at (b1, b2) = (-1.6e-07, -4.0e-04)
```

`curve="fold"` with `codim2_detectors=["fold_bogdanov_takens", "zero_hopf", "cusp"]`
continues a fold curve instead; a fold seed has no `frequency`.

**Limits.**

- The seed must be a codim-1 point of the requested kind; for a Hopf seed,
  `frequency` picks the pair.
- Detection is by sign change between accepted points. Two codim-2 points closer than
  a step can cancel, and a test function that touches zero without crossing is missed.
- The Hopf curve's eigenvectors become dependent as $\omega \to 0$: it can be
  followed *to* a Bogdanov-Takens point, not through it as a Hopf curve.
- The normalisation vector $c$ is fixed at the seed; a long curve along which $q_R$
  turns orthogonal to it becomes singular.

References: [GovaertsKuznetsovSijnave2000], [GovaertsEtAl2019]. API:
[`codim2_builder`, `codim2_config`, `hopf_curve`](../api/continuation.md).

## Normal forms at codim-2 points

**Problem.** The test functions say *where* a codim-2 point is, not *which* one. Two
Bogdanov-Takens points can unfold differently; a zero-Hopf point may or may not shed an
invariant torus; a double Hopf is "simple" or "difficult". The coefficients of the
normal form on the centre manifold decide it.

**Idea.** Kuznetsov's projection method (Kuznetsov 1999, section 4). Write the field at
the critical point as

$$
F(x) = Ax + \tfrac12 B(x, x) + \tfrac16 C(x, x, x) + \dots
$$

and look for the centre manifold $x = H(w)$ and the normal form $\dot w = G(w)$
together, from the homological equation

$$
H_w(w)\, G(w) = F(H(w)) .
$$

Collecting the $w^\nu$ terms gives a linear system $L h_\nu = R_\nu$ for each Taylor
coefficient of $H$. Where $L$ is singular, Fredholm's condition $\langle p, R_\nu \rangle = 0$
gives a normal-form coefficient, and the bordered system

$$
\begin{pmatrix} L & q \\ \bar p^\top & 0 \end{pmatrix}
\begin{pmatrix} h \\ s \end{pmatrix} =
\begin{pmatrix} R \\ 0 \end{pmatrix}
$$

gives $h$ (Kuznetsov 1999, eq. (4.6)). Nothing depends on the dimension, and the centre
manifold's curvature enters through the $h_\nu$. The forms $B$, $C$, ... are exact:
`CriticalForms` evaluates the field on Taylor jets and recovers each symmetric form by
polarisation.

| Point | Normal form (Kuznetsov 1999) | What decides the case |
|---|---|---|
| Cusp | $\dot w = c w^3$, (3.1) | sign of $c = \tfrac16\langle p, C(q,q,q) + 3B(q, h_2)\rangle$ |
| Bogdanov-Takens | $\dot w_0 = w_1$, $\dot w_1 = a w_0^2 + b w_0 w_1$, (3.2) | $s = \operatorname{sign}(ab)$ |
| Zero-Hopf | (3.4) | $s = \operatorname{sign}(bc)$, $\theta = \operatorname{Re} d / b$; a torus when $s\theta < 0$ |
| Double Hopf | (3.6), needs (3.5): $k\omega_1 \ne l\omega_2$ for $k + l \le 5$ | simple when $p_{11} p_{22} > 0$ |
| Hopf | $\dot w = i\omega w + \tfrac12 G_{21} w \lvert w \rvert^2$ | $l_1 = \operatorname{Re} G_{21} / (2\omega)$ |

```python
import numpy as np

from discrecontinual_equations.continuation.codim2_normal_form import (
    bogdanov_takens_normal_form,
)
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)

problem = TwoParameterProblem(equation, 0, 1, 0.0, AutomaticDifferentiation())
bt = bogdanov_takens_normal_form(problem, np.zeros(2), 0.0, 0.0)
print(f"a = {bt.a:.6f}, b = {bt.b:.6f}, s = {bt.s:+d}")
# a = 1.000000, b = -1.000000, s = -1

beta2 = bt.unfolding.side * 0.1
print("Hopf curve near", bt.unfolding.hopf(beta2))
print("homoclinic near", bt.unfolding.homoclinic(beta2))
# Hopf curve near [-0.01 -0.1 ]
# homoclinic near [-0.0196 -0.1   ]
```

The unfolding maps the normal form's parameters back to the system's through
$\beta_2^2$ (Bosschaert and Kuznetsov 2021, sections 2.2.3 and 2.2.7). In the normal form
$\dot w_1 = \beta_1 + \beta_2 w_1 + a w_0^2 + b w_0 w_1$ (their eq. (8)) the Hopf curve is
$\beta_1 = -(a/b^2)\beta_2^2$ and the homoclinic curve
$\beta_1 = -\tfrac{49}{25}(a/b^2)\beta_2^2 + O(\beta_2^3)$; here they come out at
$\beta_1 = -0.01$ and $-0.0196$ for $\beta_2 = -0.1$, as they should.
`zero_hopf_normal_form`, `hopf_hopf_normal_form` and `cusp_normal_form` are called the
same way, and `Codim2Driver` attaches the label and coefficients to every point it
finds.

**Normalisation.** Raw coefficients depend on how the critical eigenvectors are scaled.
The library follows MatCont: unit length, $q_1 \cdot q_0 = 0$ at a Bogdanov-Takens
point, $\langle p, q \rangle = 1$ with $\langle p, q \rangle = \bar p^\top q$, and the phase
fixed so the largest component is real and positive. The classifying quantities
($s$, $\theta$, `simple`, `torus`, the sign of $c$) do not depend on that choice.

**Errata and decisions.**

- *Lorenz-84 $G_{300}$.* Kuznetsov (1999), section 10, prints
  $G_{300} = 25602814720/36130026241 = 0.7086$ for the "new" Lorenz model. The paper's
  own section 8 formula $G_{300} = \langle p_0, C(q_0,q_0,q_0) + 3B(q_0, h_{200})\rangle$,
  which matches the $G_{300}/6$ of its normal form (3.4), gives exactly three times
  that, $2.1259$, on the paper's printed $q_0$ and $p_0$. The model has no cubic terms,
  so the printed value is $\langle p_0, B(q_0, h_{200})\rangle$: the factor 3 was dropped.
  The same computation reproduces every other printed coefficient exactly, and the
  library and an independent Poincaré-Dulac computation agree. The printed
  $e(0) = 0.2106514716$ is the formula at the printed $G_{300}$; the consistent value is
  $1.6503$. The sign of $e(0)$, and the paper's conclusion ($s = \operatorname{sign}(bc) = 1$,
  $\theta < 0$), are unchanged.
- *The cusp coefficient carries $3B(q, h_2)$* (DEQ-83). That term is the centre
  manifold's curvature (Kuznetsov 1999, section 5). Leaving it out is right only in one
  dimension; with it, a swallowtail of a system with a slaved stable mode lands where the
  reduced equation puts it.
- *First Lyapunov coefficient at the positive frequency* (DEQ-96). $l_1$ normalises
  $\langle p, q \rangle = 1$ at $+i\omega$. Normalised at whichever sign the eigen-solver
  returned, it was wrong in 31 of 34 random similarity transforms of one system. The
  library's $l_1 = \operatorname{Re} G_{21}/(2\omega)$; MatCont prints
  $\operatorname{Re} G_{21}/2$ (manual eq. (40)), which is $\omega l_1$ and is available as
  `HopfNormalForm.matcont_lyapunov`.
- *Cusp orientation by continuity* (DEQ-102, PR #73, under review). The quadratic fold
  coefficient $a = \tfrac12\langle p, B(q, q)\rangle$ is odd in $q$, so its sign along a fold
  curve depends on which way $q$ points. It is fixed by keeping $q$ continuous along the
  curve, as MatCont does, not by requiring $\langle p, q\rangle > 0$, which does not fix it.
  The cubic $c$ above is even in $q$ and is unaffected.

**Limits.**

- Bogdanov-Takens: the unfolding stops at $\beta_2^2$, and the third-order homoclinic
  predictor of Bosschaert and Kuznetsov is not implemented.
- Double Hopf: cubic order only. The fifth-order coefficients that settle the difficult
  case are not computed, and a strongly resonant point ($k + l \le 5$) raises
  `ResonanceError` and is labelled `resonant k:l`.
- Zero-Hopf: the higher-order terms that can change the topology of the unfolding
  (Kuznetsov 1999, section 3 (iv)) are not computed; `torus` is a statement about the
  truncated unfolding.
- The forms are exact only for fields written in arithmetic (`+ - * / **`); a field that
  calls a NumPy transcendental function on its state does not propagate jets.

References: [Kuznetsov1999], [BosschaertKuznetsov2021], [GovaertsEtAl2019]. API:
[Center manifolds and normal forms](../api/center-manifold.md), which also lists the
validation against published values.

## Centre-manifold reduction of a Hopf point

**Problem.** At a Hopf point of an $n$-dimensional system, is the bifurcation
supercritical (a stable cycle appears) or subcritical? And where the first Lyapunov
coefficient vanishes, what does the second say? The planar focal-value recursion
answers both, but only in the plane.

**Idea.** Change coordinates so the critical pair spans the first two, $x = x_0 + T(u, v)$,
and solve the invariance equation of the centre manifold $v = W(u)$,

$$
DW(u)\, f_c(u, W(u)) = f_h(u, W(u)),
$$

degree by degree up to six. Each degree is a linear solve that is unique because the
hyperbolic block has no eigenvalue on the imaginary axis. Substituting $W$ back gives a
planar field whose focal values $\eta_4$ and $\eta_6$ are the first and second Lyapunov
quantities. It is the graph form of the same homological equation that Kuznetsov's
method solves (Kuznetsov 1999, section 4).

```python
import numpy as np

from discrecontinual_equations.continuation.center_manifold import (
    TaylorCenterManifold,
)


class CoupledHopf(DeterministicFunction):
    """A planar Hopf point two-way coupled to a stable direction."""

    def eval(self, point, time=None):
        x, y, z = point
        r2 = x * x + y * y
        return [-y - x * r2 + z * x, x - y * r2, -2.0 * z + x * x + x * y]


field = CoupledHopf(
    variables=[State() for _ in range(3)],
    parameters=[],
    results=[State() for _ in range(3)],
    time=None,
)
jacobian = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -2.0]])
eta4, eta6 = TaylorCenterManifold().lyapunov_quantities(
    field, np.zeros(3), jacobian, 0.0
)
print(f"eta4 = {eta4:.6f}  (negative: supercritical)")
# eta4 = -0.437500  (negative: supercritical)
```

**Limits.**

- One critical pair only; a second pair, or a zero eigenvalue, makes the homological
  operator singular.
- The focal values belong to the coordinates the eigenvector split produces: $\eta_4$
  scales with the square, and $\eta_6$ with the fourth power, of that basis. Their signs
  are invariants, and so is $\eta_6$ where $\eta_4 = 0$, the only place the
  degenerate-Bautin detector reads it. On a coupled 3D Hopf point $\eta_4 = l_1/4$ to
  $10^{-9}$.
- Degree six is enough for $\eta_4$ and $\eta_6$ and no more.

References: [Kuznetsov1999]. API:
[`TaylorCenterManifold`](../api/center-manifold.md).

## Three parameters: cusp, generalized-Hopf and Bogdanov-Takens curves

**Problem.** A codim-2 point moves along a curve when a third parameter changes, and on
that curve lie codim-3 points: a swallowtail on a cusp curve (the cubic coefficient
vanishes too), a degenerate Bautin point on a generalized-Hopf curve (the second
Lyapunov quantity changes sign).

**Idea.** The same appending trick, with minimally augmented systems so the extra
equations are scalars. For the generalized-Hopf curve the eigenvector is never an
unknown: a bordered solve of $f_u^2 + \kappa I$ gives a $2 \times 2$ matrix $G$ that
vanishes when $f_u$ has eigenvalues $\pm i\sqrt{\kappa}$ (MatCont manual, eqs. (61)-(62)),
and the curve is

$$
f(u; p_a, p_b, p_c) = 0, \qquad G_{i_1 j_1} = G_{i_2 j_2} = 0, \qquad l_1(u, \kappa) = 0
$$

in the unknowns $(u, \kappa, p_a, p_b)$ (Govaerts, Kuznetsov and Sijnave 2000,
section 7.1, set $\det G = 0$; two entries, as MatCont does, are better conditioned).
The cusp curve appends the vanishing quadratic fold coefficient to the minimally
augmented fold system; the Bogdanov-Takens curve writes the Jordan chain
$f_u q = 0$, $f_u q_1 = q$ out explicitly.

A swallowtail, from the normal form $\dot x = b_1 + b_2 x + b_3 x^2 - x^4$, whose cusp
curve is $(b_1, b_2, b_3) = (3t^4, -8t^3, 6t^2)$ at $x = t$:

```python
from discrecontinual_equations.continuation.codim3_builder import (
    Codim3Driver,
    Codim3Seed,
)
from discrecontinual_equations.continuation.codim3_config import Codim3Config


class B1(Parameter, name="b1", abbreviation="b1"):
    pass


class B2(Parameter, name="b2", abbreviation="b2"):
    pass


class B3(Parameter, name="b3", abbreviation="b3"):
    pass


class Swallowtail(DeterministicFunction):
    """x' = b1 + b2 x + b3 x^2 - x^4."""

    def eval(self, point, time=None):
        b1, b2, b3 = (parameter.value for parameter in self.parameters)
        x = point[0]
        return [b1 + b2 * x + b3 * x * x - x**4]


parameters3 = [B1(value=0.1875), B2(value=-1.0), B3(value=1.5)]
swallowtail = DifferentialEquation(
    variables=[State()],
    time=Time(),
    parameters=parameters3,
    derivative=Swallowtail(
        variables=[State()], parameters=parameters3, results=[State()], time=None
    ),
)
# Seed on the cusp curve at t = 1/2.
seed = Codim3Seed(state=[0.5], parameter_a=0.1875, parameter_b=1.5)
config = Codim3Config(
    continuation_parameter_index=0,
    second_parameter_index=2,
    third_parameter_index=1,  # b2 is the curve coordinate
    curve="fold",
    codim3_curve="cusp",
    codim3_detectors=["swallowtail"],
    initial_parameter=-1.0,
    direction=1,
    parameter_lower_bound=-1.2,
    parameter_upper_bound=1.2,
    initial_step=0.01,
    maximum_step=0.02,
    maximum_points=500,
)
result = Codim3Driver.run(config, swallowtail, seed)
for point in result.points:
    print(point.kind, f"x = {point.state[0]:.1e}")
# swallowtail x = 0.0e+00
```

A degenerate Bautin point is found the same way, with `codim3_curve="generalized_hopf"`,
`codim3_detectors=["degenerate_bautin"]` and a `Codim3Seed` that carries the Hopf
`frequency`. On $\dot z = (b_1 + i) z + b_2 z|z|^2 + b_3 z|z|^4 - z|z|^6$, seeded at
$b_1 = b_2 = 0$, $b_3 = 0.6$ and continued in $b_3$, it lands at $b_3 = 0$ to round-off.

**Limits.**

- The generalized-Hopf system needs $\kappa = \omega^2 > 0$; a curve running into a
  Bogdanov-Takens point or a neutral saddle leaves the domain of its $l_1$ row.
- The cusp and generalized-Hopf systems supply analytic state columns but difference
  the three parameter columns; the Bogdanov-Takens system is differenced throughout.
- As in two parameters, detection is by sign change between accepted points.

References: [GovaertsKuznetsovSijnave2000], [GovaertsEtAl2019], [Kuznetsov1999]. API:
[`codim3_builder`, `codim3_config`, `generalized_hopf_curve`](../api/continuation.md).

## Telling a SNIC from a homoclinic orbit

**Problem.** A branch of limit cycles whose period grows without bound ends in a global
bifurcation. Two common ones look alike: a saddle-node on an invariant circle (SNIC),
where a pair of equilibria is born on the cycle itself, and a saddle homoclinic orbit.

**Idea.** Near a SNIC the flow on the circle is, after a change of coordinate and
parameter, $\dot x = \mu + x^2 + o(x^2)$, and the orbit spends almost its whole period in
the bottleneck, which takes $\pi/\sqrt{\mu}$ (Baesens and MacKay 2013, section 1). So
$T \sim (\mu - \mu_c)^{-1/2}$, while a homoclinic period grows only like
$-\log|\mu - \mu_c|$. `characterize_snic` searches for the $\mu_c$ that makes
$\log T$ linear in $\log(\mu - \mu_c)$ and reports the slope and the fit.

```python
import numpy as np

from discrecontinual_equations.continuation.snic import characterize_snic

mu = np.linspace(0.01, 0.2, 20)
# The circle x' = mu + 2 (1 - cos x): a SNIC at mu = 0, period 2 pi / sqrt(mu^2 + 4 mu).
snic = characterize_snic(list(mu), list(2 * np.pi / np.sqrt(mu**2 + 4 * mu)))
print(f"SNIC:       mu_c = {snic.critical_parameter:+.4f}, "
      f"exponent = {snic.exponent:.3f}, law holds: {snic.is_inverse_square_root}")
# SNIC:       mu_c = -0.0005, exponent = -0.514, law holds: True

# A saddle homoclinic orbit: T ~ -log(mu) diverges far more slowly.
loop = characterize_snic(list(mu), list(1.0 - np.log(mu)))
print(f"homoclinic: exponent = {loop.exponent:.3f}, "
      f"law holds: {loop.is_inverse_square_root}")
# homoclinic: exponent = -0.357, law holds: False
```

**Limits.**

- Samples must lie on one side of $\mu_c$ and close enough for the leading term to
  dominate; the example's exponent, $-0.514$ rather than $-0.5$, is the $\mu^2$ term.
- $\mu_c$ is searched for only below the smallest sample, so a branch whose period
  diverges as the parameter *increases* must be passed with the parameter negated.
- A good fit is evidence, not proof: over a narrow window a logarithm can pass for a
  power. Confirm that equilibria appear on the former cycle.
- The thresholds ($R^2 > 0.999$, $|p + \tfrac12| < 0.08$) are heuristics.

References: [BaesensMacKay2013]. API: [`snic`](../api/continuation.md).
