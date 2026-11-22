# Center manifolds and normal forms

A continued curve tells you *where* a codimension-2 point is: the test functions in
`codim2` change sign there. It does not tell you *which* one it is. Two
Bogdanov-Takens points can unfold differently, a zero-Hopf may or may not shed an
invariant torus, and a double Hopf is "simple" or "difficult". Those questions are
answered by the coefficients of the normal form on the center manifold, and this
page is about computing them.

Two reductions live in the library:

| | `TaylorCenterManifold` (`center_manifold.py`) | Normal forms (`codim2_normal_form.py`) |
|---|---|---|
| Points | Hopf (any dimension) | Hopf, Bogdanov-Takens, zero-Hopf, Hopf-Hopf, cusp |
| Center manifold | Taylor series `w = W(u)`, degree 6, solved order by order | Solved together with the normalising transformation (Kuznetsov's projection method) |
| Output | Focal values `(eta4, eta6)` | Normal-form coefficients and the case they decide |
| Used by | generalized-Hopf and degenerate-Bautin detection | `Codim2Point.classification`, the BT unfolding |

## The method

At a critical point write the field as `x' = A x + B(x, x)/2 + C(x, x, x)/6 + ...`
and look for the center manifold `x = H(w)` and the restricted field `w' = G(w)`
together, from the homological equation `H_w(w) G(w) = F(H(w))` (Kuznetsov 1999,
eq. 4.4). Collecting the `w^nu` terms gives a linear system `L h_nu = R_nu` for each
coefficient of `H`. Where `L` is singular, the Fredholm condition `<p, R_nu> = 0`
gives the normal-form coefficient and a bordered system gives `h_nu`. Nothing in
this depends on the dimension; the center manifold's curvature enters through the
`h_nu`.

The multilinear forms are exact. `CriticalForms` evaluates the field on Taylor jets
in a *joint* state-and-parameter direction (`TwoParameterProblem.joint_taylor`) and
recovers any symmetric form - `B`, `C`, the parameter derivative `J1`, the mixed
`A1(x, k)` - by polarisation. Any field written in arithmetic (`+ - * / **`) works;
transcendental functions do not.

## Usage

```python
from discrecontinual_equations.continuation import (
    bogdanov_takens_normal_form,
    hopf_hopf_normal_form,
    zero_hopf_normal_form,
)
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)

problem = TwoParameterProblem(equation, 0, 1, 0.0, AutomaticDifferentiation())

bt = bogdanov_takens_normal_form(problem, state, p_a, p_b)
bt.a, bt.b, bt.s                       # w1' = a w0^2 + b w0 w1; s = sign(ab)
bt.unfolding.homoclinic(bt.unfolding.side * 0.01)   # (p_a, p_b) on the curve

zh = zero_hopf_normal_form(problem, state, p_a, p_b)
zh.s, zh.theta, zh.torus, zh.e

hh = hopf_hopf_normal_form(problem, state, p_a, p_b)
hh.p11, hh.p12, hh.p21, hh.p22, hh.theta, hh.delta, hh.simple
```

Codim-2 points found by `Codim2Driver` are labelled automatically:

| `kind` | `classification` | `coefficients` |
|---|---|---|
| `bogdanov_takens` | `s=+1` / `s=-1` | `a`, `b`, `s` |
| `zero_hopf` | `s=±1, theta<0` / `theta>0`, plus `, torus` when `s theta < 0` | `b`, `c`, `s`, `theta`, `e`, `torus` |
| `hopf_hopf` | `simple` / `difficult`, or `resonant k:l` | `p11`, `p12`, `p21`, `p22`, `theta`, `delta` |
| `cusp` | `c<0` / `c>0` | `c` |

## What each number means

**Bogdanov-Takens.** `w0' = w1`, `w1' = a w0^2 + b w0 w1`, with
`a = <p1, B(q0, q0)>/2` and `b = <p0, B(q0, q0)> + <p1, B(q0, q1)>` over the Jordan
chains `A q0 = 0`, `A q1 = q0`, `A^T p1 = 0`, `A^T p0 = p1`. The unfolding
`w1' = beta1 + beta2 w1 + a w0^2 + b w0 w1` has a fold at `beta1 = 0`, a Hopf at
`beta1 = -(a/b^2) beta2^2` and a homoclinic at `beta1 = -(49/25)(a/b^2) beta2^2`,
the last two on the side `sign(beta2) = sign(ab)`. In Kuznetsov's coordinates
(`w1' = b1 + b2 w0 + w0^2 + s w0 w1`) these are `b1 = b2^2/4`, `b1 = 0` and
`b1 = -(6/25) b2^2`. `bt.unfolding` maps `(beta1, beta2)` back to the original
parameters through `beta2^2` (Bosschaert and Kuznetsov, section 2.2), so the
predicted curves are right to `O(beta2^3)`.

**Zero-Hopf.** The coefficients `G200 ... G021` of Kuznetsov's (3.4), and from them
`b = G200/2`, `c = G011`, `d = G110 - i omega G300 / (3 G200)` and `e`. After scaling,
the amplitude equations are `xi' = beta1 + xi^2 + s r^2`, `r' = r (beta2 + theta xi)`
with `s = sign(bc)` and `theta = Re d / b`. Their nontrivial equilibrium - a cycle of
the full system - has a Hopf bifurcation, which is a torus (Neimark-Sacker)
bifurcation of the full system, exactly when `s theta < 0`: that is `torus`.

**Hopf-Hopf.** The four resonant cubic coefficients of Kuznetsov's (3.6), with
`omega1 > omega2`. In amplitudes `r_j' = r_j (mu_j + p_j1 r1^2 + p_j2 r2^2)` with
`p11 = Re G2100 / 2`, `p12 = Re G1011`, `p21 = Re G1110`, `p22 = Re G0021 / 2`. The
case is simple when `p11 p22 > 0`; `theta = p12/p22` and `delta = p21/p11` are the
cross coefficients once the self-couplings are scaled to `-1`. The cubic normal
form needs `k omega1 != l omega2` for `k + l <= 5`; a strongly resonant point is
labelled `resonant k:l` instead (the atlas's double-Hopf example, with frequencies
1 and 2, is one).

**Hopf.** `G21` and `l1 = Re G21 / (2 omega)`, the convention of
`normal_form.first_lyapunov`. MatCont prints `Re G21 / 2` (`matcont_lyapunov`).

## Normalisation

Raw coefficients depend on how the critical eigenvectors are scaled. The library
uses MatCont's choice - unit length, and `q1 . q0 = 0` at a Bogdanov-Takens point -
with the phase fixed so the largest component is real and positive. Scaling `q0`
by `c` multiplies `(a, b)` by `c`; scaling `q1` at a zero-Hopf by `k` multiplies
`G011` by `|k|^2/c`, and so on. The classifying quantities - `s`, `theta`, `delta`,
`simple`, `torus`, `b/a` - do not depend on the choice.

The same is true of `TaylorCenterManifold`. Its focal values are for the center
coordinates the eigenvector split happens to produce, so `eta4` scales with the
square and `eta6` with the fourth power of that basis; adding stable directions, or
changing coordinates, changes the raw numbers and not their signs. And `eta6` is an
invariant only where `eta4 = 0` - the only place the detectors use it. Embedding the
same planar Bautin among 1 to 6 driven stable modes behind a random similarity gives
`eta6 / scale^4 = 0.5` every time (`TestIndependentOfDimension`).

## Validation

| Claim | Check | Result |
|---|---|---|
| Coefficients are the normal form's own | Each normal form among 1-2 extra stable modes, behind a random similarity, coefficients compared after the eigenvector scale is read back | BT, zero-Hopf, Hopf-Hopf, cusp: agree to `1e-9` relative |
| Cubic coefficients with stable feedback | Independent Poincare-Dulac normal form in eigen-coordinates on dense random cubic fields | zero-Hopf and Hopf-Hopf: agree to `1e-9` |
| `e(0)` formula | Normalising transformation computed explicitly on polynomials, three choices of the free parameter | `Re` of the `v^2 zeta` coefficient equals `e` to `1e-12` |
| Published fold-Hopf | Kuznetsov (1999), section 10, "new" Lorenz model | `G200, G110, G011, G111, G210, G021, b, c, Re d/G200` to `1e-9`; see below for `G300` and `e` |
| Published BT | MatCont tutorial IV, Bykov-Yablonskii-Kim model: `(a, b) = (-8.378442e-2, -2.136280)` | agree to all printed digits |
| Published Hopf | Same tutorial, `l1 = 10.70259` | `10.70102` at the exact Hopf; MatCont's point is `5e-7` off in `Q2`, worth `1.6e-3` |
| BT unfolding to second order | Nonlinear unfolding with a moving center manifold; distance from predicted to exact fold and Hopf | falls as `beta2^3.01` and `beta2^3.00` |
| Homoclinic asymptotics (oracle) | Continued homoclinic points of the atlas BT, mesh error removed by Richardson (121/241 nodes) | `b1/b2^2 = -6/25 + 7e-6 + 0.028 abs(b2)` in Kuznetsov's coordinates |
| Hopf regression (oracle) | `TaylorCenterManifold` on a coupled 3D Hopf, `omega = 1` and `2.5` | `eta4 = l1 / 4` to `1e-9` |

**The Lorenz-84 example's `G300`.** Every coefficient printed in Kuznetsov (1999,
section 10) is reproduced except `G300`, which the paper gives as
`25602814720/36130026241 = 0.7086`. Under the paper's own normal form (3.4), where
the `w0^3` term is `G300/6`, the coefficient is three times that, `2.1259`: the
library's value (the paper's own section 8 formula) and the independent
Poincare-Dulac reference's agree.
The printed `e(0) = 0.2106514716` is the paper's formula evaluated at the printed
`G300`; at the consistent `G300` it is `1.6503`, which is also what the explicit
normalising transformation gives. The sign, and so the paper's conclusion about the
model, is unchanged.

## Limits

- Bogdanov-Takens: the unfolding stops at `beta2^2`. The `K01` components of `K10`
  and `K02` (`delta2`, `delta3`) and `K11` are not computed; they move points along
  the curves without moving the curves at that order. The third-order homoclinic
  predictor of Bosschaert and Kuznetsov is not implemented.
- Hopf-Hopf: cubic order only. The fifth-order coefficients that settle the
  difficult case (Kuznetsov 1999, `G3200 ... G0032`) are not computed.
- The second Lyapunov coefficient comes from `TaylorCenterManifold`'s focal value,
  not from Kuznetsov's `G32`; the `r^6` coefficient at a degenerate Bautin point is
  not computed.

## References

- Yu. A. Kuznetsov, "Numerical normalization techniques for all codim 2 bifurcations
  of equilibria in ODE's", SIAM J. Numer. Anal. 36 (1999) 1104-1124.
- M. M. Bosschaert and Yu. A. Kuznetsov, "Interplay between normal forms and center
  manifold reduction for homoclinic predictors near Bogdanov-Takens bifurcation",
  SIAM J. Appl. Dyn. Syst. (2024); arXiv:2109.12570.
- Yu. A. Kuznetsov and N. Neirynck, MatCont tutorial IV (2021).
- Yu. A. Kuznetsov, *Elements of Applied Bifurcation Theory*, 3rd ed., Springer
  (2004), chapter 8.
