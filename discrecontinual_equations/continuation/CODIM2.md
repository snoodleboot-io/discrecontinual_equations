# Codimension-2 bifurcations (two-parameter continuation)

A codimension-2 bifurcation is generically met while continuing a codim-1 curve in
**two** parameters. This module continues a fold curve or a Hopf curve and scans it
for codim-2 events. It reuses the entire codim-1 engine: a bifurcation curve is
just the zero set of an augmented residual, so the same pseudo-arclength corrector,
tangent predictor, step controller, and localizer continue it unchanged.

## Method

The active parameters are `continuation_parameter_index` (call it `p_a`) and
`second_parameter_index` (`p_b`). `p_b` plays the role of the arclength parameter;
everything else - including `p_a` - is solved for along the curve.

**Fold curve** (`curve="fold"`) continues the Moore-Spence system

    f(u; p_a, p_b) = 0,   f_u q = 0,   c . q - 1 = 0

with unknowns `(u, q, p_a)`. **Hopf curve** (`curve="hopf"`) continues the real
form of `f_u q = i omega q`, with unknowns `(u, q_R, q_I, omega, p_a)`; here
`omega` is an explicit coordinate.

Along the curve, each `Codim2TestFunction` is a scalar over the original spectrum
whose sign change brackets one codim-2 type:

| Event | Curve | Test function | Registry key |
|-------|-------|---------------|--------------|
| Bogdanov-Takens | Hopf | `omega -> 0` | `bogdanov_takens` |
| Bogdanov-Takens | fold | non-null eigenvalue `-> 0` | `fold_bogdanov_takens` |
| Zero-Hopf (fold-Hopf) | Hopf | `det(f_u)` changes sign | `zero_hopf` |
| Hopf-Hopf | Hopf | second pair reaches the axis | `hopf_hopf` |

New events are added by registering another test function; the analyzer and driver
do not change.

## Usage

```python
seed = CurveSeed(state=[...], parameter_a=..., frequency=...)  # a codim-1 point
config = Codim2Config(
    continuation_parameter_index=0,
    second_parameter_index=1,
    curve="hopf",
    codim2_detectors=["bogdanov_takens"],
    initial_parameter=...,
    direction=1,
)
result = Codim2Driver.run(config, equation, seed)
result.points  # detected Codim2Point objects
result.curve  # the continued curve as an augmented Branch
```

The seed is a codim-1 fold or Hopf point (found by ordinary continuation); the
driver builds the eigen/null vector, continues the curve, and analyzes it.

## Derivatives: automatic differentiation

The curve residuals and the normal-form coefficients need derivatives of `f`. A
`DerivativeProvider` supplies them, selected by the `derivative_provider`
configuration key:

- `automatic_differentiation` (default) evaluates the field on truncated Taylor
  jets, giving **exact** derivatives to any order over real or complex directions.
  This makes the augmented curve Jacobian and every coefficient below exact.
- `finite_difference_derivatives` is a fallback (central differences, first order
  in practice, real directions only).

Automatic differentiation supports any field written in arithmetic
(`+ - * / **`), which covers all polynomial and rational vector fields. A field
built from transcendental functions needs the finite-difference provider (or a
jet-aware elementary library).

## Detectors

On the **fold** curve: `bogdanov_takens`, `zero_hopf` (shared kind with the Hopf
curve), and `cusp`. The cusp test is the quadratic coefficient
`a = (1/2) <p, B(q, q)>`; it vanishes at a cusp.

On the **Hopf** curve: `bogdanov_takens` (omega through zero), `zero_hopf`,
`hopf_hopf`, and `generalized_hopf`. The generalized-Hopf test is the first
Lyapunov coefficient `l1` (Kuznetsov's invariant formula, built from `B` and `C`
with complex eigenvectors); it vanishes at a Bautin point. Its **sign** gives
criticality (negative = supercritical), which is what the zero crossing detects,
independent of scaling convention.

## Accuracy and limits

- The eigenvalue-based tests use only the spectrum of `f_u`; detection and
  location are robust to a few parts in `1e-4` to `1e-3`.
- `cusp` and `generalized_hopf` use second and third derivatives; with the
  automatic-differentiation provider these are exact, so their accuracy is limited
  by the continuation step, not by differentiation.
- `hopf_hopf` identifies the second pair by excluding the critical pair nearest
  `+/- i omega`; if two pairs are nearly coincident the identification can swap.

## Codimension three

`Codim3Driver` continues a codim-2 curve in three parameters and detects codim-3
points; the curve is chosen by `codim3_curve`.

- `cusp` continues the **cusp curve** (the fold system augmented with `a = 0`) and
  detects the **swallowtail**, where the cubic fold coefficient
  `b = (1/6) <p, C(q, q, q)>` changes sign. Verified against the swallowtail normal
  form.
- `generalized_hopf` continues the **generalized-Hopf (Bautin) curve** (the Hopf
  system augmented with `l1 = 0`) and detects the **degenerate Bautin**, where the
  second Lyapunov quantity `l2` changes sign. Verified against an extended Bautin
  normal form.

Automatic differentiation supplies the third-, fourth-, and fifth-order
derivatives exactly, which is what makes robust codim-3 detection possible.

### The Lyapunov quantities and their validation

`l1` and `l2` are the planar focal values, computed by the classical
Lyapunov-function recursion (`focal.py`) on the exact bivariate Taylor expansion of
the field. Their signs are what the detectors use. The computation is validated as
follows:

- The **first** quantity matches, with an exact constant ratio, both the
  Guckenheimer-Holmes closed-form first Lyapunov coefficient and the independent
  Kuznetsov `l1` (from `normal_form.py`) across systems that include quadratic
  terms - so the recursion, including its handling of quadratic nonlinearities, is
  correct.
- The **second** quantity has the sign of `s` on the Bautin normal form
  `z' = (b1 + i) z + b2 z|z|^2 + s z|z|^4` for `s = +/-1`. Because the recursion is
  order-agnostic and already validated at fourth order (including quadratics), the
  sixth-order quantity is trustworthy for its sign.

The focal-value route is planar (two state variables). For systems of higher
dimension the field is first reduced to its two-dimensional center manifold and the
same recursion is applied to the reduced planar field, so the generalized-Hopf and
degenerate-Bautin detectors now apply in any dimension. The reduction is validated
against the independently verified n-dimensional first Lyapunov coefficient and by
embeddings that preserve the planar quantities exactly - see `CENTER_MANIFOLD.md`
for the method and the full validation.


