# Center manifold reduction

At a Hopf point the asymptotic dynamics of an `n`-dimensional system collapses
onto a two-dimensional invariant center manifold tangent to the eigenspace of the
critical pair `±iω`. Reducing to that manifold is what lets the first and second
Lyapunov quantities — and therefore the generalized-Hopf and degenerate-Bautin
detectors — apply in any dimension rather than only in the plane.

## Design

The reduction is a pipeline of single-responsibility collaborators rather than a
monolithic routine, so each stage can be tested and varied on its own:

- `SpectralSplit` (abstraction) chooses the coordinates that separate the center
  pair from the hyperbolic directions and returns a `CoordinateSplit` value object
  (transform, inverse, frequency). `EigenvalueSplit` is the eigenvector-based
  implementation; the real center basis `[Re v, -Im v]` puts the linear center part
  in the rotation form `[[0, -w], [w, 0]]`. `TaylorCenterManifold` depends on the
  abstraction and defaults to `EigenvalueSplit`, so an alternative split (for
  example a Schur-based one) can be injected without touching the rest.
- `FieldExpansion` evaluates the vector field on seeded `Multivariate` series in the
  split coordinates to get its exact local Taylor expansion, and packages the
  center rows, hyperbolic rows, and hyperbolic linear block as a `SplitField`.
- `ManifoldExpansion` solves the center manifold `w = W(u)` from the invariance
  equation `DW(u) . u_dot = w_dot`, delegating each degree to a
  `HomologicalEquation`. Because the hyperbolic block has no eigenvalues on the
  imaginary axis, the operator at each degree is invertible, so `W` is a unique
  linear solve at degrees 2 through 6.
- `ReducedField` substitutes `w = W(u)` back into the center rows (evaluating the
  multivariate expansion at `Bivariate` arguments), time-scales so the linear part
  is a pure rotation, and yields the reduced planar field. `TaylorCenterManifold`
  hands that field to the verified focal-value recursion (`focal.recurse`).

`CenterCoordinates` holds the two center seed series and is shared by the manifold
solve and the reduction so the substitution is defined in exactly one place.

## Status: implemented and validated

The reduction is wired into `TwoParameterProblem.lyapunov_quantities`, which uses
the planar recursion directly in two dimensions and the center manifold reduction
above in higher dimensions. It is exercised by the generalized-Hopf curve residual
and the degenerate-Bautin detector, so both now work for systems of any dimension.

Validation (see `tests/test_center_manifold.py` and the `TestCenterManifoldCodim3`
case in `tests/test_codim3.py`):

- **First quantity against an independent reference.** The reduced first Lyapunov
  quantity matches the independently verified `n`-dimensional Kuznetsov
  coefficient with the same constant ratio (`0.25`) found in the planar
  cross-check, with matching signs, across coupled three- and four-dimensional
  systems — including two-way coupling and both stable and unstable extra
  directions, where the center manifold is genuinely non-flat.
- **Second quantity, end to end.** Embeddings that leave the center dynamics
  unchanged — a fully decoupled hyperbolic direction, and a hyperbolic direction
  driven one way by the center (a non-trivial `W` with no feedback) — reproduce the
  planar `(η₄, η₆)` exactly for both stability signs, certifying the whole pipeline
  (splitting, transform, order-6 manifold solve, normalisation, recursion).
- **Codim-3 in higher dimensions.** A degenerate Bautin embedded in a
  three-dimensional system with a coupled stable direction is detected exactly at
  the origin by continuing the generalized-Hopf curve.

The one piece without a fully independent oracle is the feedback contribution of
the manifold specifically to the *second* quantity at fifth order; it rests on the
same order-agnostic invariance solve that is certified for the first quantity with
feedback and for the second quantity without feedback, together with the verified
planar recursion. A future `n`-dimensional second-Lyapunov reference (the analogue
of the Kuznetsov `l₁` check) would close that gap.
