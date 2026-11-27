"""The Freidlin-Wentzell quasipotential of a planar flow, and its minimum action paths.

The stationary density of :mod:`.fokker_planck` says where the mass sits and the
exponent of :mod:`.lyapunov` whether paths escape; neither says how a path gets from
one attractor to another. At small noise that is answered by the quasipotential.
For ``dx = f dt + sqrt(eps) G dW`` with ``D = G G^T``, the probability of a path
``psi`` is ``exp(-S / eps)`` with the action ``S = (1/2) integral (psi' - f)^T D^{-1}
(psi' - f) dt``, and the quasipotential ``U(x)`` of an attractor is the least action
over all paths that leave it and reach ``x`` in any time. Escape rates go like
``exp(-U / eps)``, the stationary density like ``exp(-U / eps)`` to leading order, and
the minimiser is the most probable transition path.

*Normalisation.* The rest of the library writes ``dx = f dt + G dW`` with no ``eps``,
so ``U`` is defined here with ``D = G G^T`` exactly as :mod:`.fokker_planck` builds it
- the ``eps = 1`` reading - and solves the Hamilton-Jacobi equation

    ``H(x, grad U) = grad U . f + (1/2) grad U^T D grad U = 0``.

In this convention a gradient drift ``f = -grad V`` with ``G = I`` has ``U = 2 V``,
the linear system ``dx = A x dt + G dW`` has ``U = (1/2) x^T S^{-1} x`` with
``A S + S A^T + D = 0``, and in both the density the grid solve returns is
``exp(-U)`` up to a constant; the tests assert all three. (The ticket wrote the
linear case as ``x^T S^{-1} x``. Under the Hamiltonian it also states that is twice
too large: substituting ``U = x^T M x`` gives ``M A + A^T M + 2 M D M = 0``, solved
by ``M = S^{-1} / 2``.) The Ito-Stratonovich correction to the drift is of order
``eps`` and does not enter ``U``: the drift here is ``f`` as written.

*Method.* ``U`` is computed on the grid of :class:`~.fokker_planck.DensityGrid` by
the ordered line integral method of Dahiya and Cameron (J. Sci. Comput. 2018,
arXiv 1706.07509), by default their OLIM-MID: midpoint quadrature, with ``U``
interpolated linearly along the front. ``U`` may instead be interpolated by a
parabola (``update_order="quadratic"``, see *Order* below), which is not a
published method and pays only on a drift that is linear or nearly so. Grid
points are accepted in increasing order of ``U``, as in Dijkstra's method; when
a point ``x0`` is accepted every unaccepted point ``x`` within ``update_radius``
cells of it is offered

* a one-point update ``U(x0) + Q(x0, x)``, and
* a triangle update for every accepted neighbour ``x1`` of ``x0``: the minimum over
  ``x_s = x0 + s (x1 - x0)``, ``s`` in ``[0, 1]``, of the interpolated ``U(x_s)``
  plus ``Q(x_s, x)``,

where ``Q(a, x) = |x - a|_A |f_m|_A - <x - a, f_m>_A`` is the geometric action of the
straight segment by the midpoint rule, in the metric ``A = D^{-1}`` taken at the
midpoint. Midpoints of two grid points always lie on the grid of half the spacing,
so ``f`` and ``A`` are evaluated once there and looked up; along a triangle's
segment they are interpolated linearly between its two end midpoints. This
matches Dahiya and Cameron's handling of a variable anisotropic ``D`` (Physica D
2018, arXiv 1806.05321, their Eqs. (19), (21), (22)): ``A`` is ``D^{-1}`` at each
midpoint, not the inverse of an averaged ``D``, and ``f`` and ``A`` are linear
along the base. How the minimum over ``s`` is found is the subject of *Triangle
minimisation* below. The update radius is what lets the
method see around an anisotropic or rotating drift: a minimum action path can meet
the accepted front at a shallow angle, and only a stencil wide enough to reach back
along it finds the right predecessor. Every triangle is tried rather than the
hierarchical subset of the paper (triangles only round the best one-point
predecessor): the subset was measured here and doubled the rotating
Ornstein-Uhlenbeck error below at 161 points (1.06e-3 against 5.0e-4).

The attractor is initialised exactly where it can be: around a stable equilibrium
``x*`` the points within a couple of cells get the quadratic quasipotential of the
linearisation, ``(1/2) (x - x*)^T S^{-1} (x - x*)`` with ``J S + S J^T + D(x*) = 0``;
around a limit cycle the points within a cell and a half get the expansion of
``U`` across the cycle, ``(1/2) G z^2 + (1/6) C z^3`` in the distance ``z``
along its normal, with ``G`` the periodic solution of Lin, Yu and Zhou's
Riccati equation along the orbit and ``C`` from the next order (see
:class:`CycleAttractor`). Before DEQ-151 it was Dahiya and Cameron's start,
which takes the drift's rotational part parallel to the drift at the nearest
point of the cycle: wrong at leading order where the rotation turns across
the cycle, and below zero inside the verification's fast elliptic cycle.

*Measured.* The error in ``U`` against the closed forms, by doubling the grid on a
fixed box, at a radius held at 8 cells (the tests repeat these):

  - double well ``V = x^4/4 - x^2/2 + y^2/2``, ``D = I``, max error of ``U - 2 (V -
    V_min)`` on the basin of ``(-1, 0)`` within ``{U <= 2}``, box ``[-2, 1] x
    [-1.5, 1.5]``: 0.153, 0.0420, 0.0113 at 41, 81, 161 points per side, observed
    orders 1.87 and 1.89;
  - rotating Ornstein-Uhlenbeck, ``A = [[-1, -2], [2, -1]]``, ``G = [[1, 0.5],
    [0, 0.8]]``, max error against ``(1/2) x^T S^{-1} x`` on ``{U <= 1.5}``, box
    ``[-1.5, 1.5]^2``: 2.27e-3, 9.68e-4, 5.00e-4, observed orders 1.23 and 0.95
    (1.70e-4, 4.25e-5, 1.16e-5 with the parabolic base, orders 2.00 and 1.87).

The error is two terms of different order. Linear interpolation of ``U`` along the
base of a triangle costs ``O(h^2)`` per update over ``O(1 / (K h))`` updates along a
path, which is first order, and is the bound Dahiya and Cameron state for every
OLIM; the rotating case showed it, and the parabolic base below removes it. The
midpoint rule on a one-point update of length up to ``K h`` costs
``O((K h)^3 |f''|)`` per update, second order overall but with a constant growing
like ``K^2``, and since the march keeps the minimum and the rule underestimates a
convex integrand, it is the long updates that are picked. The double well's
``x - x^3`` is steep enough on this box for that term to dominate: at 81 points
the error is 0.0107, 0.0112 and 0.0420 for radii 2, 4 and 8.

*Order.* With ``update_order="quadratic"`` ``U(x_s)`` along a triangle's base is
the parabola through ``x0``, ``x1 = x0 + e`` and a third accepted point on the
same line, ``x0 - e`` or else ``x0 + 2 e`` - straight only where neither is
accepted - so its interpolation error is ``O(h^3)`` rather than ``O(h^2)``.
``F(s)`` is then minimised exactly (see *Triangle minimisation*). This is not a
published method, and it is not the default. The ticket proposed a Hermite cubic
with ``grad U`` from the march; it was built first and failed: ``grad U`` at a
point can only come from the update that set it, by the momentum ``|f|_A A t /
|t|_A - A f`` along the segment's direction ``t``, which is first order (the
chord), or second order if the start's own ``grad U`` fixes the curvature (the
chord's reflection) - and then each point's error feeds the next. On the
rotating case at 41 points the median ``grad U`` was off by 25% from the chord
and 80% by reflection, and ``U`` by 0.32. Dahiya and Cameron also warn that
``U`` need not be smooth across the front (arXiv 1806.05321, Sec. 3.3.2); the
parabola uses accepted values only.

The parabola does not raise the order: the quadrature term above is ``O(K^2
h^2)`` either way. What it does is make the update exact where ``U`` is
quadratic - a linear drift with a constant ``D``. There the midpoint rule's
``Q`` is exact along the characteristic through the segment's midpoint
(``grad U`` at the midpoint is exact for a quadratic ``U``, and
``grad U . (x - a) <= Q(a, x)`` with equality along it), the parabola is
exact, and what is left is how well ``s`` is found: on the rotating
Ornstein-Uhlenbeck process the Newton steps with ``f`` and ``A`` frozen that
the parabola used before DEQ-116 left 1.70e-4, 4.25e-5 and 1.16e-5 at 41, 81
and 161 points, and the exact root leaves 3.1e-15, 4.4e-15 and 8.0e-15 -
rounding; the non-normal linear system likewise goes from 1.08e-3 and
5.16e-4 at 41 and 81 points to 1.5e-15 and 1.1e-15. A linear drift that
rotates faster than the radius reaches is still limited by the radius (SDE
(21): 1.38e-3, 4.62e-4, 5.38e-4). On a nonlinear drift the straight base's
overestimate of a convex ``U`` partly cancels the midpoint rule's
underestimate, and the parabola takes that away; the exact minimum takes
away more (below).

When it helps - linear or nearly linear drifts. Max error at 321 points,
radius 8, straight against parabolic, measured before DEQ-116 made the
parabola's minimisation exact: rotating Ornstein-Uhlenbeck 3.34e-4 and
3.41e-6; their SDE (21) 1.34e-3 and 3.73e-4; the non-normal linear system
2.36e-4 and 9.26e-5; Maier-Stein's barrier at 161 x 129 points (radius 10)
5.6e-4 and 2.0e-5. Since then the rotating and the non-normal systems are
exact to rounding at every size measured (above).

When it does not - nonlinear drifts. Before DEQ-116, at 321 points, straight
against parabolic: double well 2.85e-3 both; anisotropic gradient 2.09e-3 and
2.15e-3; SDE (24) 4.66e-3 and 5.15e-3; and on the verification's nonlinear
rotating systems (``f = -D grad V + c R grad V`` with ``c = 4``, a
multiplicative-noise one, an elliptic cycle with rotation 3) 7.1e-4 and 5.3e-4,
8.2e-4 and 1.27e-3, 8.5e-3 and 5.3e-3. The exact minimisation made the parabola
worse again on every one of these, by 1.2 to 1.45 times at 41, 81 and 161 points
(on the verification's masks, before against after: double well 0.153, 0.0420,
0.0113 against 0.210, 0.0524, 0.0136; anisotropic gradient 0.145, 0.0395,
8.70e-3 against 0.207, 0.0442, 0.0110; SDE (24) 0.381, 0.0907, 0.0219 against
0.525, 0.129, 0.0306; ``c = 4`` 0.0464, 9.85e-3, 2.23e-3 against 0.0654, 0.0143,
3.06e-3; multiplicative 0.0867, 0.0215, 5.16e-3 against 0.108, 0.0280, 6.86e-3;
elliptic cycle 0.505, 0.133, 0.0317 against 0.594, 0.161, 0.0390): the frozen
steps' shortfall had been offsetting the midpoint rule's underestimate there
too. The parabola also costs more than the straight base: 2.3 to 2.8 times its
march before DEQ-116, and the exact root a further 1.6 to 1.9 times at 61 to 161
points (best of three, under load; the march stays near linear in the number of
points, a log-log slope of 1.04 to 1.05 against 0.91 to 0.93 before), so at
equal cost the comparison is the straight base at 321 points against the
parabola at 161, and the straight base wins on every nonlinear problem measured:

  ==========================  ===============  ================
  problem                     straight, 321    parabolic, 161
  ==========================  ===============  ================
  double well                 2.85e-3          1.36e-2
  anisotropic gradient        2.09e-3          1.10e-2
  SDE (24)                    4.66e-3          3.06e-2
  rotating, c = 4             7.13e-4          3.06e-3
  multiplicative noise        8.19e-4          6.86e-3
  elliptic cycle              8.54e-3          3.90e-2
  rotating Ornstein-Uhlenbeck 3.34e-4          8.0e-15
  ==========================  ===============  ================

(The parabolic column is on the verification's masks; before DEQ-116 those
gave the same three digits as the masks the straight column was measured on.)
So the parabola is for a drift that is linear, or nearly so on the region
asked about; ``update_order="midpoint"``, the default, is OLIM-MID with the
departures listed at the end. On the noisy Hopf film's grid, near its
singular origin, the worst frame was 0.054 straight and 0.056 parabolic
(before DEQ-116).

*Triangle minimisation.* Dahiya and Cameron minimise over ``s`` exactly: they
solve ``F'(s) = 0`` for the full derivative - their Eq. (23), which includes
the change of ``f`` and ``A`` along the base - by Wilkinson's hybrid secant and
bisection method, where ``F'`` changes sign between the ends. The default
straight base here does not: it freezes ``f`` and ``A`` at the current ``s``,
where ``F'(s) = 0`` is a quadratic with a closed-form root, and refines that
twice from the middle of the segment. The exact root was built and measured
(DEQ-116; max error at 41, 81, 161 points, radius 8, on the verification's
masks, frozen against exact): on linear drifts it is better - rotating
Ornstein-Uhlenbeck 2.89e-3, 1.23e-3, 6.14e-4 against 1.90e-3, 7.53e-4,
4.10e-4; SDE (21) 5.35e-3, 3.23e-3, 2.28e-3 against 4.38e-3, 2.13e-3,
1.71e-3 - and on nonlinear ones worse at every size: double well 0.153,
0.0420, 0.0113 against 0.210, 0.0524, 0.0135; anisotropic gradient 0.145,
0.0391, 8.67e-3 against 0.203, 0.0433, 0.0106; rotating, ``c = 4``, 3.89e-2,
6.55e-3, 1.23e-3 against 5.70e-2, 1.09e-2, 1.64e-3; elliptic cycle 0.472,
0.119, 1.88e-2 against 0.566, 0.141, 2.75e-2. The frozen step stops short of
the least value, and that partly offsets the midpoint rule's underestimate of
a convex integrand, which the exact minimum takes in full. So the default keeps
the frozen step, a departure from the paper made on these measurements.

The parabolic base solves Eq. (23) - with ``U(x_s)`` the parabola - by the same
kind of hybrid: ``F'`` at the frozen root and at the end it points away from
bracket the minimum, and secant steps are taken inside the bracket, a
bisection whenever a step is not half the one before last (Brent's rule); the
least of ``F`` at that root, at the frozen root and at both ends is kept, the
ends being the one-point updates, so the triangle never offers more than they
do. Unlike the paper's, the bracket need not span the whole base: a minimum
next to the frozen root is found where ``F'`` has one sign at both ends. On a
march it reaches the least ``F`` over a fine grid of ``s`` on all but about
one triangle in ten thousand (59 of 529,376 on the rotating process at 41
points, by at most 8e-4: a second local minimum that neither bracket holds),
where the Newton steps it replaces missed it on one in five (107,958, by up to
0.21).

*Radius.* The radius is 8 cells on every grid unless ``update_radius`` says
otherwise. Dahiya and Cameron's rule of thumb ``K = 10 + 4 (round(log2 N) -
7)`` (their Eq. (28), for ``2^7 <= N <= 2^12``), ``N`` the longer side, held at
8 below 128 points, is :func:`default_radius`, to opt into. It trades the two
terms above, and the trade is not the same everywhere. At 321 points, radius 14
against 8, the straight base: rotating Ornstein-Uhlenbeck 9.30e-5 against
3.34e-4, SDE (21) 2.55e-4 against 1.34e-3, non-normal 1.04e-4 against 2.36e-4
- but the double well 9.05e-3 against 2.85e-3, the anisotropic gradient
7.44e-3 against 2.09e-3 and SDE (24) 1.84e-2 against 4.66e-3, where the longer
updates' quadrature error grows like ``K^2`` and nothing is gained from
reaching further. With the parabolic base (at radius 14: rotating
Ornstein-Uhlenbeck 2.15e-6, SDE (21) 1.16e-5, non-normal 9.26e-5; double well
9.05e-3, anisotropic gradient 7.52e-3, SDE (24) 1.85e-2) the linear cases gain
less (SDE (21) 1.16e-5 against 3.73e-4, the others within a factor of two)
and the gradient-dominated ones lose as much. So the default is 8, and
``update_radius=default_radius(grid)`` is for a drift whose
rotation-to-gradient ratio ``a`` (below) is near or above 8 - SDE (21) has
``a = 10`` - on a grid of 128 points or more; a ratio well above the radius
calls for a radius at least that large whatever the grid.

*What the radius must cover.* Below a threshold the radius does not merely cost
accuracy: the error stops converging. Near the attractor the drift is a gradient
part ``-(1/2) D grad U`` plus a rotation ``l`` along the level sets, and a minimum
action path crosses the level sets at a shallow angle - about ``1 / a`` for
``a = |l|_A / |(1/2) D grad U|_A``, ``A = D^{-1}`` - so the accepted point it
comes from lies of the order of ``a`` cells back along the front. A stencil much
narrower than that cannot reach it, and every update is taken from the wrong
place, at every spacing. On the paper's SDE (21), where ``a = 10`` everywhere
(``U = 2 x^2 + y^2`` on ``[-1, 1]^2``, max error on ``{U <= 0.9}``): radius 2
gives 0.381, 0.370 and 0.371 at 129, 257 and 513 points, and radius 4 gives
0.048 and 0.051 at 129 and 257 - neither converges - against 1.6e-3 at radius 8
and 3.1e-4 at the paper's 14 on 257 points (all with the straight base). A
radius of 8 converges there, if slowly (2.28e-3 and 1.34e-3 at 161 and 321
points, order 0.76), and comfortably on the rotating Ornstein-Uhlenbeck process
(``a = 2.36``). So the radius
should be comparable to the largest rotation-to-gradient ratio on the region
asked about, and a field that does not settle on refining the grid is the
symptom of one that is too small.


*Paths.* :meth:`OrderedLineIntegral.minimum_action_path` follows the computed field
from a chosen point back to the attractor: along a minimum action path
``psi' = f + D grad U`` (the characteristic ``dH / dp``), so the path is traced
backwards along ``-(f + D grad U)`` with ``grad U`` interpolated from the grid. In the
gradient case that is steepest descent of ``V`` run in reverse. Between two chosen
points, :class:`GeometricMinimumAction` relaxes a whole path by the geometric
minimum action method of Heymann and Vanden-Eijnden (Comm. Pure Appl. Math. 2008),
in its semi-implicit form - the descent flow as Grafke, Schafer and
Vanden-Eijnden restate it (arXiv 1604.03818, Eq. (19) for additive noise, Eqs.
(38) to (42) for a state-dependent ``D``); the letter usually cited for the
semi-implicit form is Vanden-Eijnden and Heymann, J. Chem. Phys. 128, 061103
(2008). It needs no grid and crosses saddles, which a descent on ``U`` cannot.

*Errors.* The march is a single pass: a value is final once accepted, and nothing
in it says how accurate any value is. :meth:`OrderedLineIntegral.check` says so
point by point, on request, from a value computed without the grid
(:meth:`OrderedLineIntegral.reference`): the chain of
:class:`GeometricMinimumAction` relaxed to the point from a small surface about
the attractor - a level set of the quadratic form a quarter of the way out from
an equilibrium, the curve at the band's edge beside a cycle - on which ``U`` is
the attractor's own expansion (for an equilibrium the quadratic form plus the
cubic term of the Taylor series, so wrong at fourth order). The chain's start
slides along the surface to where ``U + S`` is least. Each check also estimates
the reference's own error, from a surface at twice the distance and a chain of
half the nodes, and the error of the grid value is estimated as its distance
from the reference plus that. With a tolerance the grid value is replaced where
the estimate exceeds it and the reference is good to it. The estimate tracks the
true error on the four reference problems (the tests, at 81 points): on the
non-normal linear system, the anisotropic gradient, SDE (24) and Maier-Stein,
every estimate is above the true error or within a factor of three of it
wherever that exceeds ``1e-4``.

*Where the reference has no bound.* That estimate rests on an assumption: that
what changes as the surface shrinks is the surface's own error. Where the path
winds round the attractor faster than a chain of nodes follows, the chain
resolves less of the winding the closer it starts, and the value gets worse
going in, not better - inside the verification's fast elliptic cycle the
reference at (0.3, 0) was 1.66 off with an estimate of 1.47 (DEQ-117). The
reference therefore also relaxes from a surface at half the distance, and
reports no bound (an estimate of ``inf``) where that moves the value by more
than the estimate, or where the first two surfaces' values differ by more than
half of ``U`` on the outer one. A check never stops at a point: one without a
bound is marked :data:`UNBOUNDED`, one where no reference could be computed
:data:`FAILED`; neither is refined, :attr:`GridQuasipotential.unbounded` lists
them and a warning names them. That a finite estimate bounds the error is
measured, not proven: across the verification's systems every finite estimate
measured does (see :meth:`OrderedLineIntegral.reference`), but the rule that
decides it has been wrong once already - beside the fast ellipse, where the
three surfaces' values converge only at first order, a finite estimate of
2.6e-5 stood against an error of 2.9e-5 until the estimate took the tail of
the series (DEQ-152).

*Winding paths* (DEQ-152). That cycle's failures had two causes, both since
removed. Its band started from a wrong ``U`` (see :class:`CycleAttractor`),
and the chain was a straight line: the traced path it starts from stops where
a step would raise ``U``, which round a fast rotation is almost at once, and
81 nodes cannot hold the four turns the true path makes. The trace is now
carried on into the start surface whatever ``U`` does, with the attractor's
expansion for ``grad U`` once inside its band, and the chain gets 320 nodes
for every turn the trace makes (81 at least); a start slid along the surface
moves the whole chain with it, not its first node alone. Every reference
of the verification's set beside and inside that cycle is now finite and
bounds its error (0.3, 0: 3.2e-4 off, estimate 4.9e-3, where the grid is
9.3e-3 off), and the strongly rotating equilibria gain as much: on SDE (21)
the references went from 2.1e-2 to 4.3e-2 off to 7e-6 to 1.4e-5. Two more
points beside the cycle, where the values converge too slowly to bound, are
labelled. A winding point costs far more than one that does not: up to 100
seconds inside the cycle at 81 points, 4 to 8 beside it, against 0.4 to 18
elsewhere.

The traced path's action was the first candidate, being cheap, and was measured
and rejected: its own error, from a path that follows the grid's gradient, is
the size of the grid's (at 81 points the median 3.6e-4 against a largest grid
error of 9.0e-4 on the non-normal system, 1.0e-3 against 3.4e-2 on the
anisotropic gradient), so the gap between the two was within a factor of three
of the true error, or above it, at only 90%, 82% and 94% of the points where
that exceeds ``1e-4`` on the non-normal system, the anisotropic gradient and
SDE (24), and beside the cycle the trace does not move at all. The grid-free
value costs a few seconds a point against about ten milliseconds for a trace,
which is why it is asked for point by point and never part of the march.

*Limits.* The method is planar - a two-dimensional :class:`DensityGrid` only; the
ordered line integral idea extends to three dimensions at a much higher cost and is
not implemented. The drift and noise must be time-independent: ``U`` is a static
object and a periodically forced system needs the quasipotential of its Poincare map
instead. ``D`` must be non-singular at every grid point (scalar noise, or noise on a
subset of the components, is refused, as it is by the density solve); a point
*between* grid points where ``D`` vanishes - the origin of the conformal noise
``sigma z dB`` - is treated as impassable, which is the quasipotential's own limit
there (it diverges logarithmically), but the values in the cells around it are not
claimed to the order above. ``U`` near the boundary of the box is wrong wherever the
true minimum action path would leave the box, so the box should contain the sublevel set
of ``U`` being asked about: ``U`` increases along every minimum action path, so the
path to ``x`` stays inside ``{U <= U(x)}``.

*Errata and departures*, each measured above:

  - The linear quasipotential is ``(1/2) x^T S^{-1} x``, not ``x^T S^{-1} x``
    as the DEQ-31 ticket wrote (*Normalisation*).
  - OLIM-MID here tries every triangle, not the paper's hierarchical subset,
    which doubled the rotating Ornstein-Uhlenbeck error (*Method*).
  - Dahiya and Cameron's radius rule, Eq. (28), is opt-in through
    :func:`default_radius`: it pays on strongly rotating drifts and costs on
    nearly gradient ones (*Radius*).
  - The parabolic update is opt-in and not a published method: exact to
    rounding on a linear drift with a constant ``D``, worse than the
    straight base on every nonlinear drift measured (*Order*).
  - The default straight base finds ``s`` with ``f`` and ``A`` frozen, not by
    the paper's exact Eq. (23), which was better on linear drifts and worse
    on nonlinear ones (*Triangle minimisation*).
  - The grid-free reference's error estimate is not a bound everywhere;
    where it cannot be shown to be one it is reported as ``inf``
    (*Where the reference has no bound*).

References:
    - Dahiya, D. and Cameron, M. (2018). Ordered line integral methods for
      computing the quasi-potential. Journal of Scientific Computing.
      arXiv:1706.07509, doi:10.1007/s10915-017-0590-9. Eqs. (2), (3): the
      geometric action and the Hamilton-Jacobi equation; Sec. 2.1.2, (7)-(12):
      the start beside a cycle, replaced by DEQ-151; (14), (15), (B-7), (B-8):
      the OLIM-MID updates; (21), (24): test systems; (28): the radius rule.
    - Dahiya, D. and Cameron, M. (2018). An ordered line integral method for
      computing the quasi-potential in the case of variable anisotropic
      diffusion. Physica D. arXiv:1806.05321,
      doi:10.1016/j.physd.2018.07.002. Eqs. (14)-(18): the start at an
      equilibrium; (19), (21), (22): the updates with a matrix ``D``; (23)
      and Wilkinson's hybrid secant/bisection solver: the exact triangle
      minimisation; Sec. 3.3.2: ``U`` need not be smooth across the front.
    - Heymann, M. and Vanden-Eijnden, E. (2008). The geometric minimum action
      method: a least action principle on the space of curves. Communications
      on Pure and Applied Mathematics 61(8), 1052-1117. doi:10.1002/cpa.20238.
      The gMAM; not read here, used as Grafke et al. restate it.
    - Grafke, T., Schafer, T. and Vanden-Eijnden, E. (2017). Long term effects
      of small random perturbations on dynamical systems: theoretical and
      computational tools. Fields Institute Communications, Springer.
      arXiv:1604.03818, doi:10.1007/978-1-4939-6969-2_2. Eq. (19): the gMAM
      descent flow, with the additive-noise Hamiltonian of Eqs. (32)-(35);
      Eqs. (38)-(42): the simplified flow for a state-dependent ``D``.
    - Lin, L., Yu, H. and Zhou, X. (2019). Quasi-potential calculation and
      minimum action method for limit cycle. Journal of Nonlinear Science
      29(3), 961-991. arXiv:1806.04031, doi:10.1007/s00332-018-9509-3. Eq.
      (23): the quadratic expansion of ``U`` about a cycle; Eq. (30): the
      periodic Riccati equation for its coefficient; Sec. 4.2, Eqs. (36),
      (37): the periodic solution in the plane; Eq. (32): its momentum.
"""

import heapq
import warnings
from collections.abc import Sequence

import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.linalg import cho_factor, solve_banded, solve_continuous_lyapunov
from scipy.spatial import cKDTree

from discrecontinual_equations.continuation.fokker_planck import DensityGrid
from discrecontinual_equations.continuation.noise import (
    NoiseMatrix,
    diffusion_matrix,
    field_at,
    fields_at,
    jacobian_at,
    jacobians_along,
)
from discrecontinual_equations.function.function import Function

_PLANE = 2
# f_x, f_y, A_xx, A_xy, A_yy per half-lattice point.
_COLUMNS = 5
# The solver's radius on every grid, and Dahiya and Cameron's rule (Eq. (28)),
# which default_radius offers to opt into.
_DEFAULT_RADIUS = 8
_RULE_BASE, _RULE_SLOPE, _RULE_START = 10, 4, 7
MIDPOINT, QUADRATIC = "midpoint", "quadratic"
_EQUILIBRIUM_CELLS = 2.0
_CYCLE_CELLS = 1.5
_CYCLE_SAMPLES_PER_CELL = 4
_ELLIPTICITY_FLOOR = 1.0e-10
_REFINEMENTS = 2
# The triangle's root on its base: at most this many secant or bisection
# steps, stopping once F anywhere in the bracket is within the tolerance of
# its least value there.
_ROOT_STEPS = 40
_ROOT_TOLERANCE = 1.0e-13
_HALF = 0.5
_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_UNKNOWN, _ACCEPTED = 0, 2
_CONSTANT_TOLERANCE = 1.0e-9
_CONSTANT_PROBE = 1.0e-3
_DESCENT_STEP = 0.5
_DESCENT_MAX_STEPS = 20000
_ARRIVAL_CELLS = 2.0
_CUBIC_DIRECTIONS = 8
_TAYLOR_STEP = 1.0e-3
_REFERENCE_NODES = 81
# The fewest chain nodes per turn of a winding path (see _winding_nodes).
_NODES_PER_TURN = 320
_START_SHARE = 0.25
_SECANT_STEPS = 6
_SECANT_TOLERANCE = 1.0e-4
_SECANT_SETTLED = 1.0e-9
# After the secant, at most this many parabolic steps on the values themselves.
_PARABOLA_STEPS = 4
# The safety factor on the reference's geometric tail bound.
_TAIL_SAFETY = 2.0
# A chain that blows up is relaxed again at half the step, this many times in all.
_STEP_HALVINGS = 6
# A node moving by this share of the distance between the chain's ends in one
# step is unstable.
_JUMP_SHARE = 0.25
# The chain stops when no node moves by this much in a step. The action is
# stationary on the minimiser, so what is left of the chain's error enters the
# value squared; 1e-10, the default, costs twice the steps for nothing visible.
_REFERENCE_SETTLED = 1.0e-8
_BAND_SAMPLES_PER_CELL = 16
# Below this |a dt| the step's exponential integrals are taken by their series.
_SERIES_BELOW = 1.0e-4
GRID, CHECKED, REFINED, FAILED, UNBOUNDED = 0, 1, 2, 3, 4


def default_radius(grid: DensityGrid) -> int:
    """The update radius for a grid: ``max(8, 10 + 4 (round(log2 N) - 7))``.

    Dahiya and Cameron's rule of thumb (their Eq. (28)) for ``N`` points per
    side, ``2^7 <= N <= 2^12``, with ``N`` the larger side here. Below 128
    points it would fall under 8, and the radius must stay comparable to the
    drift's rotation-to-gradient ratio (see the module docstring), so it is
    held there. It is not the solver's default, which is 8 on every grid: pass
    it as ``update_radius`` for a strongly rotating drift on a fine grid, where
    it pays, and not for one that is nearly a gradient, where it costs (the
    module docstring's *Radius*).

    Args:
        grid: The grid the march will run on.

    Returns:
        The radius in cells, at least 8.

    References:
        - Dahiya, D. and Cameron, M. (2018). Ordered line integral methods for computing
          the quasi-potential. Journal of Scientific Computing. arXiv:1706.07509,
          doi:10.1007/s10915-017-0590-9. Eq. (28), the rule of thumb for the update
          factor ``K``. Departure: held at 8 below 128 points, and not the default.
    """
    sides = max(grid.shape)
    rule = _RULE_BASE + _RULE_SLOPE * (round(np.log2(sides)) - _RULE_START)
    return max(_DEFAULT_RADIUS, int(rule))


class EquilibriumAttractor:
    """A stable equilibrium, initialised by the quasipotential of its linearisation.

    ``radius`` is how far, in cells, the quadratic form is written before the ordered
    line integrals take over. It is a fixed number of cells, so the region shrinks
    with the spacing and the error it carries (of order ``|x - x*|^3``) shrinks
    faster than the method's own.

    Near ``x*`` the drift is ``J (x - x*)`` and the noise ``D(x*)``, and the
    quasipotential of that linear system is ``(1/2) (x - x*)^T S^{-1} (x -
    x*)`` with ``J S + S J^T + D(x*) = 0`` - the stationary covariance of the
    linearisation. Dahiya and Cameron write the same matrix in closed form
    after a change of variables; the Lyapunov solve is equivalent and works for
    any ``D``. The equilibrium must be stable: a quasipotential is measured
    from an attractor.

    References:
        - Dahiya, D. and Cameron, M. (2018). An ordered line integral method for
          computing the quasi-potential in the case of variable anisotropic diffusion.
          Physica D. arXiv:1806.05321, doi:10.1016/j.physd.2018.07.002. Eqs. (14)-(18),
          the initialisation by the quasi-potential of the linearisation.
    """

    __slots__ = ["_radius", "_state"]

    def __init__(self, state: Sequence[float], radius: float = _EQUILIBRIUM_CELLS):
        self._state = np.asarray(state, dtype=float)
        self._radius = float(radius)

    @property
    def state(self) -> np.ndarray:
        """The equilibrium."""
        return self._state

    def initial(
        self,
        grid: DensityGrid,
        drift: Function,
        noise: NoiseMatrix,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flat indices and values of the points set before the march starts."""
        _check_inside(grid, self._state)
        jacobian = jacobian_at(drift, self._state)
        if np.max(np.linalg.eigvals(jacobian).real) >= 0.0:
            message = (
                f"the equilibrium at {self._state.tolist()} is not stable; a "
                f"quasipotential is measured from an attractor"
            )
            raise ValueError(message)
        diffusion = diffusion_matrix(noise.columns_at(self._state))
        covariance = solve_continuous_lyapunov(jacobian, -diffusion)
        form = _HALF * np.linalg.inv(covariance)
        points = grid.points()
        offsets = points - self._state
        cells = np.linalg.norm(offsets / grid.spacings, axis=1)
        near = np.flatnonzero(cells <= self._radius)
        values = np.einsum("ki,ij,kj->k", offsets[near], form, offsets[near])
        return near, values

    def end(self) -> np.ndarray | None:
        """Where a path traced back to this attractor finishes."""
        return self._state

    def reached(self, state: np.ndarray, grid: DensityGrid) -> bool:
        """Whether ``state`` lies within the region the attractor initialised."""
        cells = np.linalg.norm((np.asarray(state) - self._state) / grid.spacings)
        return bool(cells <= self._radius)

    def start(
        self,
        grid: DensityGrid,  # noqa: ARG002 (a cycle's start is set in cells)
        drift: Function,
        noise: NoiseMatrix,
        target: np.ndarray,
    ) -> "_EllipseStart":
        """Where a grid-free path to ``target`` starts: a small level set of ``U``.

        The level set of the quadratic form through the point a quarter of the
        way to ``target``. It is not tied to the grid: ``U`` on it is right to
        fourth order in its size, and the start of the path, where a rotating
        equilibrium makes it wind, is better left out than resolved.
        """
        jacobian = jacobian_at(drift, self._state)
        diffusion = diffusion_matrix(noise.columns_at(self._state))
        form = _HALF * np.linalg.inv(solve_continuous_lyapunov(jacobian, -diffusion))
        cubic = _cubic_coefficients(drift, noise, self._state, form)
        offset = _START_SHARE * (np.asarray(target, dtype=float) - self._state)
        level = float(offset @ form @ offset)
        return _EllipseStart(self._state, form, cubic, level)


class _EllipseStart:
    """The level set ``(x - x*)^T M (x - x*) = c`` about an equilibrium.

    ``U`` there is the quadratic form plus the cubic term of its Taylor expansion
    (see :func:`_cubic_coefficients`), so it is wrong only at fourth order in the
    distance from the equilibrium. Points on it are ``x* + sqrt(c) W (cos t,
    sin t)`` with ``W = M^{-1/2}``.
    """

    __slots__ = ["_centre", "_cubic", "_form", "_level", "_root", "_unroot"]

    def __init__(
        self,
        centre: np.ndarray,
        form: np.ndarray,
        cubic: np.ndarray,
        level: float,
    ) -> None:
        self._centre = centre
        self._form = form
        self._cubic = cubic
        self._level = level
        values, vectors = np.linalg.eigh(form)
        self._root = vectors @ np.diag(values**-_HALF) @ vectors.T
        self._unroot = vectors @ np.diag(values**_HALF) @ vectors.T

    def shrunk(self) -> "_EllipseStart":
        """The same surface at half the distance from the equilibrium."""
        return _EllipseStart(self._centre, self._form, self._cubic, self._level / 4.0)

    @property
    def step(self) -> float:
        """A parameter step of the order of a tenth of the surface's size."""
        return 0.1

    def point(self, parameter: float) -> np.ndarray:
        """The point of the surface at angle ``parameter``."""
        circle = np.array([np.cos(parameter), np.sin(parameter)])
        return self._centre + np.sqrt(self._level) * self._root @ circle

    def parameter(self, state: np.ndarray) -> float:
        """The angle of the surface point nearest ``state`` in the form's metric."""
        unit = self._unroot @ (np.asarray(state) - self._centre)
        return float(np.arctan2(unit[1], unit[0]))

    def inside(self, state: np.ndarray) -> bool:
        """Whether ``state`` lies within the surface."""
        offset = np.asarray(state) - self._centre
        return bool(offset @ self._form @ offset < self._level)

    def value(self, state: np.ndarray) -> float:
        """``U`` at ``state``, to fourth order."""
        offset = np.asarray(state) - self._centre
        return float(offset @ self._form @ offset + _cubic_value(self._cubic, offset))

    def gradient(self, state: np.ndarray) -> np.ndarray:
        """``grad U`` at ``state``, to third order."""
        offset = np.asarray(state) - self._centre
        return 2.0 * self._form @ offset + _cubic_gradient(self._cubic, offset)


def _cubic_monomials(offset: np.ndarray) -> np.ndarray:
    first, second = offset[0], offset[1]
    return np.array([first**3, first**2 * second, first * second**2, second**3])


def _cubic_value(coefficients: np.ndarray, offset: np.ndarray) -> float:
    return float(coefficients @ _cubic_monomials(offset))


def _cubic_gradient(coefficients: np.ndarray, offset: np.ndarray) -> np.ndarray:
    first, second = offset[0], offset[1]
    by_first = np.array([3.0 * first**2, 2.0 * first * second, second**2, 0.0])
    by_second = np.array([0.0, first**2, 2.0 * first * second, 3.0 * second**2])
    return np.array([coefficients @ by_first, coefficients @ by_second])


def _cubic_coefficients(
    drift: Function,
    noise: NoiseMatrix,
    state: np.ndarray,
    form: np.ndarray,
) -> np.ndarray:
    """The cubic term ``U3`` of ``U`` at a stable equilibrium ``x*``.

    With ``x = x* + y``, ``f = J y + (1/2) F2[y, y] + ...``, ``D = D0 + D1[y] +
    ...`` and ``U = y^T M y + U3(y) + ...``, the third-order terms of
    ``grad U . f + (1/2) grad U^T D grad U = 0`` are

        ``grad U3(y) . B y = -(y^T M F2[y, y] + 2 y^T M D1[y] M y)``,
        ``B = J + 2 D0 M``,

    a linear equation for the four coefficients of ``U3``. ``B`` is the linear
    flow along which minimum action paths leave ``x*`` (eigenvalues of positive
    real part), so the derivative along it is invertible on cubics. ``F2`` and
    ``D1`` are central differences; the equation is collocated at eight
    directions and solved by least squares.
    """
    jacobian = jacobian_at(drift, state)
    centre = diffusion_matrix(noise.columns_at(state))
    flow = jacobian + 2.0 * centre @ form
    rows, right = [], []
    for angle in np.linspace(0.0, np.pi, _CUBIC_DIRECTIONS, endpoint=False):
        unit = np.array([np.cos(angle), np.sin(angle)])
        ahead, behind = state + _TAYLOR_STEP * unit, state - _TAYLOR_STEP * unit
        curvature = (
            field_at(drift, ahead)
            - 2.0 * field_at(drift, state)
            + field_at(drift, behind)
        ) / _TAYLOR_STEP**2
        change = (
            diffusion_matrix(noise.columns_at(ahead))
            - diffusion_matrix(noise.columns_at(behind))
        ) / (2.0 * _TAYLOR_STEP)
        pulled = form @ unit
        right.append(-(unit @ form @ curvature + 2.0 * pulled @ change @ pulled))
        along = flow @ unit
        first, second = unit
        rows.append(
            [
                3.0 * first**2 * along[0],
                2.0 * first * second * along[0] + first**2 * along[1],
                second**2 * along[0] + 2.0 * first * second * along[1],
                3.0 * second**2 * along[1],
            ],
        )
    return np.linalg.lstsq(np.array(rows), np.array(right), rcond=None)[0]


class CycleAttractor:
    """A stable limit cycle, given as states around it in order.

    ``U`` vanishes on the cycle, and the points within ``width`` cells of it
    start from the expansion of ``U`` about it in the distance ``z`` along the
    cycle's normal: ``U = (1/2) G(tau) z^2 + (1/6) C(tau) z^3 + O(z^4)``, with
    ``tau`` the time along the cycle. ``G`` is the periodic solution of the
    Riccati equation of Lin, Yu and Zhou (their Eq. (30)), which in the plane
    is scalar,

        ``G' = -2 M G - A G^2``,  ``M = n^T J n``,  ``A = n^T D n``,

    for the drift's Jacobian ``J`` and the unit normal ``n`` on the cycle;
    ``H = 1 / G`` then solves the linear ``H' = 2 M H + A``, whose periodic
    solution is their Eq. (37) - positive exactly when the cycle is stable
    (``integral of M`` over a period below zero) and ``A`` is not zero all the
    way round. ``C`` comes from the third-order terms of the same expansion of
    the Hamilton-Jacobi equation, derived here (see :class:`_CycleExpansion`):
    a linear periodic equation with the same stability. Both are solved round
    the cycle once, exactly for piecewise-linear coefficients.

    The start Dahiya and Cameron give (their Sec. 2.1.2, Eqs. (7) to (12)) was
    used before DEQ-151: it splits the drift as ``f = -(1/2) D grad U + l``
    and takes ``l`` parallel to ``f`` at the cycle point nearest. That is
    exact when the rotation keeps its direction across the cycle, as on SDE
    (24)'s circle, but where it turns it is wrong at the leading order: ``l``
    is large and its direction changes by ``O(z)``, so ``grad U`` is off by
    ``O(|l| z)``, as much as itself. On the verification's elliptic cycle run
    round three times faster than it is approached it left the band off by
    0.117, 0.0289 and 0.0082 at 41, 81 and 161 points, and ``U`` below zero
    inside the cycle (down to -0.080, -0.020, -0.0059). From the expansion
    the band is off by 2.9e-4, 2.5e-5 and 1.8e-6 (order 3.5 to 3.8: the
    ``z^4`` term), ``U`` is nowhere negative, and the largest error on the
    verification's mask falls from 0.472, 0.119 and 0.0188 to 0.372, 0.0911
    and 0.0119. On SDE (24)'s circle, where the old start was right in
    direction, the band is off by 2.3e-4, 1.7e-5 and 1.3e-6 against 1.6e-4,
    4.4e-5 and 9.3e-6, and the largest error moves by at most 1.7e-5 of
    itself (0.0220528 to 0.0220532 at 161 points).

    *Limit.* What is left beside a fast cycle is the march's own error, not
    the start's. The drift's rotation-to-gradient ratio grows without bound
    towards the cycle, where ``grad U`` vanishes, so no fixed radius reaches
    far enough back along the front just outside the band (see the module
    docstring's *What the radius must cover*): on the elliptic cycle the
    largest error within ``|q - 1| < 0.2`` is 3.6e-3 and 4.7e-3 at 81 and 161
    points, and neither a band of three cells nor a radius of 16 brings it
    down. Inside the cycle the largest error is 0.0106 to 0.0139 at 81
    points, against 0.0149 to 0.0276 before, but 5.2e-3 to 8.7e-3 at 161
    points, against 1.4e-3 to 6.2e-3 (by bands of ``q``): there the old
    start's error, below the true ``U``, had offset the march's, above it.

    A straight-segment action from the foot, the other obvious start, is wrong
    by ``O(h)``: across a rotating flow it pays ``|l|`` per unit length, and
    the march cannot repair the first values it accepts.

    The cycle's own states should lie on it to well within a cell: ``U`` is
    measured from the polygon through them, and a polygon off the cycle by
    ``e`` puts ``G z e`` into every value of the band.

    References:
        - Lin, L., Yu, H. and Zhou, X. (2019). Quasi-potential calculation and
          minimum action method for limit cycle. Journal of Nonlinear Science
          29, 961-991. arXiv:1806.04031, doi:10.1007/s00332-018-9509-3. Eq.
          (23), the quadratic expansion about the cycle; Eq. (30), the
          periodic Riccati equation; Eqs. (36), (37) and Sec. 4.2, its
          periodic solution in the plane; Eq. (32), the momentum of the
          expansion. The cubic term is derived here, not taken from them.
        - Dahiya, D. and Cameron, M. (2018). Ordered line integral methods for computing
          the quasi-potential. Journal of Scientific Computing. arXiv:1706.07509,
          doi:10.1007/s10915-017-0590-9. Sec. 2.1.2, Eqs. (7)-(12), the start
          next to a limit cycle this replaces; Eqs. (24)-(26), SDE (24).
    """

    __slots__ = ["_states", "_width"]

    def __init__(self, states: Sequence[Sequence[float]], width: float = _CYCLE_CELLS):
        self._states = np.asarray(states, dtype=float)
        self._width = float(width)

    def initial(
        self,
        grid: DensityGrid,
        drift: Function,
        noise: NoiseMatrix,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flat indices and values of the points set before the march starts."""
        for state in self._states:
            _check_inside(grid, state)
        cell = float(np.min(grid.spacings))
        samples = _resample_closed(self._states, cell / _CYCLE_SAMPLES_PER_CELL)
        points = grid.points()
        distance, _ = cKDTree(samples).query(points)
        near = np.flatnonzero(distance <= self._width * cell)
        expansion = _CycleExpansion(samples, drift, noise)
        return near, expansion.values(points[near])

    def end(self) -> np.ndarray | None:
        """A cycle has no single end point; a traced path stops on reaching it."""
        return None

    def reached(self, state: np.ndarray, grid: DensityGrid) -> bool:
        """Whether ``state`` lies within the band the attractor initialised."""
        cell = float(np.min(grid.spacings))
        samples = _resample_closed(self._states, cell / _CYCLE_SAMPLES_PER_CELL)
        distance, _ = cKDTree(samples).query(np.asarray(state, dtype=float))
        return bool(distance <= self._width * cell)

    def start(
        self,
        grid: DensityGrid,
        drift: Function,
        noise: NoiseMatrix,
        target: np.ndarray,
    ) -> "_BandStart":
        """Where a grid-free path to ``target`` starts: the edge of the band.

        The curve ``width`` cells from the cycle on ``target``'s side, capped at
        half the distance to ``target``, with ``U`` there from the same
        expansion the march starts from.
        """
        cell = float(np.min(grid.spacings))
        samples = _resample_closed(self._states, cell / _BAND_SAMPLES_PER_CELL)
        distance, _ = cKDTree(samples).query(np.asarray(target, dtype=float))
        offset = min(self._width * cell, _HALF * float(distance))
        return _BandStart(_CycleExpansion(samples, drift, noise), target, offset)


class _CycleExpansion:
    """``U`` about a stable cycle to third order in the distance from it.

    With ``tau`` the time along the cycle ``gamma``, ``v = f(gamma)`` of speed
    ``s``, ``e = v / s``, ``n`` the unit normal ``e`` turned by a right angle,
    and ``x = gamma(tau) + z n(tau)``, ``U = (1/2) G z^2 + (1/6) C z^3``.
    Substituting into ``grad U . f + (1/2) grad U^T D grad U = 0`` - with
    ``grad U = (dU / dtau) e / (s (1 - k z)) + (dU / dz) n``, ``k = n^T J e /
    s`` the curvature, ``f = v + z J n + (1/2) z^2 f_nn`` and ``D = D(gamma) +
    z D_n`` - the terms in ``z^2`` give Lin, Yu and Zhou's Riccati equation
    (their Eq. (30) in the plane), and those in ``z^3``

        ``C' = -3 (M + G A) C - 3 (k G' + G' (e^T J n + G e^T D n) / s
              + G n . f_nn + G^2 n^T D_n n)``.

    Both ``H = 1 / G`` and ``C`` solve ``y' = a(tau) y + b(tau)`` with
    ``integral of a`` over a period negative on a stable cycle (for ``C``,
    ``M + G A = -G'/G - M``), so each has one periodic solution,
    ``y(0) = integral of exp(integral_t^T a) b dt / (1 - exp(integral_0^T a))``
    (Lin, Yu and Zhou's Eq. (37) for ``H``). It is taken between the samples
    with ``a`` and ``b`` linear in ``tau`` (the trapezoidal rule inside the
    exponential and for ``b``), and ``tau`` from the chords at the mean of
    the speeds at their ends. The derivatives of ``f`` and ``D`` across the
    cycle are central differences.

    A point's foot is its projection on the chord from the nearest sample to
    its neighbour on the point's side; the coefficients there are linear
    between the two samples.

    On SDE (24)'s circle (``G = 4``, ``C = 12``) and the verification's
    elliptic cycle (``U = (1/2) (q - 1)^2``, ``G = |grad q|^2``) both
    coefficients match the closed forms (the tests).
    """

    __slots__ = [
        "_count",
        "_cubic",
        "_normals",
        "_quadratic",
        "_rate",
        "_samples",
        "_speeds",
        "_tangents",
        "_tree",
    ]

    def __init__(self, samples: np.ndarray, drift: Function, noise: NoiseMatrix):
        self._samples = samples
        self._count = len(samples)
        self._tree = cKDTree(samples)
        velocity = fields_at(drift, samples)
        speeds = np.linalg.norm(velocity, axis=1)
        if not np.all(speeds > 0.0):
            message = "the drift vanishes on the cycle; it is not a limit cycle"
            raise ValueError(message)
        tangents = velocity / speeds[:, None]
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        scale = _TAYLOR_STEP * np.maximum(np.max(np.abs(samples), axis=1), 1.0)
        ahead = samples + scale[:, None] * normals
        behind = samples - scale[:, None] * normals
        forward = samples + scale[:, None] * tangents
        backward = samples - scale[:, None] * tangents
        across = fields_at(drift, ahead), fields_at(drift, behind)
        along = fields_at(drift, forward), fields_at(drift, backward)
        turn_n = (across[0] - across[1]) / (2.0 * scale[:, None])
        turn_e = (along[0] - along[1]) / (2.0 * scale[:, None])
        bend = (across[0] - 2.0 * velocity + across[1]) / (scale * scale)[:, None]
        cells = _cells(noise, samples)
        change = (_cells(noise, ahead) - _cells(noise, behind)) / (2.0 * scale)[
            :, None, None
        ]
        normal_rate = np.einsum("ki,ki->k", normals, turn_n)
        shear = np.einsum("ki,ki->k", tangents, turn_n)
        curvature = np.einsum("ki,ki->k", normals, turn_e) / speeds
        spread = np.einsum("ki,kij,kj->k", normals, cells, normals)
        mixed = np.einsum("ki,kij,kj->k", tangents, cells, normals)
        spread_change = np.einsum("ki,kij,kj->k", normals, change, normals)
        bend_n = np.einsum("ki,ki->k", normals, bend)
        closed = np.vstack([samples, samples[:1]])
        lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        times = lengths / (_HALF * (speeds + np.roll(speeds, -1)))
        if float(np.sum(_HALF * (normal_rate + np.roll(normal_rate, -1)) * times)) >= 0:
            message = (
                "the cycle is not stable: the drift's normal rate n^T J n "
                "does not integrate to below zero round it"
            )
            raise ValueError(message)
        inverse = _periodic_linear(times, 2.0 * normal_rate, spread)
        if not np.all(inverse > 0.0):
            message = (
                "the Riccati equation about the cycle has no positive periodic "
                "solution: the noise does not reach across the cycle"
            )
            raise ValueError(message)
        quadratic = 1.0 / inverse
        rate = -2.0 * normal_rate * quadratic - spread * quadratic * quadratic
        source = -3.0 * (
            curvature * rate
            + rate * (shear + quadratic * mixed) / speeds
            + quadratic * bend_n
            + quadratic * quadratic * spread_change
        )
        self._cubic = _periodic_linear(
            times,
            -3.0 * (normal_rate + quadratic * spread),
            source,
        )
        self._quadratic = quadratic
        self._rate = rate
        self._speeds = speeds
        self._tangents = tangents
        self._normals = normals

    @property
    def samples(self) -> np.ndarray:
        """The cycle's samples."""
        return self._samples

    @property
    def quadratic(self) -> np.ndarray:
        """``G`` at each sample."""
        return self._quadratic

    @property
    def cubic(self) -> np.ndarray:
        """``C`` at each sample."""
        return self._cubic

    def _locate(self, points: np.ndarray):
        """The normal distance of each point and the weights of its two samples."""
        _, nearest = self._tree.query(points)
        previous = (nearest - 1) % self._count
        following = (nearest + 1) % self._count
        closest = self._samples[nearest]
        offset = points - closest
        ahead = np.einsum("ki,ki->k", offset, self._samples[following] - closest)
        behind = np.einsum("ki,ki->k", offset, self._samples[previous] - closest)
        beside = np.where(ahead >= behind, following, previous)
        chord = self._samples[beside] - closest
        share = np.einsum("ki,ki->k", offset, chord) / np.einsum(
            "ki,ki->k", chord, chord
        )
        share = np.clip(share, 0.0, 1.0)
        normal = (1.0 - share)[:, None] * self._normals[nearest] + share[
            :, None
        ] * self._normals[beside]
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        foot = closest + share[:, None] * chord
        distance = np.einsum("ki,ki->k", points - foot, normal)
        return distance, nearest, beside, share, normal

    def _blend(self, values: np.ndarray, nearest, beside, share) -> np.ndarray:
        return (1.0 - share) * values[nearest] + share * values[beside]

    def values(self, points: np.ndarray) -> np.ndarray:
        """``U`` at each row of ``points``."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        distance, nearest, beside, share, _ = self._locate(points)
        quadratic = self._blend(self._quadratic, nearest, beside, share)
        cubic = self._blend(self._cubic, nearest, beside, share)
        return distance * distance * (_HALF * quadratic + cubic * distance / 6.0)

    def gradient(self, point: np.ndarray) -> np.ndarray:
        """``grad U`` at ``point``, to second order in the distance.

        Across the cycle ``G z + (1/2) C z^2``; along it ``(1/2) G' z^2 / s``,
        Lin, Yu and Zhou's Eq. (32) in the plane.
        """
        points = np.atleast_2d(np.asarray(point, dtype=float))
        distance, nearest, beside, share, normal = self._locate(points)
        z = float(distance[0])
        quadratic = float(self._blend(self._quadratic, nearest, beside, share)[0])
        cubic = float(self._blend(self._cubic, nearest, beside, share)[0])
        rate = float(self._blend(self._rate, nearest, beside, share)[0])
        speed = float(self._blend(self._speeds, nearest, beside, share)[0])
        tangent = np.array([normal[0, 1], -normal[0, 0]])
        return (quadratic * z + _HALF * cubic * z * z) * normal[0] + (
            _HALF * rate * z * z / speed
        ) * tangent


def _cells(noise: NoiseMatrix, states: np.ndarray) -> np.ndarray:
    """``D = G G^T`` at each row of ``states``."""
    amplitude = noise.columns_along(states)
    return np.einsum("pki,pkj->pij", amplitude, amplitude)


def _periodic_linear(times: np.ndarray, rate: np.ndarray, source: np.ndarray):
    """The periodic solution of ``y' = a y + b`` at the samples of a cycle.

    ``times[k]`` is the time from sample ``k`` to the next (the last back to
    the first). Over each step ``a`` is held at the mean of its ends and ``b``
    is linear, and the step is then exact: with ``g = a dt``,
    ``y_{k+1} = P y_k + dt (b_k E1(g) + (b_{k+1} - b_k) E2(g))``, ``P = exp(g)``,
    ``E1 = (P - 1) / g``, ``E2 = (P - 1 - g) / g^2``. The periodic ``y_0`` is
    ``sum_k S_k prod_{j > k} P_j / (1 - prod_k P_k)``, summed in logarithms.
    (The trapezoidal rule for ``b`` was tried first: its error, ``g^2 / 12``
    relative, was 3e-4 in ``G`` on SDE (24)'s circle at 400 samples.)
    """
    after_source = np.roll(source, -1)
    growth = times * _HALF * (rate + np.roll(rate, -1))
    factor = np.exp(growth)
    small = np.abs(growth) < _SERIES_BELOW
    safe = np.where(small, 1.0, growth)
    first_kind = np.where(
        small,
        1.0 + growth / 2.0 + growth**2 / 6.0,
        np.expm1(safe) / safe,
    )
    second_kind = np.where(
        small,
        0.5 + growth / 6.0 + growth**2 / 24.0,
        (np.expm1(safe) - safe) / (safe * safe),
    )
    gain = times * (source * first_kind + (after_source - source) * second_kind)
    later = np.concatenate([np.cumsum(growth[::-1])[::-1][1:], [0.0]])
    first = float(np.sum(gain * np.exp(later))) / (-float(np.expm1(np.sum(growth))))
    values = np.empty_like(rate)
    values[0] = first
    for index in range(len(rate) - 1):
        values[index + 1] = factor[index] * values[index] + gain[index]
    return values


class _BandStart:
    """The curve at a fixed distance from a cycle, on one side of it.

    Points on it are ``c(s) + d n(s)`` for the cycle ``c`` by arc length ``s``
    and ``n`` its unit normal towards the side of the target, interpolated
    between the samples' normals so that the curve has no corners. ``U`` and
    ``grad U`` on it come from the cycle's expansion.
    """

    __slots__ = [
        "_arc",
        "_expansion",
        "_normals",
        "_offset",
        "_samples",
        "_target",
        "_tree",
    ]

    def __init__(
        self,
        expansion: _CycleExpansion,
        target: np.ndarray,
        offset: float,
    ) -> None:
        samples = expansion.samples
        self._expansion = expansion
        self._samples = samples
        self._target = np.asarray(target, dtype=float)
        self._offset = offset
        self._tree = cKDTree(samples)
        closed = np.vstack([samples, samples[:1]])
        chords = np.diff(closed, axis=0)
        lengths = np.linalg.norm(chords, axis=1)
        self._arc = np.concatenate([[0.0], np.cumsum(lengths)])
        across = np.column_stack([chords[:, 1], -chords[:, 0]]) / lengths[:, None]
        normals = across + np.roll(across, 1, axis=0)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        _, nearest = self._tree.query(np.asarray(target, dtype=float))
        if float((np.asarray(target) - samples[nearest]) @ normals[nearest]) < 0.0:
            normals = -normals
        self._normals = normals

    def shrunk(self) -> "_BandStart":
        """The same curve at half the distance from the cycle."""
        return _BandStart(self._expansion, self._target, _HALF * self._offset)

    @property
    def step(self) -> float:
        """A parameter step of a few times the offset."""
        return 4.0 * self._offset

    def point(self, parameter: float) -> np.ndarray:
        """The point of the curve at arc length ``parameter`` along the cycle."""
        closed = np.vstack([self._samples, self._samples[:1]])
        normals = np.vstack([self._normals, self._normals[:1]])
        length = self._arc[-1]
        place = parameter % length
        base = np.array([np.interp(place, self._arc, closed[:, a]) for a in range(2)])
        normal = np.array(
            [np.interp(place, self._arc, normals[:, a]) for a in range(2)],
        )
        return base + self._offset * normal / np.linalg.norm(normal)

    def parameter(self, state: np.ndarray) -> float:
        """The arc length of the cycle point nearest ``state``."""
        point = np.asarray(state, dtype=float)
        _, index = self._tree.query(point)
        following = (index + 1) % len(self._samples)
        chord = self._samples[following] - self._samples[index]
        share = float((point - self._samples[index]) @ chord / (chord @ chord))
        if share < 0.0:
            index = (index - 1) % len(self._samples)
            chord = (
                self._samples[(index + 1) % len(self._samples)] - self._samples[index]
            )
            share = float((point - self._samples[index]) @ chord / (chord @ chord))
        share = min(max(share, 0.0), 1.0)
        return float(self._arc[index] + share * np.linalg.norm(chord))

    def inside(self, state: np.ndarray) -> bool:
        """Whether ``state`` is nearer the cycle than the curve."""
        distance, _ = self._tree.query(np.asarray(state, dtype=float))
        return bool(distance < self._offset)

    def value(self, state: np.ndarray) -> float:
        """``U`` at ``state`` from the cycle's expansion."""
        return float(self._expansion.values(np.asarray(state, dtype=float))[0])

    def gradient(self, state: np.ndarray) -> np.ndarray:
        """``grad U`` at ``state`` from the cycle's expansion."""
        return self._expansion.gradient(np.asarray(state, dtype=float))


def _resample_closed(states: np.ndarray, spacing: float) -> np.ndarray:
    """Points about ``spacing`` apart on the periodic spline through ``states``.

    The spline, in the chord length, rather than the polygon: ``U`` beside the
    cycle is measured from these points, and a polygon cuts inside the cycle by
    its sagitta - ``3e-5`` for SDE (24)'s circle given at 400 states, which put
    ``G z`` times that into every value of the band, about as much as the
    expansion's own error (DEQ-151).
    """
    closed = np.vstack([states, states[:1]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(lengths)])
    count = max(int(np.ceil(arc[-1] / spacing)), len(states))
    target = np.linspace(0.0, arc[-1], count, endpoint=False)
    return CubicSpline(arc, closed, bc_type="periodic")(target)


def _geometric(step: np.ndarray, velocity: np.ndarray, metric: np.ndarray) -> float:
    return float(
        np.sqrt(step @ metric @ step) * np.sqrt(velocity @ metric @ velocity)
        - step @ metric @ velocity,
    )


def _warn_unchecked(failed: list[int], unbounded: list[int], *, refining: bool) -> None:
    """Say which points :meth:`OrderedLineIntegral.check` could not bound."""
    parts = []
    if failed:
        parts.append(f"no reference could be computed at {len(failed)} ({failed})")
    if unbounded:
        parts.append(
            f"the reference's own error is not a bound at {len(unbounded)} "
            f"({unbounded})",
        )
    message = "; ".join(parts) + " of the points checked"
    if refining:
        message += ", and they were not refined"
    warnings.warn(message, UserWarning, stacklevel=3)


def _check_inside(grid: DensityGrid, state: np.ndarray) -> None:
    for axis, value in enumerate(state):
        coordinates = grid.axis(axis)
        if not coordinates[0] <= value <= coordinates[-1]:
            message = f"the attractor state {state.tolist()} lies outside the grid"
            raise ValueError(message)


class GridQuasipotential:
    """``U`` sampled on a planar :class:`DensityGrid`.

    ``values`` is shaped like the grid and is ``inf`` where no path from the attractor
    reached: behind a point where the noise vanishes, or where the grid does not
    connect. ``floor`` is the largest ``U`` on the points the attractor
    initialised, as the march left them: the level below which a state is as close
    to the attractor as the grid resolves. ``attractor`` is the one the march
    started from, when it is known.

    ``errors`` and ``status`` are shaped like the grid and say how far to trust
    each value. A field fresh from the march has every status :data:`GRID` and
    every error ``nan``: nothing has been measured. After
    :meth:`OrderedLineIntegral.check` a checked point has status :data:`CHECKED`
    and an estimated error, and a point whose value was replaced by the grid-free
    one has status :data:`REFINED` and the estimated error of the new value.
    A point the check could not bound keeps the grid's value: status
    :data:`FAILED` (no grid-free value could be computed; error ``nan``) or
    :data:`UNBOUNDED` (one was, but its own error estimate is not a bound
    there; error ``inf``). :attr:`unbounded` lists both.
    """

    __slots__ = [
        "_attractor",
        "_errors",
        "_floor",
        "_grid",
        "_interpolator",
        "_status",
        "_values",
    ]

    def __init__(  # noqa: PLR0913 (the field and what is known about it)
        self,
        grid: DensityGrid,
        values: np.ndarray,
        floor: float,
        attractor: "EquilibriumAttractor | CycleAttractor | None" = None,
        errors: np.ndarray | None = None,
        status: np.ndarray | None = None,
    ) -> None:
        self._grid = grid
        self._values = np.asarray(values, dtype=float).reshape(grid.shape)
        self._floor = float(floor)
        self._attractor = attractor
        self._errors = (
            np.full(grid.shape, np.nan)
            if errors is None
            else np.asarray(errors, dtype=float).reshape(grid.shape)
        )
        self._status = (
            np.full(grid.shape, GRID, dtype=np.int8)
            if status is None
            else np.asarray(status, dtype=np.int8).reshape(grid.shape)
        )
        # Built once: a traced path asks for U at every step, and building the
        # interpolator per call was most of what tracing a path cost.
        self._interpolator = RegularGridInterpolator(
            (grid.axis(0), grid.axis(1)),
            self._values,
            bounds_error=False,
            fill_value=np.inf,
        )

    @property
    def grid(self) -> DensityGrid:
        """The grid ``U`` lives on."""
        return self._grid

    @property
    def values(self) -> np.ndarray:
        """``U``, shaped like the grid."""
        return self._values

    @property
    def floor(self) -> float:
        """The largest ``U`` the march left on the initialised points."""
        return self._floor

    @property
    def attractor(self) -> "EquilibriumAttractor | CycleAttractor | None":
        """The attractor ``U`` is measured from, if the field knows it."""
        return self._attractor

    @property
    def errors(self) -> np.ndarray:
        """The estimated error of each value, ``nan`` where none was estimated."""
        return self._errors

    @property
    def status(self) -> np.ndarray:
        """:data:`GRID`, :data:`CHECKED`, :data:`REFINED`, :data:`FAILED` or
        :data:`UNBOUNDED` per value."""
        return self._status

    @property
    def unbounded(self) -> np.ndarray:
        """Flat indices of the points a check could not bound, in order.

        Those of status :data:`FAILED` - no grid-free value could be computed -
        and :data:`UNBOUNDED` - one was, but its own error estimate is not a
        bound there. Their values are the grid's, unrefined.
        """
        flat = self._status.ravel()
        return np.flatnonzero((flat == FAILED) | (flat == UNBOUNDED))

    def at(self, state: Sequence[float] | np.ndarray) -> float:
        """``U`` at a state, bilinearly interpolated; ``inf`` off the grid."""
        point = np.asarray(state, dtype=float)[None, :]
        return float(self._interpolator(point)[0])

    def gradient(self) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
        """Interpolators for the two components of ``grad U``, central differences."""
        first, second = np.gradient(
            self._values,
            self._grid.axis(0),
            self._grid.axis(1),
        )
        axes = (self._grid.axis(0), self._grid.axis(1))
        return (
            RegularGridInterpolator(
                axes,
                first,
                bounds_error=False,
                fill_value=np.nan,
            ),
            RegularGridInterpolator(
                axes,
                second,
                bounds_error=False,
                fill_value=np.nan,
            ),
        )


class MinimumActionPath:
    """A path from an attractor, and the geometric action spent along it.

    ``states`` runs as the path was produced: from the chosen point back to the
    attractor for a traced path, from the first end to the second for a relaxed
    one. ``action`` is the action of going the uphill way, attractor to point.

    The two flags say whether to believe it, each for the method it belongs to,
    in the way :class:`~.corrector.CorrectionResult` reports a corrector that ran
    out of iterations: ``arrived`` is whether a traced path reached its attractor
    rather than stalling short of it - at a saddle, say - and ``converged`` is
    whether a relaxed chain stopped moving before its iteration limit, with
    ``iterations`` the number it took. Each is ``True`` for the method that does
    not set it.
    """

    __slots__ = ["action", "arrived", "converged", "iterations", "states"]

    def __init__(
        self,
        states: np.ndarray,
        action: float,
        *,
        arrived: bool = True,
        converged: bool = True,
        iterations: int = 0,
    ) -> None:
        self.states = np.asarray(states, dtype=float)
        self.action = float(action)
        self.arrived = arrived
        self.converged = converged
        self.iterations = iterations


class _HalfLattice:
    """``f`` and ``A = D^{-1}`` on the grid of half the spacing.

    The midpoint of any two grid points ``i`` and ``j`` is the half-lattice point
    ``i + j``, so every quadrature node the method ever needs is a lookup. ``f``
    and ``G`` are evaluated at all of them through ``eval_batch`` (see
    :func:`~.noise.fields_at` and :meth:`~.noise.NoiseMatrix.columns_along`): one
    call each for a function written with array arithmetic, one per point for
    one that keeps the base class's ``eval_batch``. The five
    numbers per point - ``f_x, f_y, A_xx, A_xy, A_yy`` - are kept side by side in one
    table so that one gather fetches all of them, and the march does its algebra on
    their columns: the einsum over ``2 x 2`` blocks it replaces cost four times as
    much at the sizes one accepted point updates.
    """

    __slots__ = ["table"]

    def __init__(self, grid: DensityGrid, drift: Function, noise: NoiseMatrix):
        axes = [
            np.linspace(grid.axis(a)[0], grid.axis(a)[-1], 2 * grid.shape[a] - 1)
            for a in range(_PLANE)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.stack([block.ravel() for block in mesh], axis=1)
        velocity = fields_at(drift, points)
        amplitude = noise.columns_along(points)
        cells = np.einsum("pki,pkj->pij", amplitude, amplitude)
        _check_nodes(cells.reshape((*mesh[0].shape, _PLANE, _PLANE))[::2, ::2])
        inverse = _inverse(cells)
        self.table = np.column_stack(
            [velocity, inverse[:, 0, 0], inverse[:, 0, 1], inverse[:, 1, 1]],
        ).reshape((*mesh[0].shape, _COLUMNS))


def _check_nodes(cells: np.ndarray) -> None:
    eigenvalues = np.linalg.eigvalsh(cells.reshape(-1, _PLANE, _PLANE))
    scale = float(np.max(eigenvalues))
    if float(np.min(eigenvalues)) <= _ELLIPTICITY_FLOOR * max(scale, 1.0):
        message = (
            "the diffusion matrix G G^T is singular at a grid point; the "
            "quasipotential needs a non-degenerate diffusion"
        )
        raise ValueError(message)


def _inverse(cells: np.ndarray) -> np.ndarray:
    """``D^{-1}`` per point, ``nan`` where ``D`` is singular: such a point is a wall."""
    eigenvalues = np.linalg.eigvalsh(cells)
    scale = max(float(np.max(eigenvalues)), 1.0)
    singular = eigenvalues[:, 0] <= _ELLIPTICITY_FLOOR * scale
    safe = cells.copy()
    safe[singular] = np.eye(_PLANE)
    inverse = np.linalg.inv(safe)
    inverse[singular] = np.nan
    return inverse


def _form(first_x, first_y, metric, second_x, second_y):
    """``u^T A v`` for columns of vectors, ``A`` read from half-lattice rows."""
    return (
        metric[:, 2] * first_x * second_x
        + metric[:, 3] * (first_x * second_y + first_y * second_x)
        + metric[:, 4] * first_y * second_y
    )


def _action(step_x, step_y, coefficients):
    """``Q`` of many segments at once; ``inf`` across a point where ``D`` vanishes.

    ``coefficients`` holds ``f`` and ``A`` at each segment's midpoint, one row per
    segment, in the half lattice's column order.
    """
    velocity_x, velocity_y = coefficients[:, 0], coefficients[:, 1]
    values = np.sqrt(
        _form(step_x, step_y, coefficients, step_x, step_y)
        * _form(velocity_x, velocity_y, coefficients, velocity_x, velocity_y),
    ) - _form(step_x, step_y, coefficients, velocity_x, velocity_y)
    return np.where(np.isnan(values), np.inf, values)


class OrderedLineIntegral:
    """The quasipotential of a planar ``dx = f dt + G dW`` by ordered line integrals.

    ``update_radius`` is the stencil radius in cells (``K`` in the paper), 8 by
    default on every grid; :func:`default_radius` gives Dahiya and Cameron's
    rule, which grows with the grid, for a strongly rotating drift (see the
    module docstring for when it pays). It must be comparable to the
    drift's rotation-to-gradient ratio on the region asked about, or the error
    does not converge as the grid is refined. ``update_order`` is
    :data:`MIDPOINT` (``U`` a straight line along the front, OLIM-MID as
    published, the default) or :data:`QUADRATIC` (a parabola, to opt into for
    a drift that is linear or nearly so; see the module docstring's *Order*
    for where it helps and where it does not). See the module docstring for
    both, the method, the normalisation, and its limits.

    The march accepts grid points in increasing order of ``U``, as Dijkstra's
    method does, and offers each unaccepted point within the radius a
    one-point update - the midpoint-rule action of the straight segment from
    the accepted point - and a triangle update from every segment between the
    accepted point and an accepted neighbour, minimised over the point where
    the path leaves that segment. It is first order in general (the straight
    interpolation of ``U`` along the front) and exact to rounding on a linear
    drift with a constant ``D`` with the parabola.

    References:
        - Dahiya, D. and Cameron, M. (2018). Ordered line integral methods for computing
          the quasi-potential. Journal of Scientific Computing. arXiv:1706.07509,
          doi:10.1007/s10915-017-0590-9. The OLIM family and OLIM-MID: the geometric
          action, Eq. (2); the Hamilton-Jacobi equation, Eq. (3); the one-point and
          triangle updates, Eqs. (14), (15), (B-7), (B-8); the update radius, Eq. (28).
          Departure: every triangle is tried, not the hierarchical subset; Eq. (28) is
          not the default.
        - Dahiya, D. and Cameron, M. (2018). An ordered line integral method for
          computing the quasi-potential in the case of variable anisotropic diffusion.
          Physica D. arXiv:1806.05321, doi:10.1016/j.physd.2018.07.002. The midpoint
          update with a matrix ``D``, Eq. (19); the triangle update with ``f`` and ``A =
          D^{-1}`` linear along the base, Eqs. (21), (22); its exact derivative, Eq.
          (23), solved by Wilkinson's hybrid secant/bisection method. Departure: the
          default straight base freezes ``f`` and ``A`` at the current ``s`` instead of
          solving Eq. (23) (measured; see the module docstring's *Triangle
          minimisation*); the parabolic update solves it.
    """

    __slots__ = ["_drift", "_grid", "_noise", "_offsets", "_quadratic", "update_radius"]

    def __init__(
        self,
        drift: Function,
        noise: NoiseMatrix,
        grid: DensityGrid,
        update_radius: int | None = None,
        update_order: str = MIDPOINT,
    ) -> None:
        if grid.dimension != _PLANE:
            message = (
                f"the ordered line integral method here is planar; the grid has "
                f"{grid.dimension} axes"
            )
            raise ValueError(message)
        if update_order not in (MIDPOINT, QUADRATIC):
            message = (
                f"the update order is {MIDPOINT!r} or {QUADRATIC!r}, "
                f"not {update_order!r}"
            )
            raise ValueError(message)
        self._drift = drift
        self._noise = noise
        self._grid = grid
        self._quadratic = update_order == QUADRATIC
        if update_radius is None:
            update_radius = _DEFAULT_RADIUS
        self.update_radius = update_radius
        reach = np.arange(-update_radius, update_radius + 1)
        first, second = np.meshgrid(reach, reach, indexing="ij")
        keep = (first * first + second * second <= update_radius * update_radius) & (
            (first != 0) | (second != 0)
        )
        self._offsets = (first[keep], second[keep])

    def solve(
        self,
        attractor: EquilibriumAttractor | CycleAttractor,
    ) -> GridQuasipotential:
        """March ``U`` out from ``attractor`` over the whole grid."""
        lattice = _HalfLattice(self._grid, self._drift, self._noise)
        march = _March(self._grid, lattice, self._offsets, quadratic=self._quadratic)
        indices, values = attractor.initial(self._grid, self._drift, self._noise)
        march.run(indices, values)
        # The floor is read after the march, not from what the attractor wrote.
        # The march still lowers some of the starting values as the earlier ones
        # are accepted - 173 of 374 at mu = 1 on the noisy Hopf film's grid, the
        # largest from 0.0230 to 0.0219, and at mu = 0.025, where the cycle is
        # three cells from the singular origin, 31 of 56, from 0.0208 to 0.0156 -
        # and what it leaves is the level the grid resolves next to the attractor.
        floor = float(np.max(march.values.ravel()[indices]))
        return GridQuasipotential(self._grid, march.values, floor, attractor)

    def minimum_action_path(
        self,
        field: GridQuasipotential,
        start: Sequence[float] | np.ndarray,
        end: Sequence[float] | np.ndarray | None = None,
        step: float | None = None,
    ) -> MinimumActionPath:
        """Trace the minimum action path from ``start`` back to the attractor.

        The path is followed along ``-(f + D grad U)`` in steps of ``step`` in arc
        length (half a cell by default) by the midpoint rule, and stops when a step
        would raise ``U``. That happens at the attractor, where ``U`` is least, but
        not only there: from a state in another basin the path runs down to the
        saddle between the basins and stalls beside it, since ``-(f + D grad U)``
        vanishes there and the grid cannot say which way to fall; it also stalls
        where the grid does not resolve ``U``, as beside a point where the noise
        vanishes. ``arrived`` on the result says which. A path is counted as
        arrived when it stopped within two cells of ``end`` - an equilibrium's own
        state, which is then appended as the last state. With no ``end`` it is
        the field's attractor that decides: an equilibrium is its own ``end``,
        and a cycle has been reached when the path stopped inside the band of
        points the attractor initialised - within its ``width`` of the cycle,
        by distance to the cycle's samples. That is a fixed place, unlike the
        level ``U`` reaches there, which moves with the march's error. (Only a
        field built without its attractor falls back on that level: arrived
        when ``U`` where the path stopped is no more than ``floor``.)
        A path that has not arrived is returned as far as it got, without ``end``.
        Paths that must cross a saddle are what :class:`GeometricMinimumAction` is
        for.

        The action is accumulated along the way by the same midpoint quadrature the
        march uses, plus ``U`` where the path stopped, so for an arrived path it is
        an independent estimate of ``U(start)``.
        """
        spacing = float(np.min(self._grid.spacings))
        length = step or _DESCENT_STEP * spacing
        attractor = field.attractor
        if end is None and attractor is not None:
            end = attractor.end()
        gradient = field.gradient()
        current = np.asarray(start, dtype=float)
        level = field.at(current)
        states = [current]
        action = 0.0
        for _ in range(_DESCENT_MAX_STEPS):
            half = current + _HALF * length * self._backward(current, gradient)
            direction = self._backward(half, gradient)
            if not np.all(np.isfinite(direction)):
                break
            following = current + length * direction
            # U falls along every minimum action path, so the path has arrived
            # when a step would raise it: at the attractor, where U is least, or
            # in a region the grid does not resolve - beside a point where the
            # noise vanishes, say - where going on would only circle.
            following_level = field.at(following)
            if not following_level < level:
                break
            level = following_level
            metric = np.linalg.inv(diffusion_matrix(self._noise.columns_at(half)))
            velocity = field_at(self._drift, half)
            action += _geometric(current - following, velocity, metric)
            current = following
            states.append(current)
        remaining = field.at(current)
        if end is None and attractor is not None:
            arrived = attractor.reached(current, self._grid)
        elif end is None:
            arrived = remaining <= field.floor
        else:
            target = np.asarray(end, dtype=float)
            arrived = bool(
                np.linalg.norm(current - target) <= _ARRIVAL_CELLS * spacing,
            )
            if arrived:
                states.append(target)
        return MinimumActionPath(np.array(states), action + remaining, arrived=arrived)

    def check(
        self,
        field: GridQuasipotential,
        indices: Sequence[int] | np.ndarray | None = None,
        stride: int = 1,
        tolerance: float | None = None,
        nodes: int = _REFERENCE_NODES,
    ) -> GridQuasipotential:
        """Estimate the error of ``field`` at chosen points, and refine the worst.

        At each point the grid-free value of :meth:`reference` is computed, and
        the error of the grid value is estimated as its distance from it plus
        the reference's own estimated error - an upper bound whenever that one
        is. ``indices`` are flat grid indices; by default every ``stride``-th
        point along each axis. Points inside the region the attractor
        initialised, and points the march did not reach, are left as they are.

        With a ``tolerance``, a point whose estimated error exceeds it has its
        value replaced by the reference, provided the reference's own estimated
        error is within the tolerance; otherwise it is only flagged, by its
        error. The field returned is a new one; ``field`` is not changed.

        A point is never the end of the check. Where no reference can be
        computed - a chain that will not stay finite at any step - the point
        gets status :data:`FAILED` and keeps an error of ``nan``; where the
        reference's own estimate is not a bound (see :meth:`reference`), it
        gets status :data:`UNBOUNDED` and an error of ``inf``. Neither is
        refined, whatever the tolerance; both are listed by
        :attr:`GridQuasipotential.unbounded`, and a ``UserWarning`` names them
        once the check is done.

        This costs a few relaxations of a chain of ``nodes`` states per point -
        of the order of a second each - against well under a millisecond per
        point for the march, which is why it is a separate step.

        Args:
            field: A field from :meth:`solve`, which knows its attractor.
            indices: Flat grid indices to check; by default every
                ``stride``-th point along each axis.
            stride: The spacing of the default points.
            tolerance: Refine where the estimated error exceeds this, if the
                reference is good to it.
            nodes: The number of states in each relaxed chain.

        Returns:
            A new field with the errors and statuses of the points checked.

        Raises:
            ValueError: If ``field`` does not know its attractor.
        """
        attractor = field.attractor
        if attractor is None:
            message = "checking a field needs the attractor it was marched from"
            raise ValueError(message)
        grid = field.grid
        points = grid.points()
        if indices is None:
            every = np.arange(points.shape[0]).reshape(grid.shape)
            indices = every[::stride, ::stride].ravel()
        values = field.values.copy().ravel()
        errors = field.errors.copy().ravel()
        status = field.status.copy().ravel()
        failed, unbounded = [], []
        for index in np.asarray(indices, dtype=int).tolist():
            target = points[index]
            if not np.isfinite(values[index]) or attractor.reached(target, grid):
                continue
            try:
                reference, own = self.reference(field, target, nodes)
            except ValueError:
                status[index] = FAILED
                failed.append(index)
                continue
            if not np.isfinite(own):
                errors[index], status[index] = np.inf, UNBOUNDED
                unbounded.append(index)
                continue
            estimate = abs(float(field.values.ravel()[index]) - reference) + own
            if tolerance is not None and estimate > tolerance and own <= tolerance:
                values[index], errors[index], status[index] = reference, own, REFINED
            else:
                errors[index], status[index] = estimate, CHECKED
        if failed or unbounded:
            _warn_unchecked(failed, unbounded, refining=tolerance is not None)
        return GridQuasipotential(
            grid,
            values,
            field.floor,
            attractor,
            errors,
            status,
        )

    def reference(
        self,
        field: GridQuasipotential,
        state: Sequence[float] | np.ndarray,
        nodes: int = _REFERENCE_NODES,
    ) -> tuple[float, float]:
        """``U`` at ``state`` without the grid, and an estimate of its error.

        The minimum action path is relaxed by :class:`GeometricMinimumAction`
        from a small surface about the attractor - a level set of the quadratic
        form about an equilibrium, a curve beside a cycle - on which ``U`` is
        known from the attractor's own expansion, to ``state``. The chain starts
        from the path traced on ``field``, and its first state slides along the
        surface to where ``U + S`` is least: there the chain's momentum matches
        ``grad U`` along the surface, found by the secant method. Starting off
        the attractor matters: the path from a rotating equilibrium or a cycle
        winds round it infinitely often, which a chain of nodes cannot follow,
        while from the surface it does not.

        The error estimate is the sum of two differences: from the value
        from a surface at twice the distance, which measures the expansion's
        error, and from the value on a chain of half the nodes, which measures
        the chain's.

        That sum is a bound only while it is the surface's error that changes
        as the surface shrinks - an assumption, not a theorem - and it fails
        where the path winds round the attractor faster than the chain can
        follow: the chain then resolves less of the winding the closer it
        starts, and gets worse, not better. Inside the fast elliptic cycle of
        the tests the reference at (0.3, 0) was 1.66 off with an estimate of
        1.47 (DEQ-117): the chain wound 0.14 times, the surface at twice the
        distance was 0.19 off, and one at half the distance 1.84 off. So the
        value is also relaxed from a third surface, at half the distance, and
        the estimate is returned as ``inf`` - no bound - where going in again
        moves the value by more than the estimate, or where the first two
        surfaces' values differ by more than half of ``U`` on the outer one,
        more than the expansion could be wrong by. Otherwise the surface's
        share of the estimate is the tail of a geometric series in the two
        moves, outer to inner and inner to inmost, doubled for safety and never
        below the outer move; where the inner move is not smaller than the
        outer one there is no bound, unless it is within the chain's own
        error, the half-nodes difference, and may be the chain's noise
        rather than the surface's error; then twice the two moves together
        stand for the surface (see :func:`_estimate`). (The larger of the
        two moves alone, used at first, did not bound the error beside the
        fast ellipse, where the values converge at first order: 2.9e-5 off
        against 2.6e-5 at (-1.1, 0.4), DEQ-152.) The start on each surface
        is found by the secant on the slope and then by parabolas on the
        values (:func:`_relax_from`).

        So that the chain can follow a winding path at all (DEQ-152), the
        trace it starts from is carried on into the inmost surface when it
        stalls outside it (:meth:`_onto`), each chain gets 320 nodes per turn
        of its trace (:func:`_winding_nodes`), and a start slid along the
        surface carries the chain with it (:func:`_moved_start`). Measured on
        77 references at 81 points on eleven sets of points - the non-normal,
        anisotropic-gradient and SDE (24) reference problems, the rotating
        Ornstein-Uhlenbeck process, SDE (21), and the verification's strongly
        rotating linear, gradient, rotating, multiplicative and elliptic-cycle
        systems, 20 of them beside or inside the elliptic cycle - every finite
        estimate bounds the reference's error, the closest by a factor of
        2.5. Before, 24 were labelled and 4 failed, the elliptic cycle's 20
        among them; now none fails and 4 are labelled, none of them the
        cycle's. On ten more points, five of them beside the cycle where the
        values converge slowly, none is finite and unbounded either (closest
        factor 1.85) and two are labelled. The relaxations of a winding path
        cost far more: up to 100 seconds a point inside the cycle, 4 to 8
        beside it, and 0.4 to 18 elsewhere (all at 81 points). This is
        measured, not proven.

        Args:
            field: A field from :meth:`solve`, which knows its attractor.
            state: The point to evaluate ``U`` at.
            nodes: The number of states in the relaxed chain.

        Returns:
            ``(value, error)``: ``U`` at ``state`` and the estimate of its
            error, ``inf`` where no bound could be established.

        Raises:
            ValueError: If ``field`` does not know its attractor, or a chain
                does not stay finite at any step.
        """
        attractor = field.attractor
        if attractor is None:
            message = "a reference value needs the attractor the field was marched from"
            raise ValueError(message)
        target = np.asarray(state, dtype=float)
        trace = self.minimum_action_path(field, target).states
        gmam = GeometricMinimumAction(
            self._drift,
            self._noise,
            tolerance=_REFERENCE_SETTLED,
        )
        outer = attractor.start(field.grid, self._drift, self._noise, target)
        inner = outer.shrunk()
        inmost = inner.shrunk()
        trace = self._onto(field, trace, inmost)
        far, _, place = _relax_from(gmam, outer, _guide(outer, trace, nodes))
        near, chain, parameter = _relax_from(gmam, inner, _guide(inner, trace, nodes))
        start = inner.point(parameter)
        coarse = _equal_arc_length(_resample(chain, len(chain) // 2 + 1))
        coarse[0] = start
        rough = inner.value(start) + gmam.path(coarse).action
        # Each difference is taken whole, not divided by 2^p - 1 as a Richardson
        # estimate would: the chain's error goes at fourth order on the linear
        # systems (1.04e-3, 8.4e-5, 5.9e-6 at 41, 81, 161 nodes, non-normal)
        # but at second order on the anisotropic gradient (-4.9e-5, -1.3e-5 at
        # 81, 161 at (-1.5, -0.8)), and the whole difference bounds both.
        own = abs(far - near) + abs(rough - near)
        # The estimate is a bound only while the surface's error is what
        # changes as the surface shrinks. Where the path winds round the
        # attractor faster than the chain resolves, the value moves further
        # on going in again than the estimate allows, or the two chains are
        # different paths altogether - further apart than U on the outer
        # surface, all the expansion could be wrong by (DEQ-117).
        level = abs(outer.value(outer.point(place)))
        if abs(far - near) > _HALF * level:
            return near, np.inf
        try:
            nearest, _, _ = _relax_from(gmam, inmost, _guide(inmost, trace, nodes))
        except ValueError:
            # A chain that will not stay finite that close in is the winding
            # the label is for.
            return near, np.inf
        if abs(near - nearest) > own:
            return near, np.inf
        return near, _estimate(far, near, nearest, rough)

    def _onto(
        self, field: GridQuasipotential, trace: np.ndarray, surface
    ) -> np.ndarray:
        """``trace`` carried on until it is inside ``surface``.

        The traced path stops where a step would raise ``U``, and round a fast
        rotation that happens far out: the path there runs almost along the
        level sets, ``U`` falls by little per step, and the grid's error in it
        is enough to stop the trace - at the verification's elliptic cycle
        (DEQ-152) after 0.02 of the 12.6 units of the path to (0.3, 0), which
        left the chain a straight line that wound 0.12 times against 3.6. So
        the trace goes on along ``-(f + D grad U)`` by the same midpoint
        steps, whatever ``U`` does: with the grid's ``grad U`` until it
        reaches the region the attractor initialised, and from there, where
        the grid has nothing more to say, with the attractor's own expansion
        (``surface.gradient``), until it enters the surface. It stops early if
        a step is not finite - where ``f + D grad U`` vanishes, so the path
        has no direction - or leaves the grid, and in any case after
        ``_DESCENT_MAX_STEPS`` steps. A trace that entered the surface is
        returned as it was.
        """
        if any(surface.inside(state) for state in trace):
            return trace
        attractor = field.attractor
        spacing = float(np.min(self._grid.spacings))
        length = _DESCENT_STEP * spacing
        lower = np.array([self._grid.axis(a)[0] for a in range(_PLANE)])
        upper = np.array([self._grid.axis(a)[-1] for a in range(_PLANE)])
        grid_slope = field.gradient()

        def from_grid(state: np.ndarray) -> np.ndarray:
            return self._backward(state, grid_slope)

        def from_expansion(state: np.ndarray) -> np.ndarray:
            return self._backward_along(state, surface.gradient(state))

        states = list(trace)
        current = np.asarray(trace[-1], dtype=float)
        backward = from_grid
        for _ in range(_DESCENT_MAX_STEPS):
            if attractor is not None and attractor.reached(current, self._grid):
                backward = from_expansion
            half = current + _HALF * length * backward(current)
            following = current + length * backward(half)
            if not (
                np.all(np.isfinite(following))
                and np.all(following >= lower)
                and np.all(following <= upper)
            ):
                break
            current = following
            states.append(current)
            if surface.inside(current):
                break
        return np.array(states)

    def _backward_along(self, state: np.ndarray, slope: np.ndarray) -> np.ndarray:
        """The unit vector along ``-(f + D grad U)`` for a given ``grad U``."""
        velocity = field_at(self._drift, state) + (
            diffusion_matrix(self._noise.columns_at(state)) @ slope
        )
        size = float(np.linalg.norm(velocity))
        if size == 0.0:
            return np.full(_PLANE, np.nan)
        return -velocity / size

    def _backward(self, state: np.ndarray, gradient) -> np.ndarray:
        slope = np.array([component(state[None, :])[0] for component in gradient])
        velocity = (
            field_at(self._drift, state)
            + diffusion_matrix(
                self._noise.columns_at(state),
            )
            @ slope
        )
        size = float(np.linalg.norm(velocity))
        if size == 0.0:
            return np.full(_PLANE, np.nan)
        return -velocity / size


class _March:
    """Dijkstra's ordering: accept the smallest tentative ``U``, update around it."""

    __slots__ = [
        "_coordinates",
        "_offsets",
        "_quadratic",
        "_shape",
        "_status",
        "_table",
        "values",
    ]

    def __init__(
        self,
        grid: DensityGrid,
        lattice: _HalfLattice,
        offsets: tuple[np.ndarray, np.ndarray],
        *,
        quadratic: bool = False,
    ) -> None:
        self._shape = grid.shape
        self._table = lattice.table
        self._offsets = offsets
        self._quadratic = quadratic
        self._coordinates = (grid.axis(0), grid.axis(1))
        self._status = np.full(self._shape, _UNKNOWN, dtype=np.int8)
        self.values = np.full(self._shape, np.inf)

    def run(self, indices: np.ndarray, values: np.ndarray) -> None:
        """Accept the initialised points in order, then march until none is left."""
        rows, columns = np.unravel_index(indices, self._shape)
        self.values[rows, columns] = values
        heap: list[tuple[float, int, int]] = []
        for order in np.argsort(values):
            self._accept(int(rows[order]), int(columns[order]), heap)
        while heap:
            value, row, column = heapq.heappop(heap)
            if self._status[row, column] == _ACCEPTED:
                continue
            if value > self.values[row, column]:
                continue
            self._accept(row, column, heap)

    def _accept(self, row: int, column: int, heap: list) -> None:
        self._status[row, column] = _ACCEPTED
        targets_x, targets_y = self._open_targets(row, column)
        if targets_x.size == 0:
            return
        first, second = self._coordinates
        reach_x = first[targets_x] - first[row]
        reach_y = second[targets_y] - second[column]
        near = self._table[row + targets_x, column + targets_y]
        own = float(self.values[row, column])
        best = own + _action(reach_x, reach_y, near)
        neighbours = self._accepted_neighbours(row, column)
        if neighbours:
            count = targets_x.size
            pairs = np.tile(np.arange(count), len(neighbours))
            others_x = np.repeat([x for x, _ in neighbours], count)
            others_y = np.repeat([y for _, y in neighbours], count)
            edge = (first[others_x] - first[row], second[others_y] - second[column])
            reach = (reach_x[pairs], reach_y[pairs])
            midpoints = (
                near[pairs],
                self._table[others_x + targets_x[pairs], others_y + targets_y[pairs]],
            )
            other = self.values[others_x, others_y]
            if self._quadratic:
                bends = np.repeat(self._bends(row, column, neighbours), count)
                triangle = _curved_triangle(own, other, bends, edge, reach, midpoints)
            else:
                triangle = _triangle(own, other, edge, reach, midpoints)
            nearest = triangle.reshape(len(neighbours), count).min(axis=0)
            best = np.minimum(best, nearest)
        better = best < self.values[targets_x, targets_y]
        for x, y, value in zip(
            targets_x[better].tolist(),
            targets_y[better].tolist(),
            best[better].tolist(),
            strict=True,
        ):
            self.values[x, y] = value
            heapq.heappush(heap, (value, x, y))

    def _bends(self, row: int, column: int, neighbours) -> np.ndarray:
        """``c`` in ``U(x_s) = U0 + s dU + c s (s - 1)`` for each base ``x0``-``x1``.

        The parabola through ``x0``, ``x1 = x0 + e`` and a third accepted point
        on the same line, ``x0 - e`` if it is accepted, else ``x0 + 2 e``; with
        neither, ``c = 0`` and the base is the straight line of OLIM-MID.
        """
        size_x, size_y = self._shape
        own = self.values[row, column]
        steps = np.asarray(neighbours) - np.array([row, column])
        rise = self.values[row + steps[:, 0], column + steps[:, 1]] - own
        bends = np.zeros(len(neighbours))
        found = np.zeros(len(neighbours), dtype=bool)
        # The point behind x0 first, then the one beyond x1.
        for reach, weight in ((-1, 1.0), (2, -2.0)):
            third_x, third_y = row + reach * steps[:, 0], column + reach * steps[:, 1]
            inside = (
                (third_x >= 0)
                & (third_x < size_x)
                & (third_y >= 0)
                & (third_y < size_y)
            )
            open_x, open_y = np.where(inside, third_x, 0), np.where(inside, third_y, 0)
            usable = inside & (self._status[open_x, open_y] == _ACCEPTED)
            usable &= ~found
            fitted = _HALF * (self.values[open_x, open_y] - own + weight * rise)
            bends = np.where(usable, fitted, bends)
            found |= usable
        return bends

    def _open_targets(self, row: int, column: int) -> tuple[np.ndarray, np.ndarray]:
        """Every unaccepted point within the update radius of ``(row, column)``."""
        size_x, size_y = self._shape
        targets_x = row + self._offsets[0]
        targets_y = column + self._offsets[1]
        inside = (
            (targets_x >= 0)
            & (targets_x < size_x)
            & (targets_y >= 0)
            & (targets_y < size_y)
        )
        targets_x, targets_y = targets_x[inside], targets_y[inside]
        unaccepted = self._status[targets_x, targets_y] != _ACCEPTED
        return targets_x[unaccepted], targets_y[unaccepted]

    def _accepted_neighbours(self, row: int, column: int) -> list[tuple[int, int]]:
        size_x, size_y = self._shape
        return [
            (row + step_x, column + step_y)
            for step_x, step_y in _NEIGHBOURS
            if 0 <= row + step_x < size_x
            and 0 <= column + step_y < size_y
            and self._status[row + step_x, column + step_y] == _ACCEPTED
        ]


def _triangle(own, other, edge, reach, midpoints):
    """The triangle update from the segment ``x0``-``x1`` to every target.

    ``edge`` is ``x1 - x0`` and ``reach`` is ``x - x0``, as component pairs;
    ``midpoints`` holds ``f`` and ``A`` at the midpoints of ``x0``-``x`` and of
    ``x1``-``x``, one half-lattice row per target each.
    With ``f`` and ``A`` frozen, ``F(s) = U(x_s) + Q(x_s, x)`` is convex in ``s`` and
    ``F'(s) = 0`` is a quadratic, so the minimiser is closed-form; ``f`` and ``A``
    are then re-read at that ``s`` and the root taken again. Two refinements from the
    middle of the segment are enough because the two end midpoints are half a cell
    apart, so ``f`` and ``A`` barely move along it.
    """
    near, far = midpoints
    rise = other - own
    fraction = np.full(rise.shape[0], _HALF)
    for _ in range(_REFINEMENTS):
        coefficients = near + fraction[:, None] * (far - near)
        fraction = _stationary_fraction(edge, reach, rise, coefficients)
    coefficients = near + fraction[:, None] * (far - near)
    step_x = reach[0] - fraction * edge[0]
    step_y = reach[1] - fraction * edge[1]
    return own + fraction * rise + _action(step_x, step_y, coefficients)


def _curved_triangle(own, other, bend, edge, reach, midpoints):  # noqa: PLR0913
    """The triangle update with ``U`` a parabola along the base, minimised exactly.

    ``U(x_s) = U0 + s dU + c s (s - 1)`` in place of the straight line, and
    ``f`` and ``A`` in ``Q`` linear in ``s`` between the two end midpoints, as
    Dahiya and Cameron take them (arXiv 1806.05321, Eqs. (21), (22)). The
    minimum of ``F(s) = U(x_s) + Q(x_s, x)`` is where ``F'(s) = 0``, with
    ``F'`` the exact derivative of that ``F`` (:func:`_base_slope`), the change
    of ``f`` and ``A`` along the base included - their Eq. (23) - and its
    root is found by :func:`_root_on_base`, a secant iteration safeguarded by
    bisection like the paper's hybrid solver, from the straight base's
    frozen-coefficient root. The value returned is the least of ``F`` at that
    root, at the starting point and at both ends of the base, the ends being
    the one-point updates through ``x0`` and ``x1``: the update never offers
    more than they do.

    On a linear drift with a constant ``D`` this ``F`` is exact - the midpoint
    rule is exact along the characteristic through the segment's midpoint and
    the parabola is exact for a quadratic ``U`` - so what was left of the error
    was the minimisation: Newton steps with ``f`` and ``A`` frozen at the
    current ``s`` left the rotating Ornstein-Uhlenbeck process 1.70e-4 off at
    41 points, and the exact root leaves it 3e-15 off.
    """
    rise = other - own
    near, far = midpoints
    start = np.full(rise.shape[0], _HALF)
    for _ in range(_REFINEMENTS):
        coefficients = near + start[:, None] * (far - near)
        start = _stationary_fraction(edge, reach, rise, coefficients)
    root = _root_on_base(rise, bend, edge, reach, midpoints, start)
    candidates = [
        _base_cost(own, rise, bend, edge, reach, midpoints, fraction)
        for fraction in (root, start, np.zeros_like(start), np.ones_like(start))
    ]
    return np.minimum.reduce(candidates)


def _base_cost(own, rise, bend, edge, reach, midpoints, fraction):  # noqa: PLR0913
    """``F(s) = U0 + s rise + bend s (s - 1) + Q(x_s, x)``."""
    near, far = midpoints
    coefficients = near + fraction[:, None] * (far - near)
    step_x = reach[0] - fraction * edge[0]
    step_y = reach[1] - fraction * edge[1]
    curve = own + fraction * rise + bend * fraction * (fraction - 1.0)
    return curve + _action(step_x, step_y, coefficients)


def _base_slope(rise, bend, edge, reach, base, fraction):  # noqa: PLR0913
    """``F'(s)`` with ``f`` and ``A`` linear along the base, not frozen.

    ``base`` is the pair ``(near, far - near)``: ``f`` and ``A`` at ``s`` are
    ``near + s (far - near)``. With ``d = reach - s edge``, ``a = |d|_A^2``,
    ``b = |f|_A^2`` and ``c = <d, f>_A``, all three depending on ``s`` through
    ``d``, ``f`` and ``A``, ``Q = sqrt(a b) - c`` and ``Q' = (a' b + a b') /
    (2 sqrt(a b)) - c'`` - Dahiya and Cameron's Eq. (23) (Physica D 2018,
    arXiv 1806.05321), their Eq. (B-8) of arXiv 1706.07509 with a matrix
    ``A``. ``nan`` where ``Q`` is not differentiable: a vanishing ``f``, or
    ``A`` undefined beside a point where ``D`` vanishes.
    """
    near, change = base
    coefficients = near + fraction[:, None] * change
    edge_x, edge_y = edge
    step_x = reach[0] - fraction * edge_x
    step_y = reach[1] - fraction * edge_y
    velocity_x, velocity_y = coefficients[:, 0], coefficients[:, 1]
    turn_x, turn_y = change[:, 0], change[:, 1]
    first, mixed, second = coefficients[:, 2], coefficients[:, 3], coefficients[:, 4]
    # A d, A f, A' d and A' f, component by component.
    pulled_x = first * step_x + mixed * step_y
    pulled_y = mixed * step_x + second * step_y
    carried_x = first * velocity_x + mixed * velocity_y
    carried_y = mixed * velocity_x + second * velocity_y
    first, mixed, second = change[:, 2], change[:, 3], change[:, 4]
    bent_x = first * step_x + mixed * step_y
    bent_y = mixed * step_x + second * step_y
    swept_x = first * velocity_x + mixed * velocity_y
    swept_y = mixed * velocity_x + second * velocity_y
    squared = step_x * pulled_x + step_y * pulled_y
    speed = velocity_x * carried_x + velocity_y * carried_y
    squared_slope = (
        step_x * bent_x
        + step_y * bent_y
        - 2.0 * (edge_x * pulled_x + edge_y * pulled_y)
    )
    speed_slope = (
        velocity_x * swept_x
        + velocity_y * swept_y
        + 2.0 * (turn_x * carried_x + turn_y * carried_y)
    )
    cross_slope = (
        turn_x * pulled_x
        + turn_y * pulled_y
        + step_x * swept_x
        + step_y * swept_y
        - edge_x * carried_x
        - edge_y * carried_y
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        action_slope = (squared_slope * speed + squared * speed_slope) / (
            2.0 * np.sqrt(squared * speed)
        ) - cross_slope
    return rise + bend * (2.0 * fraction - 1.0) + action_slope


def _root_on_base(rise, bend, edge, reach, midpoints, start):  # noqa: PLR0913
    """Where ``F'`` changes sign from negative to positive in ``[0, 1]``.

    ``F'`` is evaluated at ``start`` and at the end it points away from:
    where the two differ in sign they bracket a minimum, and a secant step
    from the last two iterates is taken whenever it falls inside the bracket
    and at most half its width from the last iterate, a bisection otherwise
    - the hybrid secant and bisection solver Dahiya and Cameron use
    (Wilkinson 1967). The bracket shrinks at every step and the iterate never
    leaves it. Where nothing is bracketed, ``start`` is returned: the caller
    compares ``F`` at the ends anyway.
    """
    base = (midpoints[0], midpoints[1] - midpoints[0])
    start_slope = _base_slope(rise, bend, edge, reach, base, start)
    end = np.where(start_slope < 0.0, 1.0, 0.0)
    end_slope = _base_slope(rise, bend, edge, reach, base, end)
    fraction = start.copy()
    active = np.flatnonzero(
        ((start_slope < 0.0) & (end_slope > 0.0))
        | ((start_slope > 0.0) & (end_slope < 0.0)),
    )
    if active.size == 0:
        return fraction
    upward = end[active] == 1.0
    low = np.where(upward, start[active], 0.0)
    high = np.where(upward, 1.0, start[active])
    low_slope = np.where(upward, start_slope[active], end_slope[active])
    high_slope = np.where(upward, end_slope[active], start_slope[active])
    # The secant starts from the two ends of the bracket, start the later.
    previous, previous_slope = end[active], end_slope[active]
    current, slope = start[active], start_slope[active]
    rise, bend = rise[active], bend[active]
    edge = (edge[0][active], edge[1][active])
    reach = (reach[0][active], reach[1][active])
    base = (base[0][active], base[1][active])
    # The last step's length and the one before it.
    last, lagged = np.full(active.size, np.inf), np.full(active.size, np.inf)
    for _ in range(_ROOT_STEPS):
        with np.errstate(divide="ignore", invalid="ignore"):
            secant = current - slope * (current - previous) / (slope - previous_slope)
        # A secant step that leaves the bracket, or is not half as long as
        # the step before last, gives way to a bisection (Brent's rule).
        inside = (secant > low) & (secant < high)
        inside &= np.abs(secant - current) < _HALF * lagged
        following = np.where(inside, secant, _HALF * (low + high))
        last, lagged = np.abs(following - current), last
        following_slope = _base_slope(rise, bend, edge, reach, base, following)
        below = following_slope < 0.0
        low = np.where(below, following, low)
        high = np.where(below, high, following)
        low_slope = np.where(below, following_slope, low_slope)
        high_slope = np.where(below, high_slope, following_slope)
        # F' rises through zero across the bracket, so F at the end where
        # |F'| is smaller is within |F'| times the width of its least value.
        closer = -low_slope < high_slope
        fraction[active] = np.where(closer, low, high)
        spread = np.minimum(-low_slope, high_slope) * (high - low)
        settled = (spread <= _ROOT_TOLERANCE) | ~np.isfinite(following_slope)
        moving = ~settled
        if not np.any(moving):
            break
        active = active[moving]
        rise, bend = rise[moving], bend[moving]
        edge = (edge[0][moving], edge[1][moving])
        reach = (reach[0][moving], reach[1][moving])
        base = (base[0][moving], base[1][moving])
        low, high = low[moving], high[moving]
        low_slope, high_slope = low_slope[moving], high_slope[moving]
        last, lagged = last[moving], lagged[moving]
        previous, previous_slope = current[moving], slope[moving]
        current, slope = following[moving], following_slope[moving]
    return fraction


def _stationary_fraction(edge, reach, rise, coefficients):
    """Root of ``F'(s)`` in ``[0, 1]`` for frozen ``f`` and ``A``.

    With ``d(s) = reach - s edge``, ``F'(s) = rise + <edge, f>_A - |f|_A <edge, d>_A /
    |d|_A``, so the root solves ``<edge, d>_A / |d|_A = r`` with ``r = (rise +
    <edge, f>_A) / |f|_A``; squaring gives a quadratic in ``s`` whose admissible root
    is the one below. Where ``r^2`` is at least ``|edge|_A^2`` the derivative has one
    sign on the whole segment and the minimum is an end.
    """
    edge_x, edge_y = edge
    reach_x, reach_y = reach
    velocity_x, velocity_y = coefficients[:, 0], coefficients[:, 1]
    length = _form(edge_x, edge_y, coefficients, edge_x, edge_y)
    projection = _form(edge_x, edge_y, coefficients, reach_x, reach_y)
    squared = _form(reach_x, reach_y, coefficients, reach_x, reach_y)
    speed = np.sqrt(_form(velocity_x, velocity_y, coefficients, velocity_x, velocity_y))
    slope = rise + _form(edge_x, edge_y, coefficients, velocity_x, velocity_y)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = slope / speed
        room = length - ratio * ratio
        root = projection / length - ratio * np.sqrt(
            np.maximum(length * squared - projection * projection, 0.0),
        ) / (length * np.sqrt(room))
    fraction = np.where(slope > 0.0, 0.0, 1.0)
    fraction = np.where(room > 0.0, root, fraction)
    fraction = np.where(np.isfinite(fraction), fraction, _HALF)
    return np.clip(fraction, 0.0, 1.0)


class GeometricMinimumAction:
    """The minimum action path between two points, by the simplified gMAM.

    The path is a chain of states with its ends fixed, parametrised by normalised arc
    length ``alpha`` in ``[0, 1]``. In coordinates ``z = L^{-1} x`` with
    ``D = L L^T`` the noise is the identity, and with ``b`` the drift and ``J`` its
    Jacobian there the steepest descent of the geometric action, preconditioned by
    ``lambda = |b| / |z'|``, is

        ``z_tau = lambda^2 z'' - lambda (J - J^T) z' - J^T b``

    up to a tangential part. Only the component of the explicit terms normal to the
    path is applied, and the reparametrisation to equal arc length after every step
    redistributes the nodes; see the comment in ``_relax`` for why the tangential
    part cannot simply be left to the reparametrisation. On the rotating
    Ornstein-Uhlenbeck process from the origin to ``(0.8, -0.3)`` the action is
    0.34%, 0.16% and 0.11% above ``(1/2) x^T S^{-1} x`` with 41, 81 and 161
    nodes, the same at steps 0.05 and 0.01; what remains is the infinitely many
    turns the exact path makes as it spirals out of the origin, which a chain of
    nodes cannot follow. The ``z''`` term is taken implicitly - a tridiagonal
    solve per component - so the step is not limited by the node spacing. The
    action reported is ``integral |b| |z'| - <b, z'> d alpha`` by the midpoint rule.

    A ``D`` that depends on the state takes the general form of the method,
    in ``x`` itself (see :meth:`_varying_relax`): the momentum has the closed
    form ``theta = D^{-1} (lambda x' - f)`` of a quadratic Hamiltonian, the
    flow gains the derivatives of ``D``, and it is preconditioned by ``D`` so
    that its stiff part is again ``lambda^2 x''``, taken implicitly. On
    ``f = -D grad V + R grad V`` with ``D`` varying by a factor of two (the
    tests' multiplicative system, ``U = 2 V``) the action at ``(-0.6, 0.5)``
    is off by 1.36e-4, 3.11e-5 and 1.04e-5 with 41, 81 and 161
    nodes. A ``D`` that is the same at every state of the chain and at the
    states moved a little along each axis is taken as constant. The drift, its
    Jacobian and ``D`` are evaluated at every state of the chain in one
    batch (:func:`~.noise.fields_at`, :func:`~.noise.jacobians_along`), which
    the long chains of a winding path need (DEQ-152): a 321-node chain into
    the verification's elliptic cycle took 105 s under a profiler one state
    at a time, 97% of it in the drift, and about 5 s batched (the action the same
    to the digits printed).

    *Limits.* A chain of nodes cannot follow a path that winds round its
    start - the minimiser out of a rotating equilibrium or cycle winds
    infinitely often - so the chain finds the least action among the paths
    it can resolve; :meth:`OrderedLineIntegral.reference` starts it off a
    small surface for that reason. Which minimiser it reaches depends on the
    starting chain, and a path that winds needs a few hundred nodes a turn:
    from the exact path into the elliptic cycle (3.6 turns) the action was
    off by 0.26, 0.21, 2.1e-2, 1.7e-3 and 1.3e-4 at 81, 161, 321, 641 and
    1281 nodes.

    References:
        - Heymann, M. and Vanden-Eijnden, E. (2008). The geometric minimum action
          method: a least action principle on the space of curves. Communications on
          Pure and Applied Mathematics 61(8), 1052-1117. doi:10.1002/cpa.20238. The
          method; not read here (the publisher refused the fetch) - used as Grafke et
          al. restate it.
        - Grafke, T., Schafer, T. and Vanden-Eijnden, E. (2017). Long term effects of
          small random perturbations on dynamical systems: theoretical and computational
          tools. Fields Institute Communications, Springer. arXiv:1604.03818,
          doi:10.1007/978-1-4939-6969-2_2. Eq. (19), the gMAM descent flow, taken with
          the additive-noise Hamiltonian of Eqs. (32)-(35) after ``z = L^{-1} x``; Eqs.
          (38)-(42), the simplified flow for a state-dependent ``D``.
        - Vanden-Eijnden, E. and Heymann, M. (2008). The geometric minimum action method
          for computing minimum energy paths. Journal of Chemical Physics 128, 061103.
          doi:10.1063/1.2833040. The semi-implicit form; not read here.
    """

    __slots__ = ["_drift", "_iterations", "_noise", "_step", "_tolerance"]

    def __init__(
        self,
        drift: Function,
        noise: NoiseMatrix,
        step: float = 0.05,
        iterations: int = 20000,
        tolerance: float = 1.0e-10,
    ) -> None:
        self._drift = drift
        self._noise = noise
        self._step = step
        self._iterations = iterations
        self._tolerance = tolerance

    @property
    def drift(self) -> Function:
        """The drift ``f``."""
        return self._drift

    @property
    def noise(self) -> NoiseMatrix:
        """The noise amplitude ``G``."""
        return self._noise

    def path(
        self,
        initial: Sequence[Sequence[float]] | np.ndarray,
    ) -> MinimumActionPath:
        """Relax ``initial`` (states from one end to the other) to the minimiser.

        The chain is relaxed until no node moves by more than ``tolerance`` in a
        step, or for ``iterations`` steps; the result's ``converged`` says which,
        and a chain still moving is returned as it stands rather than refused.

        The explicit part of the flow is stiff where the drift rotates fast
        against its gradient - on Dahiya and Cameron's SDE (21), rotation ten
        times the gradient, a chain of 41 nodes blew up at a step of 0.05 - so
        a chain that stops being finite, or jumps, is started again from
        ``initial`` at half the step, a few times over, and the step that held
        is kept for later paths. The step decides only how fast the chain
        settles, not where.
        """
        states = np.asarray(initial, dtype=float)
        factor = self._constant_factor(states)
        step = self._step
        for _ in range(_STEP_HALVINGS):
            # A chain that blows up overflows on its way to nan; that is
            # caught in _relax_all, not warned about.
            with np.errstate(over="ignore", invalid="ignore"):
                relaxed = self._relax_all(states, factor, step)
            if relaxed is not None:
                self._step = step
                return relaxed
            step *= _HALF
        message = (
            f"the minimum action path did not stay finite at any step down to "
            f"{step / _HALF:g}"
        )
        raise ValueError(message)

    def _relax_all(
        self,
        states: np.ndarray,
        factor: np.ndarray | None,
        step: float,
    ) -> MinimumActionPath | None:
        """Relax at one step, in ``z = L^{-1} x`` for a constant ``D``.

        ``None`` if the chain stopped being finite, or if one step moved a node
        by more than a quarter of the distance between its ends: no relaxation
        moves that far at once, while an unstable step does - on SDE (21) at a
        step of 0.0125 the nodes jumped by twice that distance every step,
        and the chain neither blew up nor settled.
        """
        if factor is None:
            chain = states
        else:
            inverse = np.linalg.inv(factor)
            chain = states @ inverse.T
        spacing = 1.0 / (chain.shape[0] - 1)
        reach = _JUMP_SHARE * float(np.linalg.norm(chain[-1] - chain[0]))
        converged = False
        taken = 0
        for taken in range(1, self._iterations + 1):
            if factor is None:
                following = self._varying_relax(chain, spacing, step)
            else:
                following = self._relax(chain, factor, inverse, spacing, step)
            if not np.all(np.isfinite(following)):
                return None
            following = _equal_arc_length(following)
            change = float(np.max(np.abs(following - chain)))
            # The first step may move nodes far by reparametrising a chain
            # that was spaced evenly in x rather than z; only later ones count.
            if taken > 1 and change > reach:
                return None
            chain = following
            if change < self._tolerance:
                converged = True
                break
        if factor is None:
            action = self._varying_action(chain)
        else:
            action = self._action(chain, factor, inverse)
            chain = chain @ factor.T
        return MinimumActionPath(
            chain,
            action,
            converged=converged,
            iterations=taken,
        )

    def _constant_factor(self, states: np.ndarray) -> np.ndarray | None:
        """The Cholesky factor of a constant ``D``, or ``None`` where it varies.

        ``D`` is compared at the chain's states and at the same states moved a
        little along each axis, so that a chain lying where a varying ``D``
        happens to agree with itself is not taken for a constant one.
        """
        nudge = _CONSTANT_PROBE * max(float(np.max(np.abs(states))), 1.0)
        probes = np.vstack(
            [states, states + np.array([nudge, 0.0]), states + np.array([0.0, nudge])],
        )
        cells = np.array(
            [diffusion_matrix(self._noise.columns_at(state)) for state in probes],
        )
        _check_nodes(cells)
        deviation = float(np.max(np.abs(cells - cells[0])))
        if deviation > _CONSTANT_TOLERANCE * max(float(np.max(np.abs(cells[0]))), 1.0):
            return None
        lower, _ = cho_factor(cells[0], lower=True)
        return np.tril(lower)

    def _varying_relax(
        self,
        chain: np.ndarray,
        spacing: float,
        step: float,
    ) -> np.ndarray:
        """One step of the general descent, preconditioned by ``D``.

        With ``A = D^{-1}``, ``lambda = |f|_A / |x'|_A`` and the momentum
        ``theta = A (lambda x' - f)`` (Grafke, Schafer and Vanden-Eijnden,
        Eqs. (40), (41)), the descent direction of their Eq. (42) is
        ``lambda theta' + f_x^T theta + (1/2) theta^T D_x theta``. Multiplied by
        ``D`` - a positive definite preconditioner - and with
        ``theta' = A' (lambda x' - f) + A (lambda' x' + lambda x'' - f_x x')``
        expanded, it is

            ``lambda^2 x'' - lambda D' theta - lambda f_x x'
              + D (f_x^T theta + (1/2) theta^T D_x theta)``

        up to a part along ``x'``, which is dropped as in :meth:`_relax`. The
        ``lambda^2 x''`` term is taken implicitly. ``D'`` is ``D``'s derivative
        along the chain, ``sum_k (dD / dx_k) x'_k``, and ``theta^T D_x theta``
        is the vector of ``theta^T (dD / dx_k) theta``. With a constant ``D``
        this is the flow of :meth:`_relax` in the coordinates ``x = L z``.
        """
        velocity = fields_at(self._drift, chain)
        jacobians = jacobians_along(self._drift, chain, velocity)
        cells, slopes = _diffusion_and_slopes(self._noise, chain)
        metric = np.linalg.inv(cells)
        tangent = np.gradient(chain, spacing, axis=0)
        speed = np.sqrt(np.einsum("ki,kij,kj->k", tangent, metric, tangent))
        size = np.sqrt(np.einsum("ki,kij,kj->k", velocity, metric, velocity))
        scale = size / np.where(speed > 0.0, speed, 1.0)
        momentum = np.einsum(
            "kij,kj->ki",
            metric,
            scale[:, None] * tangent - velocity,
        )
        change = np.einsum("kmij,km->kij", slopes, tangent)
        force = -scale[:, None] * (
            np.einsum("kij,kj->ki", change, momentum)
            + np.einsum("kij,kj->ki", jacobians, tangent)
        )
        pulled = np.einsum("kji,kj->ki", jacobians, momentum) + _HALF * np.einsum(
            "ki,kmij,kj->km",
            momentum,
            slopes,
            momentum,
        )
        force += np.einsum("kij,kj->ki", cells, pulled)
        length = np.linalg.norm(tangent, axis=1)
        unit = tangent / np.where(length > 0.0, length, 1.0)[:, None]
        force -= np.sum(force * unit, axis=1)[:, None] * unit
        return _implicit_step(chain, step * force, step * scale * scale, spacing)

    def _varying_action(self, chain: np.ndarray) -> float:
        """``sum |f|_A |dx|_A - <f, dx>_A`` with ``f`` and ``A`` at each midpoint."""
        middle = _HALF * (chain[1:] + chain[:-1])
        steps = np.diff(chain, axis=0)
        velocity = fields_at(self._drift, middle)
        metric = np.linalg.inv(_cells(self._noise, middle))
        size = np.sqrt(np.einsum("ki,kij,kj->k", velocity, metric, velocity))
        length = np.sqrt(np.einsum("ki,kij,kj->k", steps, metric, steps))
        return float(
            np.sum(size * length - np.einsum("ki,kij,kj->k", steps, metric, velocity)),
        )

    def _drift_at(self, chain, factor, inverse):
        states = chain @ factor.T
        fields = fields_at(self._drift, states)
        velocity = fields @ inverse.T
        jacobians = inverse @ jacobians_along(self._drift, states, fields) @ factor
        return velocity, jacobians

    def _relax(self, chain, factor, inverse, spacing, step):
        velocity, jacobians = self._drift_at(chain, factor, inverse)
        tangent = np.gradient(chain, spacing, axis=0)
        speed = np.linalg.norm(tangent, axis=1)
        scale = np.linalg.norm(velocity, axis=1) / np.where(speed > 0.0, speed, 1.0)
        skew = jacobians - np.transpose(jacobians, (0, 2, 1))
        force = -scale[:, None] * np.einsum("kij,kj->ki", skew, tangent) - np.einsum(
            "kji,kj->ki",
            jacobians,
            velocity,
        )
        # Only the normal part moves the curve; the tangential part only slides
        # nodes along it, and the reparametrisation's linear interpolation then
        # cuts the corners it slid them round. Left in, that bias balanced the
        # flow at a fixed point that depended on the step size: 0.63% and 1.02%
        # above the exact action at steps 0.05 and 0.01, and growing with the
        # node count. Projected out, the answer is the same at both steps and
        # converges as nodes are added.
        unit = tangent / np.where(speed > 0.0, speed, 1.0)[:, None]
        force -= np.sum(force * unit, axis=1)[:, None] * unit
        return _implicit_step(chain, step * force, step * scale * scale, spacing)

    def _action(self, chain, factor, inverse) -> float:
        middle = _HALF * (chain[1:] + chain[:-1])
        steps = np.diff(chain, axis=0)
        velocity = fields_at(self._drift, middle @ factor.T) @ inverse.T
        return float(
            np.sum(
                np.linalg.norm(velocity, axis=1) * np.linalg.norm(steps, axis=1)
                - np.sum(velocity * steps, axis=1),
            ),
        )


def _implicit_step(chain, push, stiffness, spacing):
    """``x + push`` with ``stiffness x''`` taken implicitly, the ends held.

    ``nan`` throughout if either is not finite, which the caller takes for a
    step too large.
    """
    explicit = chain + push
    weight = stiffness / (spacing * spacing)
    if not (np.all(np.isfinite(explicit)) and np.all(np.isfinite(weight))):
        return np.full_like(chain, np.nan)
    nodes = chain.shape[0]
    bands = np.zeros((3, nodes))
    bands[1] = 1.0 + 2.0 * weight
    bands[0, 1:] = -weight[:-1]
    bands[2, :-1] = -weight[1:]
    bands[1, 0] = bands[1, -1] = 1.0
    bands[0, 1] = 0.0
    bands[2, -2] = 0.0
    explicit[0], explicit[-1] = chain[0], chain[-1]
    return np.column_stack(
        [solve_banded((1, 1), bands, explicit[:, axis]) for axis in range(_PLANE)],
    )


def _diffusion_and_slopes(noise: NoiseMatrix, chain: np.ndarray):
    """``D`` at each state, and ``dD / dx_k`` by central differences."""
    cells = _cells(noise, chain)
    slopes = np.empty((chain.shape[0], _PLANE, _PLANE, _PLANE))
    for axis in range(_PLANE):
        shift = np.zeros(_PLANE)
        shift[axis] = _TAYLOR_STEP
        ahead = _cells(noise, chain + shift)
        behind = _cells(noise, chain - shift)
        slopes[:, axis] = (ahead - behind) / (2.0 * _TAYLOR_STEP)
    return cells, slopes


def _guide(surface, trace: np.ndarray, nodes: int) -> tuple[np.ndarray, float]:
    """A chain from ``surface`` to ``trace[0]`` along the traced path, and its start.

    The traced path runs from the target towards the attractor; it is cut where
    it enters the surface, or at its end if it stalled outside, and its last
    state outside is moved onto the surface. The chain has ``nodes`` states, or
    more where the traced path winds (:func:`_winding_nodes`).
    """
    outside = 0
    while outside < len(trace) and not surface.inside(trace[outside]):
        outside += 1
    kept = trace[: max(outside, 1)]
    parameter = surface.parameter(kept[-1])
    chain = np.vstack([surface.point(parameter), kept[::-1]])
    # The turns are counted on the traced path alone: the step from its last
    # state onto the surface is a corner, not a turn, and counted in it gave
    # the tests' non-normal system 141 nodes for a trace that turns 0.03 times.
    return _resample(chain, _winding_nodes(kept, nodes)), parameter


def _winding_nodes(trace: np.ndarray, nodes: int) -> int:
    """``nodes``, or more where ``trace`` turns more than once.

    The chain's error grows with how far its path turns per node: on the
    verification's elliptic cycle, with the exact path to (0.3, 0) from 0.06
    off the cycle (3.6 turns) as the initial chain, the relaxed value was off
    by 0.26, 0.21, 2.1e-2, 1.7e-3, 1.3e-4 and 1.3e-5 at 81, 161, 321, 641,
    1281 and 2561 nodes (DEQ-152), and at 81 it unwound a whole turn. So the
    chain gets at least ``_NODES_PER_TURN`` nodes for every turn the trace's
    direction makes, counted net, from its first step to its last.
    """
    steps = np.diff(trace, axis=0)
    steps = steps[np.linalg.norm(steps, axis=1) > 0.0]
    if len(steps) < _PLANE:
        return nodes
    angle = np.unwrap(np.arctan2(steps[:, 1], steps[:, 0]))
    turns = abs(float(angle[-1] - angle[0])) / (2.0 * np.pi)
    return max(nodes, int(np.ceil(_NODES_PER_TURN * turns)) | 1)


def _resample(chain: np.ndarray, count: int) -> np.ndarray:
    """``count`` states at equal arc length along the polygon through ``chain``."""
    lengths = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(lengths)])
    target = np.linspace(0.0, arc[-1], count)
    return np.column_stack(
        [np.interp(target, arc, chain[:, axis]) for axis in range(_PLANE)],
    )


def _momentum(drift: Function, noise: NoiseMatrix, states: np.ndarray) -> np.ndarray:
    """``p = |f|_A A t / |t|_A - A f`` at a chain's first state, ``A = D^{-1}``.

    The derivative of the geometric action by the path's velocity, which on a
    minimum action path is ``grad U``. The tangent ``t`` is the second-order
    one-sided difference, the chain being evenly spaced in its parameter.
    """
    tangent = -3.0 * states[0] + 4.0 * states[1] - states[2]
    metric = np.linalg.inv(diffusion_matrix(noise.columns_at(states[0])))
    velocity = field_at(drift, states[0])
    speed = np.sqrt(velocity @ metric @ velocity)
    return speed * metric @ tangent / np.sqrt(tangent @ metric @ tangent) - (
        metric @ velocity
    )


def _relax_from(
    gmam: "GeometricMinimumAction",
    surface,
    guide: tuple[np.ndarray, float],
) -> tuple[float, np.ndarray, float]:
    """``U(start) + S(start -> end)`` least over starts on ``surface``.

    Its derivative along the surface is ``(grad U - p) . t`` for the chain's
    momentum ``p`` and the surface's tangent ``t``, and the secant method finds
    its root, each step relaxing the previous chain from the new start. The
    slope comes from the chain's first states, and it can be wrong enough to
    stop the secant well off the least value: on the inmost surface about the
    rotating Ornstein-Uhlenbeck process at (-0.41, -0.71) it settled 0.14
    radians away, 3.4e-6 off where the least value is 1.0e-7 off (DEQ-152).
    So the values themselves are then minimised: a parabola through the least
    value found and its neighbours on either side, a few times
    (:func:`_parabolic_steps`). Returns the least value, its chain and the
    parameter of its start.
    """
    chain, parameter = guide
    nudge = surface.step * _SECANT_TOLERANCE

    def attempt(where: float, previous: np.ndarray) -> tuple[float, np.ndarray, float]:
        start = surface.point(where)
        path = gmam.path(_equal_arc_length(_moved_start(previous, start)))
        tangent = (surface.point(where + nudge) - surface.point(where - nudge)) / (
            2.0 * nudge
        )
        momentum = _momentum(gmam.drift, gmam.noise, path.states)
        slope = float((surface.gradient(start) - momentum) @ tangent)
        return surface.value(start) + path.action, path.states, slope

    best_place = parameter
    best = attempt(parameter, chain)
    tried = [(best_place, best[0], best[1])]
    behind_place, behind = best_place, best
    place = parameter - np.copysign(surface.step, best[2])
    for _ in range(_SECANT_STEPS):
        here = attempt(place, behind[1])
        tried.append((place, here[0], here[1]))
        if abs(here[2]) < abs(best[2]):
            best, best_place = here, place
        change = (here[2] - behind[2]) / (place - behind_place)
        if change == 0.0 or not np.isfinite(change):
            break
        following = place - here[2] / change
        reach = 4.0 * surface.step
        following = min(max(following, place - reach), place + reach)
        # F changes by about half the slope times the step still to go: once
        # that is far below any error asked about, the start is good enough.
        if abs(_HALF * here[2] * (following - place)) < _SECANT_SETTLED:
            break
        behind_place, behind = place, here
        place = following
    return _parabolic_steps(
        tried,
        lambda where, previous: attempt(where, previous)[:2],
        4.0 * surface.step,
    )


def _parabolic_steps(tried, attempt, reach: float):
    """The least of ``tried`` - ``(place, value, chain)`` - lowered by parabolas.

    Each step fits the parabola through the least value and the nearest
    place tried on either side of it (or the two nearest, if one side has
    none), and tries its vertex, kept within ``reach``; a parabola that does
    not open upwards steps ``reach / 4`` away from the higher neighbour
    instead. It stops after ``_PARABOLA_STEPS`` steps or once a step lowers
    the value by less than ``_SECANT_SETTLED``.
    """
    tried = list(tried)
    for _ in range(_PARABOLA_STEPS):
        tried.sort(key=lambda entry: entry[1])
        place, value, chain = tried[0]
        others = tried[1:]
        if not others:
            break
        below = [entry for entry in others if entry[0] < place]
        above = [entry for entry in others if entry[0] > place]
        if below and above:
            left = max(below, key=lambda entry: entry[0])
            right = min(above, key=lambda entry: entry[0])
        elif len(others) >= _PLANE:
            left, right = sorted(others, key=lambda entry: abs(entry[0] - place))[:2]
        else:
            break
        low, high = left[0] - place, right[0] - place
        if 0.0 in (low, high) or low == high:
            break
        # The parabola a s^2 + b s + value through the three, s from place.
        slope_low, slope_high = (left[1] - value) / low, (right[1] - value) / high
        curvature = (slope_high - slope_low) / (high - low)
        if curvature > 0.0:
            following = place - (slope_low - curvature * low) / (2.0 * curvature)
        else:
            higher = max((left, right), key=lambda entry: entry[1])
            following = place - np.copysign(0.25 * reach, higher[0] - place)
        following = min(max(following, place - reach), place + reach)
        if any(
            abs(following - entry[0]) <= _SECANT_TOLERANCE * reach for entry in tried
        ):
            break
        here, states = attempt(following, chain)
        tried.append((following, here, states))
        if value - here < _SECANT_SETTLED:
            break
    place, value, chain = min(tried, key=lambda entry: entry[1])
    return value, chain, place


def _estimate(far: float, near: float, nearest: float, rough: float) -> float:
    """The inner value's error from the three surfaces and the half-node chain.

    What is left of the inner value's surface error is bounded by the tail of
    a geometric series in the two moves, outer to inner and inner to inmost:
    with ratio ``r = |far - near| / |near - nearest|`` the error left in
    ``near`` is ``|near - nearest| r / (r - 1)``, doubled here for safety
    (``_TAIL_SAFETY``) and never taken below the outer move; the chain's own
    error, ``|rough - near|``, is added. On a smooth expansion ``r`` is near
    16 and the outer move is the larger. Beside the verification's fast
    ellipse the values converge at first order (``r`` = 1.7 and 2.4 at
    (-1.1, 0.4)), where the larger move alone, 2.6e-5, did not bound an error
    of 2.9e-5 (DEQ-152). Where going in moves the value as much as the step
    before (``r <= 1``) nothing says it converges, and there is no bound
    (``inf``) - unless the inner move is within the chain's own error: then
    the moves may be the chain's noise rather than the surface's error (SDE
    (21) at 41 points, (0.5, 0.3): moves of 5e-6 and 9e-6 against a chain term
    of 1.3e-4, and an error of 9e-6), and twice the two moves together, plus
    the chain's error, is the estimate. (The larger move alone there, tried
    first, bounded the slowly converging values beside the fast ellipse by
    only 4% at (0.2, -0.84): moves of 9.0e-6 and 9.1e-6, a chain term of
    2.3e-5 and an error of 3.1e-5.)
    """
    outward, inward = abs(far - near), abs(near - nearest)
    noise = abs(rough - near)
    if inward == 0.0:
        return outward + noise
    ratio = outward / inward
    if ratio > 1.0:
        tail = _TAIL_SAFETY * inward * ratio / (ratio - 1.0)
        return max(outward, tail) + noise
    if inward <= noise:
        return _TAIL_SAFETY * (outward + inward) + noise
    return float(np.inf)


def _moved_start(chain: np.ndarray, start: np.ndarray) -> np.ndarray:
    """``chain`` with its first state at ``start``, the move shared out along it.

    Each state moves by the start's move times the share of the arc length
    still to go, so the end stays put and no corner is made. Moving the first
    state alone - as before DEQ-152 - put a corner the size of the secant's
    step into the chain, harmless at 81 nodes but not at the thousand a
    winding path is given: inside the verification's elliptic cycle the
    chain then blew up at every step.
    """
    lengths = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(lengths)])
    share = 1.0 - arc / arc[-1] if arc[-1] > 0.0 else np.zeros_like(arc)
    return chain + share[:, None] * (np.asarray(start) - chain[0])


def _equal_arc_length(chain: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(lengths)])
    target = np.linspace(0.0, arc[-1], chain.shape[0])
    return np.column_stack(
        [np.interp(target, arc, chain[:, axis]) for axis in range(_PLANE)],
    )
