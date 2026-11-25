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
segment they are interpolated linearly between its two end midpoints, which makes
the one-dimensional minimisation over ``s`` for a straight ``U(x_s)`` a quadratic
with a closed-form root, refined twice from the midpoint of the segment. This
matches Dahiya and Cameron's handling of a variable anisotropic ``D`` (Physica D
2018, arXiv 1806.05321, their Eqs. (19), (21), (22)): ``A`` is ``D^{-1}`` at each
midpoint, not the inverse of an averaged ``D``, and ``f`` and ``A`` are linear
along the base. The update radius is what lets the
method see around an anisotropic or rotating drift: a minimum action path can meet
the accepted front at a shallow angle, and only a stencil wide enough to reach back
along it finds the right predecessor. Every triangle is tried rather than the
hierarchical subset of the paper (triangles only round the best one-point
predecessor): the subset was measured here and doubled the rotating
Ornstein-Uhlenbeck error below at 161 points (1.06e-3 against 5.0e-4).

The attractor is initialised exactly where it can be: around a stable equilibrium
``x*`` the points within a couple of cells get the quadratic quasipotential of the
linearisation, ``(1/2) (x - x*)^T S^{-1} (x - x*)`` with ``J S + S J^T + D(x*) = 0``;
around a limit cycle the points within a cell and a half get the paper's
Simpson-rule start from the nearest point of the cycle (see
:class:`CycleAttractor`), with ``grad U`` from the split of the drift into a
gradient and a rotational part.

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
``F(s)`` is then minimised by Newton steps from the straight base's closed-form
root, keeping the lower of the two. This is not a published method, and it is
not the default. The ticket proposed a Hermite cubic with ``grad U`` from the
march; it was built first and failed: ``grad U`` at a point can only come from
the update that set it, by the momentum ``|f|_A A t / |t|_A - A f`` along the
segment's direction ``t``, which is first order (the chord), or second order if
the start's own ``grad U`` fixes the curvature (the chord's reflection) - and
then each point's error feeds the next. On the rotating case at 41 points the
median ``grad U`` was off by 25% from the chord and 80% by reflection, and ``U``
by 0.32. Dahiya and Cameron also warn that ``U`` need not be smooth across the
front (arXiv 1806.05321, Sec. 3.3.2); the parabola uses accepted values only.

The parabola does not raise the order: the quadrature term above is ``O(K^2
h^2)`` either way. What it does is make the update exact where ``U`` is
quadratic - a linear drift with a constant ``D``. There the midpoint rule's
``Q`` is exact along the characteristic through the segment's midpoint
(``grad U`` at the midpoint is exact for a quadratic ``U``, and
``grad U . (x - a) <= Q(a, x)`` with equality along it), the parabola is
exact, and what is left is how well ``s`` is found: on the rotating
Ornstein-Uhlenbeck process at 41 points the parabola gives 1.70e-4 with the
Newton steps and 5.5e-8 with ``s`` minimised by brute force. On a nonlinear
drift the straight base's overestimate of a convex ``U`` partly cancels the
midpoint rule's underestimate, and the parabola takes that away.

When it helps - linear or nearly linear drifts. Max error at 321 points,
radius 8, straight against parabolic, on the sets above: rotating
Ornstein-Uhlenbeck 3.34e-4 and 3.41e-6 (94 times); their SDE (21) 1.34e-3 and
3.73e-4 (3.6 times); the non-normal linear system 2.36e-4 and 9.26e-5 (2.5
times); Maier-Stein's barrier at 161 x 129 points (radius 10) 5.6e-4 and
2.0e-5.

When it does not - nonlinear drifts. At 321 points, straight against
parabolic: double well 2.85e-3 both; anisotropic gradient 2.09e-3 and 2.15e-3;
SDE (24) 4.66e-3 and 5.15e-3; and on the verification's nonlinear rotating
systems (``f = -D grad V + c R grad V`` with ``c = 4``, a multiplicative-noise
one, an elliptic cycle with rotation 3) 7.1e-4 and 5.3e-4, 8.2e-4 and 1.27e-3,
8.5e-3 and 5.3e-3 - and at 161 points the parabola was worse on all three
(1.23e-3 and 2.23e-3, 4.28e-3 and 5.16e-3, 1.88e-2 and 3.17e-2). It costs 2.3
to 2.8 times the march's time (measured at 321 points, best of one under
load), so at equal cost the comparison is the straight base at 321 points
against the parabola at 161, and the straight base wins on every nonlinear
problem measured:

  ==========================  ===============  ================
  problem                     straight, 321    parabolic, 161
  ==========================  ===============  ================
  double well                 2.85e-3          1.13e-2
  anisotropic gradient        2.09e-3          8.70e-3
  SDE (24)                    4.66e-3          2.19e-2
  rotating, c = 4             7.13e-4          2.23e-3
  multiplicative noise        8.19e-4          5.16e-3
  elliptic cycle              8.54e-3          3.17e-2
  rotating Ornstein-Uhlenbeck 3.34e-4          1.16e-5
  ==========================  ===============  ================

So the parabola is for a drift that is linear, or nearly so on the region
asked about; ``update_order="midpoint"``, the default, is OLIM-MID as
published. On the noisy Hopf film's grid, near its singular origin, the worst
frame is 0.054 straight and 0.056 parabolic.

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
in its semi-implicit form for additive noise - the descent flow as Grafke, Schafer
and Vanden-Eijnden restate it (arXiv 1604.03818, Eq. (19)); the letter usually
cited for that form is Vanden-Eijnden and Heymann, J. Chem. Phys. 128, 061103
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
claimed to the order above. The geometric minimum action method here assumes a
constant ``D``. ``U`` near the boundary of the box is wrong wherever the true
minimum action path would leave the box, so the box should contain the sublevel set
of ``U`` being asked about: ``U`` increases along every minimum action path, so the
path to ``x`` stays inside ``{U <= U(x)}``.
"""

import heapq
from collections.abc import Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import cho_factor, solve_banded, solve_continuous_lyapunov
from scipy.spatial import cKDTree

from discrecontinual_equations.continuation.fokker_planck import DensityGrid
from discrecontinual_equations.continuation.noise import (
    NoiseMatrix,
    diffusion_matrix,
    field_at,
    fields_at,
    jacobian_at,
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
_NEWTON_STEPS = 3
_HALF = 0.5
_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_UNKNOWN, _ACCEPTED = 0, 2
_CONSTANT_TOLERANCE = 1.0e-9
_DESCENT_STEP = 0.5
_DESCENT_MAX_STEPS = 20000
_ARRIVAL_CELLS = 2.0
_CUBIC_DIRECTIONS = 8
_TAYLOR_STEP = 1.0e-3
_REFERENCE_NODES = 81
_START_SHARE = 0.25
_SECANT_STEPS = 6
_SECANT_TOLERANCE = 1.0e-4
_SECANT_SETTLED = 1.0e-9
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
GRID, CHECKED, REFINED = 0, 1, 2


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

    ``U`` vanishes on the cycle. The points within ``width`` cells of it start
    from the value Dahiya and Cameron's initialisation gives (their Eqs. (9) to
    (12)), carried over to a matrix ``D``. Near the cycle the drift splits as
    ``f = -(1/2) D grad U + l`` with ``grad U . l = 0``, and ``l`` is close to
    ``f`` on the cycle, where ``grad U`` vanishes; taking ``l`` parallel to ``f``
    at the foot ``x*`` of ``x`` on the cycle gives

        ``grad U(z) = -2 D(z)^{-1} (f(z) - c f(x*))``,
        ``c = <f(z), f(x*)>_A / <f(x*), f(x*)>_A``, ``A = D(z)^{-1}``,

    and ``U(x) = integral of grad U . dx`` from the foot, where ``U`` and
    ``grad U`` vanish, by Simpson's rule:
    ``(4 grad U(x_m) + grad U(x)) . (x - x*) / 6`` with ``x_m`` the midpoint.
    The foot is the projection of ``x`` on the chord from the nearest point of
    the cycle, resampled to a quarter of a cell, to its neighbour on ``x``'s side.

    A straight-segment action from the foot, the obvious alternative, is wrong
    by ``O(h)``: across a rotating flow it pays ``|l|`` per unit length. The
    march accepts the band in the order of these values and cannot repair the
    first ones it accepts, so that error stays. On Dahiya and Cameron's SDE (24)
    it left ``U`` next to the cycle off by 5.45e-3 and 3.27e-3 at 161 and 321
    points (order 0.74); from this start, 1.47e-3 and 3.0e-4 (order 2.3), the
    same as from the exact values.
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
        distance, nearest = cKDTree(samples).query(points)
        near = np.flatnonzero(distance <= self._width * cell)
        values = np.array(
            [
                _cycle_start(drift, noise, samples, int(index), point)
                for index, point in zip(nearest[near], points[near], strict=True)
            ],
        )
        return near, values

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
        Simpson-rule start the march uses.
        """
        cell = float(np.min(grid.spacings))
        samples = _resample_closed(self._states, cell / _BAND_SAMPLES_PER_CELL)
        distance, _ = cKDTree(samples).query(np.asarray(target, dtype=float))
        offset = min(self._width * cell, _HALF * float(distance))
        return _BandStart(samples, drift, noise, target, offset)


class _BandStart:
    """The curve at a fixed distance from a cycle, on one side of it.

    Points on it are ``c(s) + d n(s)`` for the cycle ``c`` by arc length ``s``
    and ``n`` its unit normal towards the side of the target, interpolated
    between the samples' normals so that the curve has no corners.
    """

    __slots__ = [
        "_arc",
        "_drift",
        "_noise",
        "_normals",
        "_offset",
        "_samples",
        "_target",
        "_tree",
    ]

    def __init__(
        self,
        samples: np.ndarray,
        drift: Function,
        noise: NoiseMatrix,
        target: np.ndarray,
        offset: float,
    ) -> None:
        self._samples = samples
        self._drift = drift
        self._noise = noise
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
        return _BandStart(
            self._samples,
            self._drift,
            self._noise,
            self._target,
            _HALF * self._offset,
        )

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
        """``U`` at ``state`` by the cycle's Simpson-rule start."""
        point = np.asarray(state, dtype=float)
        _, index = self._tree.query(point)
        return _cycle_start(self._drift, self._noise, self._samples, int(index), point)

    def gradient(self, state: np.ndarray) -> np.ndarray:
        """``grad U`` at ``state`` from the split of the drift at its foot."""
        point = np.asarray(state, dtype=float)
        _, index = self._tree.query(point)
        foot = _cycle_foot(self._samples, int(index), point)
        along = field_at(self._drift, foot)
        return _cycle_slope(self._drift, self._noise, along, point)


def _resample_closed(states: np.ndarray, spacing: float) -> np.ndarray:
    closed = np.vstack([states, states[:1]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(lengths)])
    count = max(int(np.ceil(arc[-1] / spacing)), len(states))
    target = np.linspace(0.0, arc[-1], count, endpoint=False)
    return np.column_stack(
        [np.interp(target, arc, closed[:, axis]) for axis in range(_PLANE)],
    )


def _cycle_start(
    drift: Function,
    noise: NoiseMatrix,
    samples: np.ndarray,
    index: int,
    point: np.ndarray,
) -> float:
    """``U`` at ``point`` from the cycle sample ``index`` nearest it (see above)."""
    foot = _cycle_foot(samples, index, point)
    along = field_at(drift, foot)
    step = point - foot
    middle = _HALF * (foot + point)
    return float(
        (
            4.0 * _cycle_slope(drift, noise, along, middle)
            + _cycle_slope(drift, noise, along, point)
        )
        @ step
        / 6.0,
    )


def _cycle_foot(samples: np.ndarray, index: int, point: np.ndarray) -> np.ndarray:
    """The projection of ``point`` on the chord from sample ``index`` to its side."""
    closest = samples[index]
    neighbours = (samples[index - 1], samples[(index + 1) % len(samples)])
    beside = max(
        neighbours,
        key=lambda other: float((point - closest) @ (other - closest)),
    )
    tangent = (beside - closest) / np.linalg.norm(beside - closest)
    return closest + float((point - closest) @ tangent) * tangent


def _cycle_slope(
    drift: Function,
    noise: NoiseMatrix,
    along: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """``grad U = -2 D^{-1} (f - c f(x*))`` at ``state``, ``along = f(x*)``."""
    metric = np.linalg.inv(diffusion_matrix(noise.columns_at(state)))
    velocity = field_at(drift, state)
    weight = float(along @ metric @ along)
    share = float(velocity @ metric @ along) / weight if weight > 0.0 else 0.0
    return -2.0 * metric @ (velocity - share * along)


def _geometric(step: np.ndarray, velocity: np.ndarray, metric: np.ndarray) -> float:
    return float(
        np.sqrt(step @ metric @ step) * np.sqrt(velocity @ metric @ velocity)
        - step @ metric @ velocity,
    )


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
        """:data:`GRID`, :data:`CHECKED` or :data:`REFINED` per value."""
        return self._status

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

        This costs a few relaxations of a chain of ``nodes`` states per point -
        of the order of a second each - against well under a millisecond per
        point for the march, which is why it is a separate step.
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
        for index in np.asarray(indices, dtype=int).tolist():
            target = points[index]
            if not np.isfinite(values[index]) or attractor.reached(target, grid):
                continue
            reference, own = self.reference(field, target, nodes)
            estimate = abs(float(field.values.ravel()[index]) - reference) + own
            if tolerance is not None and estimate > tolerance and own <= tolerance:
                values[index], errors[index], status[index] = reference, own, REFINED
            else:
                errors[index], status[index] = estimate, CHECKED
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
        far, _, _ = _relax_from(gmam, outer, _guide(outer, trace, nodes))
        near, chain, parameter = _relax_from(gmam, inner, _guide(inner, trace, nodes))
        start = inner.point(parameter)
        coarse = _equal_arc_length(_resample(chain, nodes // 2 + 1))
        coarse[0] = start
        rough = inner.value(start) + gmam.path(coarse).action
        # Each difference is taken whole, not divided by 2^p - 1 as a Richardson
        # estimate would: the chain's error goes at fourth order on the linear
        # systems (1.04e-3, 8.4e-5, 5.9e-6 at 41, 81, 161 nodes, non-normal)
        # but at second order on the anisotropic gradient (-4.9e-5, -1.3e-5 at
        # 81, 161 at (-1.5, -0.8)), and the whole difference bounds both.
        return near, abs(far - near) + abs(rough - near)

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
    """The triangle update with ``U`` a parabola along the base.

    ``U(x_s) = U0 + s dU + c s (s - 1)`` in place of the straight line, so
    ``F(s) = U(x_s) + Q(x_s, x)`` is no longer minimised in closed form. It
    starts from the straight line's root, takes Newton steps on
    ``F'(s) = dU + c (2 s - 1) + Q'(s)`` with ``f`` and ``A`` frozen at the
    current ``s``, and keeps whichever of the two values is lower.
    """
    near, far = midpoints
    rise = other - own
    fraction = np.full(rise.shape[0], _HALF)
    for _ in range(_REFINEMENTS):
        coefficients = near + fraction[:, None] * (far - near)
        fraction = _stationary_fraction(edge, reach, rise, coefficients)
    linear = fraction
    for _ in range(_NEWTON_STEPS):
        coefficients = near + fraction[:, None] * (far - near)
        reach_slope, reach_curvature = _action_derivatives(
            edge,
            reach,
            fraction,
            coefficients,
        )
        total = 2.0 * bend + reach_curvature
        with np.errstate(divide="ignore", invalid="ignore"):
            step = (rise + bend * (2.0 * fraction - 1.0) + reach_slope) / total
        step = np.where(np.isfinite(step) & (total > 0.0), step, 0.0)
        fraction = np.clip(fraction - step, 0.0, 1.0)
    value = _curved_cost(own, rise, bend, edge, reach, fraction, midpoints)
    fallback = _curved_cost(own, rise, bend, edge, reach, linear, midpoints)
    return np.minimum(value, fallback)


def _action_derivatives(edge, reach, fraction, coefficients):
    """``dQ/ds`` and ``d2Q/ds2`` along the base, ``f`` and ``A`` frozen.

    With ``d = reach - s edge``: ``Q' = <edge, f>_A - |f|_A <edge, d>_A / |d|_A``
    and ``Q'' = |f|_A (|edge|_A^2 |d|_A^2 - <edge, d>_A^2) / |d|_A^3``.
    """
    edge_x, edge_y = edge
    step_x = reach[0] - fraction * edge_x
    step_y = reach[1] - fraction * edge_y
    velocity_x, velocity_y = coefficients[:, 0], coefficients[:, 1]
    speed = np.sqrt(_form(velocity_x, velocity_y, coefficients, velocity_x, velocity_y))
    squared = _form(step_x, step_y, coefficients, step_x, step_y)
    length = np.sqrt(squared)
    along = _form(edge_x, edge_y, coefficients, step_x, step_y)
    edge_squared = _form(edge_x, edge_y, coefficients, edge_x, edge_y)
    drift = _form(edge_x, edge_y, coefficients, velocity_x, velocity_y)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = drift - speed * along / length
        curvature = (
            speed * (edge_squared * squared - along * along) / (squared * length)
        )
    return slope, curvature


def _curved_cost(own, rise, bend, edge, reach, fraction, midpoints):  # noqa: PLR0913
    near, far = midpoints
    coefficients = near + fraction[:, None] * (far - near)
    step_x = reach[0] - fraction * edge[0]
    step_y = reach[1] - fraction * edge[1]
    curve = own + fraction * rise + bend * fraction * (fraction - 1.0)
    return curve + _action(step_x, step_y, coefficients)


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

    ``D`` must be constant; a state-dependent ``D`` needs the metric's own
    derivatives in the flow and is refused rather than silently ignored.
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
        is kept for later paths. The step decides only how fast the chain settles, not
        where.
        """
        states = np.asarray(initial, dtype=float)
        factor = self._constant_factor(states)
        inverse = np.linalg.inv(factor)
        step = self._step
        for _ in range(_STEP_HALVINGS):
            # A chain that blows up overflows on its way to nan; that is
            # caught below, not warned about.
            with np.errstate(over="ignore", invalid="ignore"):
                relaxed = self._relax_all(states @ inverse.T, factor, inverse, step)
            if relaxed is not None:
                self._step = step
                break
            step *= _HALF
        else:
            message = (
                f"the minimum action path did not stay finite at any step down to "
                f"{step / _HALF:g}"
            )
            raise ValueError(message)
        chain, converged, taken = relaxed
        result = chain @ factor.T
        return MinimumActionPath(
            result,
            self._action(chain, factor, inverse),
            converged=converged,
            iterations=taken,
        )

    def _relax_all(self, chain, factor, inverse, step):
        """Relax at one step: the chain, whether it settled, the steps taken.

        ``None`` if the chain stopped being finite, or if one step moved a node
        by more than a quarter of the distance between its ends: no relaxation
        moves that far at once, while an unstable step does - on SDE (21) at a
        step of 0.0125 the nodes jumped by twice that distance every step,
        and the chain neither blew up nor settled.
        """
        spacing = 1.0 / (chain.shape[0] - 1)
        reach = _JUMP_SHARE * float(np.linalg.norm(chain[-1] - chain[0]))
        converged = False
        taken = 0
        for taken in range(1, self._iterations + 1):
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
        return chain, converged, taken

    def _constant_factor(self, states: np.ndarray) -> np.ndarray:
        cells = np.array(
            [diffusion_matrix(self._noise.columns_at(state)) for state in states],
        )
        deviation = float(np.max(np.abs(cells - cells[0])))
        if deviation > _CONSTANT_TOLERANCE * max(float(np.max(np.abs(cells[0]))), 1.0):
            message = (
                "the geometric minimum action method here needs a constant "
                "diffusion matrix G G^T"
            )
            raise ValueError(message)
        _check_nodes(cells[:1])
        lower, _ = cho_factor(cells[0], lower=True)
        return np.tril(lower)

    def _drift_at(self, chain, factor, inverse):
        states = chain @ factor.T
        fields = [field_at(self._drift, state) for state in states]
        velocity = np.array(fields) @ inverse.T
        jacobians = np.array(
            [
                inverse @ jacobian_at(self._drift, state, base) @ factor
                for state, base in zip(states, fields, strict=True)
            ],
        )
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
        explicit = chain + step * force
        weight = step * scale * scale / (spacing * spacing)
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

    def _action(self, chain, factor, inverse) -> float:
        middle = _HALF * (chain[1:] + chain[:-1])
        steps = np.diff(chain, axis=0)
        velocity = np.array(
            [inverse @ field_at(self._drift, factor @ state) for state in middle],
        )
        return float(
            np.sum(
                np.linalg.norm(velocity, axis=1) * np.linalg.norm(steps, axis=1)
                - np.sum(velocity * steps, axis=1),
            ),
        )


def _guide(surface, trace: np.ndarray, nodes: int) -> tuple[np.ndarray, float]:
    """A chain from ``surface`` to ``trace[0]`` along the traced path, and its start.

    The traced path runs from the target towards the attractor; it is cut where
    it enters the surface, or at its end if it stalled outside, and its last
    state outside is moved onto the surface.
    """
    outside = 0
    while outside < len(trace) and not surface.inside(trace[outside]):
        outside += 1
    kept = trace[: max(outside, 1)]
    parameter = surface.parameter(kept[-1])
    chain = np.vstack([surface.point(parameter), kept[::-1]])
    return _resample(chain, nodes), parameter


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
    its root, each step relaxing the previous chain from the new start. Returns
    the value, the chain and the parameter of the start with the smallest
    derivative found.
    """
    chain, parameter = guide
    nudge = surface.step * _SECANT_TOLERANCE

    def attempt(where: float, previous: np.ndarray) -> tuple[float, np.ndarray, float]:
        start = surface.point(where)
        initial = previous.copy()
        initial[0] = start
        path = gmam.path(_equal_arc_length(initial))
        tangent = (surface.point(where + nudge) - surface.point(where - nudge)) / (
            2.0 * nudge
        )
        momentum = _momentum(gmam.drift, gmam.noise, path.states)
        slope = float((surface.gradient(start) - momentum) @ tangent)
        return surface.value(start) + path.action, path.states, slope

    best_place = parameter
    best = attempt(parameter, chain)
    behind_place, behind = best_place, best
    place = parameter - np.copysign(surface.step, best[2])
    for _ in range(_SECANT_STEPS):
        here = attempt(place, behind[1])
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
    return best[0], best[1], best_place


def _equal_arc_length(chain: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(lengths)])
    target = np.linspace(0.0, arc[-1], chain.shape[0])
    return np.column_stack(
        [np.interp(target, arc, chain[:, axis]) for axis in range(_PLANE)],
    )
