"""Connecting orbits by a truncated projection boundary-value problem.

A homoclinic orbit leaves an equilibrium along its unstable manifold and returns
along its stable manifold. On a long finite interval it is approximated by a
trajectory whose departure end lies in the unstable eigenspace and whose arrival
end lies in the stable eigenspace - the projection boundary conditions - with a
phase condition to remove the time-translation freedom. The trajectory is
discretised by the trapezoidal rule and the resulting system is solved by
Gauss-Newton. The projection conditions use the left eigenvectors of the Jacobian,
the linear approximation of the manifolds parameterised exactly in ``manifold.py``.
These are Beyn's projection boundary conditions (IMA J. Numer. Anal. 10, 1990), as
HomCont uses them (Champneys, Kuznetsov & Sandstede, IJBC 6, 1996).

The Newton matrix is block-banded - each collocation row couples two neighbouring
nodes - and bordered by the boundary and phase rows, which touch only the two end
nodes and the centre. It is assembled sparse from per-node field Jacobians and
factored with a sparse LU, so an iteration costs time and memory linear in the
number of nodes.

This solves for a single connecting orbit of a fixed field; continuing it in a
parameter (the codim-1 global bifurcation) is the next step and is described in
``ROADMAP.md``. Real (saddle) spectra are supported; spiral connections need the
complex-eigenvector projection and are deferred.
"""

import math
import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import bmat, csc_matrix, csr_matrix
from scipy.sparse import identity as sparse_identity
from scipy.sparse.linalg import SuperLU, lsmr, splu

from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.function.function import Function

_IMAGINARY = 1.0e-9
_DEFAULT_TOLERANCE = 1.0e-9
_DEFAULT_ITERATIONS = 60
_STEP = 1.0e-7
# The finite-difference Jacobian carries noise of order ``_STEP``; singular
# directions below this cutoff are that noise, and solving along them turns a
# converged iterate into a divergent one. ``lstsq`` truncates them instead.
_RCOND = 1.0e-8
# The collocation system is overdetermined by the phase condition, so its
# least-squares minimum can be nonzero and ``_DEFAULT_TOLERANCE`` unreachable.
# Stop once repeated iterations stop improving on the best residual seen.
_STAGNATION_LIMIT = 3
# An iterate whose residual is above ``_DEFAULT_TOLERANCE`` is converged only if
# its Gauss-Newton correction is below this, component by component, relative to
# ``1 + |x|``: the test of AUTO's EPSU and COLNEW's CONTRL, at the low end of the
# range AUTO recommends. Measured: converged least-squares minima stop at 1e-14,
# the diverged solves this refuses at 1.1 to 14.
_CORRECTION_TOLERANCE = 1.0e-7
# How nearly a Newton correction must satisfy the normal equations, relative to
# ``|J|_F (|J|_F |dx| + |r|)``, before its size is trusted; see
# ``_relative_correction``. Neither AUTO nor COLNEW has such a test: they trust
# their own linear solves. This one exists for the fallbacks.
_NORMAL_EQUATIONS = 1.0e-11
# Convergence tolerance for the iterative least squares, the fallback when the
# sparse factorisation fails. Tight enough that the step matches a dense
# factorisation to well below the accuracy of the orbit.
_LSMR_TOLERANCE = 1.0e-13
_LSMR_ITERATIONS = 10000
# lsmr's istop value meaning it hit maxiter without converging.
_LSMR_ITERATION_LIMIT = 7
_HALF = 0.5
# Column ordering for the sparse LU of the Newton system; see ``_factor``.
_ORDERING = "COLAMD"
# Inverse-iteration steps for the smallest singular value of the Newton matrix,
# which sets the augmented system's scale. The scale need only be right to within
# a small factor: one step was measured within 4x and two within 30% of the true
# value, and with either the step error matched QR's from cond(J) = 1e2 to 1e8.
_SCALE_STEPS = 2
# Levenberg-Marquardt parameter of every Newton step, as ``(_DAMPING sigma_max)^2``.
# A connection that is one of a family has a Newton matrix singular along the
# family (measured sigma_min / sigma_max 1e-18 to 7e-17 on the sheared Kirk-Silber
# edges leaving xi2); the undamped step there divides rounding by rounding. The
# damped step filters each singular direction by ``sigma^2 / (sigma^2 + lambda)``:
# below the floor it is dropped, which makes the step the minimum-norm one (DEQ-175);
# above it the step is changed by at most ``(_DAMPING cond(J))^2``, 1e-6 at the
# ``cond(J) = 1e8`` that the augmented solve is pinned to match QR at.
_DAMPING = 1.0e-11
# Power-iteration steps for sigma_max, which only scales the damping floor.
_POWER_STEPS = 8
_ROOT_HALF = 0.5**0.5
# How far the truncation estimate extends the interval, as a fraction of the
# half-length. Truncation error decays like ``exp(-2 lambda T)``, so a fraction
# rather than a fixed time keeps the reduction comparable across problems whose
# natural time scales differ by orders of magnitude. A quarter buys roughly a
# decade of reduction on the saddle measured in the docstrings below.
_TRUNCATION_EXTENSION = 0.25
# Multiple of an interval's truncation error its residual may reach; the exact
# orbit, sampled, reaches 1.24 of it (see CollocationResidual).
_RESIDUAL_SAFETY = 2.0
# Multiple of its own error bound the deferred-corrected inconsistency may reach.
# Genuine connections measured up to 0.38 of the unscaled bound on the damped
# well, and up to 1.5 of it on Nagumo fronts at half-lengths 2 to 3, where the
# leading-order boundary miss runs low (see _manifold_miss); the factor is
# margin for the leading-order estimates the bound is made of.
_INCONSISTENCY_SAFETY = 2.0
# Differences of order 7 bound the corrected truncation estimate, so a path needs
# at least eight nodes to be checked for consistency.
_CORRECTED_ORDER = 7
_AUTODIFF = AutomaticDifferentiation()


class MeshSpec:
    """The finite-interval mesh and phase condition for a connecting orbit."""

    __slots__ = ["half_length", "intervals", "phase_index", "phase_value"]

    def __init__(
        self,
        half_length: float,
        intervals: int,
        phase_index: int = 0,
        phase_value: float = 0.0,
    ) -> None:
        self.half_length = half_length
        self.intervals = intervals
        self.phase_index = phase_index
        self.phase_value = phase_value


class OrbitSolution:
    """A computed connecting orbit: mesh times and the state at each node.

    ``residual`` is the path's :class:`CollocationResidual`, attached by the solver
    that returned it; a path assembled by hand has none until measured.
    """

    __slots__ = ["_residual", "_states", "_times"]

    def __init__(
        self,
        times: np.ndarray,
        states: np.ndarray,
        residual: "CollocationResidual | None" = None,
    ) -> None:
        self._times = times
        self._states = states
        self._residual = residual

    @property
    def residual(self) -> "CollocationResidual | None":
        """The collocation residual measured when the path was solved, if it was."""
        return self._residual

    @property
    def times(self) -> np.ndarray:
        """Mesh times from ``-T`` to ``+T``."""
        return self._times

    @property
    def states(self) -> np.ndarray:
        """State at each mesh node, shape ``(nodes, dimension)``."""
        return self._states

    def component(self, index: int) -> np.ndarray:
        """One state component along the orbit."""
        return self._states[:, index]


class OrbitResolution:
    """Measured error estimates for a connecting orbit, one per resolution parameter.

    A truncated connecting orbit has two independent resolutions and either can be
    the inadequate one: the mesh spacing, which sets the collocation error, and the
    half-length, which sets how much of the orbit's approach to the equilibrium is
    cut off. ``discretisation`` and ``truncation`` are measured separately because a
    single number cannot express both, and reporting only one certifies orbits that
    are badly wrong in the other. Measured on the exact homoclinic
    ``x = 1.5 sech^2(t/2)``:

    * at spacing ``h = 0.25``, extending the half-length from 15 to 20 shifts the
      orbit by 5.2e-13 while its true error is 7.7e-3 - truncation is genuinely
      finished there, and alone it would claim ten orders of magnitude more accuracy
      than the orbit has;
    * at half-length 5, doubling the intervals past 160 shifts the orbit by about
      5e-6 while its true error is stuck at 2.8e-4 - the boundary is the limit, and
      the discretisation estimate alone is 50x optimistic.

    ``estimate`` is therefore the larger of the two, and ``limited_by`` names which
    one it was, since that is what says where to spend the next solve.

    A component whose refinement solve did not converge is ``inf``, not a shift:
    a refused solve has no orbit to compare. On sheared Kirk-Silber both
    refinements once returned their seeds unchanged and the estimate read 0.0.
    So ``estimate`` is 0.0 only when both refinements converged and moved
    nothing, and ``valid`` says whether both converged.

    References:
        Bader and Ascher 1987 (docs/references.md, ``Bader1987``): COLNEW
        reports no solution, not its last iterate, when the Newton iteration on
        a mesh fails; a refinement here does the same.
    """

    __slots__ = ["discretisation", "truncation"]

    def __init__(self, discretisation: float, truncation: float) -> None:
        self.discretisation = discretisation
        self.truncation = truncation

    @property
    def valid(self) -> bool:
        """Whether both refinement solves converged, so both components are shifts."""
        return math.isfinite(self.discretisation) and math.isfinite(self.truncation)

    @property
    def estimate(self) -> float:
        """The larger of the two components: neither alone bounds the error."""
        return max(self.discretisation, self.truncation)

    @property
    def limited_by(self) -> str:
        """Which resolution parameter to improve first, ``"spacing"`` or ``"length"``.

        ``"spacing"`` means halve the mesh spacing, which divides the discretisation
        error by four. ``"length"`` means extend the half-length; truncation error
        falls like ``exp(-2 lambda T)``, so ``ln(4) / (2 lambda)`` more time buys the
        same factor of four, where ``lambda`` is the leading eigenvalue magnitude.
        """
        if self.truncation > self.discretisation:
            return "length"
        return "spacing"


class CollocationResidual:
    """How far a path is from satisfying the trapezoidal rule, and how far it may be.

    ``largest`` is the largest component of ``x_{i+1} - x_i - (h/2)(f_i + f_{i+1})``
    over every interval. A converged solve of a consistent discrete system drives
    it to the Newton tolerance; one that has no solution - because the orbit it
    asks for does not exist - stops at a least-squares minimum that is not.

    Each interval is held to its own allowance: ``_RESIDUAL_SAFETY`` times the
    trapezoidal rule's local truncation error ``h^3 |x'''| / 12``, read off the
    largest third difference of the path among the (up to three) stencils that
    cover the interval, plus the Newton tolerance as a floor - a residual the
    solver itself calls converged cannot refute a path. ``worst`` is the largest
    ratio of an interval's residual to its allowance, and the path is a
    trajectory when it is at most 1. ``tolerance`` is the largest allowance, for
    scale.

    Per interval, because the truncation error varies by orders of magnitude
    along a connection: near a saddle ``x'''`` is 50 to 250 times smaller than
    mid-transition, and a test against the largest third difference anywhere
    passes a defect there as large as the whole orbit's error estimate. With a
    safety factor, because the exact orbit sampled on the mesh leaves a residual
    1.005 (``h = 0.2``) to 1.24 (``h = 1.6``) times its largest third difference
    over 12, measured on the Guckenheimer-Holmes connection; per interval it
    stays within 0.62 of the allowance. As in COLSYS and bvp4c, the error is
    controlled where it is made.

    Measured on that connection at ``h = 0.2``: genuine orbits leave 1e-15 to
    3e-10, a path with no orbit behind it 1.9e-3, against a largest allowance of
    6e-5. A leak ``eps x1^2 x2^2`` out of the plane is refused from
    ``eps = 1e-3`` (3e-4 at ``h = 0.1``): the threshold falls as ``h^2``.

    The resolution estimate cannot stand in for this. It measures how much a path
    moves when the mesh changes, and a non-trajectory converged on one mesh can
    sit still on the next: the path above had an estimate of 1.9e-4.

    **Broken connections.** The per-interval test cannot see a connection that
    is broken by less than the trapezoidal rule's own error. A homoclinic loop
    broken by damping ``delta y`` leaves a least-squares minimum whose residual
    is about ``0.4 delta sqrt(h)`` spread along the whole orbit, well inside
    allowances of order ``h^3``. At ``h = 0.25`` it passed up to ``delta = 9e-3``
    (DEQ-155). A second test closes this gap. It asks whether the *discrete
    system* is consistent once its own truncation error is accounted for.

    The idea is deferred correction. The exact orbit leaves the trapezoidal rule
    a residual ``tau = -h^3 x'''/12 - h^5 x^(5)/480 - ...`` on each interval, and
    ``-D3/12 + D5/120`` reads it off the path's differences to ``O(h^7)``. The
    corrected system ``G(x) = F(x) - tau_hat(x)``, with ``F`` the solver's
    Newton residual, is then a sixth-order discretisation. One Gauss-Newton step
    on ``G`` moves the path to ``x + z``. ``inconsistency`` is the part of
    ``G(x + z)`` that no step can remove, ``|(I - J J^+) G(x + z)|``. For a
    genuine connection this is the corrected scheme's own error. For a broken
    one it is the miss.

    ``inconsistency_tolerance`` bounds the genuine part, with two terms:

    * the leftover of the difference estimate, ``|D7 x / 840|`` over the
      collocation rows, the next term of the expansion;
    * the truncation of the interval. The exact orbit misses the linear
      projection conditions by the quadratic part of the manifold. On the
      departure side that miss is ``l_k N(e) / (2 mu - lambda_k)``, where
      ``N(e) = f(p + e) - Df(p) e``, ``mu`` is the slowest unstable rate and
      ``lambda_k`` a projected eigenvalue. The arrival side is the mirror image.

    The sum is multiplied by ``_INCONSISTENCY_SAFETY``, and the Newton tolerance
    is added as a floor. A path is a trajectory only if both tests pass.

    Measured on a dissipative loop that is genuine, ``y' = x - x^2 + k y H``
    with ``H`` the saddle's energy: the inconsistency fell from 7e-5 to 1e-10
    between ``h = 0.5`` and ``h = 0.0625``, as ``h^6.5``. On solved and
    exact-sampled paths it stayed within 0.16 of its bound for half-lengths of
    6 or more, and within 0.75 down to half-length 3, where the interval's
    truncation is most of it. The broken loop's
    inconsistency is ``0.415 delta`` at ``h = 0.25``, ``0.295 delta`` at
    ``0.125`` and ``0.209 delta`` at ``0.0625``. On a quarter-decade grid, at
    half-length 10 (12 for ``h = 0.5``), it is refused from ``delta = 1e-3``
    at ``h = 0.5``, ``3.2e-5`` at ``0.25``, ``5.6e-7`` at ``0.125`` and ``1.8e-8`` at
    ``0.0625``. That is a fall of about ``h^5.5``, until the Newton-tolerance
    floor stops it near ``1e-8``. The per-interval test alone refused it only
    from ``9e-3`` at ``h = 0.25``. Below those values a broken loop and the
    corrected scheme's own error cannot be told apart on that mesh: refine it.

    A path with fewer than eight nodes is not checked for consistency, because
    ``D7`` needs eight. Its ``inconsistency`` is 0.0. The homoclinic curve
    solves a square system, which has no inconsistent direction, and it leaves
    the test out the same way.

    References:
        V. Pereyra, "Iterated deferred corrections for nonlinear operator
        equations", Numer. Math. 10, 316-323, 1967, DOI 10.1007/BF02162030:
        deferred correction, which subtracts a truncation-error estimate read
        from the computed solution. The text could not be fetched (paywalled).
        The expansion and the ``D7/840`` leftover used here were derived and
        checked in exact arithmetic. The leading term matches Atkinson (1989),
        eq. (5.1.7), as quoted in the Wikipedia article "Trapezoidal rule",
        section "Error analysis".
        W.-J. Beyn, "The numerical computation of connecting orbits in
        dynamical systems", IMA J. Numer. Anal. 10, 379-405, 1990, DOI
        10.1093/imanum/10.3.379. These are the projection boundary conditions,
        whose truncation error decays exponentially (abstract). The
        quadratic-miss estimate above is this module's own leading-order
        analysis.
        See docs/references.md, ``Pereyra1967`` and ``Beyn1990``.
    """

    __slots__ = [
        "inconsistency",
        "inconsistency_tolerance",
        "largest",
        "tolerance",
        "worst",
    ]

    def __init__(
        self,
        largest: float,
        tolerance: float,
        worst: float,
        inconsistency: float = 0.0,
        inconsistency_tolerance: float = 0.0,
    ) -> None:
        self.largest = largest
        self.tolerance = tolerance
        self.worst = worst
        self.inconsistency = inconsistency
        self.inconsistency_tolerance = inconsistency_tolerance

    @property
    def consistent(self) -> bool:
        """Whether the corrected discrete system is consistent to within its error.

        A bound that is not finite proves nothing, so it is not consistent.
        """
        return (
            math.isfinite(self.inconsistency_tolerance)
            and self.inconsistency <= self.inconsistency_tolerance
        )

    @property
    def is_trajectory(self) -> bool:
        """Whether every interval's residual is within its allowance, and the
        discrete system is consistent: no connection is broken under the mesh."""
        return self.worst <= 1.0 and self.consistent


class NotConvergedError(RuntimeError):
    """A connecting-orbit solve whose best iterate is not converged.

    ``solution`` is that iterate, with its collocation residual attached;
    ``newton_residual`` is the norm of the full Newton residual there and
    ``correction`` the largest component of its Gauss-Newton correction relative
    to ``1 + |x|`` - ``inf`` where none could be computed. They are for diagnosis:
    the iterate is not an orbit. AUTO ends such a step with ``MX`` (no
    convergence) and COLNEW with ``IFLAG = -2``; neither returns the iterate as
    a solution, and neither does :meth:`ConnectingOrbit.solve`.

    References:
        Doedel et al., *AUTO-07P: Continuation and Bifurcation Software for
        Ordinary Differential Equations*, Concordia University, 2007 (rev. 2012),
        "Tolerances": ``EPSU``, the relative convergence criterion of the
        Newton/Chord iteration, and ``ITNW``.
        G. Bader and U. Ascher, "A new basis implementation for a mixed order
        boundary value ODE solver", SIAM J. Sci. Stat. Comput. 8(4), 483-500,
        1987, DOI 10.1137/0908047; COLNEW subroutine ``CONTRL``: convergence is
        ``|dz_i| <= tol (|z_i| + 1)``, failure returns ``IFLAG = -2``.
    """

    def __init__(
        self,
        message: str,
        solution: OrbitSolution,
        *,
        newton_residual: float,
        correction: float,
    ) -> None:
        super().__init__(message)
        self.solution = solution
        self.newton_residual = newton_residual
        self.correction = correction


class ConnectingOrbit(ABC):
    """Solve for an orbit connecting an equilibrium to itself or another.

    Subclasses supply the projection boundary conditions via :meth:`_boundary`; the
    trapezoidal collocation, phase condition, and Gauss-Newton solve are shared.
    """

    __slots__ = ["_function", "_mesh"]
    _function: Function
    _mesh: MeshSpec

    def solve(self, seed: np.ndarray) -> OrbitSolution:
        """Refine ``seed`` to the connecting orbit by Gauss-Newton iteration.

        The iterate with the smallest residual is the candidate, not the last one
        computed: past the least-squares minimum the finite-difference Jacobian
        is dominated by noise and a further step degrades an already-converged
        orbit. Iteration stops early once ``_STAGNATION_LIMIT`` successive steps
        fail to improve on the best residual seen, or once a residual is not
        finite.

        The candidate is returned only if it is converged: its Newton residual is
        below ``_DEFAULT_TOLERANCE``, or the Gauss-Newton correction computed
        there is below ``_CORRECTION_TOLERANCE`` relative to ``1 + |x|`` in every
        component - a stationary point of the least squares, which a system with
        no exact solution stops at. The correction certifies only a step
        computed at full numerical rank (DEQ-180); see :meth:`_newton_step`,
        whose minimum-norm step is what lets a connection that is one of a
        family converge at all (DEQ-175). This is the test of AUTO (EPSU) and COLNEW;
        neither returns an iterate that fails it, and neither does this. Where
        Gauss-Newton diverges the best iterate is often the seed itself, and
        returning it unannounced was how a non-trajectory with an estimate of
        0.0 came back on sheared Kirk-Silber (DEQ-120). Otherwise this raises
        :class:`NotConvergedError`, which carries the candidate.

        The returned orbit carries its :class:`CollocationResidual`. Converged is
        not the same as a trajectory: the least-squares minimum of a connection
        that does not exist is converged, and its residual is what says so.

        Args:
            seed: Initial path, shape ``(intervals + 1, dimension)``.

        Returns:
            The converged path, with ``residual`` attached.

        Raises:
            NotConvergedError: The best iterate passes neither test.

        References:
            Doedel et al., *AUTO-07P: Continuation and Bifurcation Software for
            Ordinary Differential Equations*, Concordia University, 2007 (rev. 2012),
            "Tolerances": ``EPSU``, the relative convergence criterion of the
            Newton/Chord iteration, and ``ITNW``.
            G. Bader and U. Ascher, "A new basis implementation for a mixed order
            boundary value ODE solver", SIAM J. Sci. Stat. Comput. 8(4), 483-500,
            1987, DOI 10.1137/0908047; COLNEW subroutine ``CONTRL``: convergence is
            ``|dz_i| <= tol (|z_i| + 1)``, failure returns ``IFLAG = -2``.
            J. J. Moré, "The Levenberg-Marquardt algorithm: implementation and
            theory", Lecture Notes in Mathematics 630, Springer, 1978, 105-116,
            DOI 10.1007/BFb0067700, eq. (2.4): the step (see :meth:`_newton_step`).
            See docs/references.md, ``Doedel2007``, ``Bader1987``, ``More1978``.
        """
        nodes = self._mesh.intervals + 1
        times = np.linspace(-self._mesh.half_length, self._mesh.half_length, nodes)
        unknowns = seed.astype(float).flatten()
        shape = seed.shape
        best = unknowns.copy()
        best_residual = math.inf
        best_correction = math.inf
        stagnant = 0
        analytic = True
        for _ in range(_DEFAULT_ITERATIONS):
            norm, residual = self._residual_norm(unknowns, times, shape)
            if not math.isfinite(norm):
                break
            improved = norm < best_residual
            if improved:
                best, best_residual = unknowns.copy(), norm
                stagnant = 0
            else:
                stagnant += 1
                if stagnant >= _STAGNATION_LIMIT:
                    break
            if norm < _DEFAULT_TOLERANCE:
                break
            jacobian = self._jacobian(unknowns, times, shape, residual, analytic)
            if jacobian is None:
                analytic = False
                jacobian = self._numerical_jacobian(
                    unknowns,
                    times,
                    shape,
                    residual.size,
                )
            step, resolved = self._newton_step(jacobian, residual)
            if improved:
                best_correction = _relative_correction(
                    jacobian,
                    residual,
                    step,
                    unknowns,
                    resolved=resolved,
                )
            unknowns = unknowns + step
        states = best.reshape(shape)
        measured = self.collocation_residual(OrbitSolution(times, states))
        solution = OrbitSolution(times, states, measured)
        if best_residual < _DEFAULT_TOLERANCE:
            return solution
        if best_correction <= _CORRECTION_TOLERANCE:
            return solution
        message = (
            f"connecting-orbit solve did not converge: at its best iterate the "
            f"Newton residual is {best_residual:.3g} and the Gauss-Newton "
            f"correction {best_correction:.3g} of the state's scale (converged "
            f"means below {_DEFAULT_TOLERANCE:g} or {_CORRECTION_TOLERANCE:g}); "
            f"its collocation residual reaches {measured.worst:.3g} "
            f"times what the mesh allows. A seed too far from any connection, or "
            f"a connection that does not exist, is the usual cause. A correction "
            f"of inf means the step there could not be trusted: computed without "
            f"full rank, or failing the normal equations"
        )
        raise NotConvergedError(
            message,
            solution,
            newton_residual=best_residual,
            correction=best_correction,
        )

    def _residual_norm(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        shape: tuple[int, int],
    ) -> tuple[float, np.ndarray]:
        """The Newton residual and its norm, ``inf`` once an iterate overflows.

        A diverging iterate overflows the field - Python raises on ``x**3`` past
        the float range - or the norm; either ends the iteration, not the program.
        """
        try:
            residual = self._residual(unknowns, times, shape)
        except OverflowError:
            return math.inf, np.empty(0)
        with np.errstate(over="ignore", invalid="ignore"):
            return float(np.linalg.norm(residual)), residual

    def estimate_orbit_error(self, solution: OrbitSolution) -> OrbitResolution:
        """Estimate the orbit's error in both resolution parameters by re-solving.

        A converged solve proves the *discrete* system was satisfied on *this* mesh
        over *this* interval. It says nothing about whether either is adequate,
        and neither convergence nor the residual's size carries that information.

        Two solves, compared against the orbit in hand:

        * **discretisation** - re-solve on a mesh with twice the intervals over the
          same interval, seeded by interpolation, and take the largest state shift at
          the shared nodes. Second-order convergence makes this ``E - E/4``, so it
          runs about 0.75x the *coarse* error and 3x the *refined* one. Measured on
          the exact homoclinic it was 0.75x at every level from 80 to 640 intervals,
          so unlike the cycle estimator it is a usable bound rather than a screen.
        * **truncation** - re-solve on an interval longer by
          ``_TRUNCATION_EXTENSION``, at the same spacing so the original nodes are a
          subset, seeded by holding the end states, and take the largest shift on the
          overlap. Holding the end states needs no knowledge of the equilibria, so it
          serves homoclinic and heteroclinic orbits alike.

        ``estimate`` is the larger; see :class:`OrbitResolution` for why reporting
        either alone certifies orbits that are badly wrong.

        A refined solve that converges somewhere useless shows up as a *large*
        shift, which is the correct verdict. One that does not converge has no
        orbit to compare, and its component is ``inf``: the estimate is computed
        only from converged refinements (see :attr:`OrbitResolution.valid`).

        Costs two solves, one at double the intervals and one slightly longer.

        ``intervals`` must be even. The phase condition pins the centre node, and on
        an odd mesh the doubled mesh's centre node sits half a step away, so the two
        orbits come out translated relative to each other and the shift measures that
        translation instead of the error.
        """
        mesh = self._mesh
        nodes = mesh.intervals + 1
        states = solution.states
        if states.shape[0] != nodes:
            message = (
                f"solution has {states.shape[0]} nodes, but this solver has "
                f"{nodes}; pass the solution this solver returned"
            )
            raise ValueError(message)
        if mesh.intervals % 2 != 0:
            message = (
                f"intervals must be even to compare meshes, got {mesh.intervals}; "
                f"the phase condition pins the centre node and an odd mesh has no "
                f"matching centre when doubled"
            )
            raise ValueError(message)
        return OrbitResolution(
            self._discretisation_shift(solution),
            self._truncation_shift(solution),
        )

    def collocation_residual(self, solution: OrbitSolution) -> CollocationResidual:
        """The path's largest trapezoidal residual, against its truncation error.

        Only the collocation rows are measured: the boundary and phase rows say
        where the path starts, ends and is pinned, not whether it is a trajectory.
        Any path with the solver's dimension can be measured, not only one this
        solver returned. See :class:`CollocationResidual`.

        The consistency test does use the boundary and phase rows. A broken
        connection is a trajectory that cannot also meet them, and the
        inconsistency measures by how much. That costs the Jacobian at the path
        and two least-squares solves.
        """
        states, times = solution.states, solution.times
        rows = [self._collocation(states, times, i) for i in range(len(times) - 1)]
        inconsistency, bound = self._inconsistency(states, times)
        return _measure_collocation(states, np.array(rows), inconsistency, bound)

    def _inconsistency(
        self,
        states: np.ndarray,
        times: np.ndarray,
    ) -> tuple[float, float]:
        """The deferred-corrected system's irreducible residual, and its bound.

        See :class:`CollocationResidual`. ``(0.0, 0.0)`` for a path too short to
        difference; ``inf`` when the measurement itself overflows or produces
        a non-finite value, which no trajectory does.
        """
        nodes = states.shape[0]
        if nodes <= _CORRECTED_ORDER:
            return 0.0, 0.0
        shape = states.shape
        unknowns = states.astype(float).flatten()
        leftover = np.array(
            [
                _difference(states, i, _CORRECTED_ORDER) / 840.0
                for i in range(nodes - 1)
            ],
        )
        first = leftover.size
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                missed = self._boundary_truncation(states)
                corrected = self._corrected_residual(unknowns, times, shape)
                if not np.all(np.isfinite(corrected)):
                    return math.inf, _DEFAULT_TOLERANCE
                jacobian = self._jacobian(
                    unknowns,
                    times,
                    shape,
                    corrected,
                    analytic=True,
                )
                if jacobian is None:
                    jacobian = self._numerical_jacobian(
                        unknowns,
                        times,
                        shape,
                        corrected.size,
                    )
                # A boundary row's miss reaches the inconsistency only through
                # that row's share of it, |(I - J J^+) e_row|, which is small at
                # an end the orbit has nearly reached. By the triangle
                # inequality, sum |miss_row| |P e_row| bounds the miss's part.
                reached = 0.0
                for offset, size in enumerate(missed):
                    unit = np.zeros(corrected.size)
                    unit[first + offset] = 1.0
                    share = unit + jacobian @ self._least_squares(jacobian, unit)
                    reached += size * float(np.linalg.norm(share))
                bound = (
                    _INCONSISTENCY_SAFETY * (float(np.linalg.norm(leftover)) + reached)
                    + _DEFAULT_TOLERANCE
                )
                moved = unknowns + self._least_squares(jacobian, corrected)
                remaining = self._corrected_residual(moved, times, shape)
                irreducible = remaining + jacobian @ self._least_squares(
                    jacobian,
                    remaining,
                )
                inconsistency = float(np.linalg.norm(irreducible))
        except (OverflowError, ValueError):
            return math.inf, _DEFAULT_TOLERANCE
        return inconsistency, bound

    def _corrected_residual(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """The Newton residual less the path's own trapezoidal truncation error.

        ``-D3/12 + D5/120`` on each collocation row: the trapezoidal rule's local
        error ``-h^3 x'''/12 - h^5 x^(5)/480`` to ``O(h^7)``; see
        :class:`CollocationResidual`. The boundary and phase rows are unchanged.
        """
        states = unknowns.reshape(shape)
        residual = self._residual(unknowns, times, shape)
        estimate = np.array(
            [
                -_difference(states, i, 3) / 12.0 + _difference(states, i, 5) / 120.0
                for i in range(shape[0] - 1)
            ],
        )
        residual[: estimate.size] -= estimate.ravel()
        return residual

    def _shift(self, moved: np.ndarray, original: np.ndarray) -> float:
        """Largest component of the change between two paths at shared nodes.

        A solver posed in other coordinates overrides this to measure the change
        in the coordinates its caller reads the orbit in.
        """
        return float(np.max(np.abs(moved - original)))

    def _solve_on(self, mesh: MeshSpec, seed: np.ndarray) -> OrbitSolution:
        """Solve the same problem on a different mesh, leaving this solver unchanged.

        The boundary conditions depend on the two end states and not on how many
        nodes lie between them, so swapping the mesh is enough to re-pose the problem
        at another resolution - no subclass needs to know how to rebuild itself.
        """
        saved = self._mesh
        self._mesh = mesh
        try:
            return self.solve(seed)
        finally:
            self._mesh = saved

    def _discretisation_shift(self, solution: OrbitSolution) -> float:
        """Largest state shift from doubling the intervals over the same interval."""
        mesh = self._mesh
        states = solution.states
        coarse_times = solution.times
        fine_times = np.linspace(
            -mesh.half_length,
            mesh.half_length,
            2 * mesh.intervals + 1,
        )
        seed = np.column_stack(
            [
                np.interp(fine_times, coarse_times, states[:, index])
                for index in range(states.shape[1])
            ],
        )
        try:
            refined = self._solve_on(
                MeshSpec(
                    mesh.half_length,
                    2 * mesh.intervals,
                    mesh.phase_index,
                    mesh.phase_value,
                ),
                seed,
            )
        except NotConvergedError:
            return math.inf
        return self._shift(refined.states[::2], states)

    def _truncation_shift(self, solution: OrbitSolution) -> float:
        """Largest state shift on the overlap from extending the interval.

        The extension is a whole number of mesh steps, so the original nodes are a
        subset of the longer mesh and the overlap needs no interpolation.
        """
        mesh = self._mesh
        states = solution.states
        spacing = 2.0 * mesh.half_length / mesh.intervals
        extra = max(1, round(_TRUNCATION_EXTENSION * mesh.half_length / spacing))
        seed = np.vstack(
            [
                np.repeat(states[:1], extra, axis=0),
                states,
                np.repeat(states[-1:], extra, axis=0),
            ],
        )
        try:
            longer = self._solve_on(
                MeshSpec(
                    mesh.half_length + extra * spacing,
                    mesh.intervals + 2 * extra,
                    mesh.phase_index,
                    mesh.phase_value,
                ),
                seed,
            )
        except NotConvergedError:
            return math.inf
        overlap = longer.states[extra : extra + states.shape[0]]
        return self._shift(overlap, states)

    def _residual(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        shape: tuple[int, int],
    ) -> np.ndarray:
        states = unknowns.reshape(shape)
        nodes = shape[0]
        blocks = [self._collocation(states, times, i) for i in range(nodes - 1)]
        blocks.append(self._boundary(states))
        blocks.append(np.array([self._phase(states)]))
        return np.concatenate(blocks)

    def _collocation(
        self,
        states: np.ndarray,
        times: np.ndarray,
        i: int,
    ) -> np.ndarray:
        step = times[i + 1] - times[i]
        here = np.array(self._function.eval(point=list(states[i]), time=None))
        ahead = np.array(self._function.eval(point=list(states[i + 1]), time=None))
        return states[i + 1] - states[i] - 0.5 * step * (here + ahead)

    def _phase(self, states: np.ndarray) -> float:
        centre = states.shape[0] // 2
        return states[centre, self._mesh.phase_index] - self._mesh.phase_value

    def _least_squares(self, jacobian: csr_matrix, residual: np.ndarray) -> np.ndarray:
        """The Newton step of :meth:`_newton_step`, without its rank report."""
        return self._newton_step(jacobian, residual)[0]

    def _newton_step(
        self,
        jacobian: csr_matrix,
        residual: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """The minimum-norm least-squares Newton step, and whether ``J`` resolved it.

        The step ``x`` minimises ``|J x + r|^2 + lambda |x|^2``, the
        Levenberg-Marquardt subproblem, with ``lambda = (_DAMPING sigma_max)^2``
        fixed far below every singular value a well-posed solve has. In the
        singular basis it is ``-sum_i sigma_i (u_i . r) / (sigma_i^2 + lambda) v_i``:
        directions with ``sigma_i^2 >> lambda`` get the Gauss-Newton step
        unchanged, and directions with ``sigma_i^2 << lambda`` are dropped. As
        ``lambda -> 0`` this is the minimum-norm least-squares step ``-J^+ r``
        (Moré 1978, eq. (2.4)). Where the connection is one of a family, ``J`` is
        singular along the family and the Gauss-Newton step there is rounding
        divided by rounding - 10 to 95 on the sheared Kirk-Silber edges leaving
        xi2, and different on every BLAS kernel. The minimum-norm step is not:
        it moves only across the family, so the iteration is Gauss-Newton with
        the pseudo-inverse and stops on whichever member of the family it
        reaches first. Measured there: the residual falls quadratically, 0.59
        to 1e-15 in seven steps, on every OpenBLAS kernel tried, to within
        1e-12 of Gauss-Newton with a dense SVD pseudo-inverse. ``lambda`` is a
        fixed floor, not Moré's trust-region parameter: the iteration here keeps
        no radius, and a nearly singular but isolated problem still takes its
        full Gauss-Newton step.

        It is the lower half of the solution of the augmented system

            [ alpha I   J       ] [ s ]   [ -r ]
            [ J^T      -mu I    ] [ x ] = [  0 ],

        with ``alpha mu = lambda``: the first row defines ``s = (-r - J x) /
        alpha`` and the second is then the damped normal equations
        ``(J^T J + lambda I) x = -J^T r``, without forming ``J^T J``. The scale
        ``alpha = sigma_eff / sqrt(2)``, with ``sigma_eff^2 = sigma_min^2 +
        lambda`` the damped system's smallest singular value squared, keeps the
        condition number near ``sigma_max / sigma_eff``: unscaled, with
        ``sigma_min < 1`` as it is here (about ``1/N``), it is conditioned like
        ``J^T J``. Measured on synthetic problems, unscaled forward errors were
        4e-7 at ``cond(J) = 1e6`` and 1e-3 at ``1e8``, against QR's 2e-9 and 3e-5;
        scaled, they match QR's (Bjorck, BIT 7, 1967). ``sigma_min`` comes from
        ``_SCALE_STEPS`` steps of inverse iteration through an unscaled
        factorisation, so a step costs two sparse factorisations, both in fill
        linear in the nodes because only the boundary and phase rows reach across
        the block-banded collocation.

        The second value says whether ``J`` has full numerical rank,
        ``sigma_min > _RCOND sigma_max``. A step computed where it does not is
        not the Gauss-Newton step along the directions it dropped, so its size
        certifies nothing (see :func:`_relative_correction`).

        ``lsmr``, damped by the same floor, is the fallback for a factorisation
        that is non-finite, with the dense least-squares solve, which reports
        its rank, behind it; both announce themselves with a ``RuntimeWarning``.
        An exactly singular factorisation in the ``sigma_min`` estimate also
        warns, and is a rank deficiency the damped solve then handles; if
        ``lsmr`` then takes the step, its estimate of ``cond(J)`` decides the
        rank.

        References:
            J. J. Moré, "The Levenberg-Marquardt algorithm: implementation and
            theory", in G. A. Watson (ed.), *Numerical Analysis*, Lecture Notes
            in Mathematics 630, Springer, 1978, 105-116, DOI 10.1007/BFb0067700;
            Sec. 2, eq. (2.4), ``p(lambda) = -(J^T J + lambda D^T D)^{-1} J^T f``,
            and its limit for rank-deficient ``J``, ``D p(0) = lim D p(lambda) =
            -(J D^{-1})^+ f`` as ``lambda -> 0+``, here with ``D = I``; Sec. 3,
            eq. (3.1)-(3.2), the damped step as a least-squares problem rather
            than normal equations, which "are particularly unreliable when
            lambda = 0 and J is nearly rank deficient".
            See docs/references.md, ``More1978``.
        """
        rows, columns = jacobian.shape
        right = np.concatenate([-residual, np.zeros(columns)])
        largest = _largest_singular_value(jacobian)
        floor = _DAMPING * largest
        measured = True
        try:
            smallest = _smallest_singular_value(jacobian)
        except RuntimeError:
            measured = False
            warnings.warn(
                "connecting-orbit Newton step: exactly singular factorisation; "
                "taking the minimum-norm step",
                RuntimeWarning,
                stacklevel=2,
            )
            smallest = 0.0
        resolved = smallest > _RCOND * largest
        reason = "non-finite solution (near-singular factorisation)"
        try:
            scale = _ROOT_HALF * math.hypot(smallest, floor)
            augmented = _augmented(jacobian, scale, floor * floor / scale)
            solution = _factor(augmented).solve(right)
        except RuntimeError:
            reason = "exactly singular factorisation"
            solution = np.full(rows + columns, np.nan)
        step = solution[rows:]
        if np.all(np.isfinite(step)):
            return step, resolved
        warnings.warn(
            f"connecting-orbit Newton step: {reason}; falling back to lsmr",
            RuntimeWarning,
            stacklevel=2,
        )
        step, stop, _, _, _, _, condition, _ = lsmr(
            jacobian,
            -residual,
            damp=floor,
            atol=_LSMR_TOLERANCE,
            btol=_LSMR_TOLERANCE,
            maxiter=_LSMR_ITERATIONS,
        )
        if stop != _LSMR_ITERATION_LIMIT and np.all(np.isfinite(step)):
            # Without a sigma_min estimate, lsmr's own estimate of cond(J) decides.
            return step, (resolved or not measured) and condition < 1.0 / _RCOND
        warnings.warn(
            "connecting-orbit Newton step: lsmr did not converge after a singular "
            "factorisation; falling back to a dense least-squares solve",
            RuntimeWarning,
            stacklevel=2,
        )
        return _dense_step(jacobian, residual)

    def _jacobian(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        shape: tuple[int, int],
        residual: np.ndarray,
        analytic: bool,  # noqa: FBT001
    ) -> csr_matrix | None:
        """The exact Jacobian, or ``None`` when the field cannot be differentiated.

        Automatic differentiation evaluates the field on Taylor jets, which a
        transcendental or otherwise non-analytic field may reject. That is not an
        error: the caller falls back to finite differences for the rest of the solve.
        """
        if not analytic:
            return None
        try:
            return self._analytic_jacobian(unknowns, times, shape, residual.size)
        except (TypeError, ValueError, AttributeError):
            return None

    def _analytic_jacobian(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        shape: tuple[int, int],
        rows: int,
    ) -> csr_matrix:
        """Assemble the exact Jacobian block by block, in O(N) field Jacobians.

        Trapezoidal collocation couples only neighbouring nodes, so the collocation
        rows are banded: block ``i`` holds ``-I - (h/2) Df(x_i)`` against node ``i``
        and ``I - (h/2) Df(x_{i+1})`` against node ``i+1``. Each node's field
        Jacobian is needed by the two intervals that meet there, so it is evaluated
        once per node rather than once per interval. The boundary and phase rows are
        affine in the states they touch, so their exact rows come from unit
        displacements.

        The blocks go straight into a sparse matrix: nothing of size ``N^2`` is
        ever allocated.
        """
        states = unknowns.reshape(shape)
        derivatives = np.array(
            [_AUTODIFF.jacobian(self._function, state, 0.0) for state in states],
        )
        return self._assemble(unknowns, times, shape, rows, derivatives)

    def _numerical_jacobian(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        shape: tuple[int, int],
        rows: int,
    ) -> csr_matrix:
        """The same sparse assembly, with each node's field Jacobian differenced.

        The collocation rows see the field only through the node Jacobians, so
        differencing the field at each node - ``dimension`` extra evaluations per
        node - fills the same banded pattern as automatic differentiation, in O(N)
        work rather than one full residual per unknown.
        """
        states = unknowns.reshape(shape)
        derivatives = np.array([self._differenced(state) for state in states])
        return self._assemble(unknowns, times, shape, rows, derivatives)

    def _differenced(self, state: np.ndarray) -> np.ndarray:
        """Forward-difference Jacobian of the field at one node."""
        base = np.array(self._function.eval(point=list(state), time=None))
        columns = np.empty((state.size, state.size))
        for j in range(state.size):
            shifted = state.copy()
            shifted[j] += _STEP
            columns[:, j] = (
                np.array(self._function.eval(point=list(shifted), time=None)) - base
            ) / _STEP
        return columns

    def _assemble(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        shape: tuple[int, int],
        rows: int,
        derivatives: np.ndarray,
    ) -> csr_matrix:
        """Collocation blocks from the node Jacobians, bordered by the affine rows."""
        nodes, dimension = shape
        row, column, value = _collocation_triplets(derivatives, times)
        border = self._affine_triplets(unknowns, shape, (nodes - 1) * dimension)
        return csr_matrix(
            (
                np.concatenate([value, border[2]]),
                (np.concatenate([row, border[0]]), np.concatenate([column, border[1]])),
            ),
            shape=(rows, unknowns.size),
        )

    def _affine_triplets(
        self,
        unknowns: np.ndarray,
        shape: tuple[int, int],
        offset: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The boundary and phase rows, which are affine in the states, as triplets."""
        nodes, dimension = shape
        states = unknowns.reshape(shape)
        base = self._boundary(states)
        rows: list[np.ndarray] = []
        columns: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for node in (0, nodes - 1):
            for component in range(dimension):
                shifted = states.copy()
                shifted[node, component] += 1.0
                rows.append(offset + np.arange(base.size))
                columns.append(np.full(base.size, node * dimension + component))
                values.append(self._boundary(shifted) - base)
        centre = nodes // 2
        rows.append(np.array([offset + base.size]))
        columns.append(np.array([centre * dimension + self._mesh.phase_index]))
        values.append(np.array([1.0]))
        return np.concatenate(rows), np.concatenate(columns), np.concatenate(values)

    @abstractmethod
    def _boundary(self, states: np.ndarray) -> np.ndarray:
        """Projection boundary conditions at the two ends of the orbit."""
        raise NotImplementedError

    @abstractmethod
    def _boundary_truncation(self, states: np.ndarray) -> np.ndarray:
        """By how much the exact orbit through the path's ends misses
        :meth:`_boundary`, one magnitude per boundary row; see
        :func:`_manifold_miss`."""
        raise NotImplementedError


def _difference(states: np.ndarray, interval: int, order: int) -> np.ndarray:
    """The ``order``-th forward difference nearest centred on ``interval``.

    Interval ``i`` runs from node ``i`` to ``i + 1``. The centred stencil of odd
    order starts at node ``i - (order - 1)/2``. Near an end it slides inward to
    stay on the mesh, where the orbit is close to its equilibrium and the
    differences are small.
    """
    last = states.shape[0] - 1 - order
    start = min(max(interval - (order - 1) // 2, 0), last)
    weights = np.array(
        [(-1.0) ** (order - k) * math.comb(order, k) for k in range(order + 1)],
    )
    return weights @ states[start : start + order + 1]


def _manifold_miss(
    function: Function,
    end: "Terminus",
    displacement: np.ndarray,
    *,
    departing: bool,
) -> np.ndarray:
    """Leading-order miss of the linear projection conditions by the exact orbit.

    An orbit leaving the equilibrium ``p`` lies on its unstable manifold, which
    is tangent to the unstable eigenspace and curves away from it quadratically.
    Write ``e`` for the displacement from ``p`` and ``N(e) = f(p + e) - Df(p) e``
    for the nonlinear part of the field. A projected (stable) component
    ``c_k = l_k e`` then obeys ``c_k' = lambda_k c_k + l_k N(e)``, with ``N``
    growing like ``exp(2 mu t)`` and ``mu`` the slowest unstable rate. Its
    bounded solution is ``c_k = l_k N(e) / (2 mu - lambda_k)``. The arrival end
    is the same in reverse time, ``|l_k N(e)| / (2 nu + Re lambda_k)``, with
    ``nu`` the slowest stable rate. Each denominator is replaced by its real
    part, which is smaller in modulus. The manifold's cubic and higher terms
    are left out, and on short intervals they make the estimate low: on a
    Nagumo front it was 0.81 of the exact miss at half-length 2, 0.90 at 3 and
    0.97 at 5. ``_INCONSISTENCY_SAFETY`` covers that. Returns one magnitude per
    row that :func:`_projection_conditions` makes, in its order.
    """
    values, left, stable, unstable = _split_eigenspaces(end.jacobian)
    projected, kept = (stable, unstable) if departing else (unstable, stable)
    if not projected or not kept:
        return np.zeros(
            len(_projection_conditions(left, values, projected, displacement)),
        )
    slowest = min(abs(values[index].real) for index in kept)
    field = np.array(
        function.eval(point=list(end.equilibrium + displacement), time=None),
        dtype=float,
    )
    nonlinear = field - end.jacobian @ displacement
    misses: list[float] = []
    for index in projected:
        if values[index].imag < -_IMAGINARY:
            continue
        rate = 2.0 * slowest + abs(values[index].real)
        size = abs(left[index] @ nonlinear) / rate
        misses.append(size)
        if values[index].imag > _IMAGINARY:
            misses.append(size)
    return np.array(misses)


def _collocation_triplets(
    derivatives: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse triplets of the trapezoidal collocation rows, from node Jacobians.

    ``derivatives[i]`` is the field Jacobian at node ``i``. Interval ``i`` owns rows
    ``i*n .. i*n+n-1`` and holds ``-I - (h_i/2) Df(x_i)`` against node ``i`` and
    ``I - (h_i/2) Df(x_{i+1})`` against node ``i+1``: two ``n x n`` blocks per
    interval, ``2 N n^2`` nonzeros in all.
    """
    nodes, dimension, _ = derivatives.shape
    weights = _HALF * np.diff(times)[:, None, None]
    identity = np.eye(dimension)
    here = -identity - weights * derivatives[:-1]
    ahead = identity - weights * derivatives[1:]
    first = np.arange(nodes - 1)[:, None, None] * dimension
    rows = np.broadcast_to(first + np.arange(dimension)[None, :, None], here.shape)
    columns = np.broadcast_to(first + np.arange(dimension)[None, None, :], here.shape)
    return (
        np.concatenate([rows.ravel(), rows.ravel()]),
        np.concatenate([columns.ravel(), (columns + dimension).ravel()]),
        np.concatenate([here.ravel(), ahead.ravel()]),
    )


def _measure_collocation(
    states: np.ndarray,
    rows: np.ndarray,
    inconsistency: float = 0.0,
    inconsistency_tolerance: float = 0.0,
) -> CollocationResidual:
    """The :class:`CollocationResidual` of a path from its per-interval residuals.

    ``rows[i]`` is interval ``i``'s trapezoidal residual vector. Shared with the
    homoclinic curve, whose field depends on the point's parameters and whose
    square system has no inconsistency to measure.
    """
    residuals = np.max(np.abs(rows), axis=1)
    third = np.max(np.abs(np.diff(states, 3, axis=0)), axis=1, initial=0.0)
    covering = np.array(
        [
            np.max(third[max(0, i - 2) : i + 1], initial=0.0)
            for i in range(len(residuals))
        ],
    )
    allowance = _RESIDUAL_SAFETY * covering / 12.0 + _DEFAULT_TOLERANCE
    return CollocationResidual(
        float(np.max(residuals)),
        float(np.max(allowance)),
        float(np.max(residuals / allowance)),
        inconsistency,
        inconsistency_tolerance,
    )


def _relative_correction(
    jacobian: csr_matrix,
    residual: np.ndarray,
    step: np.ndarray,
    unknowns: np.ndarray,
    *,
    resolved: bool = True,
) -> float:
    """Largest ``|step_i| / (1 + |x_i|)``, or ``inf`` if the step cannot be trusted.

    COLNEW's convergence test, ``|dz_i| <= tol (|z_i| + 1)``, in one number. A
    small correction proves convergence only if it is the least-squares step.
    So the step must also satisfy the normal equations ``J^T (J dx + r) = 0``,
    to within ``_NORMAL_EQUATIONS`` of ``|J|_F (|J|_F |dx| + |r|)``. Otherwise a
    solver that returned a zero step at a non-stationary iterate would certify
    it, because the defect there is ``|J^T r|``.

    The dense fallback does exactly this along the directions it truncates.
    ``lstsq`` with ``rcond = _RCOND`` zeroes every singular direction below
    ``_RCOND sigma_max``, and a residual along such a direction leaves the
    defect ``sigma |r|``. At the old tolerance of 1e-6, a step truncated at
    ``sigma = 1e-10`` was certified with a correction of 1.6e-16, against a true
    step of 1e10 (DEQ-156). At ``1e-11`` the test sees a truncated direction
    with ``sigma > 1e-11 |J|_F`` when the residual lies along it. In general
    it sees one only when ``sigma |r_t| > 1e-11 |J|_F (|J|_F |dx| + |r|)``,
    with ``r_t`` the residual's part along it. So a residual mostly in
    resolved directions hides a truncation: with ``sigma = 1e-9`` and ten
    times as much residual elsewhere, a step whose truncated part is 0.1 is
    certified. A step truncated below the limit is invisible to any
    normal-equation test: its defect is smaller than the rounding in forming
    ``J dx``.

    So the test is only a second line. A step its solver computed without
    full numerical rank - ``resolved`` false: ``lstsq`` kept fewer singular
    values than columns, or the damped step's ``sigma_min`` estimate is below
    ``_RCOND sigma_max`` - is never trusted, whatever its defect (DEQ-180).
    Such an iterate is converged only if its residual is below tolerance.

    That rounding, about ``eps |J| |dx|``, is why ``|J|_F |dx|`` is in the
    scale. Without it, a tolerance this tight refuses a correct but large step.
    A truncated step is near zero, so the term costs it nothing. Measured over
    every step the connecting-orbit tests trust, the defect is at most 1.5e-15
    of this scale. A step that is not finite makes the defect
    NaN, and fails the test too.
    """
    if not resolved:
        return math.inf
    norm = float(np.sqrt(jacobian.multiply(jacobian).sum()))
    with np.errstate(over="ignore", invalid="ignore"):
        scale = norm * (
            norm * float(np.linalg.norm(step)) + float(np.linalg.norm(residual))
        )
        remainder = jacobian @ step + residual
        gradient = float(np.linalg.norm(jacobian.T @ remainder))
    if not gradient <= _NORMAL_EQUATIONS * scale:
        return math.inf
    return float(np.max(np.abs(step) / (1.0 + np.abs(unknowns))))


def _augmented(
    jacobian: csr_matrix,
    scale: float = 1.0,
    damping: float = 0.0,
) -> csc_matrix:
    """The augmented matrix ``[[scale I, J], [J^T, -damping I]]``.

    Its lower solution block is the least-squares solution of ``J x = -r``
    damped by ``lambda = scale * damping`` - the minimiser of ``|J x + r|^2 +
    lambda |x|^2`` - for any positive ``scale``; the scale changes only the
    conditioning. ``damping = 0`` is the undamped least-squares problem.
    """
    rows, columns = jacobian.shape
    lower = -damping * sparse_identity(columns) if damping > 0.0 else None
    return bmat(
        [[scale * sparse_identity(rows), jacobian], [jacobian.T, lower]],
        format="csc",
    )


def _largest_singular_value(jacobian: csr_matrix) -> float:
    """Estimate ``sigma_max(J)`` by ``_POWER_STEPS`` steps of power iteration.

    It only scales the damping floor and the rank test, so a small factor is
    immaterial; the estimate approaches ``sigma_max`` from below.
    """
    vector = 1.0 + 0.5 * np.cos(np.arange(jacobian.shape[1]))
    growth = 0.0
    for _ in range(_POWER_STEPS):
        vector /= np.linalg.norm(vector)
        image = jacobian.T @ (jacobian @ vector)
        growth = float(np.linalg.norm(image))
        if not np.isfinite(growth) or growth == 0.0:
            return growth
        vector = image
    return growth**0.5


def _dense_step(jacobian: csr_matrix, residual: np.ndarray) -> tuple[np.ndarray, bool]:
    """The dense least-squares step, and whether it was computed at full rank.

    ``lstsq`` with ``rcond = _RCOND`` zeroes every singular direction below
    ``_RCOND sigma_max`` and reports how many it kept. A step that dropped any
    is not the least-squares step along them, and no normal-equation test can
    always tell (DEQ-180), so it is reported as rank-deficient.
    """
    step, _, rank, _ = np.linalg.lstsq(jacobian.toarray(), -residual, rcond=_RCOND)
    return step, int(rank) == jacobian.shape[1]


def _smallest_singular_value(jacobian: csr_matrix) -> float:
    """Estimate ``sigma_min(J)`` by inverse iteration on ``J^T J``, never forming it.

    With the unscaled augmented matrix, the right-hand side ``[0, v]`` gives
    ``x = -(J^T J)^{-1} v``, so ``|x|`` grows like ``1/sigma_min^2``. The solves
    lose accuracy as ``cond(J)^2``, but the error lies along the smallest singular
    direction - the one being sought - so the estimate stays good. Raises
    ``RuntimeError`` if the matrix is exactly singular, as the step solve would.
    """
    rows, columns = jacobian.shape
    factor = _factor(_augmented(jacobian))
    vector = 1.0 + 0.5 * np.cos(np.arange(columns))
    vector /= np.linalg.norm(vector)
    growth = 1.0
    for _ in range(_SCALE_STEPS):
        image = factor.solve(np.concatenate([np.zeros(rows), vector]))[rows:]
        growth = float(np.linalg.norm(image))
        if not np.isfinite(growth) or growth == 0.0:
            message = "inverse iteration produced no finite direction"
            raise RuntimeError(message)
        vector = image / growth
    return growth**-0.5


def _factor(matrix: csc_matrix) -> SuperLU:
    """Sparse LU with a column ordering that keeps the bordered band's fill linear.

    The ordering matters more than anything else here. On the augmented saddle
    system ``COLAMD`` holds ``nnz(L + U) / nnz(A)`` at 2.6 from N = 200 to 3200;
    a minimum-degree ordering of ``A^T + A`` lets it grow with N, to 322 at 3200.
    """
    return splu(matrix, permc_spec=_ORDERING)


def _split_eigenspaces(
    jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Eigenvalues, left eigenvectors, and stable/unstable indices of a Jacobian."""
    values, vectors = np.linalg.eig(jacobian)
    left = np.linalg.inv(vectors)
    stable = [i for i in range(len(values)) if values[i].real < -_IMAGINARY]
    unstable = [i for i in range(len(values)) if values[i].real > _IMAGINARY]
    return values, left, stable, unstable


def _projection_conditions(
    left: np.ndarray,
    values: np.ndarray,
    indices: list[int],
    displacement: np.ndarray,
) -> list[float]:
    """Real scalar conditions forcing ``displacement`` out of an eigenspace.

    A real eigenvalue contributes one condition (the real projection); a complex
    conjugate pair contributes two (real and imaginary parts of one representative),
    so a spiral eigenspace is fully constrained rather than collapsed to a line.
    """
    conditions: list[float] = []
    for index in indices:
        if values[index].imag < -_IMAGINARY:
            continue  # skip the negative-frequency partner of a conjugate pair
        projection = left[index] @ displacement
        conditions.append(projection.real)
        if values[index].imag > _IMAGINARY:
            conditions.append(projection.imag)
    return conditions


class HomoclinicOrbit(ConnectingOrbit):
    """A homoclinic orbit found by a truncated projection boundary-value problem."""

    __slots__ = [
        "_equilibrium",
        "_left",
        "_stable",
        "_terminus",
        "_unstable",
        "_values",
    ]

    def __init__(
        self,
        function: Function,
        equilibrium: np.ndarray,
        jacobian: np.ndarray,
        mesh: MeshSpec,
    ) -> None:
        self._function = function
        self._equilibrium = equilibrium
        self._terminus = Terminus(equilibrium, jacobian)
        self._mesh = mesh
        self._values, self._left, self._stable, self._unstable = _split_eigenspaces(
            jacobian,
        )

    def _boundary(self, states: np.ndarray) -> np.ndarray:
        departure = states[0] - self._equilibrium
        arrival = states[-1] - self._equilibrium
        left = _projection_conditions(self._left, self._values, self._stable, departure)
        right = _projection_conditions(
            self._left,
            self._values,
            self._unstable,
            arrival,
        )
        return np.array(left + right)

    def _boundary_truncation(self, states: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                _manifold_miss(
                    self._function,
                    self._terminus,
                    states[0] - self._equilibrium,
                    departing=True,
                ),
                _manifold_miss(
                    self._function,
                    self._terminus,
                    states[-1] - self._equilibrium,
                    departing=False,
                ),
            ],
        )


class Terminus:
    """An endpoint saddle of a heteroclinic orbit: its position and Jacobian."""

    __slots__ = ["equilibrium", "jacobian"]

    def __init__(self, equilibrium: np.ndarray, jacobian: np.ndarray) -> None:
        self.equilibrium = equilibrium
        self.jacobian = jacobian


class HeteroclinicOrbit(ConnectingOrbit):
    """An orbit connecting one saddle to a different saddle.

    It leaves the ``source`` along that saddle's unstable manifold and arrives at the
    ``target`` along its stable manifold. The departure end is projected onto the
    source's stable left eigenvectors (forcing it into the unstable eigenspace) and
    the arrival end onto the target's unstable left eigenvectors.
    """

    __slots__ = [
        "_source",
        "_source_left",
        "_source_stable",
        "_source_terminus",
        "_source_values",
        "_target",
        "_target_left",
        "_target_terminus",
        "_target_unstable",
        "_target_values",
    ]

    def __init__(
        self,
        function: Function,
        source: Terminus,
        target: Terminus,
        mesh: MeshSpec,
    ) -> None:
        self._function = function
        self._mesh = mesh
        self._source = source.equilibrium
        self._target = target.equilibrium
        self._source_terminus = source
        self._target_terminus = target
        self._source_values, self._source_left, self._source_stable, _ = (
            _split_eigenspaces(source.jacobian)
        )
        self._target_values, self._target_left, _, self._target_unstable = (
            _split_eigenspaces(target.jacobian)
        )

    def _boundary(self, states: np.ndarray) -> np.ndarray:
        departure = states[0] - self._source
        arrival = states[-1] - self._target
        left = _projection_conditions(
            self._source_left,
            self._source_values,
            self._source_stable,
            departure,
        )
        right = _projection_conditions(
            self._target_left,
            self._target_values,
            self._target_unstable,
            arrival,
        )
        return np.array(left + right)

    def _boundary_truncation(self, states: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                _manifold_miss(
                    self._function,
                    self._source_terminus,
                    states[0] - self._source,
                    departing=True,
                ),
                _manifold_miss(
                    self._function,
                    self._target_terminus,
                    states[-1] - self._target,
                    departing=False,
                ),
            ],
        )
