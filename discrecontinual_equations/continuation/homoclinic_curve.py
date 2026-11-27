"""Continue a saddle homoclinic orbit in two parameters (a global codim-1 curve).

The single-orbit projection boundary-value problem in :mod:`connecting_orbit` solves
a homoclinic for a fixed field. Continuing it turns the homoclinic into a curve in a
two-parameter plane - the global bifurcation curve that, near a Bogdanov-Takens
point, emanates tangent to the Hopf curve.

The augmented system carries the orbit nodes, the saddle equilibrium, and one active
continuation parameter as unknowns; the other parameter is swept. Its equations are
the trapezoidal collocation of the orbit, the equilibrium condition, the projection
boundary conditions onto the (recomputed) stable and unstable eigenspaces of the
saddle, and a phase condition. The square system is solved by Newton's method, and
each swept point is seeded from the previous one.

The boundary conditions give one row per stable and one per unstable eigenvalue
of the saddle, so the system is square only while the saddle keeps the counts it
was seeded with. Newton refuses an iterate at which they change
(:class:`EigenvalueCountError`), and a traced curve stops there. The counts and
the projections are read from the saddle's exact (automatic-differentiation)
Jacobian, so a saddle whose eigenvalue is far smaller than a difference step is
still counted correctly; the finite-difference shifts that build the Newton
matrix keep the iterate's partition of the eigenvalues rather than re-counting.

The Newton matrix is block-banded in the nodes and bordered by the equilibrium,
boundary and phase rows and the parameter column - HomCont's structure (Champneys,
Kuznetsov & Sandstede, IJBC 6, 1996) - so it is assembled sparse and factored with
a sparse LU, in time and memory linear in the nodes.
"""

import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import splu

from discrecontinual_equations.continuation.connecting_orbit import (
    _IMAGINARY,
    CollocationResidual,
    MeshSpec,
    OrbitSolution,
    _collocation_triplets,
    _measure_collocation,
    _projection_conditions,
    _split_eigenspaces,
)
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.differential_equation import DifferentialEquation

_TOLERANCE = 1.0e-10
_ITERATIONS = 80
_STEP = 1.0e-7
_HALF = 0.5
# Column ordering for the sparse LU; see ``connecting_orbit._factor`` for why.
_ORDERING = "COLAMD"
_AUTODIFF = AutomaticDifferentiation()


class EigenvalueCountError(ValueError):
    """The saddle's stable and unstable eigenvalue counts are not the ones required.

    The projection boundary conditions put the departure off the stable
    eigenspace and the arrival off the unstable one: one row per stable and one
    per unstable eigenvalue (a complex pair gives its real and imaginary part).
    With ``N`` nodes in ``n`` dimensions the system has ``N n + s + u + 1`` rows
    against ``N n + n + 1`` unknowns, so it is square only when ``s + u = n`` -
    a hyperbolic equilibrium - and a homoclinic orbit needs ``s, u >= 1``. When
    a Newton iterate reaches a point where the counts differ from the seed's,
    the rows no longer line up and the point is refused with this error rather
    than solved as a different problem. The counts are read from the exact
    Jacobian at the saddle, so only a genuine change is refused.

    Attributes:
        expected: ``(stable, unstable)`` counts the solve requires, those of the
            seed; ``None`` when the seed itself is not a hyperbolic saddle.
        found: ``(stable, unstable)`` counts where the solve was refused.
        continuation: Value of the active continuation parameter there.
        sweep: Value of the swept parameter there.

    References:
        Champneys, A. R., Kuznetsov, Yu. A. & Sandstede, B. (1996). A numerical
            toolbox for homoclinic bifurcation analysis. *International Journal
            of Bifurcation and Chaos* 6(5), 867-887. DOI 10.1142/S0218127496000485.
            The defining system continued here.
        Beyn, W.-J. (1990). The numerical computation of connecting orbits in
            dynamical systems. *IMA Journal of Numerical Analysis* 10(3),
            379-405. DOI 10.1093/imanum/10.3.379. Projection boundary
            conditions at the truncated ends.
    """

    def __init__(
        self,
        message: str,
        expected: tuple[int, int] | None,
        found: tuple[int, int],
        parameters: tuple[float, float],
    ) -> None:
        super().__init__(message)
        self.expected = expected
        self.found = found
        self.continuation, self.sweep = parameters


def _describe(counts: tuple[int, int]) -> str:
    return f"{counts[0]} stable and {counts[1]} unstable"


def _continuous_phases(values: np.ndarray, left: np.ndarray) -> np.ndarray:
    """Fix the phase of each complex left eigenvector so it varies smoothly.

    LAPACK makes the largest component of a complex eigenvector real but leaves
    its sign free, and the sign can flip between two Jacobians ``1e-7`` apart.
    The real and imaginary projection rows then flip with it, and the
    differenced boundary rows of the Newton matrix jump by ``2 |residual| / 1e-7``
    (measured: ``1.3e5`` in the parameter column on the jerk saddle-focus).
    Rotating each complex row so its largest component is real and positive
    leaves the conditions' zero set unchanged and makes them continuous in the
    Jacobian away from ties in the largest component. Real rows are untouched.
    """
    fixed = left.copy()
    for index in np.flatnonzero(np.abs(values.imag) > _IMAGINARY):
        row = fixed[index]
        largest = row[np.argmax(np.abs(row))]
        fixed[index] = row * (np.conj(largest) / np.abs(largest))
    return fixed


def _held_partition(
    values: np.ndarray,
    reference: tuple[np.ndarray, list[int], list[int]],
) -> tuple[list[int], list[int]]:
    """The stable and unstable indices of ``values``, matched to a reference.

    A finite-difference shift of the saddle moves each eigenvalue by about the
    step times the field's curvature, and may reorder them. Pairing each shifted
    eigenvalue with the reference eigenvalue it is closest to (a minimum-cost
    assignment on ``|mu_i - lambda_j|``) lets the shifted rows inherit the
    reference's partition, in the reference's row order, even where a small
    eigenvalue's real part changes sign within the step.

    References:
        Crouse, D. F. (2016). On implementing 2D rectangular assignment
            algorithms. *IEEE Transactions on Aerospace and Electronic Systems*
            52(4), 1679-1696. DOI 10.1109/TAES.2016.140952. The modified
            Jonker-Volgenant algorithm of ``scipy.optimize.linear_sum_assignment``.
    """
    reference_values, stable, unstable = reference
    distance = np.abs(values[:, None] - reference_values[None, :])
    shifted, original = linear_sum_assignment(distance)
    matched = dict(zip(original.tolist(), shifted.tolist(), strict=True))
    return [matched[index] for index in stable], [matched[index] for index in unstable]


class HomoclinicCurvePoint:
    """A solved point on the homoclinic curve: its two parameters and orbit.

    The orbit carries its :class:`CollocationResidual` under the field at this
    point's parameters, also available as ``residual``.
    """

    __slots__ = ["_continuation", "_equilibrium", "_orbit", "_sweep"]

    def __init__(
        self,
        continuation: float,
        sweep: float,
        orbit: OrbitSolution,
        equilibrium: np.ndarray,
    ) -> None:
        self._continuation = continuation
        self._sweep = sweep
        self._orbit = orbit
        self._equilibrium = np.asarray(equilibrium, dtype=float)

    @property
    def continuation(self) -> float:
        """Value of the active continuation parameter on the curve."""
        return self._continuation

    @property
    def sweep(self) -> float:
        """Value of the swept parameter."""
        return self._sweep

    @property
    def orbit(self) -> OrbitSolution:
        """The homoclinic orbit at this point."""
        return self._orbit

    @property
    def equilibrium(self) -> np.ndarray:
        """The saddle equilibrium the orbit connects to."""
        return self._equilibrium

    @property
    def residual(self) -> CollocationResidual | None:
        """The orbit's collocation residual under the field at this point."""
        return self._orbit.residual


class HomoclinicCurve:
    """Trace a saddle homoclinic bifurcation curve in two parameters."""

    __slots__ = [
        "_continuation_index",
        "_equation",
        "_mesh",
        "_sweep_index",
        "_time",
    ]

    def __init__(
        self,
        equation: DifferentialEquation,
        continuation_index: int,
        sweep_index: int,
        mesh: MeshSpec,
        time: float = 0.0,
    ) -> None:
        self._equation = equation
        self._continuation_index = continuation_index
        self._sweep_index = sweep_index
        self._mesh = mesh
        self._time = time

    def _field(
        self,
        point: np.ndarray,
        continuation: float,
        sweep: float,
    ) -> np.ndarray:
        parameters = self._equation.derivative.parameters
        parameters[self._continuation_index].value = float(continuation)
        parameters[self._sweep_index].value = float(sweep)
        return np.array(
            self._equation.derivative.eval(point=list(point), time=self._time),
            dtype=float,
        )

    def _field_jacobian(
        self,
        point: np.ndarray,
        continuation: float,
        sweep: float,
    ) -> np.ndarray:
        base = self._field(point, continuation, sweep)
        dimension = point.size
        columns = np.empty((dimension, dimension))
        for j in range(dimension):
            shifted = point.copy()
            shifted[j] += _STEP
            columns[:, j] = (self._field(shifted, continuation, sweep) - base) / _STEP
        return columns

    def _saddle_jacobian(
        self,
        equilibrium: np.ndarray,
        continuation: float,
        sweep: float,
    ) -> np.ndarray:
        """The field Jacobian at the saddle, exact where the field allows it.

        The eigenvalue counts and the projection rows are read from it. A
        forward difference with step ``dx`` reads an eigenvalue ``lam`` as about
        ``lam + c dx`` for a field curvature ``c``, which flips the sign of any
        ``|lam| < c dx`` and so miscounts a genuine saddle; the Taylor-jet
        Jacobian has no such error. A field that cannot be evaluated on jets
        falls back to the forward difference, as the node Jacobians do.

        References:
            Griewank, A. & Walther, A. (2008). *Evaluating Derivatives: Principles
                and Techniques of Algorithmic Differentiation*, 2nd ed. SIAM.
                DOI 10.1137/1.9780898717761. Ch. 13 (Taylor and Tensor
                Coefficients): first-order Taylor propagation along each unit
                direction gives the Jacobian's columns exactly.
        """
        return self._node_jacobians(equilibrium[np.newaxis], continuation, sweep)[0]

    def _label(self, index: int) -> str:
        parameter = self._equation.derivative.parameters[index]
        return parameter.abbreviation or parameter.name or f"parameter {index}"

    def _counts(
        self,
        equilibrium: np.ndarray,
        continuation: float,
        sweep: float,
    ) -> tuple[int, int]:
        """The saddle's ``(stable, unstable)`` eigenvalue counts, read exactly."""
        _, _, stable, unstable = _split_eigenspaces(
            self._saddle_jacobian(equilibrium, continuation, sweep),
        )
        return len(stable), len(unstable)

    def _refuse(
        self,
        reason: str,
        found: tuple[int, int],
        parameters: tuple[float, float],
        expected: tuple[int, int] | None = None,
    ) -> EigenvalueCountError:
        continuation, sweep = parameters
        required = (
            "a hyperbolic saddle (some stable and some unstable, none central)"
            if expected is None
            else f"{_describe(expected)}, as at the seed"
        )
        message = (
            f"{reason}: the equilibrium has {_describe(found)} eigenvalues at "
            f"{self._label(self._continuation_index)} = {continuation:.6g} "
            f"({self._label(self._sweep_index)} = {sweep:.6g}), and the projection "
            f"boundary conditions need {required}"
        )
        return EigenvalueCountError(message, expected, found, parameters)

    def _seed_counts(
        self,
        equilibrium: np.ndarray,
        continuation: float,
        sweep: float,
    ) -> tuple[int, int]:
        """The seed's counts, refused unless it is a hyperbolic saddle."""
        counts = self._counts(equilibrium, continuation, sweep)
        stable, unstable = counts
        if stable == 0 or unstable == 0 or stable + unstable != equilibrium.size:
            reason = "the seed equilibrium is not a hyperbolic saddle"
            raise self._refuse(reason, counts, (continuation, sweep))
        return counts

    def _residual(  # noqa: PLR0913 (the Newton state plus the saddle's counts)
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        nodes: int,
        dimension: int,
        sweep: float,
        counts: tuple[int, int] | None = None,
    ) -> np.ndarray:
        states = unknowns[: nodes * dimension].reshape(nodes, dimension)
        equilibrium = unknowns[nodes * dimension : nodes * dimension + dimension]
        continuation = float(unknowns[-1])
        blocks = []
        for i in range(nodes - 1):
            step = times[i + 1] - times[i]
            here = self._field(states[i], continuation, sweep)
            ahead = self._field(states[i + 1], continuation, sweep)
            blocks.append(states[i + 1] - states[i] - 0.5 * step * (here + ahead))
        blocks.append(
            self._tail(
                states[0],
                states[-1],
                equilibrium,
                (continuation, sweep),
                counts,
            ),
        )
        centre = nodes // 2
        phase = states[centre, self._mesh.phase_index] - self._mesh.phase_value
        blocks.append(np.array([phase]))
        return np.concatenate(blocks)

    def _tail(  # noqa: PLR0913 (the tail's unknowns plus how to split the saddle)
        self,
        departure_state: np.ndarray,
        arrival_state: np.ndarray,
        equilibrium: np.ndarray,
        parameters: tuple[float, float],
        counts: tuple[int, int] | None = None,
        partition: tuple[np.ndarray, list[int], list[int]] | None = None,
    ) -> np.ndarray:
        """The equilibrium condition and the projection boundary conditions.

        These are the only rows that see the saddle's eigenvectors, and the only
        ones besides the phase condition that reach across the mesh: they touch
        the two end nodes, the equilibrium and the continuation parameter.

        A complex pair of a saddle-focus gives one real and one imaginary row,
        as for a single connecting orbit: the real parts of a conjugate pair's
        two projections are the same number, so one row per eigenvalue would
        make the Newton matrix singular.

        With ``counts`` given, a saddle whose ``(stable, unstable)`` counts differ
        is refused: its rows would not line up with the rows they are compared to.
        With ``partition`` - the eigenvalues and the stable and unstable indices
        of an unshifted saddle - the eigenvalues are not re-counted but matched
        to those (:func:`_held_partition`), so a finite-difference shift gives
        the same rows, in the same order, as the point it differences.
        """
        continuation, sweep = parameters
        values, left, stable, unstable = _split_eigenspaces(
            self._saddle_jacobian(equilibrium, continuation, sweep),
        )
        if partition is not None:
            stable, unstable = _held_partition(values, partition)
        found = (len(stable), len(unstable))
        if counts is not None and found != counts:
            reason = "the saddle's eigenvalue counts changed during the solve"
            raise self._refuse(reason, found, parameters, counts)
        left = _continuous_phases(values, left)
        departure = departure_state - equilibrium
        arrival = arrival_state - equilibrium
        boundary = _projection_conditions(left, values, stable, departure)
        boundary += _projection_conditions(left, values, unstable, arrival)
        return np.concatenate(
            [self._field(equilibrium, continuation, sweep), np.array(boundary)],
        )

    def _jacobian(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        dimension: int,
        sweep: float,
        residual: np.ndarray,
    ) -> csr_matrix:
        """Assemble the bordered block-banded Newton matrix, in O(N) field work.

        This is the defining system HomCont continues a homoclinic orbit to a
        saddle with (Champneys, Kuznetsov & Sandstede 1996): collocation, the
        equilibrium condition, Beyn's projection boundary conditions, a phase
        condition, and one free system parameter. Its Jacobian has three parts:

        * the collocation rows, banded in the nodes exactly as for a single
          connecting orbit, from one field Jacobian per node, plus the parameter
          column ``-(h/2) (f_a(x_i) + f_a(x_{i+1}))``;
        * the equilibrium and boundary rows, which touch only the two end nodes,
          the equilibrium and the parameter, and whose dependence on the saddle's
          eigenvectors is differenced - as HomCont does - in ``3n + 1``
          evaluations that do not grow with the mesh, each taking the
          eigenvectors of the exact Jacobian at the shifted saddle;
        * the phase row, a single unit entry.

        The dense finite-difference matrix it replaces took one full residual per
        unknown: ``O(N^2)`` field evaluations and ``O(N^2)`` memory per iteration.

        Each differenced tail keeps the partition of the saddle's eigenvalues at
        ``unknowns`` (:func:`_held_partition`): a shift of ``dx`` moves an
        eigenvalue by about ``c dx``, and one with ``|lam| < c dx`` would
        otherwise be re-counted on the other side and give rows of two lengths,
        though the iterate itself is a saddle with the seed's counts.
        """
        size = unknowns.size
        nodes = times.size
        states = unknowns[: nodes * dimension].reshape(nodes, dimension)
        continuation = float(unknowns[-1])
        eigenvalues, _, stable, unstable = _split_eigenspaces(
            self._saddle_jacobian(
                unknowns[nodes * dimension : nodes * dimension + dimension],
                continuation,
                sweep,
            ),
        )
        derivatives = self._node_jacobians(states, continuation, sweep)
        row, column, value = _collocation_triplets(derivatives, times)
        bands = (nodes - 1) * dimension
        sensitivity = np.array(
            [
                self._parameter_sensitivity(state, continuation, sweep)
                for state in states
            ],
        )
        weights = _HALF * np.diff(times)[:, None]
        parameter = -weights * (sensitivity[:-1] + sensitivity[1:])
        rows = [row, np.arange(bands)]
        columns = [column, np.full(bands, size - 1)]
        values = [value, parameter.ravel()]
        tail_rows = bands + np.arange(residual.size - 1 - bands)
        base = residual[bands:-1]
        for index in self._tail_columns(nodes, dimension, size):
            shifted = unknowns.copy()
            shifted[index] += _STEP
            moved = self._tail(
                shifted[:dimension],
                shifted[bands : bands + dimension],
                shifted[nodes * dimension : nodes * dimension + dimension],
                (float(shifted[-1]), sweep),
                partition=(eigenvalues, stable, unstable),
            )
            rows.append(tail_rows)
            columns.append(np.full(tail_rows.size, index))
            values.append((moved - base) / _STEP)
        rows.append(np.array([residual.size - 1]))
        columns.append(
            np.array([(nodes // 2) * dimension + self._mesh.phase_index]),
        )
        values.append(np.array([1.0]))
        return csr_matrix(
            (
                np.concatenate(values),
                (np.concatenate(rows), np.concatenate(columns)),
            ),
            shape=(residual.size, size),
        )

    @staticmethod
    def _tail_columns(nodes: int, dimension: int, size: int) -> list[int]:
        """Unknowns the tail rows depend on: both end nodes, equilibrium, parameter."""
        last = (nodes - 1) * dimension
        equilibrium = nodes * dimension
        return [
            *range(dimension),
            *range(last, last + dimension),
            *range(equilibrium, equilibrium + dimension),
            size - 1,
        ]

    def _node_jacobians(
        self,
        states: np.ndarray,
        continuation: float,
        sweep: float,
    ) -> np.ndarray:
        """The field Jacobian at every point of ``states``, exact where possible."""
        parameters = self._equation.derivative.parameters
        parameters[self._continuation_index].value = float(continuation)
        parameters[self._sweep_index].value = float(sweep)
        try:
            return np.array(
                [
                    _AUTODIFF.jacobian(self._equation.derivative, state, self._time)
                    for state in states
                ],
            )
        except (TypeError, ValueError, AttributeError):
            return np.array(
                [self._field_jacobian(state, continuation, sweep) for state in states],
            )

    def _parameter_sensitivity(
        self,
        point: np.ndarray,
        continuation: float,
        sweep: float,
    ) -> np.ndarray:
        """Forward difference of the field in the continuation parameter."""
        base = self._field(point, continuation, sweep)
        return (self._field(point, continuation + _STEP, sweep) - base) / _STEP

    @staticmethod
    def _step(jacobian: csr_matrix, residual: np.ndarray) -> np.ndarray:
        """Newton step from a sparse LU of the square bordered system.

        Falls back to the dense least-squares solve the dense Newton used when the
        matrix is singular or not square, which happens only for a degenerate
        saddle whose boundary-condition count does not match the unknowns. The
        fallback is dense, so it announces itself with a ``RuntimeWarning``.
        """
        if jacobian.shape[0] != jacobian.shape[1]:
            reason = (
                f"Newton matrix is {jacobian.shape[0]} x {jacobian.shape[1]}, not "
                f"square (a non-hyperbolic saddle)"
            )
        else:
            reason = "non-finite solution (near-singular factorisation)"
            try:
                step = splu(csc_matrix(jacobian), permc_spec=_ORDERING).solve(
                    -residual,
                )
            except RuntimeError:
                reason = "exactly singular factorisation"
                step = np.full(residual.size, np.nan)
            if np.all(np.isfinite(step)):
                return step
        warnings.warn(
            f"homoclinic-curve Newton step: {reason}; falling back to a dense "
            f"least-squares solve",
            RuntimeWarning,
            stacklevel=2,
        )
        fallback, *_ = np.linalg.lstsq(jacobian.toarray(), -residual, rcond=None)
        return fallback

    def _newton(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        dimension: int,
        sweep: float,
    ) -> np.ndarray | None:
        nodes = times.size
        counts = self._seed_counts(
            unknowns[nodes * dimension : nodes * dimension + dimension],
            float(unknowns[-1]),
            sweep,
        )
        for _ in range(_ITERATIONS):
            # Refuses an iterate whose counts are not the seed's, so the counts
            # the Jacobian holds its shifts to are the seed's as well.
            norm, residual = self._residual_norm(
                unknowns,
                times,
                dimension,
                sweep,
                counts,
            )
            if norm < _TOLERANCE:
                return unknowns
            if not np.isfinite(norm):
                return None
            jacobian = self._jacobian(
                unknowns,
                times,
                dimension,
                sweep,
                residual,
            )
            unknowns = unknowns + self._step(jacobian, residual)
        norm, _ = self._residual_norm(unknowns, times, dimension, sweep, counts)
        if norm < _TOLERANCE:
            return unknowns
        return None

    def _residual_norm(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        dimension: int,
        sweep: float,
        counts: tuple[int, int] | None = None,
    ) -> tuple[float, np.ndarray]:
        """The residual and its norm, ``inf`` once a diverging iterate overflows.

        ``counts``, when given, are the saddle's eigenvalue counts the residual
        holds the iterate to; a change raises :class:`EigenvalueCountError`.

        Overflow surfaces as Python's ``OverflowError`` from the field, or as
        numpy's refusal to decompose the saddle's non-finite Jacobian.
        """
        try:
            residual = self._residual(
                unknowns,
                times,
                times.size,
                dimension,
                sweep,
                counts,
            )
        except (OverflowError, np.linalg.LinAlgError):
            return np.inf, np.empty(0)
        with np.errstate(over="ignore", invalid="ignore"):
            return float(np.linalg.norm(residual)), residual

    def solve_point(
        self,
        sweep: float,
        orbit_seed: np.ndarray,
        equilibrium_seed: np.ndarray,
        continuation_seed: float,
    ) -> HomoclinicCurvePoint | None:
        """Solve one point of the curve at a fixed swept-parameter value.

        Returns ``None`` unless Newton's method drives the residual of the square
        system below ``_TOLERANCE``, whichever step - sparse LU or the dense
        least-squares fallback - got it there; a diverging iterate that overflows
        is a ``None`` too. The orbit carries its collocation residual.

        Newton's method on the bordered collocation system: orbit nodes, saddle
        and the continuation parameter are unknowns, the swept parameter is held.
        The saddle must keep the seed's stable and unstable eigenvalue counts
        throughout, since the boundary rows are counted from them. They are read
        from the exact (Taylor-jet) Jacobian at the saddle, with a forward
        difference only for a field that cannot be evaluated on jets.

        Args:
            sweep: Value of the swept parameter.
            orbit_seed: ``(intervals + 1, n)`` initial orbit on the mesh.
            equilibrium_seed: Initial saddle.
            continuation_seed: Initial value of the continuation parameter.

        Returns:
            The converged point, or ``None``.

        Raises:
            EigenvalueCountError: The seed is not a hyperbolic saddle, or an
                iterate changes its stable and unstable eigenvalue counts.

        References:
            Champneys, A. R., Kuznetsov, Yu. A. & Sandstede, B. (1996). A numerical
                toolbox for homoclinic bifurcation analysis. *International Journal
                of Bifurcation and Chaos* 6(5), 867-887. DOI 10.1142/S0218127496000485.
                The defining system continued here.
            Beyn, W.-J. (1990). The numerical computation of connecting orbits in
                dynamical systems. *IMA Journal of Numerical Analysis* 10(3),
                379-405. DOI 10.1093/imanum/10.3.379. Projection boundary
                conditions at the truncated ends.
            Bader and Ascher 1987 (docs/references.md, ``Bader1987``), COLNEW
            subroutine ``CONTRL``: a Newton iteration that fails returns no
            solution (``IFLAG = -2``), as this returns ``None``.
            Griewank, A. & Walther, A. (2008). *Evaluating Derivatives*, 2nd ed.
                SIAM. DOI 10.1137/1.9780898717761. Ch. 13: the Taylor-jet
                Jacobian the saddle's eigenvalue counts are read from.
        """
        nodes = self._mesh.intervals + 1
        dimension = orbit_seed.shape[1]
        times = np.linspace(-self._mesh.half_length, self._mesh.half_length, nodes)
        unknowns = np.concatenate(
            [
                orbit_seed.astype(float).flatten(),
                np.asarray(equilibrium_seed, dtype=float),
                [float(continuation_seed)],
            ],
        )
        solved = self._newton(unknowns, times, dimension, sweep)
        if solved is None:
            return None
        states = solved[: nodes * dimension].reshape(nodes, dimension)
        equilibrium = solved[nodes * dimension : nodes * dimension + dimension]
        continuation = float(solved[-1])
        rows = self._residual(solved, times, nodes, dimension, sweep)
        measured = _measure_collocation(
            states,
            rows[: (nodes - 1) * dimension].reshape(nodes - 1, dimension),
        )
        return HomoclinicCurvePoint(
            continuation,
            sweep,
            OrbitSolution(times, states, measured),
            equilibrium,
        )

    def trace(
        self,
        sweep_values: list[float],
        orbit_seed: np.ndarray,
        equilibrium_seed: np.ndarray,
        continuation_seed: float,
    ) -> list[HomoclinicCurvePoint]:
        """Trace the curve across ``sweep_values``, seeding each from the last.

        Natural-parameter continuation: each value is solved by
        :meth:`solve_point` from the previous point's orbit, saddle and
        parameter. It stops at the first value it cannot solve and returns the
        points before it, as the other curve tracers do; every point in it
        converged. A stop because the
        saddle's eigenvalue counts changed - it lost hyperbolicity, or stopped
        being a saddle - is announced with a ``RuntimeWarning`` carrying the
        :class:`EigenvalueCountError` message, which names both counts and the
        parameter values.

        Args:
            sweep_values: Swept-parameter values, in order.
            orbit_seed: Initial orbit for the first value.
            equilibrium_seed: Initial saddle for the first value.
            continuation_seed: Initial continuation parameter for the first value.

        Returns:
            The solved points, up to the first value that could not be solved.

        References:
            Champneys, A. R., Kuznetsov, Yu. A. & Sandstede, B. (1996). A numerical
                toolbox for homoclinic bifurcation analysis. *International Journal
                of Bifurcation and Chaos* 6(5), 867-887. DOI 10.1142/S0218127496000485.
                The defining system continued here.
            Beyn, W.-J. (1990). The numerical computation of connecting orbits in
                dynamical systems. *IMA Journal of Numerical Analysis* 10(3),
                379-405. DOI 10.1093/imanum/10.3.379. Projection boundary
                conditions at the truncated ends.
        """
        points: list[HomoclinicCurvePoint] = []
        orbit = orbit_seed.astype(float)
        equilibrium = np.asarray(equilibrium_seed, dtype=float)
        continuation = float(continuation_seed)
        for sweep in sweep_values:
            try:
                point = self.solve_point(sweep, orbit, equilibrium, continuation)
            except EigenvalueCountError as refusal:
                warnings.warn(
                    f"homoclinic curve stopped after {len(points)} points: {refusal}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
            if point is None:
                break
            points.append(point)
            orbit = point.orbit.states
            continuation = point.continuation
            equilibrium = point.equilibrium
        return points
