"""The homoclinic curve continued to a saddle-focus (a Shilnikov homoclinic).

The jerk system ``x' = y, y' = z, z' = -a z - b y + x - x^2`` has a saddle-focus
at the origin: one unstable real eigenvalue and a stable complex pair. A complex
pair needs a real and an imaginary projection row; two real rows are the same
row, and the bordered Newton matrix is singular. These tests hold the curve to a
well-conditioned direct solve and to an independent DOP853 shooting computation.
"""

import warnings
from unittest import TestCase, mock

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from discrecontinual_equations.continuation.connecting_orbit import MeshSpec
from discrecontinual_equations.continuation.homoclinic_curve import (
    HomoclinicCurve,
    _continuous_phases,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from tests.continuation.fields import (
    Alpha,
    BetaOne,
    ReversedTwoParameterJerkField,
    State,
    Time,
    TwoParameterJerkField,
)

_HALF = 20.0
_RTOL, _ATOL = 1.0e-12, 1.0e-15
_DEPARTURE = 1.0e-6
_CYLINDER = 1.0e-6
_LEAVE = 1.0e-2
_NEAR = 0.2
_X_MIN = 0.3
# ``a`` on the homoclinic curve at each ``b``, located by ``_jerk_splitting`` with
# brentq to xtol 1e-14 (outputs/verification/deq-85/shooting_oracle.txt). The
# splitting changes sign across +-1e-10 of each value.
_SHOOTING = {
    0.90: 0.5190497226881766,
    0.89: 0.5334680374385187,
    0.88: 0.5478809089951218,
}


def _jerk(_: float, state: np.ndarray, a: float, b: float) -> list[float]:
    x, y, z = state
    return [y, z, -a * z - b * y + x - x * x]


def _jerk_eigenvectors(
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unstable right vector (x > 0), its left row, and the stable pair's left row."""
    jacobian = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -b, -a]])
    values, vectors = np.linalg.eig(jacobian)
    left = np.linalg.inv(vectors)
    unstable = int(np.argmax(values.real))
    spiral = int(np.argmax(values.imag))
    scale = np.sign(vectors[0, unstable].real)
    return (
        scale * vectors[:, unstable].real,
        scale * left[unstable].real,
        left[spiral],
    )


def _y_section(_: float, state: np.ndarray, *__: float) -> float:
    return state[1]


def _jerk_crest(a: float, b: float) -> np.ndarray:
    """Where the unstable branch first reaches its x maximum (y = 0, x > 0.3)."""
    vector, _, _ = _jerk_eigenvectors(a, b)
    path = solve_ivp(
        _jerk,
        (0.0, 80.0),
        _DEPARTURE * vector,
        method="DOP853",
        args=(a, b),
        rtol=_RTOL,
        atol=_ATOL,
        events=_y_section,
    )
    return next(s for s in path.y_events[0] if s[0] > _X_MIN)


def _jerk_splitting(a: float, b: float) -> float:
    """Signed distance of the unstable branch from the stable manifold.

    In linear eigen-coordinates ``u = w_u . x`` and ``c = w_s . x`` the section is
    the cylinder ``|c| = 1e-6`` around the unstable axis, which the returning
    branch crosses inward once; there the stable manifold is ``u = O(|c|^2)``, so
    ``u`` at the crossing is the splitting. A branch that leaves before reaching
    the cylinder reports only its side, which is all brentq needs to bracket.
    Nothing here is shared with the collocation solver.
    """
    _, unstable, spiral = _jerk_eigenvectors(a, b)

    def cylinder(_: float, state: np.ndarray, *__: float) -> float:
        return abs(spiral @ state) - _CYLINDER

    def leave(_: float, state: np.ndarray, *__: float) -> float:
        if np.linalg.norm(state) < _NEAR:
            return abs(unstable @ state) - _LEAVE
        return -_LEAVE

    def escape(_: float, state: np.ndarray, *__: float) -> float:
        return 4.0 - abs(state[0])

    cylinder.terminal = True  # type: ignore[attr-defined]
    cylinder.direction = -1  # type: ignore[attr-defined]
    leave.direction = 1  # type: ignore[attr-defined]
    escape.terminal = True  # type: ignore[attr-defined]
    path = solve_ivp(
        _jerk,
        (0.0, 120.0),
        _jerk_crest(a, b),
        method="DOP853",
        args=(a, b),
        rtol=_RTOL,
        atol=_ATOL,
        events=(cylinder, leave, escape),
    )
    found = [
        (path.t_events[k][0], side(path.y_events[k][0]))
        for k, side in enumerate(
            (
                lambda s: float(unstable @ s),
                lambda s: float(np.sign(unstable @ s)) * _LEAVE,
                lambda s: float(np.sign(s[0])) * _LEAVE,
            ),
        )
        if path.t_events[k].size
    ]
    return min(found)[1]


def _jerk_orbit(a: float, b: float, times: np.ndarray) -> np.ndarray:
    """The shot homoclinic at ``times`` measured from its x maximum."""
    crest = _jerk_crest(a, b)
    states = np.empty((times.size, 3))
    for ahead in (True, False):
        mask = times >= 0.0 if ahead else times < 0.0
        end = times[mask].max() if ahead else times[mask].min()
        path = solve_ivp(
            _jerk,
            (0.0, end),
            crest,
            method="DOP853",
            args=(a, b),
            rtol=_RTOL,
            atol=_ATOL,
            dense_output=True,
        )
        states[mask] = path.sol(times[mask]).T
    return states


def _jerk_curve(intervals: int) -> tuple[HomoclinicCurve, np.ndarray, np.ndarray]:
    """The curve solver (``a`` continued, ``b`` swept) and a shot seed at b = 0.9."""
    parameters = [Alpha(value=0.5), BetaOne(value=0.9)]
    equation = DifferentialEquation(
        variables=[State(), State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=TwoParameterJerkField(
            variables=[State(), State(), State()],
            parameters=parameters,
            results=[State(), State(), State()],
            time=None,
        ),
    )
    curve = HomoclinicCurve(equation, 0, 1, MeshSpec(_HALF, intervals, 1, 0.0))
    times = np.linspace(-_HALF, _HALF, intervals + 1)
    return curve, times, _jerk_orbit(_SHOOTING[0.90], 0.90, times)


def _refusing_fallback():
    refuse = mock.Mock(side_effect=AssertionError("dense fallback reached"))
    return mock.patch.object(np.linalg, "lstsq", refuse)


class TestSaddleFocusNewtonMatrix(TestCase):
    def test_bordered_matrix_is_square_and_well_conditioned(self):
        """Measured: cond 1.3e15 with two real rows per pair, 101 with Re and Im."""
        curve, times, seed = _jerk_curve(100)
        unknowns = np.concatenate([seed.flatten(), np.zeros(3), [_SHOOTING[0.90]]])
        residual = curve._residual(unknowns, times, times.size, 3, 0.90)  # noqa: SLF001
        jacobian = curve._jacobian(  # noqa: SLF001
            unknowns,
            times,
            3,
            0.90,
            residual,
        )
        assert jacobian.shape == (unknowns.size, unknowns.size)
        assert np.linalg.cond(jacobian.toarray()) < 1.0e3

    def test_curve_steps_never_reach_the_dense_fallback(self):
        curve, _, seed = _jerk_curve(100)
        sweeps = [0.90, 0.89, 0.88, 0.87]
        with _refusing_fallback(), warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            points = curve.trace(sweeps, seed, np.zeros(3), _SHOOTING[0.90])
        assert [point.sweep for point in points] == sweeps

    def test_boundary_rows_are_continuous_across_an_eigenvector_sign_flip(self):
        """LAPACK flips the pair's eigenvector between a = 0.5155 and 0.5155 + 1e-7."""
        rows = []
        for a in (0.5155, 0.5155 + 1.0e-7):
            values, vectors = np.linalg.eig(
                np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -0.89, -a]]),
            )
            left = _continuous_phases(values, np.linalg.inv(vectors))
            spiral, partner = np.argmax(values.imag), np.argmin(values.imag)
            largest = left[spiral][np.argmax(np.abs(left[spiral]))]
            assert abs(largest.imag) < 1.0e-15
            assert largest.real > 0.0
            np.testing.assert_allclose(
                left[partner],
                np.conj(left[spiral]),
                atol=1.0e-15,
            )
            # still left eigenvectors: a unimodular diagonal against the right ones
            duality = left @ vectors
            np.testing.assert_allclose(np.abs(duality), np.eye(3), atol=1.0e-12)
            rows.append(left[spiral])
        assert np.max(np.abs(rows[1] - rows[0])) < 1.0e-6

    def test_real_rows_are_returned_unchanged(self):
        values = np.array([2.0, -1.0, -3.0])
        left = np.array([[0.6, -0.8, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.3, -0.4]])
        np.testing.assert_array_equal(_continuous_phases(values, left), left)


class TestUnstableSaddleFocus(TestCase):
    """The complex pair on the unstable side, so its two rows sit at the arrival end.

    Reversing time keeps the homoclinic and swaps the eigenspaces. The trapezoidal
    rule is symmetric in time, so on the mirrored mesh the reversed problem has the
    same discrete solution: measured agreement 9e-16 in ``a`` and 3e-15 in the orbit.
    """

    def test_reversed_field_gives_the_same_curve(self):
        forward, times, seed = _jerk_curve(100)
        parameters = [Alpha(value=0.5), BetaOne(value=0.9)]
        reversed_field = ReversedTwoParameterJerkField(
            variables=[State(), State(), State()],
            parameters=parameters,
            results=[State(), State(), State()],
            time=None,
        )
        backward = HomoclinicCurve(
            DifferentialEquation(
                variables=[State(), State(), State()],
                time=Time(),
                parameters=parameters,
                derivative=reversed_field,
            ),
            0,
            1,
            MeshSpec(_HALF, times.size - 1, 1, 0.0),
        )
        sweeps = [0.90, 0.89]
        ahead = forward.trace(sweeps, seed, np.zeros(3), _SHOOTING[0.90])
        with _refusing_fallback():
            behind = backward.trace(
                sweeps,
                seed[::-1].copy(),
                np.zeros(3),
                _SHOOTING[0.90],
            )
        assert [point.sweep for point in behind] == sweeps
        for there, back in zip(ahead, behind, strict=True):
            assert abs(back.continuation - there.continuation) < 1.0e-12
            assert (
                np.max(np.abs(back.orbit.states[::-1] - there.orbit.states)) < 1.0e-10
            )


class TestSaddleFocusCurveAgainstShooting(TestCase):
    """The collocated curve converges to an independent DOP853 shooting curve.

    Measured errors in ``a`` at b = 0.90, 0.89, 0.88: -1.388e-2, -1.355e-2,
    -1.322e-2 at 100 intervals and -3.502e-3, -3.420e-3, -3.337e-3 at 200 (ratios
    3.96); the Richardson value is within 4.3e-5 of the shooting value.
    """

    def test_shooting_reproduces_the_oracle_table(self):
        located = brentq(
            lambda a: _jerk_splitting(a, 0.90),
            0.50,
            0.55,
            xtol=1.0e-12,
        )
        assert abs(located - _SHOOTING[0.90]) < 1.0e-11

    def test_curve_converges_to_shooting_at_second_order(self):
        sweeps = sorted(_SHOOTING, reverse=True)
        errors = []
        for intervals in (100, 200):
            curve, _, seed = _jerk_curve(intervals)
            points = curve.trace(sweeps, seed, np.zeros(3), _SHOOTING[0.90])
            assert [point.sweep for point in points] == sweeps
            errors.append(
                np.array([p.continuation - _SHOOTING[p.sweep] for p in points]),
            )
        coarse, fine = errors
        assert np.all(np.abs(fine) < 1.0e-2 * np.array(list(_SHOOTING.values())))
        assert np.all((coarse / fine > 3.6) & (coarse / fine < 4.4))
        assert np.max(np.abs(fine + (fine - coarse) / 3.0)) < 1.0e-4
