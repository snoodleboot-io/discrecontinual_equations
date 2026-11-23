"""The Taylor chart of a saddle-focus's two-dimensional manifold.

The oracles are exact. A linear saddle-focus's stable manifold is the plane of
its complex pair, so every chart point lies in it. The nonlinear system is a
linear saddle-focus seen through the polynomial change of coordinates
``x3 = y3 + phi(x1, x2)``: its stable manifold is the graph ``x3 = phi(x1, x2)``
exactly, with ``phi`` of degree 9, so a chart of order ``N < 9`` is off it by
``phi``'s terms of degree ``N + 1`` and up - ``O(rho^(N+1))`` at radius
``rho`` - and from order 9 on it is the graph to rounding.
"""

import math
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.geodesic_level_sets import (
    GeodesicLevelSets,
    GeodesicSettings,
)
from discrecontinual_equations.continuation.manifold import (
    StableManifold,
    TaylorManifold,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.webplot.stage_builder import (
    StageSettings,
    _manifold_starts,
)
from tests.continuation.fields import Alpha, State

# The focus block of B: eigenvalues -1 +- 2i, and 1.5 off the surface.
_B = np.array([[-1.0, 2.0, 0.0], [-2.0, -1.0, 0.0], [0.0, 0.0, 1.5]])
# A skewed frame, so the linear focus's plane is no coordinate plane.
_FRAME = np.array([[1.0, 0.4, 0.3], [-0.2, 1.0, 0.5], [0.3, -0.4, 1.0]])
_LINEAR = _FRAME @ _B @ np.linalg.inv(_FRAME)
_DEGREE = 9


def _phi(x1, x2):
    """sum_{k=2}^{9} 0.7^k (x1 + 0.3 x2)^k: a surface with a term of every degree."""
    s = x1 + 0.3 * x2
    return sum((0.7**k) * s**k for k in range(2, _DEGREE + 1))


def _dphi(x1, x2):
    s = x1 + 0.3 * x2
    d = sum(k * (0.7**k) * s ** (k - 1) for k in range(2, _DEGREE + 1))
    return d, 0.3 * d


class LinearFocus(DeterministicFunction):
    """x' = A x, a saddle-focus whose stable manifold is a skewed plane."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        return [sum(_LINEAR[i, j] * point[j] for j in range(3)) for i in range(3)]


class CurvedFocus(DeterministicFunction):
    """y' = B y in x = (y1, y2, y3 + phi(y1, y2)): W^s(0) is x3 = phi(x1, x2)."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        x1, x2, x3 = point[0], point[1], point[2]
        d1 = -1.0 * x1 + 2.0 * x2
        d2 = -2.0 * x1 - 1.0 * x2
        p1, p2 = _dphi(x1, x2)
        return [d1, d2, 1.5 * (x3 - _phi(x1, x2)) + p1 * d1 + p2 * d2]


def _field(kind):
    return kind(
        variables=[State(), State(), State()],
        parameters=[Alpha(value=0.0)],
        results=[State(), State(), State()],
        time=None,
    )


def _chart(kind, order):
    function = _field(kind)
    jacobian = AutomaticDifferentiation().jacobian(function, np.zeros(3), 0.0)
    return TaylorManifold(order=order).compute(
        function,
        np.zeros(3),
        jacobian,
        StableManifold(),
    )


def _circle(chart, rho, count=24):
    return np.array(
        [
            chart.point([rho * math.cos(angle), rho * math.sin(angle)])
            for angle in np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
        ],
    )


def _graph_error(chart, rho):
    points = _circle(chart, rho)
    return float(np.max(np.abs(points[:, 2] - _phi(points[:, 0], points[:, 1]))))


class TestLinearSaddleFocus(TestCase):
    def test_the_chart_spans_the_plane_of_the_pair(self):
        # DEQ-71: the real parts of both eigenvectors of a pair are the same
        # vector, so the chart's two coordinates moved along one line.
        chart = _chart(LinearFocus, 5)
        step = 1.0e-6
        tangent = np.column_stack(
            [chart.point([step, 0.0]) / step, chart.point([0.0, step]) / step],
        )
        plane = _FRAME[:, :2]
        assert np.linalg.matrix_rank(tangent, tol=1.0e-6) == 2
        # Both columns lie in the plane of the pair.
        residual = tangent - plane @ np.linalg.lstsq(plane, tangent, rcond=None)[0]
        assert np.max(np.abs(residual)) < 1.0e-9

    def test_every_order_stays_in_the_plane(self):
        normal = np.cross(_FRAME[:, 0], _FRAME[:, 1])
        normal /= np.linalg.norm(normal)
        for order in (1, 3, 5, 7):
            assert np.max(np.abs(_circle(_chart(LinearFocus, order), 0.5) @ normal)) < (
                1.0e-14
            )

    def test_its_eigenvalues_are_the_pair(self):
        rates = _chart(LinearFocus, 3).eigenvalues
        assert rates[0] == pytest.approx(complex(-1.0, 2.0), abs=1.0e-12)
        assert rates[1] == pytest.approx(complex(-1.0, -2.0), abs=1.0e-12)


class TestCurvedSaddleFocus(TestCase):
    """Measured: at rho 0.4, 0.2, 0.1 the order-N chart is off the graph by
    1.1e-2, 1.2e-3, 1.5e-4 (N = 2) down to 6.9e-7, 1.3e-9, 2.6e-12 (N = 8):
    fitted slopes 3.13, 4.13, 5.13, 6.13, 7.12, 8.10, 9.00 for N = 2 to 8.
    From N = 9 the chart is the graph to 5e-17."""

    def test_the_error_falls_at_the_order_after_the_chart(self):
        radii = np.array([0.4, 0.2, 0.1])
        for order in range(2, _DEGREE):
            chart = _chart(CurvedFocus, order)
            errors = [_graph_error(chart, rho) for rho in radii]
            slope = np.polyfit(np.log(radii), np.log(errors), 1)[0]
            assert abs(slope - (order + 1)) < 0.2, (order, slope)

    def test_a_chart_of_the_surfaces_degree_is_the_surface(self):
        assert _graph_error(_chart(CurvedFocus, _DEGREE), 0.4) < 1.0e-15

    def test_the_chart_is_invariant_under_the_flow(self):
        # The orbit of P(theta) after 0.5 is P of theta turned and shrunk by
        # e^{lambda t}: measured 4.5e-7 at order 5 and 2.7e-14 at order 9.
        function = _field(CurvedFocus)
        for order, bound in ((5, 1.0e-6), (_DEGREE, 1.0e-12)):
            chart = _chart(CurvedFocus, order)
            theta = [0.2, 0.1]
            orbit = solve_ivp(
                lambda _t, u: function.eval(point=list(u)),
                (0.0, 0.5),
                chart.point(theta),
                rtol=1.0e-12,
                atol=1.0e-14,
            )
            assert np.linalg.norm(orbit.y[:, -1] - chart.flow_image(theta, 0.5)) < (
                bound
            )

    def test_a_resonance_is_refused(self):
        # All three eigenvalues are stable, and lambda + conj(lambda) = -2 is
        # within 1e-10 of the third: the z conj(z) coefficient's equation is
        # singular in all but rounding, which solves it to a coefficient of
        # 1e10 rather than failing.
        class Resonant(DeterministicFunction):
            def eval(self, point, time=None):  # noqa: ARG002 (base signature)
                x, y, z = point[0], point[1], point[2]
                return [-x + 2.0 * y, -2.0 * x - y, -(2.0 + 1.0e-10) * z + x * x]

        with pytest.raises(np.linalg.LinAlgError):
            _chart(Resonant, 3)


def _flat_chart(_self, _function, equilibrium, jacobian, selection):
    """The tangent plane, as the grower seeded a spiral manifold before."""
    values, vectors = np.linalg.eig(jacobian)
    first = vectors[:, selection.indices(values)[0]]
    frame = np.column_stack([first.real, first.imag])
    return (lambda theta: equilibrium + frame @ theta), "eigenspace"


def _grown(flat: bool):  # noqa: FBT001 (which seed, for the comparison)
    function = _field(CurvedFocus)
    jacobian = AutomaticDifferentiation().jacobian(function, np.zeros(3), 0.0)
    grower = GeodesicLevelSets(GeodesicSettings(0.1, 0.05, (0.025, 0.075)))
    if flat:
        with patch.object(GeodesicLevelSets, "_chart", _flat_chart):
            return grower.compute(
                function,
                np.zeros(3),
                jacobian,
                StableManifold(),
                0.3,
            )
    return grower.compute(function, np.zeros(3), jacobian, StableManifold(), 0.3)


def _ring_errors(mesh) -> np.ndarray:
    return np.array(
        [
            float(
                np.max(
                    np.abs(
                        mesh.vertices[ring][:, 2]
                        - _phi(mesh.vertices[ring][:, 0], mesh.vertices[ring][:, 1]),
                    ),
                ),
            )
            for ring in mesh.rings[:5]
        ],
    )


def test_the_grower_seeded_on_the_chart_is_ten_times_closer():
    # Measured, seed radius 0.1: the chart's first five rings are 1.6e-7 to
    # 6.4e-5 off the surface, the tangent plane's 1.1e-3 to 5.7e-3 - 31 to
    # 36,000 times further.
    curved, flat = _grown(flat=False), _grown(flat=True)
    assert curved.seed == "taylor"
    assert flat.seed == "eigenspace"
    assert np.all(_ring_errors(flat) > 10.0 * _ring_errors(curved))


def test_the_stage_fan_starts_on_a_circle_not_a_line():
    # DEQ-71: the twelve starts of the stable surface's fan, from the chart
    # at radius 1e-3, collapsed to six distinct points on one line.
    function = _field(LinearFocus)
    starts = _manifold_starts(function, np.zeros(3), StageSettings())
    fan = np.array([start for kind, start, _ in starts if kind == "stable-surface"])
    assert len(fan) == 12
    assert len({tuple(np.round(start, 12)) for start in fan}) == 12
    assert np.linalg.matrix_rank(fan, tol=1.0e-9) == 2


# A second focus, independent of the one above: lambda = -0.3 +- 5i and 0.6,
# in a skewed frame, with a cubic radial term in the pair's plane:
#   x' = -0.3 x - 5 y + x r^2,  y' = 5 x - 0.3 y + y r^2,  w' = 0.6 w + r^2.
# Its stable manifold is the graph w = h(r^2) with h solving
# 2 s (s - 0.3) h' = 0.6 h + s, which is not a polynomial:
#   h(s) = (0.3 - s) / (2 s) * (-e / (1 - e) - log(1 - e)),  e = s / 0.3.
_SKEW = np.array([[1.0, 0.5, -0.4], [0.3, 1.2, 0.6], [-0.7, 0.2, 0.9]])
_UNSKEW = np.linalg.inv(_SKEW)


def _skewed(rates):
    """``x' = F g(F^-1 x)`` for ``g`` the base field ``rates`` returns."""

    class Skewed(DeterministicFunction):
        def eval(self, point, time=None):  # noqa: ARG002 (base signature)
            base = [sum(_UNSKEW[i, j] * point[j] for j in range(3)) for i in range(3)]
            image = rates(*base)
            return [sum(_SKEW[i, j] * image[j] for j in range(3)) for i in range(3)]

    return Skewed


def _cubic_focus(x, y, w):
    r2 = x * x + y * y
    return [-0.3 * x - 5.0 * y + x * r2, 5.0 * x - 0.3 * y + y * r2, 0.6 * w + r2]


def _height(s):
    """h(s), summed as its series in e = s / 0.3, which avoids the cancellation."""
    e = s / 0.3
    n = np.arange(2, 400)
    return (0.3 - s) / (2.0 * s) * -np.sum((1.0 - 1.0 / n) * e**n)


def _off_the_log_surface(chart, rho):
    errors = []
    for angle in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
        x, y, w = _UNSKEW @ chart.point([rho * math.cos(angle), rho * math.sin(angle)])
        errors.append(abs(w - _height(x * x + y * y)))
    return max(errors)


class TestFocusOnANonPolynomialSurface(TestCase):
    """Measured: the order-N chart is off the surface as rho^(N+1) for odd N
    and rho^(N+2) for even N (the field is odd in the pair's plane, so the
    surface is even in r): slopes 2.01, 4.01, 4.01, 5.99, 6.01, 7.99, 8.01,
    9.98 for N = 1 to 8. Flowed by DOP853 for t = 1, the chart point lands on
    the chart's own image within rho^(N+1): halving theta divides the error
    by 2^2.00 ... 2^9.00 for N = 1 to 8."""

    def setUp(self):
        self.function = _field(_skewed(_cubic_focus))

    def _chart(self, order):
        jacobian = AutomaticDifferentiation().jacobian(self.function, np.zeros(3), 0.0)
        return TaylorManifold(order=order).compute(
            self.function,
            np.zeros(3),
            jacobian,
            StableManifold(),
        )

    def test_the_error_falls_at_the_surfaces_order(self):
        radii = np.array([0.05, 0.1, 0.2])
        for order in range(1, 8):
            chart = self._chart(order)
            errors = [_off_the_log_surface(chart, rho) for rho in radii]
            slope = np.polyfit(np.log(radii), np.log(errors), 1)[0]
            expected = order + 1 if order % 2 else order + 2
            assert abs(slope - expected) < 0.1, (order, slope)

    def test_the_chart_is_invariant_under_the_true_flow(self):
        for order in (1, 2, 4, 6):
            chart = self._chart(order)
            misses = []
            for theta in ([0.05, 0.03], [0.025, 0.015]):
                orbit = solve_ivp(
                    lambda _t, u: self.function.eval(point=list(u)),
                    (0.0, 1.0),
                    chart.point(theta),
                    method="DOP853",
                    rtol=1.0e-13,
                    atol=1.0e-16,
                )
                image = chart.flow_image(theta, 1.0)
                misses.append(np.linalg.norm(orbit.y[:, -1] - image))
            assert math.log2(misses[0] / misses[1]) == pytest.approx(order + 1, abs=0.1)


def _resonant(forced_along_w, forced_in_plane):
    """The focus -0.3 +- 5i with a stable third rate of exactly -0.6.

    ``z conj(z)`` has the rate ``lambda + conj(lambda) = -0.6``: its
    homological matrix is singular along ``w``. Forced along ``w`` there is
    no chart; forced only in the pair's plane, which is the matrix's range,
    the least-squares term is one.
    """

    def rates(x, y, w):
        r2 = x * x + y * y
        return [
            -0.3 * x - 5.0 * y + forced_in_plane * r2,
            5.0 * x - 0.3 * y,
            -0.6 * w + forced_along_w * r2,
        ]

    return _field(_skewed(rates))


def _resonant_chart(function):
    jacobian = AutomaticDifferentiation().jacobian(function, np.zeros(3), 0.0)
    return TaylorManifold(order=5).compute(
        function,
        np.zeros(3),
        jacobian,
        StableManifold(),
    )


class TestAResonanceThatCanBeSolved(TestCase):
    """The least-squares branch: a resonance whose remainder is in range."""

    def _miss(self, function, chart):
        theta = [0.02, 0.01, 0.015]
        orbit = solve_ivp(
            lambda _t, u: function.eval(point=list(u)),
            (0.0, 1.0),
            chart.point(theta),
            method="DOP853",
            rtol=1.0e-13,
            atol=1.0e-16,
        )
        return np.linalg.norm(orbit.y[:, -1] - chart.flow_image(theta, 1.0))

    def test_an_unforced_resonance_charts_the_linear_flow(self):
        # Measured: 7.9e-16.
        function = _resonant(0.0, 0.0)
        assert self._miss(function, _resonant_chart(function)) < 1.0e-13

    def test_a_resonance_forced_in_its_range_is_charted(self):
        # Measured: 1.0e-15.
        function = _resonant(0.0, 1.0)
        assert self._miss(function, _resonant_chart(function)) < 1.0e-13

    def test_a_resonance_forced_along_its_kernel_is_refused(self):
        with pytest.raises(np.linalg.LinAlgError, match=r"\(1, 1, 0\)"):
            _resonant_chart(_resonant(1.0, 0.0))
