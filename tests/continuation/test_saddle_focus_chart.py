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
        # the third: the z conj(z) coefficient has no equation to solve.
        class Resonant(DeterministicFunction):
            def eval(self, point, time=None):  # noqa: ARG002 (base signature)
                x, y, z = point[0], point[1], point[2]
                return [-x + 2.0 * y, -2.0 * x - y, -2.0 * z + x * x]

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
