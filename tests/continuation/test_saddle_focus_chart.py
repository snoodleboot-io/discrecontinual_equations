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

from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.manifold import (
    StableManifold,
    TaylorManifold,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.webplot.stage_builder import saddle_manifolds
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


def test_the_stage_fan_starts_on_a_circle_not_a_line():
    # DEQ-71: the twelve starts of the stable surface's fan, from the chart
    # at radius 1e-3, collapsed to six distinct points on one line.
    function = _field(LinearFocus)
    starts = []

    def capture(_function, start, *_rest):
        starts.append(np.array(start, dtype=float))
        return [[0.0, 0.0, 0.0]]

    with patch("discrecontinual_equations.webplot.stage_builder._flow", capture):
        saddle_manifolds(function, np.zeros(3), [(-1.0, 1.0)] * 3)
    fan = np.array(starts[2:])
    assert len(fan) == 12
    assert len({tuple(np.round(start, 12)) for start in fan}) == 12
    assert np.linalg.matrix_rank(fan, tol=1.0e-9) == 2
