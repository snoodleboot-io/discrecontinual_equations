"""Cusp and branch-point coefficients on coupled two-dimensional systems.

Every system is written in coordinates ``(u, v)`` with a stable mode
``v' = -v + u^2`` that the critical mode drives, and then mixed by a fixed
non-orthogonal ``x = M (u, v)``, so the null vectors have both components and the
center manifold is curved. The reference values are a center-manifold
(Poincare-Dulac) reduction done in closed form on ``(u, v)`` and checked
symbolically (``outputs/verification/deq-102/reference.py``); none of them comes
from the library.

* Cusp system ``u' = b1 + b2 u + 3 u v - u^3``: fold curve
  ``(b1, b2) = (4 s^3, -6 s^2)`` at ``u = s``, with ``q = (1, 2s)``,
  ``p = (1, 3s)`` and ``a = 6 s / (1 + 6 s^2)``; one cusp, at ``s = 0``.
* No-cusp system ``u' = b1 + b2 u - u^2 - u v``: fold curve
  ``(b1, b2) = (-s^2 - 2 s^3, 2 s + 3 s^2)``, ``q = (1, 2s)``, ``p = (1, -s)`` and
  ``a = -(1 + 3 s) / (1 - 2 s^2)``, which keeps its sign for ``s > -1/3``.
* Branch points ``u' = mu u + u v + k u^3`` at the origin: ``a = 0`` by the
  ``u -> -u`` symmetry and the center-manifold cubic is ``c = 1 + k``, while
  ``<p, C(q, q, q)> / 6 = k`` alone.

In ``x``, with ``q_x = M q / |M q|`` (``lambda = 1 / |M q|``), the coefficients
become ``a_x = lambda a`` and ``c_x = lambda^2 c``.

The singular vectors of an SVD are defined only up to sign, so
:func:`_flipping_svd` negates random pairs ``(u_i, v_i)`` of every SVD the library
takes (a valid SVD all the same): nothing reported may depend on which signs
LAPACK happened to return.
"""

import math
from unittest import TestCase, mock

import numpy as np

from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.codim2 import Codim2Analyzer, Cusp
from discrecontinual_equations.continuation.codim2_builder import (
    Codim2Driver,
    CurveSeed,
)
from discrecontinual_equations.continuation.codim2_config import Codim2Config
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.continuation.continuation_state import (
    ContinuationState,
)
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.detection import (
    BranchPoint,
    DetectionContext,
    _branch_coefficients,
)
from discrecontinual_equations.continuation.fold_curve import FoldCurveResidual
from discrecontinual_equations.continuation.jacobian import JacobianProvider
from discrecontinual_equations.continuation.normal_form import cusp_coefficient
from discrecontinual_equations.continuation.two_parameter_problem import (
    CurveDecoded,
    TwoParameterProblem,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

MIX = np.array([[1.0, 0.6], [-0.4, 1.0]])
UNMIX = np.linalg.inv(MIX)
SEEDS = range(8)
LOCATION_TOLERANCE = 2.0e-2
_SVD = np.linalg.svd


def _flipping_svd(seed: int):
    """``np.linalg.svd`` with each singular pair's sign drawn at random."""
    generator = np.random.default_rng(seed)

    def svd(matrix, *args, **kwargs):
        result = _SVD(matrix, *args, **kwargs)
        if not kwargs.get("compute_uv", True):
            return result
        left, values, right = result
        left_signs = generator.choice([-1.0, 1.0], size=left.shape[-1])
        right_signs = generator.choice([-1.0, 1.0], size=right.shape[-2])
        shared = values.shape[-1]
        right_signs[:shared] = left_signs[:shared]
        return left * left_signs, values, right * right_signs[:, None]

    return svd


def _flipped(seed: int):
    return mock.patch.object(np.linalg, "svd", _flipping_svd(seed))


class First(Parameter, name="First parameter", abbreviation="b1"):
    pass


class Second(Parameter, name="Second parameter", abbreviation="b2"):
    pass


class State(Variable, name="State", abbreviation="x"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


def _mixed(point, rows):
    """``M f(M^-1 x)``: ``rows`` maps ``(u, v)`` to ``(u', v')``."""
    u = UNMIX[0, 0] * point[0] + UNMIX[0, 1] * point[1]
    v = UNMIX[1, 0] * point[0] + UNMIX[1, 1] * point[1]
    du, dv = rows(u, v)
    return [MIX[0, 0] * du + MIX[0, 1] * dv, MIX[1, 0] * du + MIX[1, 1] * dv]


class CoupledCusp(DeterministicFunction):
    """``u' = b1 + b2 u + 3 u v - u^3``, ``v' = -v + u^2``, mixed by ``M``."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        return _mixed(
            point,
            lambda u, v: (b1 + b2 * u + 3.0 * u * v - u * u * u, -v + u * u),
        )


class CoupledFold(DeterministicFunction):
    """``u' = b1 + b2 u - u^2 - u v``, ``v' = -v + u^2``, mixed by ``M``."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        return _mixed(
            point,
            lambda u, v: (b1 + b2 * u - u * u - u * v, -v + u * u),
        )


class CoupledBranch(DeterministicFunction):
    """``u' = b1 u + u v + k u^3``, ``v' = -v + u^2``, mixed by ``M``."""

    cubic = 0.0

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        k = type(self).cubic
        return _mixed(
            point,
            lambda u, v: (mu * u + u * v + k * u * u * u, -v + u * u),
        )


class BranchNoCubic(CoupledBranch):
    """Center-manifold cubic ``c = 1``, though ``<p, C(q, q, q)> = 0``."""

    cubic = 0.0


class BranchCancelled(CoupledBranch):
    """Center-manifold cubic ``c = 0``, though ``<p, C(q, q, q)> / 6 = -1``."""

    cubic = -1.0


class BranchReinforced(CoupledBranch):
    """Center-manifold cubic ``c = 3``, ``<p, C(q, q, q)> / 6 = 2``."""

    cubic = 2.0


def _equation(function_type, first: float, second: float) -> DifferentialEquation:
    parameters = [First(value=first), Second(value=second)]
    variables = [State(), State()]
    derivative = function_type(
        variables=variables,
        parameters=parameters,
        results=[State(), State()],
        time=None,
    )
    return DifferentialEquation(
        variables=[State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=derivative,
    )


def _problem(function_type) -> TwoParameterProblem:
    return TwoParameterProblem(
        _equation(function_type, 0.0, 0.0),
        0,
        1,
        0.0,
        AutomaticDifferentiation(),
    )


def _cusp_fold(s: float) -> tuple[np.ndarray, float, float, float]:
    """State, ``b1``, ``b2`` and the reference ``a`` (``q_x`` dominant > 0)."""
    state = MIX @ np.array([s, s * s])
    image = MIX @ np.array([1.0, 2.0 * s])
    scale = 1.0 / np.linalg.norm(image)
    sign = math.copysign(1.0, image[np.argmax(np.abs(image))])
    return state, 4.0 * s**3, -6.0 * s**2, sign * scale * 6.0 * s / (1 + 6 * s * s)


def _plain_fold(s: float) -> tuple[np.ndarray, float, float, float]:
    state = MIX @ np.array([s, s * s])
    image = MIX @ np.array([1.0, 2.0 * s])
    scale = 1.0 / np.linalg.norm(image)
    sign = math.copysign(1.0, image[np.argmax(np.abs(image))])
    reference = -sign * scale * (1.0 + 3.0 * s) / (1.0 - 2.0 * s * s)
    return state, -(s**2) - 2.0 * s**3, 2.0 * s + 3.0 * s**2, reference


class TestCuspCoefficientOrientation(TestCase):
    def test_matches_the_center_manifold_reference(self):
        for function_type, fold in (
            (CoupledCusp, _cusp_fold),
            (CoupledFold, _plain_fold),
        ):
            problem = _problem(function_type)
            for s in (-0.3, -0.12, 0.05, 0.2, 0.3):
                state, b1, b2, reference = fold(s)
                value = cusp_coefficient(problem, state, b1, b2)
                assert abs(value - reference) < 1.0e-9 * max(1.0, abs(reference))

    def test_does_not_follow_the_svd_signs(self):
        problem = _problem(CoupledCusp)
        state, b1, b2, reference = _cusp_fold(0.2)
        for seed in SEEDS:
            with _flipped(seed):
                values = [cusp_coefficient(problem, state, b1, b2) for _ in range(4)]
            assert all(abs(value - reference) < 1.0e-9 for value in values)

    def test_orientation_picks_the_null_vector_sign(self):
        # a is odd in q once <p, q> = 1: the orientation names which q is meant.
        problem = _problem(CoupledCusp)
        state, b1, b2, reference = _cusp_fold(0.2)
        image = MIX @ np.array([1.0, 0.4])
        along = cusp_coefficient(problem, state, b1, b2, orientation=image)
        against = cusp_coefficient(problem, state, b1, b2, orientation=-image)
        assert abs(along - abs(reference)) < 1.0e-9
        assert abs(against + abs(reference)) < 1.0e-9

    def test_cusp_test_function_follows_the_curve_null_vector(self):
        # Along a fold curve the Moore-Spence q (c . q = 1) fixes the orientation,
        # whatever the SVD returns.
        problem = _problem(CoupledFold)
        detector = Cusp()
        state, b1, b2, _reference = _plain_fold(0.1)
        null_vector = MIX @ np.array([1.0, 0.2])
        decoded = CurveDecoded(state, b1, b2, None, null_vector=null_vector)
        flipped = CurveDecoded(state, b1, b2, None, null_vector=-null_vector)
        for seed in SEEDS:
            with _flipped(seed):
                assert detector.value(problem, decoded) < 0.0
                assert detector.value(problem, flipped) > 0.0

    def test_cusp_test_function_tracks_by_continuity_without_a_null_vector(self):
        problem = _problem(CoupledFold)
        for seed in SEEDS:
            detector = Cusp()
            with _flipped(seed):
                signs = set()
                for s in np.linspace(-0.2, 0.3, 11):
                    state, b1, b2, _reference = _plain_fold(float(s))
                    decoded = CurveDecoded(state, b1, b2, None)
                    signs.add(math.copysign(1.0, detector.value(problem, decoded)))
            assert len(signs) == 1


class _Recorder(Cusp):
    """The cusp test function, recording the orientation each scan starts from."""

    def __init__(self) -> None:
        super().__init__()
        self.starts: list[np.ndarray | None] = []

    def reset(self) -> None:
        super().reset()
        self.starts.append(self._previous)


class _Points:
    def __init__(self, points: list[tuple[np.ndarray, float, float]]) -> None:
        self.points = [_Point(state, b1, b2) for state, b1, b2 in points]


class _Point:
    def __init__(self, state: np.ndarray, b1: float, b2: float) -> None:
        self.state = [*state, b1]
        self.parameter = b2


class _Decoder:
    def decode(self, state: np.ndarray, parameter: float) -> CurveDecoded:
        return CurveDecoded(state[:2], float(state[2]), parameter, None)


class TestCuspScan(TestCase):
    def test_fold_curve_decodes_its_null_vector(self):
        # The Moore-Spence q (c . q = 1) is what orients the cusp test on a fold
        # curve, so the decoded point must carry it.
        residual = FoldCurveResidual(_problem(CoupledFold), 2, np.array([1.0, 0.0]))
        state = np.array([0.1, 0.2, 0.3, -0.4, 0.5])
        decoded = residual.decode(state, 0.6)
        assert decoded.null_vector is not None
        assert np.array_equal(decoded.null_vector, [0.3, -0.4])

    def test_each_scan_starts_without_an_orientation(self):
        problem = _problem(CoupledFold)
        detector = _Recorder()
        analyzer = Codim2Analyzer([detector])
        curve = _Points([_plain_fold(float(s))[:3] for s in (0.0, 0.1, 0.2)])
        for _ in range(2):
            analyzer.analyze(curve, _Decoder(), problem)
        assert detector.starts == [None, None]


class TestCuspDetectionUnderSvdFlips(TestCase):
    def _fold_curve(self, function_type, seed_s, fold, bounds, index):
        state, b1, b2, _reference = fold(seed_s)
        parameters = (b1, b2)
        equation = _equation(function_type, b1, b2)
        config = Codim2Config(
            continuation_parameter_index=1 - index,
            second_parameter_index=index,
            curve="fold",
            codim2_detectors=["cusp"],
            initial_parameter=parameters[index],
            direction=1,
            parameter_lower_bound=bounds[0],
            parameter_upper_bound=bounds[1],
        )
        seed = CurveSeed(state=state.tolist(), parameter_a=parameters[1 - index])
        return Codim2Driver.run(config, equation, seed)

    def test_fold_curve_without_a_cusp_reports_none(self):
        for seed in SEEDS:
            with _flipped(seed):
                result = self._fold_curve(
                    CoupledFold,
                    0.0,
                    _plain_fold,
                    (-0.3, 0.9),
                    1,
                )
            assert len(result.curve.points) > 10
            assert [p for p in result.points if p.kind == "cusp"] == []

    def test_fold_curve_with_a_cusp_locates_it_once(self):
        for seed in SEEDS:
            with _flipped(seed):
                result = self._fold_curve(
                    CoupledCusp,
                    -0.4,
                    _cusp_fold,
                    (-0.3, 0.3),
                    0,
                )
            cusps = [p for p in result.points if p.kind == "cusp"]
            assert len(cusps) == 1
            assert abs(cusps[0].parameter_a) < LOCATION_TOLERANCE
            assert abs(cusps[0].parameter_b) < LOCATION_TOLERANCE
            assert cusps[0].classification == "c>0"


class _Exact(JacobianProvider):
    """Closed-form Jacobian of ``u' = mu u + r u^2 + u v + k u^3`` in ``x``.

    ``v' = -v + u^2`` as before; it shares no code with the library.
    """

    def __init__(self, cubic: float, quadratic: float = 0.0) -> None:
        self._cubic = cubic
        self._quadratic = quadratic

    def state_jacobian(self, state: np.ndarray, parameter: float) -> np.ndarray:
        u, v = UNMIX @ state
        inner = np.array(
            [
                [
                    parameter
                    + 2.0 * self._quadratic * u
                    + v
                    + 3.0 * self._cubic * u * u,
                    u,
                ],
                [2.0 * u, -1.0],
            ],
        )
        return MIX @ inner @ UNMIX

    def parameter_derivative(self, state: np.ndarray, parameter: float) -> np.ndarray:
        _ = parameter
        u, _v = UNMIX @ state
        return MIX @ np.array([u, 0.0])


def _context(cubic: float, quadratic: float = 0.0) -> DetectionContext:
    return DetectionContext(
        ContinuationState(np.zeros(2), 0.0, np.array([0.0, 0.0, 1.0]), 0.0),
        _Exact(cubic, quadratic),
    )


def _branch_kind(cubic: float) -> str:
    return BranchPoint().refine_kind(_context(cubic))


class TestBranchPointClassification(TestCase):
    def test_coefficients_match_the_center_manifold_reference(self):
        # u' = r u^2 + c_u u^3 on the center manifold, in u = lambda w:
        # <p, B(q, q)> = 2 r lambda and <p, C + 3 B(q, h2)> = 6 c_u lambda^2.
        # (r, k) -> c_u: (0, 0) -> 1, (0, -1) -> 0, (0, 2) -> 3, (1, 0) -> 1.
        scale = 1.0 / np.linalg.norm(MIX[:, 0])
        cases = {(0.0, 0.0): 1.0, (0.0, -1.0): 0.0, (0.0, 2.0): 3.0, (1.0, 0.0): 1.0}
        for (quadratic, cubic), reduced in cases.items():
            for seed in SEEDS:
                with _flipped(seed):
                    coefficients = _branch_coefficients(_context(cubic, quadratic))
                assert coefficients is not None
                a, c = coefficients
                assert abs(abs(a) - 2.0 * quadratic * scale) < 1.0e-7
                assert abs(c - 6.0 * reduced * scale * scale) < 1.0e-6

    def test_pitchfork_needs_the_center_manifold_cubic(self):
        # <p, C(q, q, q)> = 0, but the center manifold gives c = 1.
        assert _branch_kind(0.0) == "pitchfork"

    def test_cancelled_cubic_is_degenerate(self):
        # <p, C(q, q, q)> / 6 = -1, cancelled by 3 <p, B(q, h2)> / 6 = 1.
        assert _branch_kind(-1.0) == "branch_point"

    def test_reinforced_cubic_is_a_pitchfork(self):
        assert _branch_kind(2.0) == "pitchfork"

    def test_classification_survives_svd_flips(self):
        for seed in SEEDS:
            with _flipped(seed):
                kinds = [_branch_kind(cubic) for cubic in (0.0, -1.0, 2.0)]
            assert kinds == ["pitchfork", "branch_point", "pitchfork"]

    def test_continuation_labels_the_coupled_branch_points(self):
        expected = {
            BranchNoCubic: "pitchfork",
            BranchCancelled: "branch_point",
            BranchReinforced: "pitchfork",
        }
        for function_type, kind in expected.items():
            equation = _equation(function_type, -0.5, 0.0)
            config = ContinuationConfig(
                detectors=["branch_point"],
                initial_parameter=-0.5,
                measure="component",
                initial_step=0.04,
                maximum_step=0.12,
                parameter_lower_bound=-0.5,
                parameter_upper_bound=0.5,
            )
            continuer = ContinuerBuilder.build(config, equation)
            branch = continuer.solve(equation, initial_values=[0.0, 0.0])
            kinds = [
                point.kind
                for point in branch.special_points
                if abs(point.parameter) < LOCATION_TOLERANCE
            ]
            assert kinds == [kind], (function_type.__name__, kinds)
