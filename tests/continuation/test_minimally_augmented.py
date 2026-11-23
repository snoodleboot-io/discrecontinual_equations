"""Minimally augmented cusp and generalized-Hopf curves (DEQ-51).

The analytic Jacobian columns are checked against central differences of the very
same residual at points off the curve, with random borders, on dense fields hidden
behind a random similarity (so the adjoint overlap of the Hopf eigenvectors is not
real). The cost claim is checked by counting field evaluations per Newton
iteration: they must grow linearly in ``n``, so with an ``O(n^2)`` field the
iteration is ``O(n^3)``. The located codim-3 points are checked in
:mod:`tests.continuation.test_codim3`.
"""

from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.cusp_curve import CuspCurveResidual
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.generalized_hopf_curve import (
    GeneralizedHopfCurveResidual,
)
from discrecontinual_equations.continuation.jacobian import (
    AnalyticJacobian,
    FiniteDifferenceJacobian,
)
from discrecontinual_equations.continuation.normal_form import first_lyapunov
from discrecontinual_equations.continuation.residual import EquationResidual
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable
from tests.continuation.normal_form_systems import random_transform
from tests.continuation.test_codim3 import (
    CoupledExtendedBautin,
    CoupledSwallowtail,
    _coupled_cusp_point,
)
from tests.continuation.test_codim3 import _equation as _codim3_equation

_STEP = 1.0e-6
_REFERENCE_STEP = 1.0e-4
_AGREEMENT = 1.0e-7


class First(Parameter, name="First", abbreviation="b1"):
    pass


class Second(Parameter, name="Second", abbreviation="b2"):
    pass


class Third(Parameter, name="Third", abbreviation="b3"):
    pass


class State(Variable, name="State", abbreviation="s"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


class _Dense(DeterministicFunction):
    """``X' = T g(T^-1 X - c)`` with every parameter entering nonlinearly."""

    transform = np.eye(2)
    centre = np.zeros(2)
    calls = 0

    def core(self, z: list, b: list) -> list:
        raise NotImplementedError

    def eval(
        self,
        point: list,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list:
        cls = type(self)
        cls.calls += 1
        inverse = np.linalg.inv(cls.transform)
        size = cls.transform.shape[0]
        z = [
            sum(float(inverse[i, k]) * point[k] for k in range(size))
            - float(cls.centre[i])
            for i in range(size)
        ]
        field = self.core(z, [parameter.value for parameter in self.parameters])
        return [
            sum(float(cls.transform[i, k]) * field[k] for k in range(size))
            for i in range(size)
        ]


class DenseCusp(_Dense):
    """A swallowtail block with a slaved, feeding-back stable chain."""

    def core(self, z: list, b: list) -> list:
        x, s = z[0], z[1]
        rest = [
            -(1.0 + 0.3 * i) * z[i] + x * x + 0.4 * s * z[i] + b[2] * x * z[i]
            for i in range(2, len(z))
        ]
        return [
            b[0] + b[1] * x + b[2] * x * x - x**4 + x * s + 0.2 * x * z[-1],
            x * x - s + b[0] * s,
            *rest,
        ]


class DenseBautin(_Dense):
    """An extended Bautin block driving, and driven by, a stable chain."""

    def core(self, z: list, b: list) -> list:
        x, y = z[0], z[1]
        radius = x * x + y * y
        rate = b[0] + b[1] * radius + b[2] * radius * radius - radius**3
        rest = [
            -(1.0 + 0.3 * i) * z[i] + x * x + y * z[i] + b[1] * x * y
            for i in range(2, len(z))
        ]
        return [
            rate * x - y + 0.3 * x * z[-1],
            x + rate * y - 0.2 * y * z[-1],
            *rest,
        ]


def _equation(field: type[_Dense], size: int) -> DifferentialEquation:
    parameters = [First(value=0.1), Second(value=-0.2), Third(value=0.3)]
    derivative = field(
        variables=[State() for _ in range(size)],
        parameters=parameters,
        results=[State() for _ in range(size)],
        time=None,
    )
    return DifferentialEquation(
        variables=[State() for _ in range(size)],
        time=Time(),
        parameters=parameters,
        derivative=derivative,
    )


def _cusp(size: int, seed: int) -> tuple[CuspCurveResidual, np.ndarray, float]:
    generator = np.random.default_rng(seed)
    DenseCusp.transform = random_transform(size, seed)
    DenseCusp.centre = 0.2 * generator.normal(size=size)
    equation = _equation(DenseCusp, size)
    problem = TwoParameterProblem(equation, 0, 2, 0.0, AutomaticDifferentiation())
    residual = CuspCurveResidual(
        problem,
        size,
        generator.normal(size=size),
        generator.normal(size=size),
        equation.derivative.parameters[1],
    )
    state = np.concatenate([0.4 * generator.normal(size=size), [0.1, 0.25]])
    return residual, state, -0.3


def _bautin(
    size: int,
    seed: int,
) -> tuple[GeneralizedHopfCurveResidual, np.ndarray, float]:
    generator = np.random.default_rng(seed)
    DenseBautin.transform = random_transform(size, seed)
    DenseBautin.centre = 0.2 * generator.normal(size=size)
    equation = _equation(DenseBautin, size)
    problem = TwoParameterProblem(equation, 0, 1, 0.0, AutomaticDifferentiation())
    residual = GeneralizedHopfCurveResidual(
        problem,
        size,
        generator.normal(size=(size, 2)),
        generator.normal(size=(size, 2)),
        equation.derivative.parameters[2],
    )
    equilibrium = DenseBautin.transform @ DenseBautin.centre
    state = np.concatenate(
        [equilibrium + 0.05 * generator.normal(size=size), [1.1, 0.05, 0.1]],
    )
    return residual, state, 0.4


def _reference(residual, state: np.ndarray, parameter: float) -> np.ndarray:
    """Central differences, Richardson-extrapolated to fourth order."""
    coarse = FiniteDifferenceJacobian(residual, _REFERENCE_STEP)
    fine = FiniteDifferenceJacobian(residual, _REFERENCE_STEP / 2.0)
    return (
        4.0 * fine.extended_jacobian(state, parameter)
        - coarse.extended_jacobian(state, parameter)
    ) / 3.0


def _relative_error(analytic: np.ndarray, reference: np.ndarray) -> float:
    scale = np.maximum(np.abs(reference).max(axis=1, keepdims=True), 1.0)
    return float(np.max(np.abs(analytic - reference) / scale))


class TestAnalyticColumns(TestCase):
    def test_cusp_rows_match_differences_of_the_residual(self):
        for size, seed in [(2, 1), (5, 2), (6, 3)]:
            residual, state, parameter = _cusp(size, seed)
            analytic = AnalyticJacobian(residual, _STEP).extended_jacobian(
                state,
                parameter,
            )
            reference = _reference(residual, state, parameter)
            assert _relative_error(analytic, reference) < _AGREEMENT, (size, seed)

    def test_generalized_hopf_rows_match_differences_of_the_residual(self):
        for size, seed in [(2, 4), (4, 5), (5, 6)]:
            residual, state, parameter = _bautin(size, seed)
            residual.adapt(state, parameter)
            analytic = AnalyticJacobian(residual, _STEP).extended_jacobian(
                state,
                parameter,
            )
            reference = _reference(residual, state, parameter)
            assert _relative_error(analytic, reference) < _AGREEMENT, (size, seed)

    def test_the_test_function_is_first_lyapunov_at_sqrt_kappa(self):
        residual, state, parameter = _bautin(4, 7)
        value = residual.evaluate(state, parameter)[-1]
        equation = _equation(DenseBautin, 4)
        equation.derivative.parameters[2].value = parameter
        problem = TwoParameterProblem(equation, 0, 1, 0.0, AutomaticDifferentiation())
        expected = first_lyapunov(
            problem,
            state[:4],
            state[5],
            state[6],
            float(np.sqrt(state[4])),
        )
        assert value == expected

    def test_analytic_jacobian_needs_a_differentiable_residual(self):
        equation = _equation(DenseCusp, 2)
        with pytest.raises(TypeError):
            AnalyticJacobian(EquationResidual(equation, 0, 0.0), _STEP)


class TestBorders(TestCase):
    def test_adapting_the_borders_keeps_the_zero_set(self):
        # The unknowns hold no auxiliary vectors, so a curve point stays a zero of
        # the system when the borders move. Exact curve points: the coupled
        # swallowtail's cusp curve at x = t, and the Bautin curve b1 = b2 = 0.
        g, t = CoupledSwallowtail.feedback, 0.6
        b1, b2, b3 = _coupled_cusp_point(t, g)
        equation = _codim3_equation(CoupledSwallowtail, 2, b1, b2, b3)
        problem = TwoParameterProblem(equation, 0, 2, 0.0, AutomaticDifferentiation())
        generator = np.random.default_rng(12)
        cusp = CuspCurveResidual(
            problem,
            2,
            generator.normal(size=2),
            generator.normal(size=2),
            equation.derivative.parameters[1],
        )
        equilibrium = CoupledSwallowtail.transform @ np.array([t, t * t])
        on_cusp = np.concatenate([equilibrium, [b1, b3]])
        equation = _codim3_equation(CoupledExtendedBautin, 3, 0.0, 0.0, 0.4)
        problem = TwoParameterProblem(equation, 0, 1, 0.0, AutomaticDifferentiation())
        bautin = GeneralizedHopfCurveResidual(
            problem,
            3,
            generator.normal(size=(3, 2)),
            generator.normal(size=(3, 2)),
            equation.derivative.parameters[2],
        )
        on_bautin = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        for residual, state, parameter in [
            (cusp, on_cusp, b2),
            (bautin, on_bautin, 0.4),
        ]:
            assert np.linalg.norm(residual.evaluate(state, parameter)) < 1.0e-12
            residual.adapt(state, parameter)
            assert np.linalg.norm(residual.evaluate(state, parameter)) < 1.0e-12


def _conditioning(residual, state: np.ndarray, parameter: float) -> float:
    return float(
        np.linalg.cond(
            AnalyticJacobian(residual, _STEP).state_jacobian(
                state,
                parameter,
            ),
        ),
    )


def _nearly_orthogonal(vectors: np.ndarray, generator) -> np.ndarray:
    """A border almost orthogonal to ``vectors``, so the bordering is near-singular."""
    basis = np.linalg.qr(vectors.reshape(vectors.shape[0], -1))[0]
    border = generator.normal(size=basis.shape)
    border = border - basis @ (basis.T @ border)
    return border + 1.0e-7 * basis


class TestAdaptation(TestCase):
    def test_adapting_either_cusp_border_restores_the_conditioning(self):
        g, t = CoupledSwallowtail.feedback, 0.6
        b1, b2, b3 = _coupled_cusp_point(t, g)
        equation = _codim3_equation(CoupledSwallowtail, 2, b1, b2, b3)
        problem = TwoParameterProblem(equation, 0, 2, 0.0, AutomaticDifferentiation())
        state = np.concatenate(
            [CoupledSwallowtail.transform @ np.array([t, t * t]), [b1, b3]],
        )
        matrix = problem.state_jacobian(state[:2], b1, b3)
        right = np.linalg.svd(matrix)[2][-1]
        left = np.linalg.svd(matrix.T)[2][-1]
        generator = np.random.default_rng(13)
        for borders in [
            (_nearly_orthogonal(right, generator).ravel(), left),
            (right, _nearly_orthogonal(left, generator).ravel()),
        ]:
            residual = CuspCurveResidual(
                problem,
                2,
                *borders,
                equation.derivative.parameters[1],
            )
            before = _conditioning(residual, state, b2)
            residual.adapt(state, b2)
            after = _conditioning(residual, state, b2)
            assert after < 1.0e-4 * before, (before, after)

    def test_adapting_the_hopf_borders_restores_the_conditioning(self):
        equation = _codim3_equation(CoupledExtendedBautin, 3, 0.0, 0.0, 0.4)
        problem = TwoParameterProblem(equation, 0, 1, 0.0, AutomaticDifferentiation())
        matrix = problem.state_jacobian(np.zeros(3), 0.0, 0.0)
        square = matrix @ matrix + np.eye(3)
        right = np.linalg.svd(square)[2][-2:].T
        left = np.linalg.svd(square.T)[2][-2:].T
        generator = np.random.default_rng(14)
        state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        for borders in [
            (_nearly_orthogonal(right, generator), left),
            (right, _nearly_orthogonal(left, generator)),
        ]:
            residual = GeneralizedHopfCurveResidual(
                problem,
                3,
                *borders,
                equation.derivative.parameters[2],
            )
            before = _conditioning(residual, state, 0.4)
            residual.adapt(state, 0.4)
            after = _conditioning(residual, state, 0.4)
            assert after < 1.0e-4 * before, (before, after)

    def test_adapt_keeps_the_best_conditioned_pair_of_schur_entries(self):
        residual, state, parameter = _bautin(4, 15)
        residual.adapt(state, parameter)
        chosen = residual._entries  # noqa: SLF001 (the selection under test)
        smallest = {}
        for pair in [
            [(0, 0), (0, 1)],
            [(0, 0), (1, 0)],
            [(0, 0), (1, 1)],
            [(0, 1), (1, 0)],
            [(0, 1), (1, 1)],
            [(1, 0), (1, 1)],
        ]:
            residual._entries = pair  # noqa: SLF001
            jacobian = AnalyticJacobian(residual, _STEP).extended_jacobian(
                state,
                parameter,
            )
            smallest[tuple(pair)] = np.linalg.svd(jacobian, compute_uv=False)[-1]
        assert smallest[tuple(chosen)] >= max(smallest.values()) * (1.0 - 1.0e-6)
        assert len(set(np.round(list(smallest.values()), 6))) > 1


class TestCost(TestCase):
    def test_field_evaluations_per_iteration_grow_linearly(self):
        counts = {}
        for kind, build in [("cusp", _cusp), ("hopf", _bautin)]:
            for size in (4, 8):
                residual, state, parameter = build(size, 10 + size)
                if kind == "hopf":
                    residual.adapt(state, parameter)
                field = DenseCusp if kind == "cusp" else DenseBautin
                field.calls = 0
                residual.evaluate(state, parameter)
                AnalyticJacobian(residual, _STEP).extended_jacobian(state, parameter)
                counts[kind, size] = field.calls
            # Linear: doubling n at most doubles the count (plus a constant);
            # differencing the columns made it quadruple.
            ratio = counts[kind, 8] / counts[kind, 4]
            assert ratio < 2.5, (kind, counts)
