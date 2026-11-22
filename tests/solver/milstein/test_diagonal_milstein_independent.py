"""Diagonal-noise Milstein against oracles written apart from the solver (DEQ-48).

The DEQ-48 tests compare the fast path with the general step, which shares the
solver's difference quotients and its commutativity test. These checks do not:

* a Milstein step written from Kloeden and Platen's general multi-driver form,
  ``x + a h + G dW + sum_{j,k} (L^j G_k) I_(j,k)`` with the Jacobians exact and
  ``I_(j,k) = (1/2) dW_j dW_k - (1/2) h delta_jk + A_jk`` (Ito) or without the
  ``h`` term (Stratonovich), against which the solver is compared step by step on
  coupling just below and just above the commutativity tolerance - below, the
  areas are left out and the cross product kept; above, the areas are drawn;
* which way the solver went, read off the source it asks for increments, not off
  the result;
* the strong order on geometric Brownian motion per component, whose exact
  solution is a function of the path, declared diagonal and not;
* the per-path step against a one-path ensemble driven by the same path, bit for
  bit, which the two diagonal corrections claim.
"""

import warnings

import numpy as np
import pytest

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.solver.stochastic.ensemble import StochasticEnsemble
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config import (  # noqa: E501
    EulerMaruyamaConfig,
)
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_solver import (  # noqa: E501
    EulerMaruyamaSolver,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_config import (
    MilsteinConfig,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_solver import (
    MilsteinSolver,
)
from discrecontinual_equations.solver.stochastic.wiener import (
    BrownianPath,
    WienerIncrements,
    WienerSource,
)
from discrecontinual_equations.variable import Variable

_STEP = 2.0**-5
_RESOLUTION = 2.0**-8
_TOLERANCE = 1.0e-6  # the solver's commutativity tolerance, relative
_START = np.array([0.5, 1.0, 1.5])


def _b(x):
    return 0.3 * np.sin(x) + 0.2 * x


def _db(x):
    return 0.3 * np.cos(x) + 0.2


class CoupledNoise(StochasticFunction):
    """``dx_i = -x_i dt + b(x_i) dW_i``, except ``b_0`` also reads ``c x_1``.

    ``coupling`` is set per subclass; zero is diagonal noise.
    """

    coupling = 0.0

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[float]:
        return [-value for value in point]

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[float]:
        values = [float(_b(value)) for value in point]
        values[0] += self.coupling * point[1]
        return values

    def diffusion_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        values = _b(points)
        values[:, 0] += self.coupling * points[:, 1]
        return values


def _exact_products(x: np.ndarray, coupling: float) -> np.ndarray:
    """``P[j, k] = b_j d b_k / d x_j``, exact: ``L^j G_k = P[j, k] e_k``."""
    b = _b(x)
    b[0] += coupling * x[1]
    jacobian = np.diag(_db(x))  # jacobian[k, j] = d b_k / d x_j
    jacobian[0, 1] = coupling
    return b[:, None] * jacobian.T


def _coupling_at_ratio(ratio: float) -> float:
    """The ``c`` whose cross product is ``ratio`` times the largest product at start."""
    scale = float(np.max(np.abs(_b(_START) * _db(_START))))
    return ratio * scale / float(_b(_START[1]))


def _noise(coupling: float, *, declared: bool = False) -> type[CoupledNoise]:
    return type(
        "Noise",
        (CoupledNoise,),
        {"coupling": coupling, "diagonal_noise": declared},
    )


def _equation(function_type, dimension: int) -> DifferentialEquation:
    time = Variable(name="t")
    variables = [Variable(name=f"x{i}") for i in range(dimension)]
    results = [Variable(name=f"r{i}") for i in range(dimension)]
    return DifferentialEquation(
        variables=variables,
        time=time,
        parameters=[],
        derivative=function_type(variables, [], results, time),
    )


class RecordingSource(WienerSource):
    """A Brownian path that records whether each step asked for Levy areas."""

    def __init__(self, path: BrownianPath) -> None:
        self._path = path
        self.asked: list[bool] = []

    def increments(
        self,
        time: float,
        step: float,
        dimension: int,
        *,
        levy_area: bool = False,
    ) -> WienerIncrements:
        self.asked.append(levy_area)
        return self._path.increments(time, step, dimension, levy_area=levy_area)


def _reference_step(  # noqa: PLR0913 (a state, a step and how to take it)
    x,
    h,
    increments,
    coupling,
    calculus,
    *,
    with_area,
):
    """One general Milstein step from the formula, with exact Jacobians."""
    b = _b(x)
    b[0] += coupling * x[1]
    dw = increments.delta_w
    integrals = 0.5 * np.outer(dw, dw)
    if calculus == "ito":
        integrals -= 0.5 * h * np.eye(x.size)
    if with_area:
        integrals = integrals + increments.levy_area
    products = _exact_products(x, coupling)
    # sum_{j,k} P[j, k] e_k I_(j,k): component k gets sum_j P[j, k] I_(j,k).
    correction = np.einsum("jk,jk->k", products, integrals)
    return x - x * h + b * dw + correction


def _one_step(coupling, calculus, *, declared=False):
    path = BrownianPath.sample(0.0, 1.0, _RESOLUTION, 3, seed=21)
    source = RecordingSource(path)
    config = MilsteinConfig(
        start_time=0.0,
        end_time=_STEP,
        step_size=_STEP,
        calculus=calculus,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        curve = MilsteinSolver(config, wiener=source).solve(
            _equation(_noise(coupling, declared=declared), 3),
            _START.tolist(),
        )
    final = np.array([result.discretization[-1] for result in curve.results])
    increments = path.increments(0.0, _STEP, 3, levy_area=True)
    return final, increments, source.asked, caught


# Agreement of the solver's central differences (step 1e-6) with exact slopes,
# after one step: the products' error is about eps |b| / 1e-6 ~ 1e-11, times dW^2.
_ONE_STEP = 1.0e-11


@pytest.mark.parametrize("calculus", ["ito", "stratonovich"])
class TestAroundTheTolerance:
    def test_just_below_commutes_keeps_the_cross_product_and_draws_no_area(
        self,
        calculus,
    ):
        # Arrange: the cross product half the tolerance times the largest.
        coupling = _coupling_at_ratio(0.5 * _TOLERANCE)

        # Act
        final, increments, asked, caught = _one_step(coupling, calculus)

        # Assert: no areas asked for, no warning, and the formula without areas.
        assert asked == [False]
        assert not [w for w in caught if "does not commute" in str(w.message)]
        expected = _reference_step(
            _START,
            _STEP,
            increments,
            coupling,
            calculus,
            with_area=False,
        )
        np.testing.assert_allclose(final, expected, rtol=0.0, atol=_ONE_STEP)
        # and the cross product is not dropped: without it the step is farther off.
        dropped = _reference_step(
            _START,
            _STEP,
            increments,
            0.0,
            calculus,
            with_area=False,
        )
        dropped[0] += coupling * _START[1] * increments.delta_w[0]
        assert np.abs(final - dropped).max() > 10 * _ONE_STEP

    def test_just_above_does_not_commute_and_takes_the_levy_area_step(
        self,
        calculus,
    ):
        # Arrange: the cross product twice the tolerance times the largest.
        coupling = _coupling_at_ratio(2.0 * _TOLERANCE)

        # Act
        final, increments, asked, caught = _one_step(coupling, calculus)

        # Assert: areas asked for, the cost warned of, and the formula with them.
        assert asked == [True]
        assert [w for w in caught if "does not commute" in str(w.message)]
        expected = _reference_step(
            _START,
            _STEP,
            increments,
            coupling,
            calculus,
            with_area=True,
        )
        np.testing.assert_allclose(final, expected, rtol=0.0, atol=_ONE_STEP)
        without = _reference_step(
            _START,
            _STEP,
            increments,
            coupling,
            calculus,
            with_area=False,
        )
        assert np.abs(final - without).max() > 10 * _ONE_STEP

    def test_exact_diagonal_noise_is_the_formula(self, calculus):
        # Act
        final, increments, asked, _ = _one_step(0.0, calculus)
        declared, _, asked_declared, _ = _one_step(0.0, calculus, declared=True)

        # Assert
        assert asked == asked_declared == [False]
        expected = _reference_step(
            _START,
            _STEP,
            increments,
            0.0,
            calculus,
            with_area=False,
        )
        np.testing.assert_allclose(final, expected, rtol=0.0, atol=_ONE_STEP)
        assert [v.hex() for v in declared] == [v.hex() for v in final]


@pytest.mark.parametrize("calculus", ["ito", "stratonovich"])
@pytest.mark.parametrize(
    "coupling_ratio",
    [0.0, 0.5 * _TOLERANCE],
    ids=["diagonal", "below"],
)
def test_a_one_path_ensemble_is_the_per_path_step_bit_for_bit(calculus, coupling_ratio):
    # Arrange: one Brownian path drives both.
    coupling = _coupling_at_ratio(coupling_ratio)
    path = BrownianPath.sample(0.0, 0.5, _RESOLUTION, 3, seed=31)
    config = MilsteinConfig(
        start_time=0.0,
        end_time=0.5,
        step_size=_STEP,
        calculus=calculus,
    )
    equation = _equation(_noise(coupling), 3)

    # Act
    curve = MilsteinSolver(config, wiener=path).solve(equation, _START.tolist())
    ensemble = StochasticEnsemble(config, scheme="milstein", wiener=path).solve(
        equation,
        _START,
        paths=1,
    )

    # Assert
    per_path = [result.discretization[-1] for result in curve.results]
    assert [float(v).hex() for v in ensemble.final[0]] == [
        float(v).hex() for v in per_path
    ]


def test_ensemble_refuses_coupling_just_above_the_tolerance():
    config = MilsteinConfig(start_time=0.0, end_time=_STEP, step_size=_STEP)
    with pytest.raises(ValueError, match="commuting noise"):
        StochasticEnsemble(config, scheme="milstein").solve(
            _equation(_noise(_coupling_at_ratio(2.0 * _TOLERANCE)), 3),
            _START,
            paths=2,
        )


def test_declared_stratonovich_drift_correction_is_the_undeclared_one():
    # Euler-Maruyama reads the Stratonovich correction (1/2) b_i b_i' through the
    # declared shortcut too; it must be the same numbers.
    config = EulerMaruyamaConfig(
        start_time=0.0,
        end_time=0.5,
        step_size=_STEP,
        random_seed=4,
        calculus="stratonovich",
    )
    found, declared = (
        EulerMaruyamaSolver(config).solve(
            _equation(_noise(0.0, declared=d), 3),
            _START.tolist(),
        )
        for d in (False, True)
    )
    assert [r.discretization for r in found.results] == [
        r.discretization for r in declared.results
    ]


# Geometric Brownian motion per component: dX_i = mu_i X_i dt + s_i X_i dW_i.
_MU = np.array([0.5, -0.3, 1.0])
_SIGMA = np.array([0.6, 0.9, 0.4])


class GeometricNoise(StochasticFunction):
    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[float]:
        return (_MU * np.asarray(point)).tolist()

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[float]:
        return (_SIGMA * np.asarray(point)).tolist()


class DeclaredGeometricNoise(GeometricNoise):
    diagonal_noise = True


def _strong_errors(  # noqa: PLR0913 (a problem, a scheme, and a grid)
    function_type,
    calculus,
    solver_type,
    config_type,
    steps,
    paths,
):
    errors = dict.fromkeys(steps, 0.0)
    x0 = np.ones(3)
    for seed in range(paths):
        path = BrownianPath.sample(0.0, 1.0, _RESOLUTION, 3, seed=1000 + seed)
        drift = _MU if calculus == "stratonovich" else _MU - 0.5 * _SIGMA**2
        exact = x0 * np.exp(drift + _SIGMA * path.value(1.0))
        for step in steps:
            config = config_type(
                start_time=0.0,
                end_time=1.0,
                step_size=step,
                calculus=calculus,
            )
            curve = solver_type(config, wiener=path).solve(
                _equation(function_type, 3),
                x0.tolist(),
            )
            final = np.array([result.discretization[-1] for result in curve.results])
            errors[step] += float(np.mean(np.abs(final - exact))) / paths
    return errors


def _slope(errors):
    steps = np.array(sorted(errors))
    return float(np.polyfit(np.log(steps), np.log([errors[s] for s in steps]), 1)[0])


_STEPS = [2.0**-k for k in range(2, 7)]


@pytest.mark.parametrize("calculus", ["ito", "stratonovich"])
@pytest.mark.parametrize(
    "function_type",
    [GeometricNoise, DeclaredGeometricNoise],
    ids=["found", "declared"],
)
def test_strong_order_one_on_geometric_brownian_motion_per_component(
    function_type,
    calculus,
):
    # Act
    milstein = _strong_errors(
        function_type,
        calculus,
        MilsteinSolver,
        MilsteinConfig,
        _STEPS,
        paths=30,
    )
    euler = _strong_errors(
        function_type,
        calculus,
        EulerMaruyamaSolver,
        EulerMaruyamaConfig,
        _STEPS,
        paths=30,
    )

    # Assert: order one, where the Euler-Maruyama yardstick on the same paths is
    # one half, so the test tells the two apart.
    assert 0.85 < _slope(milstein) < 1.2
    assert 0.35 < _slope(euler) < 0.7
    assert milstein[_STEPS[-1]] < 0.5 * euler[_STEPS[-1]]
