"""SRK3 for several noise channels (DEQ-66): order 1.5 with the triple integrals.

* **Done when, order.** ``dx = dW_1``, ``dz = mu z dt + z x dW_2`` is
  non-commuting (``L^1 b^2 = (0, z) != 0 = L^2 b^1``) and its triple-integral
  terms do not vanish (``L^2 L^1 b^2``, ``L^1 L^2 b^2``, ``L^2 L^2 b^2``). Its
  exact solution ``z = exp(mu t + int x dW_2 - (1/2) int x^2 dt)`` is computed
  here from the cells of an :class:`IteratedIntegralPath` the test builds itself,
  and srk3 is measured against it and against srk3 on a fine step of the same
  path, at five step sizes. The cosine system is a nonlinear check against the
  fine step.
* **Each coefficient.** One step of the scheme against one step of the order 1.5
  Ito-Taylor scheme (Kuznetsov, arXiv:1802.00888, (5)) written out here with
  analytic derivatives, on a time-dependent full-matrix noise: the difference
  must fall like ``h^2``. A coefficient off by a constant factor gives ``h`` (an
  ``I_(j,k)`` term) or ``h^(3/2)`` (an ``I_(j,k,l)`` or ``I_(j,0)`` one).
* **Done when, single channel.** A scalar run is the run on main, bit for bit
  (hashes recorded on main @ 8a76b6a), and a ``1 x 1`` matrix noise through the
  several-channel step agrees with it to rounding.
"""

import hashlib
import warnings

import numpy as np
import pytest

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.solver.stochastic.iterated_integrals import (
    IteratedIntegralPath,
    IteratedIntegrals,
    IteratedIntegralSampler,
    IteratedIntegralSource,
)
from discrecontinual_equations.solver.stochastic.srk3.srk3_config import SRK3Config
from discrecontinual_equations.solver.stochastic.srk3.srk3_solver import SRK3Solver
from discrecontinual_equations.solver.stochastic.wiener import BrownianPath
from discrecontinual_equations.variable import Variable
from tests.solver.geometric_brownian_motion import GeometricBrownianMotionCase, slope

_MU = 0.5
_X0 = 0.5
_CELLS = 10
_STEPS = [2.0**-k for k in range(3, 8)]
_REFERENCE = 2.0**-9


class ExponentialOfIteratedIntegral(StochasticFunction):
    """``dx = dW_1``, ``dz = mu z dt + z x dW_2``."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[float]:
        return [0.0, _MU * point[1]]

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[list[float]]:
        return [[1.0, 0.0], [0.0, point[1] * point[0]]]


class CrossCosineNoise(StochasticFunction):
    """``dx = -x/2 dt + cos(y) dW_1``, ``dy = -y/2 dt + cos(x) dW_2``."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[float]:
        return [-0.5 * point[0], -0.5 * point[1]]

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[list[float]]:
        return [[np.cos(point[1]), 0.0], [0.0, np.cos(point[0])]]


class TimeDependentMatrixNoise(StochasticFunction):
    """A full ``2 x 2`` noise, a drift with ``t``, every Taylor term nonzero."""

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        x, y = point
        return [-0.5 * x + 0.3 * np.sin(time), -0.5 * y + 0.2 * x * x]

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[list[float]]:
        x, y = point
        return [[np.cos(y), 0.3 * x], [0.2, np.cos(x)]]


def _equation(function_type, size):
    time = Variable(name="t", discretization=[])
    variables = [Variable(name=f"v{i}", discretization=[]) for i in range(size)]
    results = [Variable(name=f"d{i}", discretization=[]) for i in range(size)]
    return DifferentialEquation(
        variables=variables,
        time=time,
        parameters=[],
        derivative=function_type(variables, [], results, time),
    )


def _final(step, wiener, function_type, initial, start=0.0, end=1.0):  # noqa: PLR0913
    solver = SRK3Solver(
        SRK3Config(start_time=start, end_time=end, step_size=step),
        wiener=wiener,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.solve(_equation(function_type, len(initial)), list(initial))
    return np.array([r.discretization[-1] for r in solver.solution.results])


def _cells(seed):
    """Cell integrals of a ``2^-10`` grid, drawn by the sampler at ``q = 4``."""
    zeta = np.random.default_rng(seed).standard_normal((2**_CELLS, 5, 2))
    return IteratedIntegralSampler.assemble(zeta, 2.0**-_CELLS, 4, 4)


def _exact_z(cells):
    """``exp(mu + int x dW_2 - (1/2) int x^2 dt)`` from the cells, by Chen here.

    Per cell ``int x dW_2 = x(t_c) dW_2 + I_(1,2)`` and ``int x^2 dt = x(t_c)^2 d
    + 2 x(t_c) I_(1,0) + int (W_1 - W_1(t_c))^2``, whose last term is ``d^2 / 2 +
    2 I_(1,1,0)``; ``I_(1,1,0)`` is not held, has mean zero, and over the path
    adds up to an error of order ``d^(3/2)``.
    """
    width = 2.0**-_CELLS
    x = _X0 + np.cumsum(cells.delta_w[:, 0]) - cells.delta_w[:, 0]
    ito = np.sum(x * cells.delta_w[:, 1] + cells.double[:, 0, 1])
    square = np.sum(x**2 * width + 2.0 * x * cells.delta_z[:, 0] + 0.5 * width**2)
    return np.exp(_MU + ito - 0.5 * square)


@pytest.fixture(scope="module")
def errors():
    exact, fine, cosine = ({h: [] for h in _STEPS} for _ in range(3))
    for seed in range(16):
        cells = _cells(seed)
        path = IteratedIntegralPath(
            0.0,
            2.0**-_CELLS,
            cells.delta_w,
            cells.delta_z,
            cells.double,
            cells.triple,
        )
        z = _exact_z(cells)
        reference = _final(_REFERENCE, path, ExponentialOfIteratedIntegral, [_X0, 1.0])
        cos_reference = _final(_REFERENCE, path, CrossCosineNoise, [0.3, -0.2])
        for h in _STEPS:
            value = _final(h, path, ExponentialOfIteratedIntegral, [_X0, 1.0])[1]
            exact[h].append(abs(value - z))
            fine[h].append(abs(value - reference[1]))
            cos = _final(h, path, CrossCosineNoise, [0.3, -0.2])
            cosine[h].append(np.max(np.abs(cos - cos_reference)))
    return {
        name: {h: float(np.mean(v)) for h, v in table.items()}
        for name, table in [("exact", exact), ("fine", fine), ("cosine", cosine)]
    }


class TestSeveralChannelsConvergence:
    def test_strong_order_one_and_a_half_against_the_exact_solution(self, errors):
        measured = slope(errors["exact"])
        assert 1.3 < measured < 1.9, errors["exact"]

    def test_strong_order_one_and_a_half_against_a_fine_step(self, errors):
        measured = slope(errors["fine"])
        assert 1.3 < measured < 1.9, errors["fine"]

    def test_nonlinear_noise_against_a_fine_step(self, errors):
        measured = slope(errors["cosine"])
        assert 1.25 < measured < 1.9, errors["cosine"]

    def test_diagonal_noise_converges_to_geometric_brownian_motion(self):
        # dX_i = X_i dt + 0.5 X_i dW_i: X_i(1) = exp(0.875 + 0.5 W_i(1)).
        case = GeometricBrownianMotionCase(dimension=2)
        table = {h: [] for h in _STEPS}
        for seed in range(8):
            path = IteratedIntegralPath.sample(0.0, 1.0, 2.0**-_CELLS, 2, seed=seed)
            exact = np.exp(0.875 + 0.5 * path.values[-1])
            for h in _STEPS:
                solver = SRK3Solver(
                    SRK3Config(end_time=1.0, step_size=h),
                    wiener=path,
                )
                solver.solve(case.equation(), [1.0, 1.0])
                final = np.array(
                    [r.discretization[-1] for r in solver.solution.results]
                )
                table[h].append(np.max(np.abs(final - exact)))
        assert slope({h: float(np.mean(v)) for h, v in table.items()}) > 1.3


def _taylor_step(point, time, step, integrals):
    """Kuznetsov (5) / Kloeden-Platen (10.4.6) for TimeDependentMatrixNoise."""
    x, y = point
    a = np.array([-0.5 * x + 0.3 * np.sin(time), -0.5 * y + 0.2 * x * x])
    da = np.array([[-0.5, 0.0], [0.4 * x, -0.5]])
    da_dt = np.array([0.3 * np.cos(time), 0.0])
    ha = np.array([np.zeros((2, 2)), [[0.4, 0.0], [0.0, 0.0]]])
    b = np.array([[np.cos(y), 0.2], [0.3 * x, np.cos(x)]])  # b[j] is column j
    db = np.array(
        [[[0.0, -np.sin(y)], [0.0, 0.0]], [[0.3, 0.0], [-np.sin(x), 0.0]]],
    )  # db[j][i, k] = d b^j_i / d x_k
    hb = np.zeros((2, 2, 2, 2))
    hb[0, 0] = [[0.0, 0.0], [0.0, -np.cos(y)]]
    hb[1, 1] = [[-np.cos(x), 0.0], [0.0, 0.0]]

    def second(hessian, u, v):
        return np.einsum("ikl,k,l->i", hessian, u, v)

    l0a = da_dt + da @ a + 0.5 * sum(second(ha, b[k], b[k]) for k in range(2))
    out = point + a * step + b.T @ integrals.delta_w + 0.5 * l0a * step**2
    mixed = step * integrals.delta_w - integrals.delta_z
    for j in range(2):
        l0b = db[j] @ a + 0.5 * sum(second(hb[j], b[k], b[k]) for k in range(2))
        out = out + (da @ b[j]) * integrals.delta_z[j] + l0b * mixed[j]
    for j1 in range(2):
        for j2 in range(2):
            out = out + (db[j2] @ b[j1]) * integrals.double[j1, j2]
            for j3 in range(2):
                # L^j1 L^j2 b^j3 = b^j3''[b^j2, b^j1] + b^j3' (b^j2' b^j1)
                term = second(hb[j3], b[j2], b[j1]) + db[j3] @ (db[j2] @ b[j1])
                out = out + term * integrals.triple[j1, j2, j3]
    return out


class _Fixed(IteratedIntegralSource):
    def __init__(self, integrals: IteratedIntegrals) -> None:
        self._integrals = integrals

    def iterated_integrals(self, time, step, dimension):  # noqa: ARG002
        return self._integrals


class TestAgainstTheTaylorScheme:
    def test_one_step_differs_from_the_taylor_step_by_h_squared(self):
        generator = np.random.default_rng(0)
        sampler = IteratedIntegralSampler(generator)
        table = {}
        for k in range(3, 9):
            h = 2.0**-k
            differences = []
            for _ in range(150):
                u = generator.standard_normal((2, 2))
                integrals = sampler.complete(
                    np.sqrt(h) * u[0],
                    0.5 * h**1.5 * (u[0] + u[1] / np.sqrt(3.0)),
                    h,
                    terms=(4, 4),
                )
                point = generator.uniform(-1.0, 1.0, 2)
                scheme = _final(
                    h,
                    _Fixed(integrals),
                    TimeDependentMatrixNoise,
                    point,
                    start=0.5,
                    end=0.5 + h,
                )
                differences.append(scheme - _taylor_step(point, 0.5, h, integrals))
            table[h] = float(np.sqrt(np.mean(np.sum(np.square(differences), axis=1))))
        measured = slope(table)
        assert measured > 1.85, table


_MAIN_HASHES = [
    "f0e077d20d0f39df",
    "9954679e7aaca142",
    "ec87e4889b1ff8ae",
    "301a0800658f76f1",
    "ef2356cbd7d83e49",
    "ded78480fe1f6bfb",
    "7090d4873835f78d",
]


def _digest(solver):
    values = np.array(solver.solution.results[0].discretization)
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


class TestSingleChannelUnchanged:
    def test_scalar_runs_are_bit_identical_to_main(self):
        digests = []
        for calculus in ["ito", "stratonovich"]:
            for seed in [0, 7, 12345]:
                solver = SRK3Solver(
                    SRK3Config(
                        end_time=1.0,
                        step_size=0.05,
                        random_seed=seed,
                        calculus=calculus,
                    ),
                )
                solver.solve(
                    GeometricBrownianMotionCase(calculus=calculus).equation(),
                    [1.0],
                )
                digests.append(_digest(solver))
        path = BrownianPath.sample(0.0, 1.0, 2**-8, 1, seed=3)
        solver = SRK3Solver(
            SRK3Config(end_time=1.0, step_size=2**-4, random_seed=1),
            wiener=path,
        )
        solver.solve(GeometricBrownianMotionCase().equation(), [1.0])
        digests.append(_digest(solver))
        assert digests == _MAIN_HASHES

    def test_one_by_one_matrix_noise_is_the_scalar_step_to_rounding(self):
        class MatrixGeometric(StochasticFunction):
            def eval(self, point, time=None):  # noqa: ARG002
                return [point[0]]

            def diffusion(self, point, time=None):  # noqa: ARG002
                return [[0.5 * point[0]]]

        path = IteratedIntegralPath.sample(0.0, 1.0, 2.0**-8, 1, seed=6)
        matrix = _final(2.0**-4, path, MatrixGeometric, [1.0])
        solver = SRK3Solver(SRK3Config(end_time=1.0, step_size=2.0**-4), wiener=path)
        solver.solve(GeometricBrownianMotionCase().equation(), [1.0])
        scalar = solver.solution.results[0].discretization[-1]
        assert matrix[0] == pytest.approx(scalar, rel=1.0e-12)


class TestSources:
    def test_default_source_integrates_several_channels_and_repeats(self):
        def run(seed):
            solver = SRK3Solver(
                SRK3Config(end_time=1.0, step_size=0.125, random_seed=seed),
            )
            solver.solve(_equation(CrossCosineNoise, 2), [0.3, -0.2])
            return [r.discretization[-1] for r in solver.solution.results]

        assert run(3) == run(3)
        assert run(3) != run(4)

    def test_a_path_without_triple_integrals_is_refused_for_several_channels(self):
        path = BrownianPath.sample(0.0, 1.0, 2.0**-6, 2, seed=1)
        solver = SRK3Solver(SRK3Config(end_time=1.0, step_size=0.25), wiener=path)
        with pytest.raises(ValueError, match=r"I_\(j,k,l\).*IteratedIntegralPath"):
            solver.solve(_equation(CrossCosineNoise, 2), [0.3, -0.2])
