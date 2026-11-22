"""Milstein with Levy areas: non-commuting matrix noise at strong order 1.0 (DEQ-36).

Three yardsticks.

* ``dx = dW_1, dy = x dW_2`` (oracle 2 of the issue): ``y(T) = int W_1 dW_2 =
  I_(1,2)(0, T)``. The Milstein step is ``y + x dW_2 + I_(1,2)`` over each step,
  which by Chen's relation sums to exactly ``I_(1,2)(0, T)`` when the step's
  ``I_(1,2)`` is exact - so on a Brownian path that carries its areas the scheme
  is exact to rounding at every step size, which is more than order 1.0 and
  pins the index convention (``I_(2,1)`` instead would be order 0.5); "to
  rounding" means to the central-difference Jacobians' ``1e-10``. Euler-
  Maruyama omits the iterated integral and measures 0.5. Driven by the sampler,
  the only error left is the sampler's, bounded by ``sqrt(T) h`` with the
  truncation rule's ``c1 = 1``; it is measured against a coupled reference.
* ``G = [[cos y, 0], [0, cos x]]``, a nonlinear non-commuting noise with a
  linear drift, measured against the same scheme on the path's own grid (an
  order-1.0 reference, 16 times finer than the finest step measured).
* Conformal (commuting) noise: the step is the commuting one, bit for bit.
"""

import warnings

import numpy as np
import pytest

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config import (
    EulerMaruyamaConfig,
)
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_solver import (
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
    GaussianWienerSource,
    LevyAreaSampler,
    WienerIncrements,
    WienerSource,
)
from discrecontinual_equations.variable import Variable
from tests.solver.conformal_geometric_brownian_motion import ConformalCase
from tests.solver.coupled_levy_area import coupled_areas
from tests.solver.geometric_brownian_motion import slope


class IteratedIntegralNoise(StochasticFunction):
    """``dx = dW_1``, ``dy = x dW_2``: ``G = [[1, 0], [0, x]]``, no drift."""

    def eval(
        self,
        point: list[float],  # noqa: ARG002
        time: float | None = None,  # noqa: ARG002
    ) -> list[float]:
        return [0.0, 0.0]

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[list[float]]:
        return [[1.0, 0.0], [0.0, point[0]]]


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


def _equation(function_type) -> DifferentialEquation:
    time = Variable(name="t", discretization=[])
    variables = [
        Variable(name="x", discretization=[]),
        Variable(name="y", discretization=[]),
    ]
    results = [
        Variable(name="dx", discretization=[]),
        Variable(name="dy", discretization=[]),
    ]
    return DifferentialEquation(
        variables=variables,
        time=time,
        parameters=[],
        derivative=function_type(variables, [], results, time),
    )


def _final(  # noqa: PLR0913 (a solver, its config, a step, a path, a problem)
    solver_type,
    config_type,
    step,
    wiener,
    function_type,
    initial,
    **extra,
):
    config = config_type(start_time=0.0, end_time=1.0, step_size=step, **extra)
    solver = solver_type(config, wiener=wiener)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.solve(_equation(function_type), initial)
    return np.array([r.discretization[-1] for r in solver.solution.results])


_STEPS = [2.0**-k for k in range(2, 8)]


class TestIteratedIntegralOracle:
    def test_milstein_is_exact_on_a_path_with_areas_euler_is_order_half(self):
        euler = dict.fromkeys(_STEPS, 0.0)
        paths = 40
        for index in range(paths):
            path = BrownianPath.sample(0.0, 1.0, 2.0**-8, 2, seed=index)
            exact = path.increments(0.0, 1.0, 2, levy_area=True).iterated_integrals(
                1.0,
            )[0, 1]
            for step in _STEPS:
                milstein = _final(
                    MilsteinSolver,
                    MilsteinConfig,
                    step,
                    path,
                    IteratedIntegralNoise,
                    [0.0, 0.0],
                )
                # Exact but for the central-difference Jacobian (relative error
                # about 1e-10) that every step takes: 1e-9 against an Euler error
                # of 0.1 and an error of the same size for I_(2,1) in place of
                # I_(1,2).
                assert abs(milstein[1] - exact) < 1e-9, (step, milstein, exact)
                euler[step] += (
                    abs(
                        _final(
                            EulerMaruyamaSolver,
                            EulerMaruyamaConfig,
                            step,
                            path,
                            IteratedIntegralNoise,
                            [0.0, 0.0],
                        )[1]
                        - exact,
                    )
                    / paths
                )
        assert 0.35 < slope(euler) < 0.65, euler

    def test_milstein_with_the_sampler_is_within_the_order_one_bound(self):
        # Driven by the sampler the scheme's only error is the sampler's, and the
        # truncation rule bounds it by h^(3/2) per step in L2, sqrt(T) h at T.
        steps = [2.0**-k for k in range(4, 9)]
        errors = {}
        for step in steps:
            squared = 0.0
            paths = 20
            for index in range(paths):
                source = CoupledSource(seed=index)
                final = _final(
                    MilsteinSolver,
                    MilsteinConfig,
                    step,
                    source,
                    IteratedIntegralNoise,
                    [0.0, 0.0],
                )
                squared += (final[1] - source.exact_y) ** 2 / paths
            errors[step] = float(np.sqrt(squared))
            assert errors[step] < step, errors
        assert slope(errors) > 0.6, errors


class CoupledSource(WienerSource):
    """Sampler areas with a coupled reference, for the iterated-integral oracle.

    Each step's area is the production sampler's at the truncation the rule
    gives; alongside, the source keeps the reference ``I_(1,2)`` built from 64
    times as many Fourier coefficients, and accumulates the exact ``y(T) = sum
    W_1(t_n) dW_2 + I_(1,2)`` the scheme is measured against.
    """

    def __init__(self, seed: int) -> None:
        self._generator = np.random.default_rng(seed)
        self._w1 = 0.0
        self.exact_y = 0.0

    def increments(
        self,
        time: float,  # noqa: ARG002
        step: float,
        dimension: int,
        *,
        levy_area: bool = False,
    ) -> WienerIncrements:
        assert dimension == 2
        assert levy_area
        delta_w = np.sqrt(step) * self._generator.standard_normal((1, 2))
        terms = LevyAreaSampler.terms(step, 2)
        approximation, reference = coupled_areas(
            self._generator,
            delta_w,
            step,
            terms,
            reference_terms=64 * terms,
        )
        self.exact_y += self._w1 * delta_w[0, 1] + (
            0.5 * delta_w[0, 0] * delta_w[0, 1] + reference[0, 0, 1]
        )
        self._w1 += delta_w[0, 0]
        return WienerIncrements(
            delta_w=delta_w[0],
            delta_z=np.zeros(2),
            levy_area=approximation[0],
        )


class TestNonCommutingConvergence:
    def test_strong_order_one_on_cross_cosine_noise(self):
        steps = [2.0**-k for k in range(2, 7)]
        resolution = 2.0**-10
        milstein = dict.fromkeys(steps, 0.0)
        euler = dict.fromkeys(steps, 0.0)
        paths = 30
        initial = [0.3, -0.2]
        for index in range(paths):
            path = BrownianPath.sample(0.0, 1.0, resolution, 2, seed=100 + index)
            reference = _final(
                MilsteinSolver,
                MilsteinConfig,
                resolution,
                path,
                CrossCosineNoise,
                initial,
            )
            for step in steps:
                milstein[step] += (
                    np.linalg.norm(
                        _final(
                            MilsteinSolver,
                            MilsteinConfig,
                            step,
                            path,
                            CrossCosineNoise,
                            initial,
                        )
                        - reference,
                    )
                    / paths
                )
                euler[step] += (
                    np.linalg.norm(
                        _final(
                            EulerMaruyamaSolver,
                            EulerMaruyamaConfig,
                            step,
                            path,
                            CrossCosineNoise,
                            initial,
                        )
                        - reference,
                    )
                    / paths
                )
        assert 0.85 < slope(milstein) < 1.3, milstein
        assert slope(euler) < 0.7, euler
        assert milstein[steps[-1]] < 0.5 * euler[steps[-1]], (milstein, euler)


class TestNonCommutingInterface:
    def test_non_commuting_noise_warns_about_cost_and_integrates(self):
        config = MilsteinConfig(start_time=0.0, end_time=1.0, step_size=0.01)
        solver = MilsteinSolver(config)
        with pytest.warns(UserWarning, match="Levy areas"):
            solver.solve(_equation(CrossCosineNoise), [0.3, -0.2])
        assert len(solver.solution.results[0].discretization) == 101
        assert np.all(np.isfinite(solver.solution.results[1].discretization))

    def test_seeded_non_commuting_runs_repeat_and_differ_by_seed(self):
        def run(seed):
            config = MilsteinConfig(
                start_time=0.0,
                end_time=1.0,
                step_size=0.05,
                random_seed=seed,
            )
            solver = MilsteinSolver(config)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solver.solve(_equation(IteratedIntegralNoise), [0.0, 0.0])
            return [r.discretization[-1] for r in solver.solution.results]

        assert run(3) == run(3)
        assert run(3) != run(4)

    def test_own_generator_draws_the_same_increments_as_euler(self):
        # The areas come from a stream of their own, so the increments a seeded
        # Milstein run draws are those a seeded Euler-Maruyama run draws: x(T),
        # which is W_1(T) for both, agrees bit for bit.
        milstein = MilsteinSolver(
            MilsteinConfig(end_time=1.0, step_size=0.05, random_seed=8),
        )
        euler = EulerMaruyamaSolver(
            EulerMaruyamaConfig(end_time=1.0, step_size=0.05, random_seed=8),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            milstein.solve(_equation(IteratedIntegralNoise), [0.0, 0.0])
        euler.solve(_equation(IteratedIntegralNoise), [0.0, 0.0])
        assert (
            milstein.solution.results[0].discretization
            == euler.solution.results[0].discretization
        )

    def test_a_gaussian_source_supplies_the_areas(self):
        final = _final(
            MilsteinSolver,
            MilsteinConfig,
            0.05,
            GaussianWienerSource(seed=2),
            IteratedIntegralNoise,
            [0.0, 0.0],
        )
        assert np.all(np.isfinite(final))


# Recorded on main @ 4ca6382, before Levy areas existed: the commuting step must
# not change by a single bit (oracle 3).
_CONFORMAL_HEX = {
    ("ito", "generator"): ["0x1.c556e13b672a3p+0", "-0x1.f6c14b7038f73p-1"],
    ("ito", "path"): ["0x1.ac2d30d49a5a1p+0", "0x1.ed41cca16e038p-2"],
    ("ito", "gaussian"): ["0x1.119d7f79c993cp+1", "-0x1.1b0eb8fbfc06bp-1"],
    ("stratonovich", "generator"): ["0x1.c556e13b554bdp+0", "-0x1.f6c14b703e463p-1"],
    ("stratonovich", "path"): ["0x1.ac2d30d48ad86p+0", "0x1.ed41cca166c5dp-2"],
    ("stratonovich", "gaussian"): ["0x1.119d7f79c6e8ep+1", "-0x1.1b0eb8fbf769ap-1"],
}


class TestCommutingCaseUnchanged:
    @pytest.mark.parametrize(("calculus", "source"), list(_CONFORMAL_HEX))
    def test_conformal_noise_is_hex_identical_to_before(self, calculus, source):
        case = ConformalCase(calculus=calculus)
        wiener = {
            "generator": None,
            "path": BrownianPath.sample(0.0, 1.0, 2**-8, 2, seed=12),
            "gaussian": GaussianWienerSource(13),
        }[source]
        config = MilsteinConfig(
            start_time=0.0,
            end_time=1.0,
            step_size=2**-5,
            random_seed=11 if source == "generator" else None,
            calculus=calculus,
        )
        solver = MilsteinSolver(config, wiener=wiener)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            solver.solve(case.equation(), [1.0, 0.0])
        final = [float(r.discretization[-1]).hex() for r in solver.solution.results]
        assert final == _CONFORMAL_HEX[(calculus, source)]
