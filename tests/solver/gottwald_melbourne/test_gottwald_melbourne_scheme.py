"""The Gottwald-Melbourne fast-slow map against its paper (DEQ-108).

Gottwald & Melbourne, "Simulation of non-Lipschitz stochastic differential
equations driven by α-stable noise: a method based on deterministic
homogenisation", arXiv:2004.09914 (MMS 19(2), 2021). Every expected value
below is written from the paper's formulas, not read off the solver: the
constants (2.2), (2.4), (2.5) and the observable of Section 4.1, the clock
ẑ_ε(t) = z_⌊t/ε⌋ of Theorem 4.1, the initial state of Remark 4.4, the
perturbation of Section 4.2, and the stable law of Section 2.1.
"""

import math

import numpy as np
import pytest

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.solver.stochastic.gottwald_melbourne.gottwald_melbourne_config import (
    GottwaldMelbourneConfig,
)
from discrecontinual_equations.solver.stochastic.gottwald_melbourne.gottwald_melbourne_solver import (
    GottwaldMelbourneSolver,
)
from discrecontinual_equations.variable import Variable

_ALPHAS = [1.2, 1.5, 1.8]


class _Linear(StochasticFunction):
    """dZ = (c - λZ) dt + σ ⋄ dW^α,η,β, with the slow time of each drift call kept."""

    def __init__(self, drift_constant=0.0, rate=0.0, sigma=1.0):
        time = Variable(name="t")
        variables = [Variable(name="Z")]
        super().__init__(variables, [], [Variable(name="dZ/dt")], time)
        self.drift_constant, self.rate, self.sigma = drift_constant, rate, sigma
        self.times: list[float] = []

    def eval(self, point, time=None):
        self.times.append(time)
        return [self.drift_constant - self.rate * point[0]]

    def diffusion(self, point, time=None):
        return [self.sigma]


def _equation(function: _Linear) -> DifferentialEquation:
    return DifferentialEquation(
        variables=function.variables,
        time=function.time,
        parameters=[Parameter(name="sigma", value=function.sigma)],
        derivative=function,
    )


def _config(**overrides) -> GottwaldMelbourneConfig:
    settings = {
        "alpha": 1.5,
        "eta": 1.0,
        "beta": 0.0,
        "epsilon": 0.01,
        "start_time": 0.0,
        "end_time": 1.0,
        "step_size": 0.1,
        "random_seed": 1,
    }
    settings.update(overrides)
    return GottwaldMelbourneConfig(**settings)


def _d_alpha(alpha: float) -> float:
    """(2.4) with (2.5): d_α = α^α (1-γ)/(2^{1-γ}-1) g_α, g_α = Γ(1-α) cos(απ/2)."""
    gamma = 1 / alpha
    g_alpha = math.gamma(1 - alpha) * math.cos(math.pi * alpha / 2)
    return alpha**alpha * (1 - gamma) / (2 ** (1 - gamma) - 1) * g_alpha


def _stable_cf(t: np.ndarray, alpha: float, eta: float, beta: float) -> np.ndarray:
    """E exp(itX_α,η,β), Section 2.1."""
    skew = 1 - 1j * beta * np.sign(t) * math.tan(alpha * math.pi / 2)
    return np.exp(-(eta**alpha) * np.abs(t) ** alpha * skew)


_CF_POINTS = np.linspace(0.25, 2.0, 8)


def _cf_error(samples: np.ndarray, alpha: float, eta: float, beta: float) -> float:
    """RMS over a few t of |empirical cf - stable cf|, less the 1/N sampling floor."""
    empirical = np.exp(1j * np.outer(_CF_POINTS, samples)).mean(axis=1)
    target = _stable_cf(_CF_POINTS, alpha, eta, beta)
    squares = np.abs(empirical - target) ** 2
    squares -= (1 - np.abs(empirical) ** 2) / samples.size
    return math.sqrt(max(float(squares.mean()), 0.0))


class TestTheConstants:
    @pytest.mark.parametrize("alpha", _ALPHAS)
    def test_x_star_solves_2_2(self, alpha):
        gamma = 1 / alpha
        x_star = GottwaldMelbourneSolver(_config(alpha=alpha))._x_star

        assert x_star ** (1 - gamma) + (1 + x_star) ** (1 - gamma) == pytest.approx(
            2.0,
            abs=1e-14,
        )

    @pytest.mark.parametrize("alpha", _ALPHAS)
    def test_d_alpha_is_2_4(self, alpha):
        solver = GottwaldMelbourneSolver(_config(alpha=alpha))

        assert solver._d_alpha == pytest.approx(_d_alpha(alpha), rel=1e-12)

    @pytest.mark.parametrize("alpha", _ALPHAS)
    def test_d_alpha_over_g_alpha_is_the_tail_constant_of_proposition_5_5(self, alpha):
        # Arrange: e_α = α^α / µ(Y) with µ(Y) = ∫_{x*}^1 h, h = x^-γ + (1+x)^-γ
        # (2.3), integrated here by quadrature rather than in closed form.
        gamma = 1 / alpha
        solver = GottwaldMelbourneSolver(_config(alpha=alpha))
        grid = np.linspace(solver._x_star, 1.0, 200_001)
        mass = np.trapezoid(grid**-gamma + (1 + grid) ** -gamma, grid)
        g_alpha = math.gamma(1 - alpha) * math.cos(math.pi * alpha / 2)

        # Assert
        assert solver._d_alpha / g_alpha == pytest.approx(alpha**alpha / mass, rel=1e-8)

    @pytest.mark.parametrize("alpha", _ALPHAS)
    @pytest.mark.parametrize("eta", [0.5, 2.0])
    def test_v_on_the_laminar_branch_is_section_4_1s(self, alpha, eta):
        # v = η d_α^-γ (1 - 2^{γ-1})^-γ ṽ, with ṽ = 1 on [0, x*].
        gamma = 1 / alpha
        solver = GottwaldMelbourneSolver(_config(alpha=alpha, eta=eta))
        expected = eta * _d_alpha(alpha) ** -gamma * (1 - 2 ** (gamma - 1)) ** -gamma

        assert solver._compute_observable(0.25 * solver._x_star, 1) == pytest.approx(
            expected,
            rel=1e-12,
        )

    @pytest.mark.parametrize("alpha", _ALPHAS)
    def test_v_on_the_hyperbolic_branch_is_section_4_1s(self, alpha):
        # ṽ = (1 - 2^{1-γ})^{-1} on (x*, 1]: negative, since 2^{1-γ} > 1.
        gamma = 1 / alpha
        solver = GottwaldMelbourneSolver(_config(alpha=alpha))
        laminar = solver._compute_observable(0.5 * solver._x_star, 1)
        hyperbolic = solver._compute_observable(0.5 * (1 + solver._x_star), 1)

        assert hyperbolic < 0
        assert hyperbolic / laminar == pytest.approx(
            1 / (1 - 2 ** (1 - gamma)),
            rel=1e-12,
        )

    @pytest.mark.parametrize("alpha", _ALPHAS)
    def test_the_sign_multiplies_both_branches(self, alpha):
        solver = GottwaldMelbourneSolver(_config(alpha=alpha))
        for x in (0.5 * solver._x_star, 0.5 * (1 + solver._x_star)):
            assert solver._compute_observable(x, -1) == -solver._compute_observable(
                x,
                1,
            )

    @pytest.mark.parametrize("alpha", _ALPHAS)
    def test_v_has_mean_zero_under_the_invariant_measure(self, alpha):
        # Arrange: ∫ v h dx over [0, 1] with x = s^p, p = 1/(1-γ), which turns
        # h dx into p (1 + (s^p/(1+s^p))^γ ... ) ds without the x^-γ singularity.
        gamma = 1 / alpha
        power = 1 / (1 - gamma)
        solver = GottwaldMelbourneSolver(_config(alpha=alpha))
        s = np.linspace(0.0, 1.0, 400_001)
        x = s**power
        weight = power * (1 + (x / (1 + x)) ** gamma)
        values = np.array([solver._compute_observable(point, 1) for point in x])

        # Act
        mean = np.trapezoid(values * weight, s) / np.trapezoid(weight, s)

        # Assert: zero to quadrature accuracy, against a value of order one.
        assert abs(mean) < 1e-4 * abs(values[0])

    @pytest.mark.parametrize("alpha", _ALPHAS)
    def test_the_map_is_2_1(self, alpha):
        # (2.1): T x = x (1 + (x/(1+x))^{γ-1} - x^{γ-1})^{1/(1-γ)} mod 1, read
        # away from 0, where its two large terms do not cancel.
        gamma = 1 / alpha
        solver = GottwaldMelbourneSolver(_config(alpha=alpha))
        for x in np.linspace(0.05, 0.95, 37):
            image = x * (1 + (x / (1 + x)) ** (gamma - 1) - x ** (gamma - 1)) ** (
                1 / (1 - gamma)
            )
            expected = image % 1.0
            assert solver._thaler_map(float(x)) == pytest.approx(expected, abs=1e-12)


class TestTheClock:
    """ẑ_ε(t_i) = z_{n_i} with n_i = ⌊(t_i - t_0)/ε⌋ (Theorem 4.1)."""

    @pytest.mark.parametrize(
        ("step_size", "epsilon", "end_time", "expected"),
        [
            # h = 1.5ε: int(h/ε) = 1 ran the map at 2/3 speed.
            (0.015, 0.01, 0.045, [0.0, 0.01, 0.03, 0.04]),
            # 0.3/0.1 is 2.9999999999999996 in floating point.
            (0.3, 0.1, 0.6, [0.0, 0.3, 0.6]),
            (0.1, 0.01, 0.3, [0.0, 0.1, 0.2, 0.3]),
        ],
    )
    def test_the_path_at_each_node_has_taken_floor_t_over_epsilon_steps(
        self,
        step_size,
        epsilon,
        end_time,
        expected,
    ):
        # Arrange: dZ = dt, no noise, so z_n = n ε exactly counts the fast steps.
        function = _Linear(drift_constant=1.0, sigma=0.0)
        config = _config(step_size=step_size, epsilon=epsilon, end_time=end_time)

        # Act
        curve = GottwaldMelbourneSolver(config).solve(_equation(function), [0.0])

        # Assert
        assert curve.results[0].discretization == pytest.approx(expected, abs=1e-12)

    def test_the_drift_is_evaluated_at_each_fast_steps_own_time(self):
        function = _Linear(drift_constant=1.0, sigma=0.0)
        config = _config(step_size=0.015, epsilon=0.01, end_time=0.045)

        GottwaldMelbourneSolver(config).solve(_equation(function), [0.0])

        assert function.times == pytest.approx([0.0, 0.01, 0.02, 0.03], abs=1e-12)


class TestTheSign:
    """χ: a fresh δ on each entry to Y = (x*, 1] signs that whole excursion."""

    @pytest.mark.parametrize("alpha", _ALPHAS)
    def test_beta_minus_one_is_beta_one_reflected(self, alpha):
        # Remark 2.1: X_α,η,-β = -X_α,η,β. With the same seed, β = ±1 draw the
        # same fast orbit and δ ≡ ±1, so dZ = dW must come out exactly negated.
        # A product of the δ, as Section 4.1 prints χ, alternates for β = -1.
        function = _Linear()
        up = GottwaldMelbourneSolver(_config(alpha=alpha, beta=1.0)).solve(
            _equation(function),
            [0.0],
        )
        down = GottwaldMelbourneSolver(_config(alpha=alpha, beta=-1.0)).solve(
            _equation(function),
            [0.0],
        )

        assert down.results[0].discretization == [
            -value for value in up.results[0].discretization
        ]


class TestTheFastState:
    def test_x0_is_the_10000th_image_of_a_uniform_draw(self):
        # Remark 4.4: x_0 = T^10000 x_0', x_0' uniform on [0, 1].
        solver = GottwaldMelbourneSolver(_config(random_seed=5))
        uniform = np.random.default_rng(5).random()
        expected = uniform
        for _ in range(10_000):
            expected = solver._thaler_map(expected)

        assert solver._initial_fast_state() == expected

    def test_paths_with_one_sided_noise_differ_between_seeds(self):
        # With β = 1 every δ is +1: only x_0 can tell two seeds apart.
        function = _Linear()
        paths = [
            GottwaldMelbourneSolver(_config(beta=1.0, random_seed=seed))
            .solve(_equation(function), [0.0])
            .results[0]
            .discretization[-1]
            for seed in (1, 2)
        ]

        assert paths[0] != paths[1]

    def test_the_perturbation_waits_10000_iterations(self):
        # Section 4.2: N(0, 1e-20), mod 1, on entering (x*, 1] at least 10^4
        # iterations after the previous one.
        solver = GottwaldMelbourneSolver(_config())
        x = 0.5 * solver._x_star
        while solver._thaler_map(x) <= solver._x_star:
            x = solver._thaler_map(x)

        early, since_early = solver._advance(x, 9_998)
        late, since_late = solver._advance(x, 9_999)

        assert (early, since_early) == (solver._thaler_map(x), 9_999)
        assert since_late == 0
        assert late != solver._thaler_map(x)
        assert abs(late - solver._thaler_map(x)) < 1e-8

    def test_no_perturbation_off_the_hyperbolic_branch(self):
        solver = GottwaldMelbourneSolver(_config())
        x = 0.9
        while solver._thaler_map(x) > solver._x_star:
            x = solver._thaler_map(x)

        assert solver._advance(x, 50_000) == (solver._thaler_map(x), 50_001)


def _final_values(config: GottwaldMelbourneConfig, function: _Linear, paths: int):
    finals = np.empty(paths)
    for path in range(paths):
        config.random_seed = 1_000 + path
        curve = GottwaldMelbourneSolver(config).solve(_equation(function), [0.0])
        finals[path] = curve.results[0].discretization[-1]
    return finals


_PATHS = 800


class TestTheLaw:
    """ẑ_ε(1) against the stable law Theorem 4.1 converges to, at ε = 0.01.

    The cf error at ε = 0.01 is about 0.07-0.1 for the paper's scheme, with a
    sampling noise of about 0.03 at 800 paths; the solver before DEQ-108 was off
    by 0.8-1.2. The rate as ε → 0 is measured in outputs/verification/deq-108.
    """

    def test_dz_equals_dw_is_the_stable_law(self):
        # Z(1) = W(1) =d X_1.5,1,0.
        finals = _final_values(_config(), _Linear(), _PATHS)

        assert _cf_error(finals, 1.5, 1.0, 0.0) < 0.15
        assert abs(np.median(finals)) < 0.25

    def test_skewed_noise_is_the_skewed_stable_law(self):
        # β = 0.5: the per-excursion sign gives X_1.5,1,0.5; a product of the
        # δ gives the symmetric law whatever β.
        finals = _final_values(_config(beta=0.5), _Linear(), _PATHS)

        skewed = _cf_error(finals, 1.5, 1.0, 0.5)
        assert skewed < 0.15
        assert skewed < _cf_error(finals, 1.5, 1.0, 0.0)

    def test_the_linear_equation_has_its_stable_law(self):
        # dZ = -Z dt + dW, Z(0) = 0: Z(1) = ∫_0^1 e^{-(1-s)} dW(s) =d
        # X_α,η',β with η' = η ((1 - e^{-α})/α)^{1/α}, for a positive kernel.
        alpha, eta, beta = 1.5, 1.0, 0.5
        finals = _final_values(_config(beta=beta), _Linear(rate=1.0), _PATHS)
        scale = eta * ((1 - math.exp(-alpha)) / alpha) ** (1 / alpha)

        assert _cf_error(finals, alpha, scale, beta) < 0.15
