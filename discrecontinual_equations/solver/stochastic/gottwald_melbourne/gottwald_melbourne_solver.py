import math

import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.solver.stochastic.gottwald_melbourne.gottwald_melbourne_config import (
    GottwaldMelbourneConfig,
    epsilon_exceeds_step,
)

# Remark 4.4: x_0 is the image of a uniform draw under this many iterations.
_BURN_IN = 10_000
# Section 4.2: a perturbation of standard deviation 1e-10, at least this far apart.
_PERTURBATION_INTERVAL = 10_000
_PERTURBATION_SIZE = 1e-10


class GottwaldMelbourneSolver(Solver):
    """
    Gottwald-Melbourne method for solving stochastic differential equations driven by α-stable Lévy processes.

    This solver implements the deterministic homogenisation approach based on the Thaler map,
    which approximates Marcus SDEs with non-Lipschitz coefficients driven by α-stable noise
    for α ∈ (1,2).

    The method uses a fast-slow system where the slow dynamics approximate the SDE solution,
    and the fast dynamics on the Thaler map generate the required α-stable increments.

    References:
    - Gottwald, G. A. and Melbourne, I. "Simulation of non-Lipschitz stochastic
      differential equations driven by α-stable noise: a method based on deterministic
      homogenisation", Multiscale Model. Simul. 19(2), 2021; arXiv:2004.09914
    """

    def __init__(self, solver_config: GottwaldMelbourneConfig):
        super().__init__(solver_config=solver_config)

        # Validate parameters
        if not (1 < self.solver_config.alpha < 2):
            raise ValueError("α must be in the range (1, 2)")
        if not (-1 <= self.solver_config.beta <= 1):
            raise ValueError("β must be in the range [-1, 1]")
        if self.solver_config.eta <= 0:
            raise ValueError("η must be positive")
        if self.solver_config.epsilon <= 0:
            raise ValueError("ε must be positive")
        # With ε above the step, each grid step still took one fast step, so the
        # path ran ε/step_size times faster than its own clock: ε = 1 on a grid of
        # 1/20000 ran 20,000 units of slow time and overflowed in the z update.
        if self.solver_config.epsilon > self.solver_config.step_size:
            raise ValueError(
                epsilon_exceeds_step(
                    self.solver_config.epsilon,
                    self.solver_config.step_size,
                ),
            )

        # A generator of this solver's own, never np.random.seed. Seeding the global
        # stream is a correctness hazard rather than a convenience: it changes every
        # draw made anywhere afterwards, so one solver's result can depend on how many
        # draws something else happened to take first, and two solvers constructed with
        # the same seed share one stream instead of repeating one another. A seed of
        # None still means fresh entropy, as before.
        self._generator = np.random.default_rng(self.solver_config.random_seed)

        # Compute Thaler map parameters
        self._gamma = self.solver_config.gamma
        self._x_star = self._compute_x_star()
        self._d_alpha = self._compute_d_alpha()
        # v on [0, x*] and on (x*, 1] (Section 4.1).
        self._v_laminar = (
            self.solver_config.eta
            * self._d_alpha ** (-self._gamma)
            * (1 - 2 ** (self._gamma - 1)) ** (-self._gamma)
        )
        self._v_hyperbolic = self._v_laminar / (1 - 2 ** (1 - self._gamma))

    def _compute_x_star(self) -> float:
        """Compute the fixed point x* of the Thaler map."""
        gamma = self._gamma

        # Solve x*^{1-γ} + (1 + x*)^{1-γ} = 2
        # Use numerical solution
        def equation(x):
            return x ** (1 - gamma) + (1 + x) ** (1 - gamma) - 2

        # Binary search for solution
        low, high = 0.0, 1.0
        for _ in range(100):
            mid = (low + high) / 2
            if equation(mid) < 0:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    def _compute_d_alpha(self) -> float:
        """d_α = α^α (1-γ)/(2^{1-γ}-1) g_α, g_α = Γ(1-α) cos(απ/2) (paper (2.4), (2.5))."""
        alpha = self.solver_config.alpha
        gamma = self._gamma
        g_alpha = math.gamma(1 - alpha) * math.cos(math.pi * alpha / 2)
        return alpha**alpha * (1 - gamma) / (2 ** (1 - gamma) - 1) * g_alpha

    def _thaler_map(self, x: float) -> float:
        """
        The Thaler map T(x) = (x^{1-γ} + (1+x)^{1-γ} - 1)^{1/(1-γ)} mod 1 (Remark 2.2).

        (1+x)^{1-γ} - 1 is formed by expm1/log1p so that it keeps its digits near
        the neutral fixed point at 0, where the orbit spends most of its time.
        """
        gamma = self._gamma
        inner = x ** (1 - gamma) + math.expm1((1 - gamma) * math.log1p(x))
        return inner ** (1 / (1 - gamma)) % 1.0

    def _compute_observable(self, x: float, chi: int) -> float:
        """
        The observable χ v(x) of the fast dynamics (Section 4.1).

        v = η d_α^{-γ} (1 - 2^{γ-1})^{-γ} ṽ with ṽ = 1 on [0, x*] and
        ṽ = (1 - 2^{1-γ})^{-1} on (x*, 1], which is negative: v has mean zero
        under the invariant measure. The sign χ multiplies both branches.
        """
        if x <= self._x_star:
            return chi * self._v_laminar
        return chi * self._v_hyperbolic

    def _draw_sign(self) -> int:
        """δ = ±1 with P(δ = ±1) = (1 ± β)/2."""
        return 1 if self._generator.random() < (1 + self.solver_config.beta) / 2 else -1

    def _initial_fast_state(self) -> float:
        """x_0 = T^10000 x_0' with x_0' uniform on [0, 1] (Remark 4.4)."""
        x = self._generator.random()
        for _ in range(_BURN_IN):
            x = self._thaler_map(x)
        return x

    def _advance(self, x: float, since_perturbation: int) -> tuple[float, int]:
        """
        One step of the fast map, and the iterations since the last perturbation.

        In floating point the orbit can be caught on a spurious periodic orbit of
        the map, so on entering (x*, 1] at least 10^4 iterations after the previous
        one it gets a N(0, 10^-20) perturbation, mod 1 (Section 4.2).
        """
        x = self._thaler_map(x)
        since_perturbation += 1
        if x > self._x_star and since_perturbation >= _PERTURBATION_INTERVAL:
            x = (x + self._generator.normal(0.0, _PERTURBATION_SIZE)) % 1.0
            since_perturbation = 0
        return x, since_perturbation

    def _fast_steps(self, times: np.ndarray) -> list[int]:
        """
        n_i = ⌊(t_i - t_0)/ε⌋, the fast steps taken by each grid time (Theorem 4.1).

        A quotient that falls a few ulps short of a whole number, as 0.3/0.1 does,
        is counted as that number.
        """
        epsilon = self.solver_config.epsilon
        start = times[0]
        counts = []
        for t in times:
            quotient = (t - start) / epsilon
            counts.append(math.floor(quotient * (1 + 1e-12) + 1e-12))
        return counts

    def solve(
        self,
        equation: DifferentialEquation,
        initial_values: list[float],
    ) -> Curve:
        """
        Solve the SDE by the fast-slow map (4.2) of Gottwald & Melbourne.

        z_{n+1} = z_n + ε a(z_n) + ε^γ b(z_n) χ_n v(x_n), x_{n+1} = T x_n, and the
        value at grid time t_i is z_{n_i} with n_i = ⌊(t_i - t_0)/ε⌋ (Theorem 4.1).
        χ_n is the δ drawn when the orbit last entered (x*, 1]: one sign for the
        hyperbolic step and the laminar steps that follow it.

        Args:
            equation: The SDE to solve (must have stochastic derivative function)
            initial_values: Initial values for all variables
        """
        self.solution = self._new_solution(equation)

        start = self.solver_config.start_time
        epsilon = self.solver_config.epsilon
        kick = epsilon**self._gamma
        z = np.array(initial_values, dtype=float)

        x = self._initial_fast_state()
        # The excursion under way at x_0 began with a δ of its own.
        chi = self._draw_sign()
        since_perturbation = 0

        self.solution.append([start, [0] * len(initial_values), z.tolist()])

        # The time grid is read once: ``times`` builds it afresh on every read,
        # so reading it per step made a path of K steps cost O(K^2) (DEQ-47).
        times = self.solver_config.times
        steps = self._fast_steps(times)
        n = 0
        for i in range(1, len(times)):
            while n < steps[i]:
                if x > self._x_star:
                    chi = self._draw_sign()
                v = self._compute_observable(x, chi)

                # The drift's time is the fast step's own slow time, t_0 + n ε.
                t_n = start + n * epsilon
                drift = np.array(equation.derivative.eval(point=z.tolist(), time=t_n))
                diffusion = np.array(
                    equation.derivative.diffusion(point=z.tolist(), time=t_n),
                )
                z = z + epsilon * drift + kick * diffusion * v

                x, since_perturbation = self._advance(x, since_perturbation)
                n += 1

            self.solution.append([times[i], [0] * len(initial_values), z.tolist()])
        return self.solution
