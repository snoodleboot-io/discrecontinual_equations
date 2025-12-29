from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config import (
    EulerMaruyamaConfig,
)
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_solver import (
    EulerMaruyamaSolver,
)
from discrecontinual_equations.variable import Variable


class StochasticTestFunction(StochasticFunction):
    """Test SDE function: dX = -α X dt + σ dW (Ornstein-Uhlenbeck process)"""

    def __init__(self, variables, parameters, time=None):
        results = [Variable(name="dX/dt", discretization=[])]
        super().__init__(variables, parameters, results, time)
        self.alpha = parameters[0].value  # drift coefficient
        self.sigma = parameters[1].value  # diffusion coefficient

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        """Drift term: μ(x,t) = -α x"""
        x = point[0]
        return [-self.alpha * x]

    def diffusion(self, point: list[float], time: float | None = None) -> list[float]:
        """Diffusion term: σ(x,t) = σ"""
        return [self.sigma]


class TestEulerMaruyamaSolver:
    def test_solve_stochastic_ode(self):
        # Arrange - Ornstein-Uhlenbeck process: dX = -α X dt + σ dW
        time = Variable(name="t", discretization=[])
        variables = [Variable(name="X", discretization=[])]
        parameters = [
            Parameter(name="alpha", value=1.0),  # drift coefficient
            Parameter(name="sigma", value=0.5),  # diffusion coefficient
        ]

        func = StochasticTestFunction(variables, parameters, time)
        equation = DifferentialEquation(
            variables=variables,
            time=time,
            parameters=parameters,
            derivative=func,
        )

        config = EulerMaruyamaConfig(
            start_time=0.0,
            end_time=1.0,
            step_size=0.01,
            random_seed=42,  # For reproducible results
        )
        solver = EulerMaruyamaSolver(config)

        initial_values = [1.0]  # X(0) = 1

        # Act
        solver.solve(equation, initial_values)

        # Assert
        assert solver.solution is not None
        assert len(solver.solution.time.discretization) == 101  # 0 to 1 with step 0.01
        assert len(solver.solution.results[0].discretization) == 101

        # Check that solution exists and is reasonable
        final_value = solver.solution.results[0].discretization[-1]
        assert isinstance(final_value, float)

        # For Ornstein-Uhlenbeck, the solution should be mean-reverting
        # With α=1, σ=0.5, starting at x=1, it should tend toward 0
        # But due to stochasticity, we just check it's a reasonable value
        assert -2.0 < final_value < 3.0  # Allow reasonable range due to noise
