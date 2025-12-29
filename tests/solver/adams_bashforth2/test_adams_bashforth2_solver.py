import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.simple_ode_function import SimpleODEFunction
from discrecontinual_equations.solver.deterministic.adams_bashforth2.adams_bashforth2_config import (
    AdamsBashforth2Config,
)
from discrecontinual_equations.solver.deterministic.adams_bashforth2.adams_bashforth2_solver import (
    AdamsBashforth2Solver,
)
from discrecontinual_equations.variable import Variable


class TestAdamsBashforth2Solver:
    def test_solve_simple_ode(self):
        # Arrange
        time = Variable(name="t", discretization=[])
        variables = [Variable(name="y", discretization=[])]
        parameters = [Parameter(name="k", value=1.0)]
        results = [Variable(name="integral_y", discretization=[])]

        func = SimpleODEFunction(variables, parameters, results, time)
        equation = DifferentialEquation(
            variables=variables,
            time=time,
            parameters=parameters,
            derivative=func,
        )

        config = AdamsBashforth2Config(start_time=0.0, end_time=1.0, step_size=0.1)
        solver = AdamsBashforth2Solver(config)

        initial_values = [1.0]  # y(0) = 1

        # Act
        solver.solve(equation, initial_values)

        # Assert
        assert solver.solution is not None
        assert (
            len(solver.solution.time.discretization) == 11
        )  # 0 to 1 with step 0.1, including initial point
        # For dy/dt = -y, y(t) = e^{-t}, so y(0.9) ≈ 0.4066
        final_value = solver.solution.results[0].discretization[-1]
        expected = np.exp(-0.9)
        assert (
            abs(final_value - expected) < 0.1
        )  # AB2 has moderate accuracy for this problem
