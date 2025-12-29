import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.simple_ode_function import SimpleODEFunction
from discrecontinual_equations.solver.deterministic.cash_karp.cash_karp_config import (
    CashKarpConfig,
)
from discrecontinual_equations.solver.deterministic.cash_karp.cash_karp_solver import (
    CashKarpSolver,
)
from discrecontinual_equations.variable import Variable


class TestCashKarpSolver:
    def test_solve_simple_ode(self):
        # Arrange
        time = Variable(name="t", discretization=[])
        variables = [Variable(name="y", discretization=[])]
        parameters = [Parameter(name="k", value=1.0)]
        results = [Variable(name="integral_y", discretization=[])]

        func = SimpleODEFunction(variables, parameters, results, time)
        equation = DifferentialEquation(variables=variables, time=time, parameters=parameters, derivative=func)

        config = CashKarpConfig(
            start_time=0.0,
            end_time=1.0,
            initial_step_size=0.1,
            absolute_tolerance=1e-8,
            relative_tolerance=1e-8,
        )
        solver = CashKarpSolver(config)

        initial_values = [1.0]  # y(0) = 1

        # Act
        solver.solve(equation, initial_values)

        # Assert
        assert solver.solution is not None
        assert len(solver.solution.time.discretization) > 2  # Should have adaptive steps
        # For dy/dt = -y, y(t) = e^{-t}, so y(1) ≈ 0.3679
        final_value = solver.solution.results[0].discretization[-1]
        expected = np.exp(-1.0)
        assert abs(final_value - expected) < 1e-6  # Should be very accurate due to adaptive stepping
