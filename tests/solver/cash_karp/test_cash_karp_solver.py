import numpy as np
import pytest

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
from tests.solver.deterministic_orders import (
    Mixed,
    Oscillating,
    forced,
    orders,
    rk_fixed,
)


class TestCashKarpSolver:
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
        assert (
            len(solver.solution.time.discretization) > 2
        )  # Should have adaptive steps
        # For dy/dt = -y, y(t) = e^{-t}, so y(1) ≈ 0.3679
        final_value = solver.solution.results[0].discretization[-1]
        expected = np.exp(-1.0)
        assert (
            abs(final_value - expected) < 1e-6
        )  # Should be very accurate due to adaptive stepping


STEPS = [0.1, 0.05, 0.025, 0.0125]


class TestCashKarpOrder:
    """Fifth order with local extrapolation, fourth for the embedded row (DEQ-138).

    With the rows swapped the solver advanced with the fourth-order row: orders
    4.06, 4.03, 4.02 on ``y' = y cos t``.
    """

    @pytest.mark.parametrize("field", [Oscillating, Mixed])
    def test_solution_is_fifth_order(self, field):
        observed = orders(forced(CashKarpSolver, CashKarpConfig), field, STEPS)
        assert all(4.8 < p < 5.6 for p in observed), observed

    @pytest.mark.parametrize("field", [Oscillating, Mixed])
    def test_embedded_row_is_fourth_order(self, field):
        solver = CashKarpSolver(CashKarpConfig())
        observed = rk_fixed(solver.a, solver.c, solver.b4, field, STEPS)
        assert all(3.8 < p < 4.3 for p in observed), observed
