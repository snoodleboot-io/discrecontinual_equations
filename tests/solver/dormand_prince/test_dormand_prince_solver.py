import numpy as np
import pytest

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.simple_ode_function import SimpleODEFunction
from discrecontinual_equations.solver.deterministic.dormand_prince.dormand_prince_config import (
    DormandPrinceConfig,
)
from discrecontinual_equations.solver.deterministic.dormand_prince.dormand_prince_solver import (
    DormandPrinceSolver,
)
from discrecontinual_equations.variable import Variable
from tests.solver.deterministic_orders import (
    Mixed,
    Oscillating,
    forced,
    orders,
    rk_fixed,
)


class TestDormandPrinceSolver:
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

        config = DormandPrinceConfig(
            start_time=0.0,
            end_time=1.0,
            initial_step_size=0.1,
            absolute_tolerance=1e-8,
            relative_tolerance=1e-8,
        )
        solver = DormandPrinceSolver(config)

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


class TestDormandPrinceOrder:
    """Fifth order on fields that depend on t, fourth for the embedded row (DEQ-137).

    With ``c_5 = 7/9`` the fifth stage saw the wrong time and both rows fell to
    first order on these fields: errors 1.28e-2, 6.36e-3, 3.16e-3, 1.58e-3 on
    ``y' = y cos t``.
    """

    @pytest.mark.parametrize("field", [Oscillating, Mixed])
    def test_solution_is_fifth_order_on_a_time_dependent_field(self, field):
        observed = orders(
            forced(DormandPrinceSolver, DormandPrinceConfig), field, STEPS
        )
        assert all(4.8 < p < 6.0 for p in observed), observed

    @pytest.mark.parametrize("field", [Oscillating, Mixed])
    def test_embedded_row_is_fourth_order_on_a_time_dependent_field(self, field):
        solver = DormandPrinceSolver(DormandPrinceConfig())
        observed = rk_fixed(solver.a, solver.c, solver.b4, field, STEPS)
        assert all(3.6 < p < 4.4 for p in observed), observed
