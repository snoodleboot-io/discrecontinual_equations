import numpy as np
import pytest

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.simple_ode_function import SimpleODEFunction
from discrecontinual_equations.solver.deterministic.dormand_prince89.dormand_prince89_config import (
    DormandPrince89Config,
)
from discrecontinual_equations.solver.deterministic.dormand_prince89.dormand_prince89_solver import (
    DormandPrince89Solver,
)
from discrecontinual_equations.solver.deterministic.prince_dormand87.prince_dormand87_config import (
    PrinceDormand87Config,
)
from discrecontinual_equations.solver.deterministic.prince_dormand87.prince_dormand87_solver import (
    PrinceDormand87Solver,
)
from discrecontinual_equations.variable import Variable
from tests.solver.deterministic_orders import (
    END,
    Autonomous,
    Oscillating,
    equation,
    forced,
    orders,
    rk_fixed,
)


class TestPrinceDormand87Solver:
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

        config = PrinceDormand87Config(
            start_time=0.0,
            end_time=1.0,
            initial_step_size=0.1,
            absolute_tolerance=1e-8,
            relative_tolerance=1e-8,
        )
        solver = PrinceDormand87Solver(config)

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
        )  # Should be very accurate due to high order


STEPS = [0.5, 0.4, 0.25, 0.2]


class TestPrinceDormand87Order:
    """Eighth order to advance, seventh embedded, down to rounding (DEQ-139).

    With ``b8[7] = 561292735/797845732`` the weights summed to ``1 - 3.1e-7``
    and the error stalled near 7e-8 on ``y' = -y^2`` whatever the step.
    """

    @pytest.mark.parametrize("field", [Oscillating, Autonomous])
    def test_solution_is_eighth_order(self, field):
        observed = orders(
            forced(PrinceDormand87Solver, PrinceDormand87Config),
            field,
            STEPS,
        )
        assert all(6.8 < p < 8.6 for p in observed), observed

    def test_embedded_row_is_seventh_order(self):
        solver = PrinceDormand87Solver(PrinceDormand87Config())
        observed = rk_fixed(solver.a, solver.c, solver.b7, Autonomous, STEPS)
        assert all(6.7 < p < 7.3 for p in observed), observed

    def test_error_reaches_rounding_level(self):
        curve = forced(PrinceDormand87Solver, PrinceDormand87Config)(1 / 16).solve(
            equation(Autonomous),
            [1.0],
        )
        error = abs(curve.results[0].discretization[-1] - Autonomous.exact(END))
        assert error < 1e-14


class TestDeprecatedNames:
    """``DormandPrince89Solver`` and its config still work, and warn."""

    def test_old_solver_name_warns_and_is_the_new_solver(self):
        with pytest.warns(DeprecationWarning, match="PrinceDormand87Config"):
            config = DormandPrince89Config(end_time=2.0)
        with pytest.warns(DeprecationWarning, match="PrinceDormand87Solver"):
            solver = DormandPrince89Solver(config)
        assert isinstance(solver, PrinceDormand87Solver)
        assert isinstance(config, PrinceDormand87Config)
        curve = solver.solve(equation(Autonomous), [1.0])
        new = PrinceDormand87Solver(PrinceDormand87Config(end_time=2.0)).solve(
            equation(Autonomous),
            [1.0],
        )
        assert curve.results[0].discretization == new.results[0].discretization

    def test_old_b9_attribute_warns_and_holds_the_old_values(self):
        # The values the alias exposed as ``b9`` before DEQ-139 (git dbc47d2~25),
        # which RKSUITE's CONST lists as the seventh-order weights.
        old_b9 = [
            13451932 / 455176623, 0, 0, 0, 0,
            -808719846 / 976000145, 1757004468 / 5645159321,
            656045339 / 265891186, -3867574721 / 1518517206,
            465885868 / 322736535, 53011238 / 667516719, 2 / 45, 0,
        ]  # fmt: skip
        with pytest.warns(DeprecationWarning, match="deprecated"):
            solver = DormandPrince89Solver(DormandPrince89Config(end_time=2.0))
        with pytest.warns(DeprecationWarning, match="b7"):
            b9 = solver.b9
        np.testing.assert_allclose(b9, solver.b7, rtol=0, atol=0)
        np.testing.assert_allclose(b9, old_b9, rtol=1e-15, atol=1e-18)
