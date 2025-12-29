from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.simple_ode_function import SimpleODEFunction
from discrecontinual_equations.variable import Variable


class TestSimpleODEFunction:
    def test_eval(self):
        # Arrange
        variables = [Variable(name="y", discretization=[])]
        parameters = [Parameter(name="k", value=1.0)]
        results = [Variable(name="dy/dt", discretization=[])]
        func = SimpleODEFunction(variables, parameters, results)

        # Act
        result = func.eval([2.0], 0.0)

        # Assert
        assert result == [-2.0]
