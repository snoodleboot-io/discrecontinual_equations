"""A differential equation: its variables, time, parameters and right-hand side.

``DifferentialEquation`` bundles what a solver needs for ``y' = f(t, y; p)``:
the state ``variables``, the ``time`` variable, the ``parameters`` ``p`` and the
``derivative`` function ``f`` (a
:class:`~discrecontinual_equations.function.function.Function`, whose ``eval``
the solvers call). For an SDE, ``derivative`` is a stochastic function that
also supplies the diffusion.
"""

from pydantic import BaseModel, ConfigDict, Field

from discrecontinual_equations.function.function import Function
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class DifferentialEquation(BaseModel):
    """An equation ``y' = f(t, y; p)`` for a solver.

    Attributes:
        variables: The state variables ``y``.
        time: The time variable ``t``.
        parameters: The parameters ``p``.
        derivative: The right-hand side ``f``; solvers call
            ``derivative.eval(point, time)``.
    """

    variables: list[Variable] = Field(
        description="List of variables in the differential equation",
    )
    time: Variable = Field(description="Time variable for the differential equation")
    parameters: list[Parameter] = Field(
        description="List of parameters in the differential equation",
    )
    derivative: Function = Field(description="Function defining the derivative terms")

    model_config = ConfigDict(arbitrary_types_allowed=True)
