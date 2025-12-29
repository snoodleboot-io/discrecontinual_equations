from pydantic import BaseModel, ConfigDict, Field

from discrecontinual_equations.function.function import Function
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class DifferentialEquation(BaseModel):
    variables: list[Variable] = Field(
        description="List of variables in the differential equation",
    )
    time: Variable = Field(description="Time variable for the differential equation")
    parameters: list[Parameter] = Field(
        description="List of parameters in the differential equation",
    )
    derivative: Function = Field(description="Function defining the derivative terms")

    model_config = ConfigDict(arbitrary_types_allowed=True)
