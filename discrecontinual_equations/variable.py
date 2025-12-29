from pydantic import BaseModel, Field


class Variable(BaseModel):
    """A variable is a that changes with the vector of time."""

    discretization: list[float] = Field(
        default=[],
        description="Values of the variable",
    )
    range: list[float] = Field(
        default=[],
        description="Range of values of the variable",
    )
    name: str | None = Field(
        default=None,
        alias="name",
        description="Name of the variable",
    )
    abbreviation: str | None = Field(
        default=None,
        alias="abbreviation",
        description="Abbreviation of the variable used in plotting",
    )

    def __init_subclass__(
        cls,
        abbreviation: str | None = None,
        name: str | None = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        cls.name = name
        cls.abbreviation = abbreviation
