"""A quantity that varies along a solution, with the values it has taken.

A ``Variable`` is a named list: ``discretization`` holds its values at the
nodes of a solve, ``range`` an optional plotting range, and ``name`` and
``abbreviation`` its labels. Solvers copy an equation's variables with an empty
``discretization`` for each solve and append to the copies.
"""

from pydantic import BaseModel, Field


class Variable(BaseModel):
    """A variable is a quantity that changes with time.

    Attributes:
        discretization: Its values, one per node.
        range: A range of values, used in plotting.
        name: The name.
        abbreviation: A short label used in plotting.
    """

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
