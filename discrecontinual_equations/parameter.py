"""A named constant of an equation, such as a rate or a coupling.

A ``Parameter`` holds a ``value`` fixed before a solve, with a ``name`` and a
plotting ``abbreviation``. A subclass may set defaults for both with class
keywords: ``class Rate(Parameter, name="rate", abbreviation="k")``.
"""

from pydantic import BaseModel, Field


class Parameter(BaseModel):
    """A parameter is a constant value in an equation that is set before execution time.

    Attributes:
        value: The value.
        name: The name.
        abbreviation: A short label used in plots.
    """

    value: float | None = Field(default=None, description="Value of the parameter")
    name: str | None = Field(default=None, description="Name of the parameter")
    abbreviation: str | None = Field(
        default=None,
        description="Abbreviation of the parameter - used in plots.",
    )

    def __init_subclass__(
        cls,
        abbreviation: str | None = None,
        name: str | None = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)

        cls.__default_name = name
        cls.__abbreviation = abbreviation

        if name is not None:
            cls.model_fields["name"].default = name
        if abbreviation is not None:
            cls.model_fields["abbreviation"].default = abbreviation

        cls.model_rebuild()
