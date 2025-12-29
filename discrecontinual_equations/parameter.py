from pydantic import BaseModel, Field


class Parameter(BaseModel):
    """A parameter is a constant value in an equation that is set before execution time."""

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
