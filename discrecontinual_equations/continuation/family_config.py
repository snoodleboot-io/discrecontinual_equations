"""Configuration for evolving a bifurcation diagram over a second parameter.

A :class:`FamilyConfig` is an ordinary one-parameter continuation configuration
(the diagram is traced in ``continuation_parameter_index``) plus the identity of a
second, *family* parameter and the values at which to slice it. The family driver
re-runs the continuation at each slice, so the result is a one-parameter family of
bifurcation diagrams - the diagram evolving as the family parameter varies.
"""

from pydantic import Field

from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)


class FamilyConfig(ContinuationConfig):
    """A continuation configuration swept over a second parameter."""

    family_parameter_index: int = Field(
        description="Index of the parameter the diagram is evolved over",
    )
    family_values: list[float] = Field(
        description="Values of the family parameter at which to trace a diagram",
    )
