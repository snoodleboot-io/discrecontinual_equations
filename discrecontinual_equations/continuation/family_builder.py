"""Trace a one-parameter family of bifurcation diagrams.

:class:`FamilyDriver` sweeps the family parameter through the configured slice
values and, at each slice, continues the equilibrium branch in the diagram
parameter by reusing the ordinary :class:`~.builder.ContinuerBuilder`. The
equilibrium found at one slice seeds the next, so the starting point is itself
continued in the family parameter (natural parameter continuation of the seed),
which keeps the family on the same sheet of equilibria as it evolves.
"""

from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.family_config import FamilyConfig
from discrecontinual_equations.differential_equation import DifferentialEquation


class FamilySlice:
    """One diagram in the family: the family value and the branch traced there."""

    __slots__ = ["_branch", "_value"]

    def __init__(self, value: float, branch: Branch) -> None:
        self._value = value
        self._branch = branch

    @property
    def value(self) -> float:
        """Value of the family parameter for this slice."""
        return self._value

    @property
    def branch(self) -> Branch:
        """Equilibrium branch traced at this slice."""
        return self._branch


class ParameterFamily:
    """A one-parameter family of bifurcation diagrams."""

    __slots__ = ["_family_index", "_slices"]

    def __init__(self, slices: list[FamilySlice], family_index: int) -> None:
        self._slices = slices
        self._family_index = family_index

    @property
    def slices(self) -> list[FamilySlice]:
        """The traced diagrams, in family-parameter order."""
        return self._slices

    @property
    def family_index(self) -> int:
        """Index of the parameter the family is evolved over."""
        return self._family_index

    @property
    def values(self) -> list[float]:
        """The family-parameter values, in order."""
        return [item.value for item in self._slices]


class FamilyDriver:
    """Evolve a bifurcation diagram over a second parameter."""

    @staticmethod
    def run(
        config: FamilyConfig,
        equation: DifferentialEquation,
        seed: list[float],
    ) -> ParameterFamily:
        """Trace one diagram per configured family value and collect them."""
        family_index = config.family_parameter_index
        current = list(seed)
        slices: list[FamilySlice] = []
        for value in config.family_values:
            equation.derivative.parameters[family_index].value = value
            continuer = ContinuerBuilder.build(config, equation)
            branch = continuer.solve(equation, current)
            slices.append(FamilySlice(value, branch))
            if len(branch) > 0:
                current = list(branch[0].state)
        return ParameterFamily(slices, family_index)
