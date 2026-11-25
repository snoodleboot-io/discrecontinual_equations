"""Trace a one-parameter family of bifurcation diagrams.

:class:`FamilyDriver` sweeps the family parameter through the configured slice
values and, at each slice, continues the equilibrium branch in the diagram
parameter by reusing the ordinary :class:`~.builder.ContinuerBuilder`. The
equilibrium found at one slice seeds the next, so the starting point is itself
continued in the family parameter (natural parameter continuation of the seed),
which keeps the family on the same sheet of equilibria as it evolves.

Seeding each slice from the previous slice's first point is a zeroth-order
(trivial) predictor in the family parameter followed by the slice's own Newton seed
polish - natural-parameter continuation in Kuznetsov's sense (1998, eq. (10.33),
with the family parameter the fixed coordinate). It inherits that method's limit:
if the seed equilibrium passes a fold in the family parameter between two slices,
the polish converges to another equilibrium or fails, and the family jumps sheets.
Keep the family values close together near such a fold, or continue the fold curve
in two parameters instead.

References:
- Kuznetsov, Yu. A. "Elements of Applied Bifurcation Theory", 2nd ed., Applied
  Mathematical Sciences 112, Springer, 1998, doi:10.1007/b98848, Section 10.2.1,
  eq. (10.33).
"""

from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.family_config import FamilyConfig
from discrecontinual_equations.differential_equation import DifferentialEquation


class FamilySlice:
    """One diagram in the family: the family value and the branch traced there.

    Args:
        value: The family parameter's value.
        branch: The branch traced at it.
    """

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
    """A one-parameter family of bifurcation diagrams.

    Args:
        slices: The traced diagrams, in family order.
        family_index: Index of the family parameter.
    """

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
        """Trace one diagram per configured family value and collect them.

        Args:
            config: The diagram's continuation settings plus the family parameter
                and its values.
            equation: The system; its family parameter is left at the last value.
            seed: An equilibrium near the first slice's starting point.

        Returns:
            The family of branches.

        References:
        - Kuznetsov (1998), eq. (10.33) (natural continuation of the seed).
        """
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
