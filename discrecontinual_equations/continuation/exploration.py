"""Automatic recursive exploration of a bifurcation structure.

Continuation is inherently recursive: a codimension-``k`` bifurcation is found as a
degeneracy along the continuation of a codimension-``(k-1)`` manifold. This module
automates that. Starting from an equilibrium it continues in the first parameter
and runs every applicable codimension-1 detector; each detected fold or Hopf is
escalated by continuing its curve in a second parameter, where every applicable
codimension-2 degeneracy is sought; each cusp or generalized Hopf is escalated
again into a third parameter, where the codimension-3 points are sought. The result
is a tree of continuations - the bifurcation structure discovered without being
told in advance which bifurcations to expect.

"Every applicable" is dimension-aware: a detector is only run where its defining
eigenvalue configuration can exist (a Hopf needs two dimensions, a zero-Hopf three,
a Hopf-Hopf four), which keeps scalar and planar systems from reporting impossible
degeneracies. Each continuation is swept in both directions from its seed so a
degeneracy on either side is found. Escalation reaches as far as the library has
defining systems for - fold and Hopf curves, then cusp and generalized-Hopf curves;
degeneracies without a continuation system yet are recorded as terminal.
"""

from pydantic import BaseModel, ConfigDict, Field

from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.codim2 import Codim2Point
from discrecontinual_equations.continuation.codim2_builder import (
    Codim2Driver,
    CurveSeed,
)
from discrecontinual_equations.continuation.codim2_config import Codim2Config
from discrecontinual_equations.continuation.codim3 import Codim3Point
from discrecontinual_equations.continuation.codim3_builder import (
    Codim3Driver,
    Codim3Seed,
)
from discrecontinual_equations.continuation.codim3_config import Codim3Config
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.differential_equation import DifferentialEquation

_CODIM1_DETECTORS = ["fold", "branch_point", "hopf"]
_FOLD_CURVE_DETECTORS = ["cusp", "fold_bogdanov_takens", "zero_hopf"]
_HOPF_CURVE_DETECTORS = [
    "generalized_hopf",
    "bogdanov_takens",
    "zero_hopf",
    "hopf_hopf",
]
_CURVE_FROM_CODIM1 = {"fold": "fold", "hopf": "hopf"}
_CURVE_DETECTORS = {"fold": _FOLD_CURVE_DETECTORS, "hopf": _HOPF_CURVE_DETECTORS}
_CODIM3_FROM_CODIM2 = {
    "cusp": ("cusp", ["swallowtail"]),
    "generalized_hopf": ("generalized_hopf", ["degenerate_bautin"]),
    "bogdanov_takens": (
        "bogdanov_takens",
        ["degenerate_bogdanov_takens", "degenerate_bogdanov_takens_b"],
    ),
}
_MINIMUM_DIMENSION = {
    "fold": 1,
    "branch_point": 1,
    "cusp": 1,
    "swallowtail": 1,
    "hopf": 2,
    "bogdanov_takens": 2,
    "fold_bogdanov_takens": 2,
    "generalized_hopf": 2,
    "degenerate_bautin": 2,
    "degenerate_bogdanov_takens": 2,
    "degenerate_bogdanov_takens_b": 2,
    "zero_hopf": 3,
    "hopf_hopf": 4,
}
_PROVIDER = "automatic_differentiation"
_MAX_BRANCHES = 6
_MAX_POINTS_PER_KIND = 6
_DIRECTIONS = (1, -1)
_ROUNDING = 3
_CUSP_BT_SEPARATION = 0.1
_CODIM2 = 2
_CODIM3 = 3


class ExplorationConfig(BaseModel):
    """What to explore: the ordered active parameters and their search ranges."""

    parameters: list[int] = Field(
        description="Ordered parameter indices to activate as codimension grows",
    )
    ranges: list[tuple[float, float]] = Field(
        description="Search bounds for each active parameter, in the same order",
    )
    initial_parameter: float = Field(description="Start value of the first parameter")
    initial_step: float = Field(default=0.01, description="Initial arclength step")
    maximum_step: float = Field(default=0.05, description="Maximum arclength step")
    maximum_points: int = Field(default=1500, description="Maximum points per curve")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DetectedBifurcation:
    """A single bifurcation found during exploration."""

    __slots__ = ["codimension", "frequency", "kind", "parameters", "state"]

    def __init__(
        self,
        kind: str,
        codimension: int,
        state: list[float],
        parameters: tuple[float, ...],
        frequency: float | None,
    ) -> None:
        self.kind = kind
        self.codimension = codimension
        self.state = state
        self.parameters = parameters
        self.frequency = frequency


class ExplorationNode:
    """One continuation in the tree: its detected points and escalated children."""

    __slots__ = [
        "children",
        "codimension",
        "depth",
        "description",
        "points",
        "segments",
    ]

    def __init__(
        self,
        description: str,
        codimension: int,
        points: list[DetectedBifurcation],
        children: list["ExplorationNode"],
    ) -> None:
        self.description = description
        self.codimension = codimension
        self.points = points
        self.children = children
        self.segments: list[list[tuple[float, ...]]] = []
        self.depth: float | None = None

    def flatten(self) -> list[DetectedBifurcation]:
        """All bifurcations in this subtree, depth-first."""
        found = list(self.points)
        for child in self.children:
            found.extend(child.flatten())
        return found


class BifurcationExplorer:
    """Continue an equilibrium and recursively escalate every degeneracy found."""

    @staticmethod
    def explore(
        config: ExplorationConfig,
        equation: DifferentialEquation,
        seed: list[float],
    ) -> ExplorationNode:
        """Return the tree of continuations rooted at the equilibrium branch."""
        base = _snapshot(equation, config.parameters)
        points = BifurcationExplorer._equilibrium_points(config, equation, seed)
        children = [
            child
            for point in points[:_MAX_BRANCHES]
            if (child := BifurcationExplorer._escalate(point, config, equation, base))
        ]
        return ExplorationNode("equilibrium branch", 1, points, children)

    @staticmethod
    def _equilibrium_points(
        config: ExplorationConfig,
        equation: DifferentialEquation,
        seed: list[float],
    ) -> list[DetectedBifurcation]:
        dimension = len(equation.variables)
        detectors = _applicable(_CODIM1_DETECTORS, dimension)
        lower, upper = config.ranges[0]
        found: list[DetectedBifurcation] = []
        for direction in _DIRECTIONS:
            continuation = ContinuationConfig(
                continuation_parameter_index=config.parameters[0],
                initial_parameter=config.initial_parameter,
                direction=direction,
                measure="norm",
                parameter_lower_bound=lower,
                parameter_upper_bound=upper,
                initial_step=config.initial_step,
                maximum_step=config.maximum_step,
                maximum_points=config.maximum_points,
                detectors=detectors,
            )
            branch = ContinuerBuilder.build(continuation, equation).solve(
                equation,
                seed,
            )
            found.extend(_from_branch(branch))
        return _dedup(found)

    @staticmethod
    def _escalate(
        point: DetectedBifurcation,
        config: ExplorationConfig,
        equation: DifferentialEquation,
        base: list[float],
    ) -> ExplorationNode | None:
        if point.kind not in _CURVE_FROM_CODIM1 or len(config.parameters) < _CODIM2:
            return None
        return BifurcationExplorer._codim2_curve(point, config, equation, base)

    @staticmethod
    def _codim2_curve(
        point: DetectedBifurcation,
        config: ExplorationConfig,
        equation: DifferentialEquation,
        base: list[float],
    ) -> ExplorationNode:
        dimension = len(equation.variables)
        curve = _CURVE_FROM_CODIM1[point.kind]
        second = config.parameters[1]
        lower, upper = config.ranges[1]
        detectors = _applicable(_CURVE_DETECTORS[curve], dimension)
        seed = CurveSeed(
            state=point.state,
            parameter_a=point.parameters[0],
            frequency=point.frequency,
        )
        start = list(base)
        start[0] = point.parameters[0]
        found: list[DetectedBifurcation] = []
        segments: list[list[tuple[float, ...]]] = []
        for direction in _DIRECTIONS:
            _restore(equation, config.parameters, start)
            codim2 = Codim2Config(
                continuation_parameter_index=config.parameters[0],
                second_parameter_index=second,
                curve=curve,
                codim2_detectors=detectors,
                derivative_provider=_PROVIDER,
                initial_parameter=base[1],
                direction=direction,
                parameter_lower_bound=lower,
                parameter_upper_bound=upper,
                initial_step=config.initial_step,
                maximum_step=config.maximum_step,
                maximum_points=config.maximum_points,
            )
            result = Codim2Driver.run(codim2, equation, seed)
            found.extend(_from_codim2(item) for item in result.points)
            segments.append([(a, b) for a, b, _omega in result.curve_parameters])
        points = _filter_cusp_near_bt(_suppress_floods(_dedup(found)))
        children = [
            child
            for item in points[:_MAX_BRANCHES]
            if (child := BifurcationExplorer._codim3(item, config, equation, base))
        ]
        node = ExplorationNode(f"{curve} curve", _CODIM2, points, children)
        node.segments = segments
        node.depth = base[2] if len(base) > _CODIM2 else None
        return node

    @staticmethod
    def _codim3(
        point: DetectedBifurcation,
        config: ExplorationConfig,
        equation: DifferentialEquation,
        base: list[float],
    ) -> ExplorationNode | None:
        dimension = len(equation.variables)
        if len(config.parameters) < _CODIM3 or point.kind not in _CODIM3_FROM_CODIM2:
            return None
        curve, detectors = _CODIM3_FROM_CODIM2[point.kind]
        detectors = _applicable(detectors, dimension)
        third = config.parameters[2]
        lower, upper = config.ranges[2]
        seed = Codim3Seed(
            state=point.state,
            parameter_a=point.parameters[0],
            parameter_b=point.parameters[1],
            frequency=point.frequency,
        )
        start = list(base)
        start[0] = point.parameters[0]
        start[1] = point.parameters[1]
        found: list[DetectedBifurcation] = []
        segments: list[list[tuple[float, ...]]] = []
        for direction in _DIRECTIONS:
            _restore(equation, config.parameters, start)
            codim3 = Codim3Config(
                continuation_parameter_index=config.parameters[0],
                second_parameter_index=config.parameters[1],
                third_parameter_index=third,
                codim3_curve=curve,
                codim3_detectors=detectors,
                derivative_provider=_PROVIDER,
                initial_parameter=base[2],
                direction=direction,
                parameter_lower_bound=lower,
                parameter_upper_bound=upper,
                initial_step=config.initial_step,
                maximum_step=config.maximum_step,
                maximum_points=config.maximum_points,
            )
            result = Codim3Driver.run(codim3, equation, seed)
            found.extend(_from_codim3(item) for item in result.points)
            segments.append([(a, b, c) for a, b, c in result.curve_parameters])
        node = ExplorationNode(
            f"{curve} curve",
            _CODIM3,
            _suppress_floods(_dedup(found)),
            [],
        )
        node.segments = segments
        return node


def _snapshot(equation: DifferentialEquation, indices: list[int]) -> list[float]:
    return [equation.derivative.parameters[i].value for i in indices]


def _restore(
    equation: DifferentialEquation,
    indices: list[int],
    values: list[float],
) -> None:
    for index, value in zip(indices, values, strict=True):
        equation.derivative.parameters[index].value = value


def _applicable(detectors: list[str], dimension: int) -> list[str]:
    return [d for d in detectors if _MINIMUM_DIMENSION.get(d, 1) <= dimension]


def _suppress_floods(
    points: list[DetectedBifurcation],
) -> list[DetectedBifurcation]:
    counts: dict[str, int] = {}
    for point in points:
        counts[point.kind] = counts.get(point.kind, 0) + 1
    return [p for p in points if counts[p.kind] <= _MAX_POINTS_PER_KIND]


def _filter_cusp_near_bt(
    points: list[DetectedBifurcation],
) -> list[DetectedBifurcation]:
    bogdanov = [p for p in points if p.kind == "bogdanov_takens"]
    kept = []
    for point in points:
        if point.kind == "cusp" and any(
            _near(point.parameters, other.parameters) for other in bogdanov
        ):
            continue
        kept.append(point)
    return kept


def _near(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return all(
        abs(a - b) < _CUSP_BT_SEPARATION for a, b in zip(left, right, strict=True)
    )


def _dedup(points: list[DetectedBifurcation]) -> list[DetectedBifurcation]:
    seen: set[tuple] = set()
    unique = []
    for point in points:
        key = (point.kind, tuple(round(v, _ROUNDING) for v in point.parameters))
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return unique


def _from_branch(branch: Branch) -> list[DetectedBifurcation]:
    return [
        DetectedBifurcation(
            point.kind,
            1,
            list(point.state),
            (point.parameter,),
            point.frequency,
        )
        for point in branch.special_points
    ]


def _from_codim2(point: Codim2Point) -> DetectedBifurcation:
    return DetectedBifurcation(
        point.kind,
        _CODIM2,
        list(point.state),
        (point.parameter_a, point.parameter_b),
        point.omega,
    )


def _from_codim3(point: Codim3Point) -> DetectedBifurcation:
    return DetectedBifurcation(
        point.kind,
        _CODIM3,
        list(point.state),
        (point.parameter_a, point.parameter_b, point.parameter_c),
        None,
    )
