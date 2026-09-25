"""Render-agnostic description of a stage: a continuation as a film.

A :class:`StageScene` is everything a browser needs to play a one-parameter
continuation as motion: the bifurcation diagram (the :class:`Timeline`), the
cycle branch (a :class:`CycleBranch`), and for each :class:`Frame` along the
parameter the equilibria with their eigenvalues, the saddle manifolds, and the
cycle at that frame, if any. A planar system's vector field is sampled on the
:class:`Lattice` for particles to flow through; a higher-dimensional system is
given a :class:`View` instead - a projection onto the plane and its field as
polynomial :class:`Term` lists - and the particles are integrated in full
dimension and drawn projected. Like :class:`~.scene.Scene` it holds no drawing
code, so a stage can be built and unit-tested without a browser, and
:func:`stage_payload` turns it into the plain JSON the renderer embeds.
"""

from collections.abc import Sequence

Pairs = list[tuple[float, float]]
Box = tuple[tuple[float, float], tuple[float, float]]


class StageSystem:
    """What the page says about the system: its name, equation, and story."""

    __slots__ = ["equation", "note", "parameter", "subtitle", "title"]

    def __init__(
        self,
        title: str,
        subtitle: str,
        parameter: str,
        equation: str | None = None,
        note: str | None = None,
    ) -> None:
        self.title = title
        self.subtitle = subtitle
        self.parameter = parameter
        self.equation = equation
        self.note = note


class Term:
    """One monomial of a polynomial field: ``coefficient * p^q * prod x_j^e_j``.

    ``exponents`` has one entry per state variable and ``parameter_power`` is the
    power of the continuation parameter, so a field's dependence on the parameter
    is exact in the browser rather than interpolated between frames.
    """

    __slots__ = ["coefficient", "exponents", "parameter_power"]

    def __init__(
        self,
        coefficient: float,
        exponents: Sequence[int],
        parameter_power: int = 0,
    ) -> None:
        self.coefficient = coefficient
        self.exponents = list(exponents)
        self.parameter_power = parameter_power


class View:
    """How a higher-dimensional system is seen on the plane.

    ``matrix`` is the ``2 x N`` linear projection from state to plane; ``bounds``
    the ``N`` ranges particles are spawned in and culled outside of; ``field``
    the system's polynomial right-hand side as one :class:`Term` list per
    component, which the page integrates in full dimension.
    """

    __slots__ = ["bounds", "field", "matrix"]

    def __init__(
        self,
        matrix: Sequence[Sequence[float]],
        bounds: Sequence[tuple[float, float]],
        field: Sequence[Sequence[Term]],
    ) -> None:
        self.matrix = [list(row) for row in matrix]
        self.bounds = list(bounds)
        self.field = [list(component) for component in field]

    @property
    def dimension(self) -> int:
        return len(self.matrix[0])

    def project(self, state: Sequence[float]) -> tuple[float, float]:
        """The plane coordinates of ``state``."""
        u = sum(m * float(s) for m, s in zip(self.matrix[0], state, strict=True))
        v = sum(m * float(s) for m, s in zip(self.matrix[1], state, strict=True))
        return u, v


def evaluate_terms(
    field: Sequence[Sequence[Term]],
    state: Sequence[float],
    parameter: float,
) -> list[float]:
    """Evaluate a polynomial field the way the page does, for checking it."""
    values = []
    for component in field:
        total = 0.0
        for term in component:
            product = term.coefficient * parameter**term.parameter_power
            for value, power in zip(state, term.exponents, strict=True):
                if power:
                    product *= float(value) ** power
            total += product
        values.append(total)
    return values


class Lattice:
    """The plane the film is shot on: its box, sampling grid, and axis names.

    A ``view`` makes the plane a projection of a higher-dimensional system; then
    no field is sampled on the grid, and the particles are integrated in full
    dimension from the view's polynomial field.
    """

    __slots__ = ["box", "grid", "view", "x_label", "y_label"]

    def __init__(
        self,
        box: Box,
        grid: int,
        x_label: str = "x",
        y_label: str = "y",
        view: View | None = None,
    ) -> None:
        self.box = box
        self.grid = grid
        self.x_label = x_label
        self.y_label = y_label
        self.view = view


class Equilibrium:
    """An equilibrium at one frame, with the eigenvalues the branch recorded."""

    __slots__ = ["eigenvalues", "stability", "x", "y"]

    def __init__(self, x: float, y: float, stability: str, eigenvalues: Pairs) -> None:
        self.x = x
        self.y = y
        self.stability = stability
        self.eigenvalues = eigenvalues


class Manifold:
    """One branch of a saddle's stable or unstable manifold, as a polyline."""

    __slots__ = ["kind", "points"]

    def __init__(self, kind: str, points: Pairs) -> None:
        self.kind = kind
        self.points = points


class Cycle:
    """A cycle on the branch: its orbit, period, and Floquet multipliers.

    ``error`` is the trivial multiplier's distance from one, which bounds how
    far every multiplier is from its true value; it is drawn with them.
    """

    __slots__ = ["amplitude", "error", "multipliers", "parameter", "period", "states"]

    def __init__(
        self,
        parameter: float,
        period: float,
        amplitude: float,
        multipliers: Pairs,
        states: Pairs,
    ) -> None:
        self.parameter = parameter
        self.period = period
        self.amplitude = amplitude
        self.multipliers = multipliers
        self.states = states
        self.error: float = 0.0


class CycleBranch:
    """The cycle branch as drawn, and what it runs into where it ends, if known.

    ``terminus`` names the end of the branch - a homoclinic, say - and is drawn
    at its tip on the timeline.
    """

    __slots__ = ["cycles", "terminus"]

    def __init__(self, cycles: list[Cycle], terminus: str | None = None) -> None:
        self.cycles = cycles
        self.terminus = terminus


class Frame:
    """One parameter value: the field, and the skeleton the flow reveals.

    ``field`` is the sampled vector field, ``2 * grid * grid`` floats in
    row-major order (``u, v`` per node, rows of constant ``y``), or empty when
    the lattice has a view. ``cycles`` indexes every cycle of the scene's
    branch that exists at this parameter - a folded branch has two, a stable
    and an unstable one - and is empty where none does. ``label`` is an
    optional sentence naming the regime; the page composes one otherwise.
    """

    __slots__ = ["cycles", "equilibria", "field", "label", "manifolds", "parameter"]

    def __init__(
        self,
        parameter: float,
        field: list[float],
        equilibria: list[Equilibrium],
        manifolds: list[Manifold],
        cycles: list[int],
    ) -> None:
        self.parameter = parameter
        self.field = field
        self.equilibria = equilibria
        self.manifolds = manifolds
        self.cycles = cycles
        self.label: str | None = None


class BranchPoint:
    """A point of an equilibrium branch, as the timeline draws it."""

    __slots__ = ["eigenvalues", "kind", "parameter", "stability", "x"]

    def __init__(
        self,
        parameter: float,
        x: float,
        stability: str,
        kind: str,
        eigenvalues: Pairs,
    ) -> None:
        self.parameter = parameter
        self.x = x
        self.stability = stability
        self.kind = kind
        self.eigenvalues = eigenvalues


class SpecialPoint:
    """A detected bifurcation on a branch, with an optional display label."""

    __slots__ = ["kind", "label", "parameter", "x"]

    def __init__(
        self,
        parameter: float,
        x: float,
        kind: str,
        label: str | None = None,
    ) -> None:
        self.parameter = parameter
        self.x = x
        self.kind = kind
        self.label = label


class Timeline:
    """The equilibrium branches as the timeline draws them, with bifurcations.

    Each arc is one continued branch; the timeline draws them separately so a
    second branch does not get joined to the first by a spurious segment.
    """

    __slots__ = ["arcs", "special"]

    def __init__(
        self,
        arcs: list[list[BranchPoint]],
        special: list[SpecialPoint],
    ) -> None:
        self.arcs = arcs
        self.special = special


class StageScene:
    """A continuation as a film: the timeline and every frame along it."""

    __slots__ = ["cycles", "frames", "lattice", "system", "timeline"]

    def __init__(
        self,
        system: StageSystem,
        lattice: Lattice,
        timeline: Timeline,
        cycles: CycleBranch,
        frames: list[Frame],
    ) -> None:
        self.system = system
        self.lattice = lattice
        self.timeline = timeline
        self.cycles = cycles
        self.frames = frames


# The page draws to screen precision, and the field is interpolated anyway, so
# the payload carries a few decimals rather than seventeen: it is the bulk of a
# self-contained page, and full precision made it two and a half times larger.
_FIELD_DECIMALS = 4
_DECIMALS = 6


def _pairs(values: Pairs) -> list[list[float]]:
    return [[round(float(a), _DECIMALS), round(float(b), _DECIMALS)] for a, b in values]


def _num(value: float) -> float:
    return round(float(value), _DECIMALS)


def _field(values: list[float]) -> list[float]:
    return [round(float(value), _FIELD_DECIMALS) for value in values]


def _view_payload(view: View | None) -> dict | None:
    if view is None:
        return None
    return {
        "matrix": [[float(m) for m in row] for row in view.matrix],
        "bounds": [[float(lo), float(hi)] for lo, hi in view.bounds],
        "field": [
            [
                {
                    "c": float(term.coefficient),
                    "e": list(term.exponents),
                    "q": int(term.parameter_power),
                }
                for term in component
            ]
            for component in view.field
        ],
    }


def stage_payload(scene: StageScene) -> dict:
    """The scene as plain JSON-ready data, in the shape the page script reads."""
    system, lattice = scene.system, scene.lattice
    return {
        "system": {
            "title": system.title,
            "subtitle": system.subtitle,
            "parameter": system.parameter,
            "x_label": lattice.x_label,
            "y_label": lattice.y_label,
            "cycle_terminus": scene.cycles.terminus,
        },
        "box": {"x": list(lattice.box[0]), "y": list(lattice.box[1])},
        "grid": {"nx": lattice.grid, "ny": lattice.grid},
        "view": _view_payload(lattice.view),
        "branch": {
            "arcs": [
                [
                    {
                        "p": _num(point.parameter),
                        "x": _num(point.x),
                        "stability": point.stability,
                        "kind": point.kind,
                        "eig": _pairs(point.eigenvalues),
                    }
                    for point in arc
                ]
                for arc in scene.timeline.arcs
            ],
            "special": [
                {
                    "p": _num(point.parameter),
                    "x": _num(point.x),
                    "kind": point.kind,
                    "label": point.label,
                }
                for point in scene.timeline.special
            ],
        },
        "cycles": [
            {
                "p": _num(cycle.parameter),
                "period": _num(cycle.period),
                "amplitude": _num(cycle.amplitude),
                "multipliers": _pairs(cycle.multipliers),
                "error": _num(cycle.error),
                "states": _pairs(cycle.states),
            }
            for cycle in scene.cycles.cycles
        ],
        "frames": [
            {
                "p": _num(frame.parameter),
                "field": _field(frame.field),
                "equilibria": [
                    {
                        "x": _num(item.x),
                        "y": _num(item.y),
                        "stability": item.stability,
                        "eig": _pairs(item.eigenvalues),
                    }
                    for item in frame.equilibria
                ],
                "manifolds": [
                    {"kind": item.kind, "points": _pairs(item.points)}
                    for item in frame.manifolds
                ],
                "cycles": list(frame.cycles),
                "label": frame.label,
            }
            for frame in scene.frames
        ],
    }
