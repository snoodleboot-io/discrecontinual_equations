"""Render-agnostic description of a stage: a continuation as a film.

A :class:`StageScene` is everything a browser needs to play a one-parameter
continuation of a planar system as motion: the bifurcation diagram (the
:class:`Timeline`), the cycle branch (a :class:`CycleBranch`), and for each
:class:`Frame` along the parameter the vector field sampled on a
:class:`Lattice` for particles to flow through, the equilibria with their
eigenvalues, the saddle manifolds, and the cycle at that frame, if any. Like
:class:`~.scene.Scene` it holds no drawing code, so a stage can be built and
unit-tested without a browser, and :func:`stage_payload` turns it into the plain
JSON the renderer embeds.
"""

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


class Lattice:
    """The plane the film is shot on: its box, sampling grid, and axis names."""

    __slots__ = ["box", "grid", "x_label", "y_label"]

    def __init__(
        self,
        box: Box,
        grid: int,
        x_label: str = "x",
        y_label: str = "y",
    ) -> None:
        self.box = box
        self.grid = grid
        self.x_label = x_label
        self.y_label = y_label


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
    """A cycle on the branch: its orbit, period, and Floquet multipliers."""

    __slots__ = ["amplitude", "multipliers", "parameter", "period", "states"]

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
    row-major order (``u, v`` per node, rows of constant ``y``). ``cycle``
    indexes the scene's cycle branch, or is ``None`` where no cycle exists.
    ``label`` is an optional sentence naming the regime; the page composes one
    otherwise.
    """

    __slots__ = ["cycle", "equilibria", "field", "label", "manifolds", "parameter"]

    def __init__(
        self,
        parameter: float,
        field: list[float],
        equilibria: list[Equilibrium],
        manifolds: list[Manifold],
        cycle: int | None,
    ) -> None:
        self.parameter = parameter
        self.field = field
        self.equilibria = equilibria
        self.manifolds = manifolds
        self.cycle = cycle
        self.label: str | None = None


class BranchPoint:
    """A point of the equilibrium branch, as the timeline draws it."""

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
    """A detected bifurcation on the branch, with an optional display label."""

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
    """The equilibrium branch as the timeline draws it, with its bifurcations."""

    __slots__ = ["points", "special"]

    def __init__(self, points: list[BranchPoint], special: list[SpecialPoint]) -> None:
        self.points = points
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
        "branch": {
            "points": [
                {
                    "p": _num(point.parameter),
                    "x": _num(point.x),
                    "stability": point.stability,
                    "kind": point.kind,
                    "eig": _pairs(point.eigenvalues),
                }
                for point in scene.timeline.points
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
                "states": _pairs(cycle.states),
            }
            for cycle in scene.cycles.cycles
        ],
        "frames": [
            {
                "p": _num(frame.parameter),
                "field": [
                    round(float(value), _FIELD_DECIMALS) for value in frame.field
                ],
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
                "cycle": frame.cycle,
                "label": frame.label,
            }
            for frame in scene.frames
        ],
    }
