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

A stochastic system is the same scene with two things added after it is built:
a :class:`StochasticStage` on the scene, carrying the noise and the top
Lyapunov exponent across the parameter, and a :class:`Density` on each frame.
Both are optional, and a deterministic scene's payload does not mention them.
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


class Projection:
    """One ``2 x N`` way of looking at a state, and what to call it.

    ``labels`` name the two plane axes and ``box`` is the window they are drawn
    in - both belong to the projection rather than the scene, since the plane
    of ``y`` against ``z`` is a different picture at a different scale from the
    plane of ``x`` against ``y``. Either may be left out, and the lattice's own
    labels and box are then used.
    """

    __slots__ = ["box", "labels", "matrix", "name"]

    def __init__(
        self,
        name: str,
        matrix: Sequence[Sequence[float]],
        labels: tuple[str, str] = ("", ""),
        box: Box | None = None,
    ) -> None:
        self.name = name
        self.matrix = [list(row) for row in matrix]
        self.labels = labels
        self.box = box


class View:
    """How a higher-dimensional system is seen on the plane.

    ``projections`` is one or more ``2 x N`` linear projections from state to
    plane; the first is what the page opens on, and the page offers the rest to
    switch between, since no single plane shows a system of three or more
    dimensions. A bare ``2 x N`` matrix is accepted and read as the only
    projection. ``bounds`` are the ``N`` ranges particles are spawned in and
    culled outside of; ``field`` the system's polynomial right-hand side as one
    :class:`Term` list per component, which the page integrates in full
    dimension.

    Switching is only possible because the page is given full states rather
    than plane coordinates: the particles were always integrated in full
    dimension, and the equilibria and cycles now travel unprojected too, so
    every projection is applied in the browser.
    """

    __slots__ = ["bounds", "field", "projections"]

    def __init__(
        self,
        matrix: Sequence[Sequence[float]] | Sequence[Projection],
        bounds: Sequence[tuple[float, float]],
        field: Sequence[Sequence[Term]],
    ) -> None:
        first = matrix[0] if len(matrix) else None
        if isinstance(first, Projection):
            self.projections = list(matrix)  # type: ignore[arg-type]
        else:
            self.projections = [Projection("", matrix)]  # type: ignore[arg-type]
        self.bounds = list(bounds)
        self.field = [list(component) for component in field]

    @property
    def matrix(self) -> list[list[float]]:
        """The projection the page opens on."""
        return self.projections[0].matrix

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
    """An equilibrium at one frame, with the eigenvalues the branch recorded.

    ``x`` and ``y`` are its plane coordinates under the view's opening
    projection. ``state`` is the full state it was projected from, kept so the
    page can project it again when the viewer switches plane; it is empty for a
    planar system, where the state and the plane are the same thing.
    """

    __slots__ = ["eigenvalues", "stability", "state", "x", "y"]

    def __init__(
        self,
        x: float,
        y: float,
        stability: str,
        eigenvalues: Pairs,
        state: Sequence[float] = (),
    ) -> None:
        self.x = x
        self.y = y
        self.stability = stability
        self.eigenvalues = eigenvalues
        self.state = [float(value) for value in state]


class Manifold:
    """One branch of a saddle's stable or unstable manifold, as a polyline.

    ``points`` is the branch in the plane under the view's opening projection.
    ``curve``, set after construction, holds the full states it came from, so
    the page can project the branch again when the viewer switches plane; it is
    empty for a planar system, where the two are the same.

    Only a manifold whose eigenspace is one-dimensional is a curve and can be
    drawn this way. A two-dimensional one is a surface, and the builder leaves
    it out rather than drawing a single arbitrary trajectory across it.
    """

    __slots__ = ["curve", "kind", "points"]

    def __init__(self, kind: str, points: Pairs) -> None:
        self.kind = kind
        self.points = points
        self.curve: list[list[float]] = []


class Cycle:
    """A cycle on the branch: its orbit, period, and Floquet multipliers.

    ``error`` is the trivial multiplier's distance from one, which bounds how
    far every multiplier is from its true value; it is drawn with them.

    ``states`` is the orbit in the plane under the view's opening projection.
    ``orbit``, set after construction the way ``error`` is, holds the full
    states it came from, so the page can project the orbit again when the
    viewer switches plane; it is empty for a planar system, where the two
    are the same.
    """

    __slots__ = [
        "amplitude",
        "error",
        "multipliers",
        "orbit",
        "parameter",
        "period",
        "states",
    ]

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
        self.orbit: list[list[float]] = []
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
    ``density`` is the stationary density at this parameter, set only on the
    frames of a stochastic stage.
    """

    __slots__ = [
        "cycles",
        "density",
        "equilibria",
        "field",
        "label",
        "manifolds",
        "parameter",
    ]

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
        # Set after construction, like ``label``, and only by a stochastic
        # stage: the stationary density at this parameter.
        self.density: Density | None = None


class Density:
    """The stationary density at one frame, as the density panel draws it.

    ``values`` is the density on the scene's density window, scaled so its
    peak is one: ``ny * nx`` floats in row-major order (rows of constant
    ``y``), the layout the field uses. It is empty where the frame has no
    density on the plane at all - below the dynamical threshold of an
    invariant point every path falls into it and the whole mass sits on one
    point - which the page draws as a point rather than a field. ``peak`` is
    the value the rest were scaled by. ``maxima`` are the interior maxima the
    solve found and ``mode`` the global one. ``crest`` is the radius of the
    ring the maxima form once the peak has left the reference state, and
    ``None`` while it still sits there: it is the single number the
    phenomenological bifurcation changes, so the timeline draws it against
    the parameter.
    """

    __slots__ = ["crest", "maxima", "mode", "peak", "values"]

    def __init__(
        self,
        values: list[float],
        maxima: Pairs,
        mode: tuple[float, float] | None,
        crest: float | None,
        peak: float = 0.0,
    ) -> None:
        self.values = values
        self.maxima = maxima
        self.mode = mode
        self.crest = crest
        self.peak = peak


class PathModel:
    """How the page integrates sample paths of ``dx = f dt + G dW``.

    ``noise`` is the amplitude ``G`` as polynomial terms: one :class:`Term`
    list per state component, per independent Brownian driver. With the drift
    terms the lattice's view already carries, the page steps paths by
    Euler-Maruyama with step ``step``, drawing every random number from a
    generator it seeds with ``seed``, so the film is the same on every build
    and every reload and the paths are the real paths of the equation rather
    than a precomputed replay. ``convention`` names the reading - Ito or
    Stratonovich - the equation is written in; the builder checks that the
    view's drift terms are the Ito drift of the system under that reading,
    since that is what the scheme integrates.
    """

    __slots__ = ["convention", "noise", "seed", "step"]

    def __init__(
        self,
        noise: Sequence[Sequence[Sequence[Term]]],
        convention: str,
        seed: int = 0,
        step: float = 0.02,
    ) -> None:
        self.noise = [[list(component) for component in driver] for driver in noise]
        self.convention = convention
        self.seed = seed
        self.step = step


class DensityWindow:
    """The window every frame's :class:`Density` is sampled on.

    ``box`` is the extent of the cells drawn and ``shape`` their count per
    axis; ``centre`` is the reference state the crest radius is measured
    from. ``exact_crest`` is an optional closed form for that radius against
    the parameter, drawn beside the measured one so the page shows its own
    error rather than hiding it.
    """

    __slots__ = ["box", "centre", "exact_crest", "shape"]

    def __init__(
        self,
        box: Box,
        shape: tuple[int, int],
        centre: tuple[float, float] = (0.0, 0.0),
        exact_crest: Pairs | None = None,
    ) -> None:
        self.box = box
        self.shape = shape
        self.centre = (float(centre[0]), float(centre[1]))
        self.exact_crest = exact_crest


class ExponentCurve:
    """The top Lyapunov exponent across the parameter, which replaces the clock.

    For a stochastic system the dynamical bifurcation changes the sign of this
    one real number, not of a spectrum. ``samples`` are the computed values
    and ``exact`` an optional closed form drawn beside them.
    """

    __slots__ = ["exact", "samples"]

    def __init__(self, samples: Pairs, exact: Pairs | None = None) -> None:
        self.samples = samples
        self.exact = exact


class StochasticStage:
    """What makes a staged system stochastic: its paths, density and exponent."""

    __slots__ = ["exponent", "paths", "window"]

    def __init__(
        self,
        paths: PathModel,
        window: DensityWindow,
        exponent: ExponentCurve,
    ) -> None:
        self.paths = paths
        self.window = window
        self.exponent = exponent


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

    __slots__ = ["cycles", "frames", "lattice", "stochastic", "system", "timeline"]

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
        # Set after construction, and only by a stochastic stage.
        self.stochastic: StochasticStage | None = None


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
        "projections": [
            {
                "name": projection.name,
                "matrix": [[float(m) for m in row] for row in projection.matrix],
                "x_label": projection.labels[0],
                "y_label": projection.labels[1],
                "box": (
                    {"x": list(projection.box[0]), "y": list(projection.box[1])}
                    if projection.box
                    else None
                ),
            }
            for projection in view.projections
        ],
        "bounds": [[float(lo), float(hi)] for lo, hi in view.bounds],
        "field": _terms_payload(view.field),
    }


def _terms_payload(field: Sequence[Sequence[Term]]) -> list[list[dict]]:
    return [
        [
            {
                "c": float(term.coefficient),
                "e": list(term.exponents),
                "q": int(term.parameter_power),
            }
            for term in component
        ]
        for component in field
    ]


def _density_payload(density: Density) -> dict:
    return {
        "values": _field(density.values),
        "peak": float(density.peak),
        "maxima": _pairs(density.maxima),
        "mode": [_num(v) for v in density.mode] if density.mode is not None else None,
        "crest": _num(density.crest) if density.crest is not None else None,
    }


def _stochastic_payload(stochastic: StochasticStage) -> dict:
    paths, window, exponent = stochastic.paths, stochastic.window, stochastic.exponent
    return {
        "convention": paths.convention,
        "seed": int(paths.seed),
        "step": float(paths.step),
        "centre": [_num(v) for v in window.centre],
        "noise": [_terms_payload(driver) for driver in paths.noise],
        "density": {
            "box": {"x": list(window.box[0]), "y": list(window.box[1])},
            "nx": int(window.shape[0]),
            "ny": int(window.shape[1]),
        },
        "exponent": _pairs(exponent.samples),
        "exact_exponent": _pairs(exponent.exact) if exponent.exact else None,
        "exact_crest": _pairs(window.exact_crest) if window.exact_crest else None,
    }


def _frame_payload(frame: Frame) -> dict:
    """One frame as the page reads it.

    The density key is written only where the frame has one, so a
    deterministic film's payload is exactly what it was before densities
    existed: the deterministic pages are the oracle that nothing here moved.
    """
    payload = {
        "p": _num(frame.parameter),
        "field": _field(frame.field),
        "equilibria": [
            {
                "x": _num(item.x),
                "y": _num(item.y),
                "stability": item.stability,
                "eig": _pairs(item.eigenvalues),
                "state": [_num(value) for value in item.state],
            }
            for item in frame.equilibria
        ],
        "manifolds": [
            {
                "kind": item.kind,
                "points": _pairs(item.points),
                "curve": [[_num(v) for v in state] for state in item.curve],
            }
            for item in frame.manifolds
        ],
        "cycles": list(frame.cycles),
        "label": frame.label,
    }
    if frame.density is not None:
        payload["density"] = _density_payload(frame.density)
    return payload


def stage_payload(scene: StageScene) -> dict:
    """The scene as plain JSON-ready data, in the shape the page script reads.

    A stochastic stage adds one ``stochastic`` block, and a density to each
    frame that has one; a deterministic stage's payload has neither key, not
    even a null, so it is byte for byte what it was.
    """
    system, lattice = scene.system, scene.lattice
    payload = {
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
                "orbit": [[_num(v) for v in state] for state in cycle.orbit],
            }
            for cycle in scene.cycles.cycles
        ],
        "frames": [_frame_payload(frame) for frame in scene.frames],
    }
    if scene.stochastic is not None:
        payload["stochastic"] = _stochastic_payload(scene.stochastic)
    return payload
