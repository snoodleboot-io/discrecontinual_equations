"""Build render-agnostic scenes from continuation results.

These functions turn the domain objects (a continued branch, a codim-2 or codim-3
result) into a :class:`~.scene.Scene` the D3 renderer can draw. Axis labels and the
pretty names of the bifurcations are supplied by the caller (or defaulted), so the
plots read in the system's own notation. They depend only on the render-agnostic
diagram view and the result value objects, never on a drawing library.
"""

from itertools import pairwise

from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.codim2_builder import Codim2Result
from discrecontinual_equations.continuation.codim3_builder import Codim3Result
from discrecontinual_equations.continuation.diagram import BifurcationDiagram
from discrecontinual_equations.continuation.exploration import ExplorationNode
from discrecontinual_equations.continuation.family_builder import ParameterFamily
from discrecontinual_equations.webplot.scene import (
    Axes,
    Gradient,
    Locus,
    Region,
    Scene,
    SceneExtras,
    Series,
    Surface,
    Tree,
    TreeNode,
)

_NAMES = {
    "fold": "fold",
    "branch_point": "branch point",
    "hopf": "Hopf",
    "bogdanov_takens": "Bogdanov\u2013Takens",
    "zero_hopf": "zero-Hopf",
    "hopf_hopf": "Hopf\u2013Hopf",
    "cusp": "cusp",
    "generalized_hopf": "generalized Hopf",
    "swallowtail": "swallowtail",
    "degenerate_bautin": "degenerate Bautin",
}

_LOW_COLOUR = "#38bdf8"
_HIGH_COLOUR = "#fbbf24"
_MIN_BRANCH_POINTS = 2
_SPATIAL_DIMENSION = 3
_MONOTONE_TOLERANCE = 1.0e-12


def with_equation(scene: Scene, equation: str) -> Scene:
    """Return a copy of ``scene`` carrying a LaTeX equation for display."""
    return Scene(
        scene.title,
        scene.subtitle,
        Axes(scene.x_label, scene.y_label),
        scene.series,
        SceneExtras(
            gradient=scene.gradient,
            surface=scene.surface,
            tree=scene.tree,
            equation=equation,
            regions=scene.regions,
        ),
    )


def with_regions(scene: Scene, regions: list[tuple[str, float, float]]) -> Scene:
    """Return a copy of ``scene`` annotated with labelled parameter-plane regions."""
    return Scene(
        scene.title,
        scene.subtitle,
        Axes(scene.x_label, scene.y_label),
        scene.series,
        SceneExtras(
            gradient=scene.gradient,
            surface=scene.surface,
            tree=scene.tree,
            equation=scene.equation,
            regions=[Region(label, x, y) for label, x, y in regions],
        ),
    )


def bifurcation_scene(
    branch: Branch,
    title: str,
    subtitle: str,
    x_label: str = "parameter",
    y_label: str = "state measure",
) -> Scene:
    """A codim-1 bifurcation diagram: measure versus continuation parameter."""
    diagram = BifurcationDiagram(branch)
    series: list[Series] = []
    seen: set[str] = set()
    for segment in diagram.segments():
        series.append(
            Series(
                role="line",
                kind=segment.stability,
                label=segment.stability if segment.stability not in seen else "",
                points=list(zip(segment.parameters, segment.measures, strict=True)),
            ),
        )
        seen.add(segment.stability)
    series.extend(_grouped_markers(_special_marker_points(diagram)))
    return Scene(title, subtitle, Axes(x_label, y_label), series)


def codim2_scene(
    result: Codim2Result,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
) -> Scene:
    """A codim-2 curve in the parameter plane with detected points."""
    curve = [(b, a) for a, b, _omega in result.curve_parameters]
    grouped: dict[str, list[tuple[float, float]]] = {}
    for point in result.points:
        grouped.setdefault(point.kind, []).append(
            (point.parameter_b, point.parameter_a),
        )
    series = [
        Series(role="line", kind="curve", label="bifurcation curve", points=curve),
        *_grouped_markers(grouped),
    ]
    return Scene(title, subtitle, Axes(x_label, y_label), series)


def codim3_scene(
    result: Codim3Result,
    projection: tuple[int, int],
    title: str,
    subtitle: str,
    labels: tuple[str, str, str],
) -> Scene:
    """A codim-3 curve projected onto two parameters, with detected points."""
    horizontal, vertical = projection
    curve = [
        (triple[horizontal], triple[vertical]) for triple in result.curve_parameters
    ]
    grouped: dict[str, list[tuple[float, float]]] = {}
    for point in result.points:
        triple = (point.parameter_a, point.parameter_b, point.parameter_c)
        grouped.setdefault(point.kind, []).append(
            (triple[horizontal], triple[vertical]),
        )
    series = [
        Series(role="line", kind="curve", label="bifurcation curve", points=curve),
        *_grouped_markers(grouped),
    ]
    return Scene(title, subtitle, Axes(labels[horizontal], labels[vertical]), series)


def _special_marker_points(
    diagram: BifurcationDiagram,
) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for point in diagram.special_points():
        grouped.setdefault(point.kind, []).append((point.parameter, point.measure))
    return grouped


def _grouped_markers(grouped: dict[str, list[tuple[float, float]]]) -> list[Series]:
    return [
        Series(role="marker", kind=kind, label=_pretty(kind), points=coordinates)
        for kind, coordinates in grouped.items()
    ]


def _pretty(kind: str) -> str:
    return _NAMES.get(kind, kind.replace("_", " "))


def family_scene(
    family: ParameterFamily,
    labels: tuple[str, str, str],
    title: str,
    subtitle: str,
) -> Scene:
    """Overlay a one-parameter family of diagrams, coloured by the family value.

    ``labels`` are the diagram-parameter, state-measure, and family-parameter
    names. Each slice is drawn in a colour interpolated along the family scale, and
    the bifurcations detected across all slices are overlaid as markers so their
    locus - the codim-2 curve - is visible as the diagram evolves.
    """
    values = family.values
    low, high = min(values), max(values)
    series: list[Series] = []
    grouped: dict[str, list[tuple[float, float]]] = {}
    for item in family.slices:
        colour = _interpolate(low, high, item.value)
        series.extend(_slice_series(item.branch, colour))
        for point in item.branch.special_points:
            grouped.setdefault(point.kind, []).append((point.parameter, point.measure))
    series.extend(_grouped_markers(grouped))
    gradient = Gradient(labels[2], low, high, _LOW_COLOUR, _HIGH_COLOUR)
    return Scene(
        title,
        subtitle,
        Axes(labels[0], labels[1]),
        series,
        SceneExtras(gradient=gradient),
    )


def _slice_series(branch: Branch, colour: str) -> list[Series]:
    diagram = BifurcationDiagram(branch)
    return [
        Series(
            role="line",
            kind=segment.stability,
            label="",
            points=list(zip(segment.parameters, segment.measures, strict=True)),
            colour=colour,
        )
        for segment in diagram.segments()
    ]


def _interpolate(low: float, high: float, value: float) -> str:
    span = (high - low) or 1.0
    fraction = (value - low) / span
    start = _channels(_LOW_COLOUR)
    end = _channels(_HIGH_COLOUR)
    mixed = [round(a + (b - a) * fraction) for a, b in zip(start, end, strict=True)]
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def _channels(colour: str) -> tuple[int, int, int]:
    value = int(colour.lstrip("#"), 16)
    return (value >> 16) & 255, (value >> 8) & 255, value & 255


def surface_scene(
    family: ParameterFamily,
    labels: tuple[str, str, str],
    captions: tuple[str, str],
    samples: int = 40,
) -> Scene:
    """Assemble the family into a surface in (diagram, family, measure) space.

    ``labels`` are the diagram-parameter, family-parameter, and measure names, and
    ``captions`` is ``(title, subtitle)``. Each slice branch is resampled by
    arclength to ``samples`` nodes so the slices align into a quad mesh; the
    bifurcations detected across the slices are joined, per kind, into loci - the
    continued bifurcation curves running over the surface.
    """
    grid: list[list[tuple[float, float, float]]] = []
    for item in family.slices:
        resampled = _resample_branch(item.branch, samples)
        grid.append([(x, item.value, z) for x, z in resampled])
    surface = Surface(grid, _continued_loci(family), labels[0], labels[1], labels[2])
    return Scene(
        captions[0],
        captions[1],
        Axes(labels[0], labels[1]),
        [],
        SceneExtras(surface=surface),
    )


def _resample_branch(branch: Branch, samples: int) -> list[tuple[float, float]]:
    raw = [(point.parameter, point.measure) for point in branch.points]
    if len(raw) < _MIN_BRANCH_POINTS:
        return raw * samples if raw else []
    measures = [measure for _parameter, measure in raw]
    if _is_monotone(measures):
        return _resample_by_measure(raw, samples)
    return _resample_by_arclength(raw, samples)


def _is_monotone(values: list[float]) -> bool:
    steps = [b - a for a, b in pairwise(values)]
    return all(s >= -_MONOTONE_TOLERANCE for s in steps) or all(
        s <= _MONOTONE_TOLERANCE for s in steps
    )


def _resample_by_measure(
    raw: list[tuple[float, float]],
    samples: int,
) -> list[tuple[float, float]]:
    ordered = sorted(raw, key=lambda pair: pair[1])
    low, high = ordered[0][1], ordered[-1][1]
    span = (high - low) or 1.0
    result = []
    for i in range(samples):
        measure = low + span * i / (samples - 1)
        parameter = _interpolate_measure(ordered, measure)
        result.append((parameter, measure))
    return result


def _interpolate_measure(ordered: list[tuple[float, float]], measure: float) -> float:
    for (p0, m0), (p1, m1) in pairwise(ordered):
        if m1 >= measure:
            span = (m1 - m0) or 1.0
            return p0 + (measure - m0) / span * (p1 - p0)
    return ordered[-1][0]


def _resample_by_arclength(
    raw: list[tuple[float, float]],
    samples: int,
) -> list[tuple[float, float]]:
    lengths = [0.0]
    for (x0, z0), (x1, z1) in pairwise(raw):
        lengths.append(lengths[-1] + ((x1 - x0) ** 2 + (z1 - z0) ** 2) ** 0.5)
    total = lengths[-1] or 1.0
    targets = [total * i / (samples - 1) for i in range(samples)]
    return [_interpolate_arclength(raw, lengths, target) for target in targets]


def _interpolate_arclength(
    raw: list[tuple[float, float]],
    lengths: list[float],
    target: float,
) -> tuple[float, float]:
    for i in range(1, len(lengths)):
        if lengths[i] >= target:
            span = lengths[i] - lengths[i - 1] or 1.0
            fraction = (target - lengths[i - 1]) / span
            x = raw[i - 1][0] + fraction * (raw[i][0] - raw[i - 1][0])
            z = raw[i - 1][1] + fraction * (raw[i][1] - raw[i - 1][1])
            return (x, z)
    return raw[-1]


def _continued_loci(family: ParameterFamily) -> list[Locus]:
    grouped: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
    for item in family.slices:
        counts: dict[str, int] = {}
        for point in item.branch.special_points:
            occurrence = counts.get(point.kind, 0)
            counts[point.kind] = occurrence + 1
            grouped.setdefault((point.kind, occurrence), []).append(
                (point.parameter, item.value, point.measure),
            )
    return [
        Locus(kind, _pretty(kind), points)
        for (kind, _occurrence), points in sorted(grouped.items())
    ]


_EXPLORATION_CURVE_COLOUR = {
    "fold curve": "#f472b6",
    "hopf curve": "#fbbf24",
    "cusp curve": "#f472b6",
    "generalized_hopf curve": "#fbbf24",
}
_EXPLORATION_POINT_COLOUR = {
    "cusp": "#f9a8d4",
    "bogdanov_takens": "#c084fc",
    "generalized_hopf": "#fde68a",
    "zero_hopf": "#67e8f9",
    "hopf_hopf": "#5eead4",
    "swallowtail": "#f472b6",
    "degenerate_bautin": "#fbbf24",
}


def exploration_diagram(
    root: ExplorationNode,
    axes: tuple[str, str],
    captions: tuple[str, str],
) -> Scene:
    """A two-parameter bifurcation diagram from an exploration tree.

    The codimension-2 curves (fold, Hopf) are drawn in the first two parameters,
    and every codimension-2 and codimension-3 point discovered is marked where the
    curves meet. Codimension-3 points are projected onto the plane.
    """
    lines: list[tuple[str, str, list[tuple[float, float]], str | None]] = []
    markers_by_kind: dict[str, list[tuple[float, float]]] = {}
    for node in _codim_two_nodes(root):
        colour = _EXPLORATION_CURVE_COLOUR.get(node.description, "#cbd5e1")
        for segment in node.segments:
            if len(segment) >= _MIN_BRANCH_POINTS:
                planar = [(point[0], point[1]) for point in segment]
                lines.append((node.description, node.description, planar, colour))
        for point in node.flatten():
            location = (point.parameters[0], point.parameters[1])
            markers_by_kind.setdefault(point.kind, []).append(location)
    markers = [
        (kind, _pretty(kind), spots, _EXPLORATION_POINT_COLOUR.get(kind, "#e5e7eb"))
        for kind, spots in markers_by_kind.items()
    ]
    return curve_scene(lines, markers, axes, captions[0], captions[1])


def exploration_skeleton(
    root: ExplorationNode,
    labels: tuple[str, str, str],
    captions: tuple[str, str],
) -> Scene:
    """A rotatable three-parameter skeleton of the exploration.

    Codimension-2 curves are lifted into the third parameter at the value they were
    continued at, codimension-3 curves are drawn with their full depth, and the
    codimension-3 organizing points are marked where a curve collapses.
    """
    loci: list[Locus] = []
    for node in _codim_two_nodes(root):
        depth = node.depth if node.depth is not None else 0.0
        for segment in node.segments:
            if len(segment) >= _MIN_BRANCH_POINTS:
                lifted = [(point[0], point[1], depth) for point in segment]
                loci.append(
                    Locus(_curve_kind(node.description), node.description, lifted),
                )
        for child in node.children:
            for segment in child.segments:
                if len(segment) >= _MIN_BRANCH_POINTS:
                    spatial = [(p[0], p[1], p[2]) for p in segment]
                    loci.append(
                        Locus(
                            _curve_kind(child.description),
                            child.description,
                            spatial,
                        ),
                    )
    surface = Surface([], loci, labels[0], labels[1], labels[2])
    surface.markers = _organizing_markers(root)
    return Scene(
        captions[0],
        captions[1],
        Axes(labels[0], labels[1]),
        [],
        SceneExtras(surface=surface),
    )


def _organizing_markers(root: ExplorationNode) -> list[Locus]:
    markers: list[Locus] = []
    for node in _codim_two_nodes(root):
        for child in node.children:
            for point in child.points:
                if len(point.parameters) >= _SPATIAL_DIMENSION:
                    location = (
                        point.parameters[0],
                        point.parameters[1],
                        point.parameters[2],
                    )
                    markers.append(Locus(point.kind, _pretty(point.kind), [location]))
    return markers


def exploration_tree(
    root: ExplorationNode,
    captions: tuple[str, str],
) -> Scene:
    """A node-link view of the exploration: what escalated into what.

    Each continuation is a node placed in a column by codimension, annotated with
    the bifurcations it detected; edges join a curve to the continuations it seeded.
    """
    nodes: list[TreeNode] = []
    edges: list[tuple[int, int]] = []
    counter = [0]
    leaves = [0.0]

    def visit(node: ExplorationNode, parent: int | None) -> None:
        identifier = counter[0]
        counter[0] += 1
        kind = "root" if node.codimension == 1 else _curve_kind(node.description)
        tree_node = TreeNode(
            identifier,
            node.codimension - 1,
            node.description,
            _point_summary(node.points),
            kind,
        )
        nodes.append(tree_node)
        if parent is not None:
            edges.append((parent, identifier))
        if node.children:
            positions = []
            for child in node.children:
                visit(child, identifier)
                positions.append(nodes[edges[-1][1]].order)
            tree_node.order = sum(positions) / len(positions)
        else:
            tree_node.order = leaves[0]
            leaves[0] += 1.0

    visit(root, None)
    tree = Tree(nodes, edges)
    return Scene(
        captions[0],
        captions[1],
        Axes("", ""),
        [],
        SceneExtras(tree=tree),
    )


def _point_summary(points: list) -> str:
    if not points:
        return "no degeneracies"
    counts: dict[str, int] = {}
    for point in points:
        counts[point.kind] = counts.get(point.kind, 0) + 1
    return ", ".join(
        f"{count}\u00d7 {_pretty(kind)}" if count > 1 else _pretty(kind)
        for kind, count in counts.items()
    )


def _codim_two_nodes(root: ExplorationNode) -> list[ExplorationNode]:
    return list(root.children)


def _curve_kind(description: str) -> str:
    return description.split(" ", maxsplit=1)[0]


def curve_scene(
    lines: list[tuple[str, str, list[tuple[float, float]], str | None]],
    markers: list[tuple[str, str, list[tuple[float, float]], str | None]],
    axes: tuple[str, str],
    title: str,
    subtitle: str,
) -> Scene:
    """A plot of labelled curves and marker points in arbitrary coordinates.

    ``lines`` and ``markers`` are ``(kind, label, points, colour)`` tuples; used for
    phase portraits, invariant-manifold charts, and connecting orbits, which live in
    state space rather than a parameter-versus-measure diagram.
    """
    series = [
        Series("line", kind, label, points, colour=colour)
        for kind, label, points, colour in lines
    ]
    series.extend(
        Series("marker", kind, label, points, colour=colour)
        for kind, label, points, colour in markers
    )
    return Scene(title, subtitle, Axes(axes[0], axes[1]), series)


def curve_family_scene(
    curves: list[list[tuple[float, float]]],
    values: list[float],
    axes: tuple[str, str],
    gradient_label: str,
    captions: tuple[str, str],
) -> Scene:
    """A family of curves coloured along a continuous family scale.

    Each curve corresponds to one ``value``; ``captions`` is ``(title, subtitle)``.
    Used for stationary-density families as a phenomenological parameter varies.
    """
    low, high = min(values), max(values)
    series = [
        Series(
            "line",
            "curve",
            "",
            points,
            colour=_interpolate(low, high, value),
        )
        for value, points in zip(values, curves, strict=True)
    ]
    gradient = Gradient(gradient_label, low, high, _LOW_COLOUR, _HIGH_COLOUR)
    return Scene(
        captions[0],
        captions[1],
        Axes(axes[0], axes[1]),
        series,
        SceneExtras(gradient=gradient),
    )
