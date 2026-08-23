"""Render-agnostic description of a plot.

A :class:`Scene` is a renderer-independent bundle of what to draw: a title, a pair
of :class:`Axes`, a list of :class:`Series`, and optional :class:`SceneExtras` -
a continuous :class:`Gradient` colour scale, or a three-dimensional
:class:`Surface`. Each series is a polyline (``role="line"``) or a marker set
(``role="marker"``), coloured by kind or by an explicit colour. Keeping this free
of any drawing library lets the scene be built and unit-tested on its own.
"""


class Axes:
    """The labelled axes of a plot."""

    __slots__ = ["x_label", "y_label"]

    def __init__(self, x_label: str, y_label: str) -> None:
        self.x_label = x_label
        self.y_label = y_label


class Series:
    """One polyline or marker set within a scene."""

    __slots__ = ["colour", "kind", "label", "points", "role"]

    def __init__(
        self,
        role: str,
        kind: str,
        label: str,
        points: list[tuple[float, float]],
        colour: str | None = None,
    ) -> None:
        self.role = role
        self.kind = kind
        self.label = label
        self.points = points
        self.colour = colour


class Gradient:
    """A continuous colour scale for a family parameter, for the colour bar."""

    __slots__ = ["high", "high_colour", "label", "low", "low_colour"]

    def __init__(
        self,
        label: str,
        low: float,
        high: float,
        low_colour: str,
        high_colour: str,
    ) -> None:
        self.label = label
        self.low = low
        self.high = high
        self.low_colour = low_colour
        self.high_colour = high_colour


class Locus:
    """A bifurcation curve continued through a family, in (x, y, z) space."""

    __slots__ = ["kind", "label", "points"]

    def __init__(
        self,
        kind: str,
        label: str,
        points: list[tuple[float, float, float]],
    ) -> None:
        self.kind = kind
        self.label = label
        self.points = points


class Surface:
    """A parameter-family surface: a grid of (x, y, z) with bifurcation loci."""

    __slots__ = ["grid", "loci", "markers", "x_label", "y_label", "z_label"]

    def __init__(
        self,
        grid: list[list[tuple[float, float, float]]],
        loci: list[Locus],
        x_label: str,
        y_label: str,
        z_label: str,
    ) -> None:
        self.grid = grid
        self.loci = loci
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label
        self.markers: list[Locus] = []


class TreeNode:
    """A positioned node in an exploration-tree diagram."""

    __slots__ = [
        "codimension",
        "detail",
        "identifier",
        "kind",
        "label",
        "level",
        "order",
    ]

    def __init__(
        self,
        identifier: int,
        level: int,
        label: str,
        detail: str,
        kind: str,
    ) -> None:
        self.identifier = identifier
        self.level = level
        self.order: float = 0.0
        self.label = label
        self.detail = detail
        self.kind = kind
        self.codimension = level + 1


class Tree:
    """A node-link exploration tree: positioned nodes and parent-child edges."""

    __slots__ = ["edges", "nodes"]

    def __init__(
        self,
        nodes: list[TreeNode],
        edges: list[tuple[int, int]],
    ) -> None:
        self.nodes = nodes
        self.edges = edges


class Region:
    """A labelled annotation placed inside a region of a parameter-plane diagram."""

    __slots__ = ["label", "x", "y"]

    def __init__(self, label: str, x: float, y: float) -> None:
        self.label = label
        self.x = x
        self.y = y


class SceneExtras:
    """Optional overlays for a scene: a gradient, a surface, a tree, an equation."""

    __slots__ = ["equation", "gradient", "regions", "surface", "tree"]

    def __init__(
        self,
        gradient: Gradient | None = None,
        surface: Surface | None = None,
        tree: "Tree | None" = None,
        equation: str | None = None,
        regions: "list[Region] | None" = None,
    ) -> None:
        self.gradient = gradient
        self.surface = surface
        self.tree = tree
        self.equation = equation
        self.regions = regions if regions is not None else []


class Scene:
    """Title, axes, series, and optional overlays."""

    __slots__ = ["_extras", "axes", "series", "subtitle", "title"]

    def __init__(
        self,
        title: str,
        subtitle: str,
        axes: Axes,
        series: list[Series],
        extras: SceneExtras | None = None,
    ) -> None:
        self.title = title
        self.subtitle = subtitle
        self.axes = axes
        self.series = series
        self._extras = extras if extras is not None else SceneExtras()

    @property
    def x_label(self) -> str:
        """Horizontal axis label."""
        return self.axes.x_label

    @property
    def y_label(self) -> str:
        """Vertical axis label."""
        return self.axes.y_label

    @property
    def gradient(self) -> Gradient | None:
        """The colour-bar gradient, if any."""
        return self._extras.gradient

    @property
    def surface(self) -> Surface | None:
        """The three-dimensional surface, if any."""
        return self._extras.surface

    @property
    def tree(self) -> "Tree | None":
        """The exploration tree, if any."""
        return self._extras.tree

    @property
    def equation(self) -> str | None:
        """The governing equation as a LaTeX string, if any."""
        return self._extras.equation

    @property
    def regions(self) -> "list[Region]":
        """Labelled region annotations for a parameter-plane diagram."""
        return self._extras.regions
