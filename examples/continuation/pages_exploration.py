"""Automated exploration, and the sensor maps it produces.

Split out of webplot_examples, which assembles the atlas from these.
"""

import math

import numpy as np

from discrecontinual_equations.continuation.exploration import (
    BifurcationExplorer,
    ExplorationConfig,
)
from discrecontinual_equations.continuation.family_builder import FamilyDriver
from discrecontinual_equations.continuation.family_config import FamilyConfig
from discrecontinual_equations.systems import (
    FerroelectricRing,
)
from discrecontinual_equations.webplot.scene import Scene
from discrecontinual_equations.webplot.scene_builder import (
    curve_scene,
    exploration_diagram,
    exploration_skeleton,
    exploration_tree,
    surface_scene,
)

try:  # python -m examples.continuation.pages_exploration
    from examples.continuation.systems import (
        First,
        Second,
        Swallowtail,
        Third,
        _equation,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from systems import (
        First,
        Second,
        Swallowtail,
        Third,
        _equation,
    )


MU = "\u03bc"
ALPHA = "\u03b1"
LAMBDA = "\u03bb"
_FOLD_WINDOW = 0.02
_LOOP_ESCAPE = 50.0
_LOOP_DEPARTURE = 0.6
_LOOP_RETURN = 0.25
_IMAG_TOLERANCE = 1.0e-9
_MODE_IMAG = 1.0e-6
_PERIOD_TOLERANCE = 1.0e-2
BETA1, BETA2, BETA3 = "\u03b2\u2081", "\u03b2\u2082", "\u03b2\u2083"
NORM = "\u2016x\u2016"
_IMAGINARY_TOLERANCE = 1.0e-9

BETA1, BETA2, BETA3 = "\u03b2\u2081", "\u03b2\u2082", "\u03b2\u2083"
BETA1, BETA2, BETA3 = "\u03b2\u2081", "\u03b2\u2082", "\u03b2\u2083"


def _exploration_diagram() -> Scene:
    equation = _equation(
        Swallowtail,
        1,
        [First(value=0.1875), Second(value=-1.0), Third(value=1.0)],
    )
    config = ExplorationConfig(
        parameters=[0, 1, 2],
        ranges=[(-1.4, 1.4), (-1.4, 1.4), (-1.4, 1.4)],
        initial_parameter=0.1875,
        initial_step=0.01,
        maximum_step=0.02,
        maximum_points=800,
    )
    root = BifurcationExplorer.explore(config, equation, [0.5])
    return exploration_diagram(
        root,
        (BETA1, BETA2),
        (
            "Automatic exploration: two-parameter diagram",
            "Fold curve continued from a detected fold; its two cusps are marked.",
        ),
    )


def _exploration_skeleton() -> Scene:
    equation = _equation(
        Swallowtail,
        1,
        [First(value=0.1875), Second(value=-1.0), Third(value=1.0)],
    )
    config = ExplorationConfig(
        parameters=[0, 1, 2],
        ranges=[(-1.4, 1.4), (-1.4, 1.4), (-1.4, 1.4)],
        initial_parameter=0.1875,
        initial_step=0.01,
        maximum_step=0.02,
        maximum_points=800,
    )
    root = BifurcationExplorer.explore(config, equation, [0.5])
    return exploration_skeleton(
        root,
        (BETA1, BETA2, "b\u2083"),
        (
            "Automatic exploration: 3-parameter skeleton",
            "The cusp curve descends through b\u2083 to the swallowtail, rotatable.",
        ),
    )


def _exploration_tree() -> Scene:
    equation = _equation(
        Swallowtail,
        1,
        [First(value=0.1875), Second(value=-1.0), Third(value=1.0)],
    )
    config = ExplorationConfig(
        parameters=[0, 1, 2],
        ranges=[(-1.4, 1.4), (-1.4, 1.4), (-1.4, 1.4)],
        initial_parameter=0.1875,
        initial_step=0.01,
        maximum_step=0.02,
        maximum_points=800,
    )
    root = BifurcationExplorer.explore(config, equation, [0.5])
    return exploration_tree(
        root,
        (
            "Automatic exploration: escalation tree",
            "Each continuation and the degeneracies it found; edges show escalation.",
        ),
    )


def _newton_symmetric(effective_gain: float, field: float) -> float:
    x = field
    for _ in range(80):
        residual = effective_gain * x - x**3 + field
        derivative = effective_gain - 3.0 * x * x
        x -= residual / derivative
    return x


def _efield_operating_map() -> Scene:
    equation = _equation(
        FerroelectricRing,
        3,
        [First(value=0.0), Second(value=1.0), Third(value=0.0)],
    )
    values = [round(float(v), 3) for v in np.linspace(0.4, 1.4, 11)]
    config = FamilyConfig(
        continuation_parameter_index=0,
        family_parameter_index=1,
        family_values=values,
        initial_parameter=1.6,
        direction=-1,
        measure="norm",
        parameter_lower_bound=-3.2,
        parameter_upper_bound=1.7,
        initial_step=0.01,
        maximum_step=0.03,
        maximum_points=2500,
        detectors=["branch_point", "hopf"],
    )
    family = FamilyDriver.run(config, equation, [0.0, 0.0, 0.0])
    breaking, oscillation = [], []
    for item in family.slices:
        for point in item.branch.special_points:
            if point.kind == "branch_point":
                breaking.append((point.parameter, item.value))
            elif point.kind == "hopf":
                oscillation.append((point.parameter, item.value))
    lines = [
        ("branch_point", "symmetry breaking (\u03bb=a)", breaking, "#a3e635"),
        ("hopf", "oscillation onset (\u03bb=-2a)", oscillation, "#fbbf24"),
    ]
    return curve_scene(
        lines,
        [],
        ("\u03bb", "a"),
        "Electric-field sensor: operating map",
        "Ferroelectric ring: oscillations for \u03bb<-2a, broken symmetry \u03bb>a.",
    )


def _efield_detection() -> Scene:
    equation = _equation(
        FerroelectricRing,
        3,
        [First(value=-0.5), Second(value=1.0), Third(value=0.0)],
    )
    values = [round(float(v), 3) for v in np.linspace(-0.6, 0.7, 12)]
    config = FamilyConfig(
        continuation_parameter_index=2,
        family_parameter_index=0,
        family_values=values,
        initial_parameter=-1.3,
        direction=1,
        measure="component",
        parameter_lower_bound=-1.4,
        parameter_upper_bound=1.4,
        initial_step=0.005,
        maximum_step=0.02,
        maximum_points=3000,
        detectors=["fold"],
    )
    seed = _newton_symmetric(1.0 - (-0.5), -1.3)
    family = FamilyDriver.run(config, equation, [seed, seed, seed])
    return surface_scene(
        family,
        ("\u03b5", "\u03bb", "x"),
        (
            "Electric-field sensor: detection surface",
            "Target field \u03b5 sweeps the well; coupling \u03bb tunes the folds.",
        ),
    )


def _ring_mode(ratio: float, count: int) -> int:
    modes = {
        mode: 1.0 / math.cos(2.0 * math.pi * mode / count)
        for mode in range(1, count // 2 + 1)
    }
    return min(modes, key=lambda mode: abs(ratio - modes[mode]))


def _sevencell_exploration() -> Scene:
    count = 7
    equation = _equation(
        FerroelectricRing,
        count,
        [First(value=0.0), Second(value=1.0), Third(value=0.0)],
    )
    config = ExplorationConfig(
        parameters=[0, 1, 2],
        ranges=[(-3.0, 3.0), (0.4, 1.8), (-1.0, 1.0)],
        initial_parameter=0.0,
        initial_step=0.03,
        maximum_step=0.06,
        maximum_points=150,
    )
    root = BifurcationExplorer.explore(config, equation, [0.0] * count)
    return exploration_diagram(
        root,
        ("\u03bb", "a"),
        (
            "Seven-cell ring: high-dimensional exploration",
            "Codim-2 Hopf curves continued from a seven-dimensional equilibrium.",
        ),
    )


def _sevencell_ring_map() -> Scene:
    count = 7
    equation = _equation(
        FerroelectricRing,
        count,
        [First(value=0.0), Second(value=1.0), Third(value=0.0)],
    )
    values = [round(0.4 + 0.1 * i, 3) for i in range(13)]
    branch: list[tuple[float, float]] = []
    hopf: dict[int, list[tuple[float, float]]] = {}
    for direction in (1, -1):
        config = FamilyConfig(
            continuation_parameter_index=0,
            family_parameter_index=1,
            family_values=values,
            initial_parameter=0.0,
            direction=direction,
            measure="norm",
            parameter_lower_bound=-5.5,
            parameter_upper_bound=3.6,
            initial_step=0.02,
            maximum_step=0.05,
            maximum_points=2000,
            detectors=["branch_point", "hopf"],
        )
        family = FamilyDriver.run(config, equation, [0.0] * count)
        for item in family.slices:
            for point in item.branch.special_points:
                if point.kind == "branch_point":
                    branch.append((point.parameter, item.value))
                else:
                    mode = _ring_mode(point.parameter / item.value, count)
                    hopf.setdefault(mode, []).append((point.parameter, item.value))
    palette = {1: "#fbbf24", 2: "#fb923c", 3: "#f472b6"}
    lines = [
        (
            "branch_point",
            "symmetry breaking (\u03bb=a)",
            sorted(branch, key=lambda pair: pair[1]),
            "#a3e635",
        ),
    ]
    lines.extend(
        (
            f"hopf{mode}",
            f"Hopf k={mode} (\u03bb=a/cos 2\u03c0k/7)",
            sorted(hopf[mode], key=lambda pair: pair[1]),
            palette.get(mode, "#fbbf24"),
        )
        for mode in sorted(hopf)
    )
    return curve_scene(
        lines,
        [],
        ("\u03bb", "a"),
        "Seven-cell ferroelectric ring: operating map",
        "A seven-dimensional sensor array; three Hopf lines from the ring spectrum.",
    )
