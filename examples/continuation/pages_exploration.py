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


# Where the rotating wave appears, in units of the gain: measured by integration
# at a = 1 (fixed point at 0.33, wave at 0.35; 1/T^2 extrapolates to 0.340).
_WAVE_ONSET = 0.34


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
    breaking, crossing = [], []
    for item in family.slices:
        for point in item.branch.special_points:
            # The detector is asked for "branch_point" and reports the kind it
            # classified, "pitchfork" here; matching the request left this line
            # empty on every published map until DEQ-14.
            if point.kind in ("branch_point", "pitchfork"):
                breaking.append((point.parameter, item.value))
            elif point.kind == "hopf":
                crossing.append((point.parameter, item.value))
    # The symmetric states +-s(1, 1, 1), continued upward from lambda = -1 where
    # s = sqrt(a + 1) sits near the guess for every a in the family. The branch
    # folds through the pitchfork at lambda = a and returns as its mirror, so each
    # Hopf is met twice; one copy per slice is kept.
    symmetric = FamilyDriver.run(
        FamilyConfig(
            continuation_parameter_index=0,
            family_parameter_index=1,
            family_values=values,
            initial_parameter=-1.0,
            direction=1,
            measure="norm",
            parameter_lower_bound=-1.2,
            parameter_upper_bound=1.7,
            initial_step=0.01,
            maximum_step=0.03,
            maximum_points=2500,
            detectors=["hopf"],
        ),
        equation,
        [1.3, 1.3, 1.3],
    )
    symmetric_hopf = []
    for item in symmetric.slices:
        hopfs = [p.parameter for p in item.branch.special_points if p.kind == "hopf"]
        if hopfs:
            symmetric_hopf.append((min(hopfs), item.value))
    # The wave's onset is not a local bifurcation the continuer can detect: the
    # period diverges there (1/T^2 linear in lambda; DEQ-14). Measured by
    # integration at a = 1 as lambda_c = 0.34, and exactly proportional to a since
    # rescaling x by sqrt(a) and t by 1/a leaves the ring depending on lambda/a.
    onset = [(_WAVE_ONSET * value, value) for value in values]
    lines = [
        ("branch_point", "symmetry breaking (\u03bb=a)", breaking, "#a3e635"),
        (
            "hopf",
            "trivial state: k=1 pair crosses (\u03bb=-2a); unstable throughout",
            crossing,
            "#fbbf24",
        ),
        (
            "hopf",
            "symmetric states lose stability (\u03bb=4a/7)",
            symmetric_hopf,
            "#60a5fa",
        ),
        (
            "bifurcation",
            "rotating wave onset, infinite period (\u03bb\u22480.34a)",
            onset,
            "#f472b6",
        ),
    ]
    return curve_scene(
        lines,
        [],
        ("\u03bb", "a"),
        "Electric-field sensor: operating map",
        "Ferroelectric ring at zero field, a>0. The symmetric states are the "
        "attractors for weak coupling; the rotating wave appears at \u03bb\u22480.34a "
        "with infinite period and coexists with them until their Hopf at \u03bb=4a/7; "
        "they merge with the trivial state at \u03bb=a. The trivial state is unstable "
        "throughout, so its k=1 crossing at \u03bb=-2a is not an onset here (it is "
        "one for a<0, where the cells are monostable).",
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
                # The detector reports "pitchfork" for what was requested as
                # "branch_point"; matched literally, every pitchfork fell through
                # to the Hopf branch and was filed under a mode (DEQ-14).
                if point.kind in ("branch_point", "pitchfork"):
                    branch.append((point.parameter, item.value))
                elif point.kind == "hopf":
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
        "A seven-dimensional sensor array: the trivial state's three Hopf lines "
        "from the ring spectrum. For a>0 that state is unstable throughout, so "
        "these are crossings, not onsets (see the three-cell map).",
    )
