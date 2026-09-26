"""The codimension-two points, and the curves through them.

Split out of webplot_examples, which assembles the atlas from these.
"""

import math

import numpy as np

from discrecontinual_equations.continuation.codim2_builder import (
    Codim2Driver,
    CurveSeed,
)
from discrecontinual_equations.continuation.codim2_config import Codim2Config
from discrecontinual_equations.continuation.codim3_builder import (
    Codim3Driver,
    Codim3Seed,
)
from discrecontinual_equations.continuation.codim3_config import Codim3Config
from discrecontinual_equations.continuation.connecting_orbit import (
    MeshSpec,
)
from discrecontinual_equations.continuation.exploration import (
    BifurcationExplorer,
    ExplorationConfig,
)
from discrecontinual_equations.continuation.homoclinic_curve import HomoclinicCurve
from discrecontinual_equations.continuation.region_analysis import describe_region
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.webplot.scene import Scene
from discrecontinual_equations.webplot.scene_builder import (
    codim2_scene,
    codim3_scene,
    curve_scene,
    exploration_skeleton,
    with_regions,
)

try:  # python -m examples.continuation.pages_codim2
    from examples.continuation.systems import (
        Bautin,
        BogdanovTakens,
        BogdanovTakensPlanar,
        BogdanovTakensTypeChange,
        BogdanovTakensUnfolding,
        CuspForm,
        ExtendedBautin,
        First,
        HopfHopf,
        Second,
        State,
        Swallowtail,
        Third,
        Time,
        ZeroHopf,
        _equation,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from systems import (
        Bautin,
        BogdanovTakens,
        BogdanovTakensPlanar,
        BogdanovTakensTypeChange,
        BogdanovTakensUnfolding,
        CuspForm,
        ExtendedBautin,
        First,
        HopfHopf,
        Second,
        State,
        Swallowtail,
        Third,
        Time,
        ZeroHopf,
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

MU = "\u03bc"
_IMAGINARY_TOLERANCE = 1.0e-9
BETA1, BETA2, BETA3 = "\u03b2\u2081", "\u03b2\u2082", "\u03b2\u2083"
BETA1, BETA2, BETA3 = "\u03b2\u2081", "\u03b2\u2082", "\u03b2\u2083"
BETA1, BETA2, BETA3 = "\u03b2\u2081", "\u03b2\u2082", "\u03b2\u2083"


def _bogdanov_takens() -> Scene:
    equation = _equation(BogdanovTakens, 2, [First(value=-0.25), Second(value=-0.5)])
    seed = CurveSeed(state=[-0.5, 0.0], parameter_a=-0.25, frequency=1.0)
    config = Codim2Config(
        continuation_parameter_index=0,
        second_parameter_index=1,
        curve="hopf",
        codim2_detectors=["bogdanov_takens"],
        initial_parameter=-0.5,
        direction=1,
        parameter_lower_bound=-0.6,
        parameter_upper_bound=0.05,
    )
    result = Codim2Driver.run(config, equation, seed)
    return codim2_scene(
        result,
        "Hopf curve to Bogdanov-Takens",
        "Along the Hopf curve the frequency falls to zero at the codim-2 point.",
        BETA2,
        BETA1,
    )


def _bogdanov_takens_portrait() -> Scene:
    parameters = [First(value=0.0), Second(value=0.0)]
    equation = DifferentialEquation(
        variables=[State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=BogdanovTakensPlanar(
            variables=[State(), State()],
            parameters=parameters,
            results=[State(), State()],
            time=None,
        ),
    )
    homoclinic: list[tuple[float, float]] = [(0.0, 0.0)]
    for step in range(1, 13):
        epsilon = 0.12 + 0.02 * step
        half = 5.0 * math.sqrt(2.0) / epsilon
        nodes = 121
        times = np.linspace(-half, half, nodes)
        scaled = epsilon * times / math.sqrt(2.0)
        profile = 1.0 / np.cosh(scaled) ** 2
        orbit = np.column_stack(
            [
                epsilon**2 * (1.0 - 3.0 * profile),
                epsilon**3 * 3.0 * math.sqrt(2.0) * profile * np.tanh(scaled),
            ],
        )
        mesh = MeshSpec(half, nodes - 1, phase_index=1, phase_value=0.0)
        sweep = (5.0 / 7.0) * epsilon**2
        point = HomoclinicCurve(equation, 0, 1, mesh).solve_point(
            sweep,
            orbit,
            np.array([epsilon**2, 0.0]),
            -(epsilon**4),
        )
        if point is not None:
            homoclinic.append((point.sweep, point.continuation))
    span = homoclinic[-1][0]
    grid = [span * k / 40.0 for k in range(41)]
    fold = [(value, 0.0) for value in grid]
    hopf = [(value, -(value**2)) for value in grid]
    lines = [
        ("fold", "fold (saddle-node)", fold, None),
        ("hopf", "Hopf", hopf, None),
        ("bogdanov_takens", "homoclinic", homoclinic, None),
    ]
    markers = [("bogdanov_takens", "Bogdanov-Takens", [(0.0, 0.0)], None)]
    scene = curve_scene(
        lines,
        markers,
        (BETA2, BETA1),
        "Bogdanov-Takens organizing centre",
        "Three curves emanate from the codim-2 point: fold, Hopf, and the homoclinic "
        "curve, tangent to the Hopf with the 5/7 slope ratio.",
    )
    seeds = [np.array([0.3, 0.0]), np.array([-0.3, 0.0])]
    samples = [
        (0.040, 0.006, None),
        (0.078, -0.0028, np.array([-0.03, 0.0])),
        (0.050, -0.012, np.array([-0.09, 0.0])),
    ]
    regions = []
    for sweep, first, cycle_start in samples:
        parameters[0].value = first
        parameters[1].value = sweep
        label = describe_region(equation.derivative, seeds, cycle_start=cycle_start)
        regions.append((label, sweep, first))
    return with_regions(scene, regions)


def _zero_hopf() -> Scene:
    equation = _equation(ZeroHopf, 3, [First(value=0.5), Second(value=0.0)])
    seed = CurveSeed(state=[0.5**0.5, 0.0, 0.0], parameter_a=0.0, frequency=1.0)
    config = Codim2Config(
        continuation_parameter_index=1,
        second_parameter_index=0,
        curve="hopf",
        codim2_detectors=["zero_hopf"],
        initial_parameter=0.5,
        direction=-1,
        parameter_lower_bound=-0.1,
        parameter_upper_bound=0.6,
    )
    result = Codim2Driver.run(config, equation, seed)
    return codim2_scene(
        result,
        "Hopf curve to zero-Hopf",
        "The Hopf curve meets the fold in x, giving a zero-Hopf point.",
        MU + "\u2081",
        MU + "\u2082",
    )


def _hopf_hopf() -> Scene:
    equation = _equation(HopfHopf, 4, [First(value=0.0), Second(value=0.5)])
    seed = CurveSeed(state=[0.0, 0.0, 0.0, 0.0], parameter_a=0.0, frequency=1.0)
    config = Codim2Config(
        continuation_parameter_index=0,
        second_parameter_index=1,
        curve="hopf",
        codim2_detectors=["hopf_hopf"],
        initial_parameter=0.5,
        direction=-1,
        parameter_lower_bound=-0.3,
        parameter_upper_bound=0.6,
    )
    result = Codim2Driver.run(config, equation, seed)
    return codim2_scene(
        result,
        "Hopf curve to Hopf-Hopf",
        "Continuing the first Hopf curve, the second pair reaches the axis.",
        MU + "\u2082",
        MU + "\u2081",
    )


def _cusp() -> Scene:
    equation = _equation(CuspForm, 1, [First(value=-0.25), Second(value=0.75)])
    seed = CurveSeed(state=[0.5], parameter_a=0.75)
    config = Codim2Config(
        continuation_parameter_index=1,
        second_parameter_index=0,
        curve="fold",
        codim2_detectors=["cusp"],
        initial_parameter=-0.25,
        direction=1,
        parameter_lower_bound=-0.35,
        parameter_upper_bound=0.35,
    )
    result = Codim2Driver.run(config, equation, seed)
    return codim2_scene(
        result,
        "Fold curve to cusp",
        "The fold curve turns through the cusp where the coefficient a vanishes.",
        BETA1,
        BETA2,
    )


def _generalized_hopf() -> Scene:
    equation = _equation(Bautin, 2, [First(value=0.0), Second(value=0.5)])
    seed = CurveSeed(state=[0.0, 0.0], parameter_a=0.0, frequency=1.0)
    config = Codim2Config(
        continuation_parameter_index=0,
        second_parameter_index=1,
        curve="hopf",
        codim2_detectors=["generalized_hopf"],
        initial_parameter=0.5,
        direction=-1,
        parameter_lower_bound=-0.6,
        parameter_upper_bound=0.6,
    )
    result = Codim2Driver.run(config, equation, seed)
    return codim2_scene(
        result,
        "Hopf curve to generalized Hopf",
        "Along the Hopf curve the first Lyapunov coefficient vanishes (Bautin).",
        BETA2,
        BETA1,
    )


def _swallowtail() -> Scene:
    equation = _equation(
        Swallowtail,
        1,
        [First(value=0.1875), Second(value=-1.0), Third(value=1.5)],
    )
    seed = Codim3Seed(state=[0.5], parameter_a=0.1875, parameter_b=1.5)
    config = Codim3Config(
        continuation_parameter_index=0,
        second_parameter_index=2,
        third_parameter_index=1,
        codim3_curve="cusp",
        codim3_detectors=["swallowtail"],
        initial_parameter=-1.0,
        direction=1,
        parameter_lower_bound=-1.2,
        parameter_upper_bound=1.2,
        initial_step=0.01,
        maximum_step=0.02,
        maximum_points=500,
    )
    result = Codim3Driver.run(config, equation, seed)
    return codim3_scene(
        result,
        (1, 2),
        "Cusp curve to swallowtail",
        "Cusp curve projected onto its plane; the swallowtail sits at the origin.",
        (BETA1, BETA2, BETA3),
    )


def _degenerate_bautin() -> Scene:
    equation = _equation(
        ExtendedBautin,
        2,
        [First(value=0.0), Second(value=0.0), Third(value=0.6)],
    )
    seed = Codim3Seed(
        state=[0.0, 0.0],
        parameter_a=0.0,
        parameter_b=0.0,
        frequency=1.0,
    )
    config = Codim3Config(
        continuation_parameter_index=0,
        second_parameter_index=1,
        third_parameter_index=2,
        codim3_curve="generalized_hopf",
        codim3_detectors=["degenerate_bautin"],
        initial_parameter=0.6,
        direction=-1,
        parameter_lower_bound=-0.7,
        parameter_upper_bound=0.7,
        initial_step=0.02,
        maximum_step=0.05,
        maximum_points=400,
    )
    result = Codim3Driver.run(config, equation, seed)
    return codim3_scene(
        result,
        (2, 1),
        "Generalized-Hopf curve to degenerate Bautin",
        "The second Lyapunov quantity vanishes along the generalized-Hopf curve.",
        (BETA1, BETA2, BETA3),
    )


def _bogdanov_takens_curve() -> Scene:
    equation = _equation(
        BogdanovTakensUnfolding,
        2,
        [First(value=-0.4), Second(value=0.32), Third(value=0.4)],
    )
    config = ExplorationConfig(
        parameters=[0, 1, 2],
        ranges=[(-0.8, 0.3), (-0.2, 0.9), (-0.6, 0.7)],
        initial_parameter=-0.4,
        initial_step=0.01,
        maximum_step=0.03,
        maximum_points=700,
    )
    roots = [
        value.real
        for value in np.roots([0.4, 0.32, -0.4])
        if abs(value.imag) < _IMAGINARY_TOLERANCE
    ]
    root = BifurcationExplorer.explore(config, equation, [roots[0], 0.0])
    return exploration_skeleton(
        root,
        (BETA1, BETA2, "b\u2083"),
        (
            "Bogdanov-Takens curve with a degenerate BT",
            "The BT curve's quadratic coefficient vanishes at the marked point.",
        ),
    )


def _bogdanov_takens_type_curve() -> Scene:
    equation = _equation(
        BogdanovTakensTypeChange,
        2,
        [First(value=0.6), Second(value=2.0), Third(value=0.4)],
    )
    config = ExplorationConfig(
        parameters=[0, 1, 2],
        ranges=[(0.2, 1.6), (1.2, 2.8), (-0.7, 0.7)],
        initial_parameter=0.6,
        initial_step=0.01,
        maximum_step=0.03,
        maximum_points=800,
    )
    roots = [
        value.real
        for value in np.roots([1.0, 2.0, 0.6])
        if abs(value.imag) < _IMAGINARY_TOLERANCE
    ]
    root = BifurcationExplorer.explore(config, equation, [roots[0], 0.0])
    return exploration_skeleton(
        root,
        (BETA1, BETA2, "b\u2083"),
        (
            "Bogdanov-Takens type change (b = 0)",
            "The BT curve's cross coefficient vanishes at the marked point.",
        ),
    )
