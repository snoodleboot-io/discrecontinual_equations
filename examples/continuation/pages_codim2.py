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
from discrecontinual_equations.continuation.codim2_normal_form import (
    BogdanovTakensNormalForm,
    bogdanov_takens_normal_form,
)
from discrecontinual_equations.continuation.codim3_builder import (
    Codim3Driver,
    Codim3Seed,
)
from discrecontinual_equations.continuation.codim3_config import Codim3Config
from discrecontinual_equations.continuation.connecting_orbit import (
    MeshSpec,
)
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.exploration import (
    BifurcationExplorer,
    ExplorationConfig,
)
from discrecontinual_equations.continuation.homoclinic_curve import HomoclinicCurve
from discrecontinual_equations.continuation.region_analysis import describe_region
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)
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
# Predicted curves are drawn dashed (kind "unstable") in this one colour.
_PREDICTED = "#8a8a8a"
_HOMOCLINIC_RATIO = 49.0 / 25.0
_KUZNETSOV_HOMOCLINIC = 6.0 / 25.0
_HOMOCLINIC_NODES = 121
_RICHARDSON_AMPLITUDES = (0.16, 0.24, 0.32)
# Dense enough that reading the prediction between samples costs < 1e-9.
_PREDICTED_SAMPLES = 2000
_ROUNDING = 1.0e-12
_DRAWN_EVERY = 50

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


def _continued_bt_curve(
    equation: DifferentialEquation,
    curve: str,
    span: float,
) -> list[tuple[float, float]]:
    """The fold or Hopf curve of the planar BT, continued from b2 = span to 0.

    Seeds are the exact codim-1 points at ``b2 = span`` (fold: ``x = 0``; Hopf:
    ``x = -b2``, ``omega^2 = 2 b2``); the continuation, not the formula, draws the
    curve, and stops at the Bogdanov-Takens point.
    """
    if curve == "fold":
        seed = CurveSeed(state=[0.0, 0.0], parameter_a=0.0)
    else:
        seed = CurveSeed(
            state=[-span, 0.0],
            parameter_a=-(span**2),
            frequency=math.sqrt(2.0 * span),
        )
    config = Codim2Config(
        continuation_parameter_index=0,
        second_parameter_index=1,
        curve=curve,
        codim2_detectors=[],
        initial_parameter=span,
        direction=-1,
        parameter_lower_bound=0.0,
        parameter_upper_bound=span,
        initial_step=0.004,
        maximum_step=0.01,
    )
    result = Codim2Driver.run(config, equation, seed)
    return sorted((float(b2), float(b1)) for b1, b2, _omega in result.curve_parameters)


def _predicted_bt_curves(
    equation: DifferentialEquation,
    span: float,
) -> tuple[dict[str, list[tuple[float, float]]], BogdanovTakensNormalForm]:
    """Fold, Hopf and homoclinic predicted from the normal form at the BT point."""
    problem = TwoParameterProblem(equation, 0, 1, 0.0, AutomaticDifferentiation())
    form = bogdanov_takens_normal_form(problem, np.zeros(2), 0.0, 0.0)
    unfolding = form.unfolding
    steps = [
        unfolding.side * span * k / _PREDICTED_SAMPLES
        for k in range(_PREDICTED_SAMPLES + 1)
    ]
    curves = {
        name: [
            (float(alpha[1]), float(alpha[0]))
            for alpha in (getattr(unfolding, name)(step) for step in steps)
        ]
        for name in ("fold", "hopf", "homoclinic")
    }
    return curves, form


def _figure(value: float) -> str:
    return "rounding" if value < _ROUNDING else f"{value:.0e}"


def _deviation(
    continued: list[tuple[float, float]],
    predicted: list[tuple[float, float]],
) -> float:
    """Largest ``|b1_continued - b1_predicted|``, the prediction read at the same b2."""
    sweep = np.array([b2 for b2, _ in predicted])
    value = np.array([b1 for _, b1 in predicted])
    order = np.argsort(sweep)
    reach = float(sweep.max())
    return max(
        abs(b1 - float(np.interp(b2, sweep[order], value[order])))
        for b2, b1 in continued
        if 0.0 < b2 <= reach
    )


def _homoclinic_point(
    equation: DifferentialEquation,
    epsilon: float,
    nodes: int,
) -> tuple[float, float] | None:
    """One continued homoclinic point ``(b2, b1)``, seeded from the BT asymptotics."""
    half = 5.0 * math.sqrt(2.0) / epsilon
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
    point = HomoclinicCurve(equation, 0, 1, mesh).solve_point(
        (5.0 / 7.0) * epsilon**2,
        orbit,
        np.array([epsilon**2, 0.0]),
        -(epsilon**4),
    )
    return None if point is None else (point.sweep, point.continuation)


def _homoclinic_order(
    equation: DifferentialEquation,
    form: BogdanovTakensNormalForm,
) -> tuple[float, float]:
    """``(intercept, slope)`` of ``b1/b2^2 + 6/25`` in Kuznetsov's coordinates.

    Each point is a Richardson step between the drawn mesh and one twice as fine,
    which removes the mesh's second-order error (a constant offset in this ratio);
    the line through what is left is the asymptotics' own remainder.
    """
    sizes, gaps = [], []
    for epsilon in _RICHARDSON_AMPLITUDES:
        ratios = []
        for nodes in (_HOMOCLINIC_NODES, 2 * _HOMOCLINIC_NODES - 1):
            point = _homoclinic_point(equation, epsilon, nodes)
            if point is None:
                return math.nan, math.nan
            beta1, beta2 = form.unfolding.kuznetsov_parameters(
                np.array([point[1], point[0]]),
            )
            ratios.append(beta1 / beta2**2 + _KUZNETSOV_HOMOCLINIC)
        sizes.append(abs(beta2))
        gaps.append((4.0 * ratios[1] - ratios[0]) / 3.0)
    slope, intercept = np.polyfit(sizes, gaps, 1)
    return float(intercept), float(slope)


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
        point = _homoclinic_point(equation, 0.12 + 0.02 * step, _HOMOCLINIC_NODES)
        if point is not None:
            homoclinic.append(point)
    span = homoclinic[-1][0]
    fold = _continued_bt_curve(equation, "fold", span)
    hopf = _continued_bt_curve(equation, "hopf", span)
    predicted, form = _predicted_bt_curves(equation, span)
    drawn = {name: points[::_DRAWN_EVERY] for name, points in predicted.items()}
    parameters[0].value = 0.0
    parameters[1].value = 0.0
    lines = [
        ("fold", "fold (continued)", fold, None),
        ("hopf", "Hopf (continued)", hopf, None),
        ("bogdanov_takens", "homoclinic (continued)", homoclinic, None),
        ("unstable", "fold (normal form)", drawn["fold"], _PREDICTED),
        ("unstable", "Hopf (normal form)", drawn["hopf"], _PREDICTED),
        ("unstable", "homoclinic (normal form)", drawn["homoclinic"], _PREDICTED),
    ]
    markers = [("bogdanov_takens", "Bogdanov-Takens", [(0.0, 0.0)], None)]
    homoclinic_gap = max(
        abs(b1 / (-_HOMOCLINIC_RATIO * b2 * b2) - 1.0)
        for b2, b1 in homoclinic
        if b2 > 0.0
    )
    intercept, slope = _homoclinic_order(equation, form)
    agreement = (
        f"Dashed: predicted by the normal form at the point (a = {form.a:.3g}, "
        f"b = {form.b:.3g}, s = {form.s:+d}). Against the continued curves for "
        f"\u03b2\u2082 \u2264 {span:.3f}: fold within "
        f"{_figure(_deviation(fold, predicted['fold']))} and Hopf within "
        f"{_figure(_deviation(hopf, predicted['hopf']))}; homoclinic within "
        f"{homoclinic_gap:.1%}. With the mesh error removed, b\u2081/b\u2082\u00b2 "
        f"in Kuznetsov's coordinates is -6/25 {intercept:+.0e} "
        f"{slope:+.3f}|b\u2082|: the -(6/25) limit, approached at first order."
    )
    scene = curve_scene(
        lines,
        markers,
        (BETA2, BETA1),
        "Bogdanov-Takens organizing centre",
        "Three curves emanate from the codim-2 point: fold, Hopf, and the homoclinic "
        "curve, tangent to the Hopf with the 5/7 slope ratio. " + agreement,
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
