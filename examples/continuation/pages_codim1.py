"""The codimension-one bifurcations, each in one parameter.

Split out of webplot_examples, which assembles the atlas from these.
"""

from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.webplot.scene import Scene
from discrecontinual_equations.webplot.scene_builder import (
    bifurcation_scene,
)

try:  # python -m examples.continuation.pages_codim1
    from examples.continuation.systems import (
        First,
        Hopf,
        Hysteresis,
        Pitchfork,
        SaddleNode,
        Transcritical,
        _equation,
        _solve,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from systems import (
        First,
        Hopf,
        Hysteresis,
        Pitchfork,
        SaddleNode,
        Transcritical,
        _equation,
        _solve,
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
NORM = "\u2016x\u2016"


def _saddle_node() -> Scene:
    equation = _equation(SaddleNode, 1, [First(value=0.0)])
    config = ContinuationConfig(
        detectors=["fold"],
        initial_parameter=1.2,
        direction=-1,
        measure="component",
        parameter_lower_bound=-1.0,
        parameter_upper_bound=1.4,
    )
    branch = _solve(equation, config, [1.2**0.5])
    return bifurcation_scene(
        branch,
        "Saddle-node (fold)",
        "x' = " + MU + " - x^2. Stable and unstable branches meet at the fold.",
        MU,
        "x",
    )


def _transcritical() -> Scene:
    equation = _equation(Transcritical, 1, [First(value=0.0)])
    config = ContinuationConfig(
        detectors=["branch_point"],
        initial_parameter=1.0,
        direction=-1,
        measure="component",
        parameter_lower_bound=-1.2,
        parameter_upper_bound=1.2,
    )
    branch = _solve(equation, config, [1.0])
    return bifurcation_scene(
        branch,
        "Transcritical",
        "x' = " + MU + " x - x^2. Two branches cross and exchange stability.",
        MU,
        "x",
    )


def _pitchfork() -> Scene:
    equation = _equation(Pitchfork, 1, [First(value=0.0)])
    config = ContinuationConfig(
        detectors=["branch_point"],
        initial_parameter=-1.2,
        direction=1,
        measure="component",
        parameter_lower_bound=-1.2,
        parameter_upper_bound=1.2,
    )
    branch = _solve(equation, config, [0.0])
    return bifurcation_scene(
        branch,
        "Pitchfork",
        "x' = " + MU + " x - x^3. A symmetric pair splits off the trivial branch; "
        "the quadratic coefficient vanishes so it is not a transcritical crossing.",
        MU,
        "x",
    )


def _hysteresis() -> Scene:
    equation = _equation(Hysteresis, 1, [First(value=0.0)])
    config = ContinuationConfig(
        detectors=["fold"],
        initial_parameter=3.0,
        direction=-1,
        measure="component",
        parameter_lower_bound=-3.2,
        parameter_upper_bound=3.2,
    )
    branch = _solve(equation, config, [2.1038])
    return bifurcation_scene(
        branch,
        "Hysteresis (two folds)",
        "x' = " + MU + " + 3x - x^3. An S-shaped branch with two folds.",
        MU,
        "x",
    )


def _hopf() -> Scene:
    equation = _equation(Hopf, 2, [First(value=0.0)])
    config = ContinuationConfig(
        detectors=["hopf"],
        initial_parameter=0.6,
        direction=-1,
        measure="norm",
        parameter_lower_bound=-0.6,
        parameter_upper_bound=0.7,
    )
    branch = _solve(equation, config, [0.0, 0.0])
    return bifurcation_scene(
        branch,
        "Hopf bifurcation",
        "Planar Hopf normal form; a periodic orbit is born at " + MU + " = 0.",
        MU,
        NORM,
    )
