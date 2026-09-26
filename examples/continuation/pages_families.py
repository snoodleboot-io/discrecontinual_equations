"""Parameter families and the surfaces they sweep out.

Split out of webplot_examples, which assembles the atlas from these.
"""

import numpy as np

from discrecontinual_equations.continuation.family_builder import FamilyDriver
from discrecontinual_equations.continuation.family_config import FamilyConfig
from discrecontinual_equations.systems import (
    FluxgateRing,
)
from discrecontinual_equations.webplot.scene import Scene
from discrecontinual_equations.webplot.scene_builder import (
    family_scene,
    surface_scene,
)

try:  # python -m examples.continuation.pages_families
    from examples.continuation.systems import (
        BogdanovTakensFamily,
        CuspForm,
        First,
        Second,
        Third,
        _equation,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from systems import (
        BogdanovTakensFamily,
        CuspForm,
        First,
        Second,
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

NORM = "\u2016x\u2016"
BETA1, BETA2, BETA3 = "\u03b2\u2081", "\u03b2\u2082", "\u03b2\u2083"
BETA1, BETA2, BETA3 = "\u03b2\u2081", "\u03b2\u2082", "\u03b2\u2083"


def _cusp_family() -> Scene:
    equation = _equation(CuspForm, 1, [First(value=0.0), Second(value=0.0)])
    config = FamilyConfig(
        continuation_parameter_index=0,
        family_parameter_index=1,
        family_values=[-0.5, -0.25, 0.0, 0.5, 1.0, 1.5],
        initial_parameter=-3.0,
        direction=1,
        measure="component",
        parameter_lower_bound=-3.2,
        parameter_upper_bound=3.2,
        initial_step=0.01,
        maximum_step=0.05,
        maximum_points=1200,
        detectors=["fold"],
    )
    family = FamilyDriver.run(config, equation, [-(3.0 ** (1 / 3))])
    return family_scene(
        family,
        (BETA1, "x", BETA2),
        "Cusp: a diagram family",
        "As " + BETA2 + " passes the cusp, a fold pair is born in the diagram.",
    )


def _fluxgate_family() -> Scene:
    equation = _equation(
        FluxgateRing,
        3,
        [First(value=0.0), Second(value=3.0), Third(value=0.0)],
    )
    config = FamilyConfig(
        continuation_parameter_index=0,
        family_parameter_index=1,
        family_values=[2.0, 2.5, 3.0, 3.5, 4.0],
        initial_parameter=0.0,
        direction=-1,
        measure="norm",
        parameter_lower_bound=-5.0,
        parameter_upper_bound=0.2,
        initial_step=0.01,
        maximum_step=0.03,
        maximum_points=2000,
        detectors=["branch_point", "fold", "hopf"],
    )
    family = FamilyDriver.run(config, equation, [0.0, 0.0, 0.0])
    return family_scene(
        family,
        ("\u03bb", NORM, "c"),
        "Coupled-core fluxgate: symmetry breaking vs gain",
        "N = 3 ring; the symmetry-breaking point tracks " + "\u03bb = 1 - c.",
    )


def _cusp_surface() -> Scene:
    equation = _equation(CuspForm, 1, [First(value=0.0), Second(value=0.0)])
    values = [round(float(v), 3) for v in np.linspace(-0.3, 1.5, 16)]
    config = FamilyConfig(
        continuation_parameter_index=0,
        family_parameter_index=1,
        family_values=values,
        initial_parameter=-3.0,
        direction=1,
        measure="component",
        parameter_lower_bound=-3.2,
        parameter_upper_bound=3.2,
        initial_step=0.01,
        maximum_step=0.05,
        maximum_points=1500,
        detectors=["fold"],
    )
    family = FamilyDriver.run(config, equation, [-(3.0 ** (1 / 3))])
    return surface_scene(
        family,
        (BETA1, BETA2, "x"),
        (
            "Cusp: equilibrium surface with continued fold curve",
            "The fold locus (pink) is the semicubical cusp curve over the plane.",
        ),
    )


def _fluxgate_surface() -> Scene:
    equation = _equation(
        FluxgateRing,
        3,
        [First(value=0.0), Second(value=3.0), Third(value=0.0)],
    )
    values = [round(float(v), 3) for v in np.linspace(2.0, 4.0, 12)]
    config = FamilyConfig(
        continuation_parameter_index=0,
        family_parameter_index=1,
        family_values=values,
        initial_parameter=0.0,
        direction=-1,
        measure="norm",
        parameter_lower_bound=-5.0,
        parameter_upper_bound=0.2,
        initial_step=0.01,
        maximum_step=0.03,
        maximum_points=2000,
        detectors=["branch_point"],
    )
    family = FamilyDriver.run(config, equation, [0.0, 0.0, 0.0])
    return surface_scene(
        family,
        ("\u03bb", "c", NORM),
        (
            "Coupled-core fluxgate: equilibrium surface",
            "The symmetry-breaking locus (green) tracks \u03bb = 1 - c over the sheet.",
        ),
    )


def _bogdanov_takens_config(values: list[float]) -> FamilyConfig:
    return FamilyConfig(
        continuation_parameter_index=0,
        family_parameter_index=1,
        family_values=values,
        initial_parameter=-1.5,
        direction=1,
        measure="component",
        parameter_lower_bound=-1.7,
        parameter_upper_bound=0.05,
        initial_step=0.005,
        maximum_step=0.02,
        maximum_points=3000,
        detectors=["fold", "hopf"],
    )


def _bogdanov_takens_family() -> Scene:
    equation = _equation(BogdanovTakensFamily, 2, [First(value=0.0), Second(value=0.5)])
    values = [0.2, 0.4, 0.6, 0.8, 1.0]
    family = FamilyDriver.run(
        _bogdanov_takens_config(values),
        equation,
        [-(1.5**0.5), 0.0],
    )
    return family_scene(
        family,
        (BETA1, "x", BETA2),
        "Fold and Hopf in a family, meeting at Bogdanov-Takens",
        "Each slice has a fold and a Hopf; they merge at the BT point as b2\u21920.",
    )


def _bogdanov_takens_surface() -> Scene:
    equation = _equation(
        BogdanovTakensFamily,
        2,
        [First(value=0.0), Second(value=0.15)],
    )
    values = [round(float(v), 3) for v in np.linspace(0.15, 1.1, 12)]
    family = FamilyDriver.run(
        _bogdanov_takens_config(values),
        equation,
        [-(1.5**0.5), 0.0],
    )
    return surface_scene(
        family,
        (BETA1, BETA2, "x"),
        (
            "Bogdanov-Takens: equilibrium surface with two continued loci",
            "Fold curve (pink) and Hopf curve (amber) meet at the codim-2 point.",
        ),
    )


def _fluxgate_bias_surface() -> Scene:
    equation = _equation(
        FluxgateRing,
        3,
        [First(value=0.0), Second(value=2.0), Third(value=0.0)],
    )
    values = [round(float(v), 3) for v in np.linspace(0.8, 3.0, 12)]
    config = FamilyConfig(
        continuation_parameter_index=2,
        family_parameter_index=1,
        family_values=values,
        initial_parameter=-1.2,
        direction=1,
        measure="component",
        parameter_lower_bound=-1.4,
        parameter_upper_bound=1.4,
        initial_step=0.005,
        maximum_step=0.02,
        maximum_points=3000,
        detectors=["fold"],
    )
    seed = -0.9
    family = FamilyDriver.run(config, equation, [seed, seed, seed])
    return surface_scene(
        family,
        ("\u03b5", "c", "x"),
        (
            "Fluxgate: bias-field surface over gain",
            "Hysteresis in the bias \u03b5 emerges once the gain c exceeds one.",
        ),
    )
