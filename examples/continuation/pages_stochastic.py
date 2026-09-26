"""The stochastic pages: density, exponent, both thresholds.

Split out of webplot_examples, which assembles the atlas from these.
"""

import numpy as np

from discrecontinual_equations.continuation.lyapunov import (
    LyapunovSettings,
    StochasticLyapunovSettings,
    StochasticSystem,
    lyapunov_spectrum,
    stochastic_lyapunov,
)
from discrecontinual_equations.continuation.stochastic import (
    Ito,
    LyapunovExponent,
    StationaryDensity,
    Stratonovich,
)
from discrecontinual_equations.webplot.scene import Scene
from discrecontinual_equations.webplot.scene_builder import (
    curve_family_scene,
    curve_scene,
    with_regions,
)

try:  # python -m examples.continuation.pages_stochastic
    from examples.continuation.systems import (
        First,
        NoisyDiffusion,
        NoisyDrift,
        Second,
        SpiralSystem,
        State,
        _equation,
        _LorenzDrift,
        _LorenzNoise,
        _stochastic_diffusion,
        _stochastic_drift,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from systems import (
        First,
        NoisyDiffusion,
        NoisyDrift,
        Second,
        SpiralSystem,
        State,
        _equation,
        _LorenzDrift,
        _LorenzNoise,
        _stochastic_diffusion,
        _stochastic_drift,
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
ALPHA = "\u03b1"


def _stochastic_bifurcation() -> Scene:
    settings = StochasticLyapunovSettings(dt=0.005, horizon=80000, transient=0, seed=7)
    curves: list[tuple[str, str, list[tuple[float, float]], str | None]] = []
    for sigma, kind, label in (
        (0.0, "stable", "no noise (s = 0)"),
        (0.6, "unstable", "noise s = 0.6"),
    ):
        points: list[tuple[float, float]] = []
        for step in range(13):
            alpha = -0.1 + 0.03 * step
            system = StochasticSystem(
                _stochastic_drift(alpha),
                _stochastic_diffusion(sigma),
                Ito(),
            )
            top = stochastic_lyapunov(system, np.array([0.0]), settings=settings).top
            points.append((alpha, top))
        curves.append((kind, label, points, None))
    markers = [("bifurcation", "shifted threshold", [(0.18, 0.0)], None)]
    scene = curve_scene(
        curves,
        markers,
        (ALPHA, "top Lyapunov exponent"),
        "Stochastic (D) bifurcation: noise shifts the threshold",
        "The top Lyapunov exponent of the trivial solution, simulated by the "
        "stochastic Benettin method. Without noise it crosses zero at alpha = 0; "
        "multiplicative noise shifts the crossing to alpha = s^2 / 2 = 0.18.",
    )
    return with_regions(
        scene,
        [("noise stabilises: alpha in (0, s^2/2)", 0.09, -0.04)],
    )


def _stochastic_attractor() -> Scene:
    def top(sigma: float) -> float:
        drift = _LorenzDrift(
            variables=[State(), State(), State()],
            parameters=[],
            results=[State(), State(), State()],
            time=None,
        )
        noise = _LorenzNoise(
            variables=[State(), State(), State()],
            parameters=[First(value=sigma)],
            results=[State(), State(), State()],
            time=None,
        )
        settings = StochasticLyapunovSettings(
            dt=0.005,
            horizon=12000,
            transient=4000,
            seed=7,
        )
        system = StochasticSystem(drift, noise, Ito())
        return stochastic_lyapunov(
            system,
            np.array([1.0, 1.0, 1.0]),
            count=1,
            settings=settings,
        ).top

    points = [(0.5 * k, top(0.5 * k)) for k in range(5)]
    lines = [("unstable", "top Lyapunov exponent", points, None)]
    markers = [("bifurcation", "chaos threshold", [(0.0, 0.0)], None)]
    return curve_scene(
        lines,
        markers,
        ("noise amplitude", "top Lyapunov exponent"),
        "Chaos persists under noise: Lorenz top Lyapunov vs noise",
        "Top Lyapunov exponent of the Lorenz attractor under additive noise, from "
        "the stochastic Benettin estimator. It stays near the deterministic value "
        "and well above zero, so the bounded chaotic attractor survives the noise.",
    )


def _dynamical_bifurcation() -> Scene:
    equation = _equation(SpiralSystem, 2, [First(value=-0.3)])
    field = equation.derivative
    settings = LyapunovSettings(dt=0.01, horizon=6000, transient=100)
    curve: list[tuple[float, float]] = []
    for step in range(19):
        mu = -0.3 + 0.03 * step
        field.parameters[0].value = mu
        spectrum = lyapunov_spectrum(field, np.array([1.0, 0.0]), settings=settings)
        curve.append((mu, spectrum.top))
    lines = [("stable", "top Lyapunov exponent", curve, None)]
    markers = [("bifurcation", "D-bifurcation", [(0.0, 0.0)], None)]
    scene = curve_scene(
        lines,
        markers,
        (MU, "Lyapunov exponent"),
        "Dynamical (D) bifurcation: top Lyapunov exponent",
        "The top Lyapunov exponent is computed numerically from the linearised flow "
        "by the Benettin QR method; it passes through zero at mu = 0, where the "
        "reference motion loses stability.",
    )
    return with_regions(
        scene,
        [
            ("stable: exponent < 0", -0.18, curve[2][1]),
            ("unstable: exponent > 0", 0.16, curve[-3][1]),
        ],
    )


def _density_family() -> Scene:
    sigma = 0.6
    equation = _equation(NoisyDrift, 1, [First(value=0.0), Second(value=sigma)])
    drift = equation.derivative
    diffusion = _equation(NoisyDiffusion, 1, [First(value=0.0), Second(value=sigma)])
    diffusion_function = diffusion.derivative
    diffusion_function.parameters[1].value = sigma
    grid = np.linspace(1.0e-3, 2.6, 1600)
    values = [0.0, 0.1, 0.18, 0.3, 0.5, 0.7]
    curves = []
    for alpha in values:
        drift.parameters[0].value = alpha
        _grid, density = StationaryDensity(
            drift,
            diffusion_function,
            Stratonovich(),
            grid,
        ).evaluate()
        peak = float(np.max(density))
        curves.append(
            [(float(x), float(d / peak)) for x, d in zip(grid, density, strict=True)],
        )
    return curve_family_scene(
        curves,
        values,
        ("x", "stationary density (scaled)"),
        "\u03b1",
        (
            "Stochastic P-bifurcation: stationary density",
            "The density mode leaves the origin at \u03b1 = \u03c3\u00b2/2 = 0.18.",
        ),
    )


def _lyapunov() -> Scene:
    sigma = 0.6
    drift = _equation(NoisyDrift, 1, [First(value=0.0), Second(value=sigma)]).derivative
    diffusion = _equation(
        NoisyDiffusion,
        1,
        [First(value=0.0), Second(value=sigma)],
    ).derivative
    diffusion.parameters[1].value = sigma
    exponent = LyapunovExponent(drift, diffusion, Stratonovich())
    curve = []
    for alpha in np.linspace(-0.4, 0.4, 41):
        drift.parameters[0].value = float(alpha)
        curve.append((float(alpha), float(exponent.at(0.0))))
    markers = [("zero_crossing", "D-bifurcation (\u03bb = 0)", [(0.0, 0.0)], None)]
    return curve_scene(
        [("curve", "Lyapunov exponent", curve, None)],
        markers,
        ("\u03b1", "top Lyapunov exponent \u03bb"),
        "Stochastic D-bifurcation: Lyapunov exponent",
        "The linear top Lyapunov exponent changes sign at \u03b1 = 0.",
    )
