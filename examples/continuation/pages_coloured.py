"""The coloured-noise page: both thresholds of the noisy pitchfork against tau.

The white pitchfork ``dx = (a x - x^3) dt + s x o dW`` has its dynamical threshold at
``a = 0`` and its phenomenological one at ``a = s^2 / 2``. Colour the noise by
augmentation (``ColouredNoise``) and ask the same two questions of the pair
``(x, eta)``, with the same locators: ``DynamicalThreshold`` over ``MeanTopExponent``
of the augmented system, and ``ModeCountThreshold`` over a kernel estimate of an
ensemble of paths - the sample-based route, since the augmented pair's noise is
singular and the grid Fokker-Planck solver refuses it.

What the page shows, and why it is not a pair of curves bending away from the white
values: at this normalisation (fixed intensity, Hänggi and Jung's ``D = 1/2``) both
thresholds are exactly independent of ``tau``. The trivial solution linearises to
``(a + s eta) dx`` and ``eta`` averages to zero, so its exponent is ``a``; and near
the origin the stationary density behaves as ``|x|^(2a/s^2 - 1)`` at every ``tau``
(the tilted solution ``exp(theta log|x| - tau eta^2 - theta s tau eta)`` of the pair's
Fokker-Planck equation has ``theta = 2a/s^2`` whatever ``tau`` is), so the density
at the origin turns from infinite to zero at ``a = s^2 / 2`` at every ``tau``. What
colour changes is the rest of the density: above ``tau ~ 1`` side peaks grow while
the origin is still a (weak, integrable) peak, a three-mode regime the white system
does not have (Hänggi and Jung, Fig. 7.1b, in the intensity ``I = x^2``); the
unified coloured-noise approximation places their onset. The sampled detector cannot
see a singularity narrower than its bandwidth, so it reads the kernel-smoothed
density: at small ``tau`` its onset is the white ``s^2 / 2 + h^2`` plus a detection
lag, and as ``tau`` grows the mass moving off the origin makes the smoothed density
bimodal earlier - already below ``tau ~ 1``, where the UCNA has no side peaks yet -
until it meets the side peaks above it. Where it sits is therefore a property of the
density at scale ``h``, not a third threshold.
"""

import numpy as np

from discrecontinual_equations.continuation.fokker_planck import DensityGrid
from discrecontinual_equations.continuation.lyapunov import (
    MatrixNoiseSystem,
    StochasticLyapunovSettings,
)
from discrecontinual_equations.continuation.noise import NoiseMatrix
from discrecontinual_equations.continuation.root_finder import (
    Bisection,
    RootFinderSettings,
    Secant,
)
from discrecontinual_equations.continuation.stochastic import Ito
from discrecontinual_equations.continuation.stochastic_threshold import (
    DynamicalThreshold,
    MeanTopExponent,
)
from discrecontinual_equations.continuation.topological_modes import (
    KernelDensity,
    ModeCountThreshold,
    TopologicalModes,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.coloured_noise import ColouredNoise
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.solver.solver_config import StochasticConfig
from discrecontinual_equations.solver.stochastic.ensemble import StochasticEnsemble
from discrecontinual_equations.variable import Variable
from discrecontinual_equations.webplot.scene import Scene
from discrecontinual_equations.webplot.scene_builder import curve_scene, with_regions

TAU = "τ"
SIGMA = 0.6
_TAUS = (0.05, 0.3, 1.0, 2.0, 4.0)
_PATHS = 6000
_BANDWIDTH = 0.1
_BRACKET = (0.02, 0.4)
_SEED = 21


class _Rate(Parameter, name="a", abbreviation="a"):
    """Pitchfork parameter."""


class _Strength(Parameter, name="s", abbreviation="s"):
    """Noise amplitude."""


class NoisyPitchfork(StochasticFunction):
    """``dx = (a x - x^3) dt + s x dW``, with array forms for the ensemble."""

    def eval(self, point, time=None):  # noqa: ARG002
        return [self.parameters[0].value * point[0] - point[0] ** 3]

    def diffusion(self, point, time=None):  # noqa: ARG002
        return [self.parameters[1].value * point[0]]

    def eval_batch(self, points, time=None):  # noqa: ARG002
        return self.parameters[0].value * points - points**3

    def diffusion_batch(self, points, time=None):  # noqa: ARG002
        return self.parameters[1].value * points


def _coloured(rate: float, tau: float) -> ColouredNoise:
    white = NoisyPitchfork(
        variables=[Variable(name="x", discretization=[])],
        parameters=[_Rate(value=rate), _Strength(value=SIGMA)],
        results=[Variable(name="dx/dt", discretization=[])],
        time=None,
    )
    return ColouredNoise(white, tau)


def _step(tau: float) -> float:
    return min(0.01, tau / 10.0)


def _sampled_modes(tau: float):
    """Modes of a kernel estimate of an ensemble's final states, common seeds."""
    grid = DensityGrid.box([-2.5], [2.5], [251])
    signs = np.where(np.arange(_PATHS) % 2 == 0, 1.0, -1.0)[:, None]
    horizon = max(25.0, 10.0 * tau)

    def modes_at(rate: float) -> TopologicalModes:
        coloured = _coloured(rate, tau)
        initial = coloured.stationary_drivers(
            0.5 * signs,
            np.random.default_rng(_SEED),
        )
        config = StochasticConfig(
            start_time=0.0,
            end_time=horizon,
            step_size=_step(tau),
            random_seed=_SEED + 1,
        )
        equation = DifferentialEquation(
            variables=coloured.variables,
            time=Variable(name="t", discretization=[]),
            parameters=coloured.parameters,
            derivative=coloured,
        )
        final = StochasticEnsemble(config).solve(equation, initial).final
        return KernelDensity(final[:, 0], grid, bandwidth=_BANDWIDTH).modes()

    return modes_at


def _exponent(tau: float) -> MeanTopExponent:
    def family(rate: float) -> MatrixNoiseSystem:
        coloured = _coloured(rate, tau)
        return MatrixNoiseSystem(
            coloured,
            NoiseMatrix(coloured.driver_columns()),
            Ito(),
        )

    settings = StochasticLyapunovSettings(
        dt=_step(tau),
        horizon=int(100.0 / _step(tau)),
        transient=0,
    )
    return MeanTopExponent(family, np.array([0.0, 0.0]), settings, seeds=(3,))


def _ucna_side_onset(tau: float) -> float | None:
    """Smallest ``a`` at which the UCNA density has a peak off the origin.

    UCNA for this system (Hänggi and Jung eq. 5.40, here ``f' - f g'/g = -2 x^2``):
    ``p ~ (1 + 2 tau x^2) |x|^(2a/s^2 - 1) exp(((2 tau a - 1) x^2 - tau x^4) / s^2)``.
    """
    x = np.geomspace(1.0e-4, 3.0, 20000)
    for rate in np.linspace(0.0005, SIGMA**2 / 2.0, 360):
        log_density = (
            np.log1p(2.0 * tau * x * x)
            + (2.0 * rate / SIGMA**2 - 1.0) * np.log(x)
            + ((2.0 * tau * rate - 1.0) * x * x - tau * x**4) / SIGMA**2
        )
        rises = np.diff(log_density)
        if np.any((rises[:-1] > 0.0) & (rises[1:] <= 0.0)):
            return float(rate)
    return None


def _coloured_thresholds() -> Scene:
    bisection = Bisection(RootFinderSettings(1.0e-14, 4.0e-3, 30))
    secant = Secant(RootFinderSettings(1.0e-10, 1.0e-6, 20))
    sampled: list[tuple[float, float]] = []
    dynamical: list[tuple[float, float]] = []
    for tau in _TAUS:
        located = ModeCountThreshold(_sampled_modes(tau), bisection).locate(*_BRACKET)
        if located is not None:
            sampled.append((tau, located))
        onset = DynamicalThreshold(_exponent(tau).at, secant).locate(-0.3, 0.3)
        if onset is not None:
            dynamical.append((tau, onset))
    span = (0.0, _TAUS[-1])
    side = [
        (float(tau), onset)
        for tau in np.linspace(0.9, _TAUS[-1], 24)
        if (onset := _ucna_side_onset(float(tau))) is not None
    ]
    lines = [
        (
            "unstable",
            "origin P-threshold s²/2 (exact, every τ)",
            [(span[0], SIGMA**2 / 2.0), (span[1], SIGMA**2 / 2.0)],
            None,
        ),
        (
            "stable",
            "D-threshold 0 (exact, every τ)",
            [(span[0], 0.0), (span[1], 0.0)],
            None,
        ),
        ("curve", "side-peak onset (UCNA, approximate)", side, None),
    ]
    markers = [
        ("bifurcation", "two or more modes, from samples (h = 0.1)", sampled, None),
        (
            "zero_crossing",
            "D-threshold, located on the augmented pair",
            dynamical,
            None,
        ),
    ]
    scene = curve_scene(
        lines,
        markers,
        (f"noise correlation time {TAU}", "threshold in a"),
        "Coloured noise: the pitchfork's P- and D-thresholds against τ",
        "dx = (a x - x³) dt + s x η dt with η an Ornstein-Uhlenbeck "
        "process of intensity fixed as τ varies (s = 0.6), so τ → 0 is "
        "the white Stratonovich pitchfork. Neither threshold moves: the origin's "
        "exponent is a and its density exponent is 2a/s² - 1 at every τ. "
        "Colour reshapes the rest of the density: above τ ≈ 1 side peaks "
        "grow while the origin is still a peak (UCNA estimate). The topological "
        "detector reads samples smoothed at h = 0.1, so its two-mode onset starts "
        "at the white s²/2 + h² (plus a detection lag) and falls as τ moves "
        "mass off the origin.",
    )
    return with_regions(
        scene,
        [
            ("white limit: s²/2 + h² from samples", 0.4, 0.26),
            ("three modes: origin and side peaks", 3.0, 0.16),
        ],
    )


if __name__ == "__main__":  # pragma: no cover
    for tau in _TAUS:
        print(tau, _ucna_side_onset(tau))
