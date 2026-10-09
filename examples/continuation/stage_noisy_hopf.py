"""The noisy Hopf as a film: two bifurcations, at two parameter values.

For ``dx = f(x) dt + G(x) dW`` two different things deserve the name
bifurcation. The *phenomenological* (P) one is a change in the shape of the
stationary density: its peak leaves the origin and becomes a ring. The
*dynamical* (D) one is the top Lyapunov exponent of the invariant origin
changing sign: paths that fell into it start to escape. They are kept as
separate objects in ``continuation/stochastic.py`` because they happen at
different parameter values, and this film is where a reader watches that.

The drift is the supercritical Hopf normal form of ``stage_hopf``. The noise
is the *conformal* multiplicative noise ``sigma z dB`` - in real coordinates
``G = sigma [[x, -y], [y, x]]``, two independent Brownian motions - and the
choice is forced. Additive noise, the first thing one writes, has a P
threshold at ``mu = 0`` and no D threshold at all: the top exponent of the
noisy Hopf with additive isotropic noise and no shear is negative for every
``mu`` (measured with the library's own estimator at sigma = 0.7: from -1.23
at mu = -0.5 to -0.30 at mu = 1.5), so there is only one event to film. A
dynamical bifurcation needs an invariant reference state, which needs the
noise to vanish there; the one isotropic noise that does so smoothly is the
conformal one. It leaves ``D = G G^T = sigma^2 r^2 I`` isotropic, so the
rotation drops out of the stationary equation and both thresholds have closed
forms the film is checked against:

* ``p(x, y) = C r^(2 mu / sigma^2 - 2) exp(-r^2 / sigma^2)`` for ``mu > 0``
  (no density on the plane below that: the whole mass sits on the origin),
  whose peak leaves the origin at **``mu = sigma^2``**. Below it the density is
  an integrable spike at the origin; above it a crater with its crest at
  ``r = sqrt(mu - sigma^2)``, strictly inside the deterministic cycle at
  ``sqrt(mu)``.
* The origin's top Lyapunov exponent is exactly ``mu``, so D is at **``mu =
  0``**: ``log|z|`` of the linearisation is ``mu t + sigma W_1(t)`` because the
  complex increment satisfies ``(dB)^2 = 0``.

The Ito and Stratonovich readings of this equation are the same equation: the
correction ``(1/2) sum_k (dG_k/dx) G_k`` vanishes identically for conformal
noise, which ``check_noise`` verifies, so neither threshold moves between
them. The page says so, and names the Ito reading it is integrated in.

``mu`` is scrubbed at fixed ``sigma = 0.7``: D at 0 and P at 0.49 both sit
inside ``[-0.5, 1.5]`` with a quarter of the axis between them. Scrubbing
``sigma`` at fixed ``mu`` would move P (at ``sigma = sqrt(mu)``) but never D,
which is at ``mu = 0`` for every sigma, so only one threshold would pass.
"""

import math
import sys

import numpy as np

from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    StationaryFokkerPlanck,
)
from discrecontinual_equations.continuation.lyapunov import (
    MatrixNoiseSystem,
    StochasticLyapunovSettings,
)
from discrecontinual_equations.continuation.noise import NoiseMatrix
from discrecontinual_equations.continuation.root_finder import (
    RootFinderSettings,
    Secant,
)
from discrecontinual_equations.continuation.stochastic import Ito
from discrecontinual_equations.continuation.stochastic_threshold import (
    MeanTopExponent,
    RadialCrater,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable
from discrecontinual_equations.webplot.report import AtlasEntry
from discrecontinual_equations.webplot.stage import (
    Frame,
    Lattice,
    Projection,
    StageScene,
    StageSystem,
    Term,
    View,
)
from discrecontinual_equations.webplot.stage_builder import Continued, Film, stage_scene
from discrecontinual_equations.webplot.stochastic_stage_builder import (
    ClosedForms,
    Diagnostics,
    Measurements,
    NoiseModel,
    Thresholds,
    stochastic_stage,
)
from discrecontinual_equations.webplot.stochastic_stage_renderer import (
    StochasticStageRenderer,
)

try:  # python -m examples.continuation.stage_noisy_hopf
    from examples.continuation.stage_hopf import Hopf, cycle_branch
    from examples.continuation.stage_support import (
        Mu,
        State,
        continue_branch,
        equation,
        publish,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from stage_hopf import Hopf, cycle_branch
    from stage_support import Mu, State, continue_branch, equation, publish

P_LO, P_HI, FRAMES = -0.5, 1.5, 81
BOX = ((-1.5, 1.5), (-1.5, 1.5))
SIGMA = 0.7
# Closed forms the film is checked against: the density craters at sigma^2,
# the origin's exponent is mu itself.
P_EXACT = SIGMA * SIGMA
D_EXACT = 0.0
# The density is solved on a box wide enough that its reflecting boundary
# sits where the density is negligible (below 1e-3 of the crest at mu = 1.5),
# and drawn on the stage's own box. An even count keeps every cell centre off
# the origin, where this noise vanishes and the diffusion is singular.
DENSITY_BOX = 2.5
DENSITY_CELLS = 100
# The threshold is located on a finer grid, and several cells out from the
# origin, because the finite-volume solution overshoots the cells around a
# singular origin by a fixed factor per cell however fine the grid; see the
# builder's docstring. On this grid at this offset the located P sits about
# 0.017 above the closed form, 0.01 of it the probe's own O(r^2) bias.
LOCATING_CELLS = 200
PROBE_OFFSET = 4
# Where each threshold is looked for. The phenomenological bracket starts
# above zero because there is no density on the plane at or below it.
P_BRACKET = (0.15, 1.0)
D_BRACKET = (-0.5, 0.5)
# The exponent is a simulated path and costs seconds; it is drawn at these
# values, plus whatever the root finder evaluates on its way to the zero.
EXPONENT_SAMPLES = (-0.5, 0.0, 0.5, 1.0, 1.5)
# Two paths of 200 time units each, with common random numbers across mu. The
# estimate is mu plus one Monte-Carlo offset shared by every mu, of standard
# deviation sigma / sqrt(2 T) = 0.035, which displaces the located D by that
# much; the closed form is drawn beside it so the offset is visible, not hidden.
EXPONENT_SETTINGS = StochasticLyapunovSettings(dt=0.01, horizon=20000, transient=0)
EXPONENT_SEEDS = (3, 5)
# The page's own generator seed and Euler-Maruyama step.
PATH_SEED = 7
PATH_STEP = 0.02
_IDENTITY = [[1.0, 0.0], [0.0, 1.0]]


class Sigma(Parameter, name="Sigma", abbreviation="sigma"):
    """The noise amplitude, held fixed while mu is scrubbed."""


class ConformalColumn(DeterministicFunction):
    """Column ``index`` of ``sigma [[x, -y], [y, x]]``: the noise ``sigma z dB``.

    Each column is driven by its own Brownian motion. Together they are
    multiplication of ``z = x + i y`` by the complex increment
    ``sigma (dW_1 + i dW_2)``, which is why the Stratonovich correction cancels:
    the Jacobian of the first column is ``sigma I`` and of the second a rotation
    by a right angle, and ``sigma I (x, y) + sigma R (-y, x) = 0``.
    """

    def __init__(
        self,
        index: int,
        variables: list[Variable],
        parameters: list[Parameter],
        results: list[Variable],
        time: Variable | None = None,
    ) -> None:
        super().__init__(variables, parameters, results, time)
        self._index = index

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        sigma = self.parameters[1].value
        x, y = point[0], point[1]
        if self._index == 0:
            return [sigma * x, sigma * y]
        return [-sigma * y, sigma * x]


def _parameters(mu: float) -> list[Parameter]:
    return [Mu(value=mu), Sigma(value=SIGMA)]


def _noise(parameters: list[Parameter]) -> NoiseMatrix:
    return NoiseMatrix(
        [
            ConformalColumn(index, [State(), State()], parameters, [State(), State()])
            for index in range(2)
        ],
    )


def _drift(parameters: list[Parameter]) -> Hopf:
    return Hopf(
        variables=[State(), State()],
        parameters=parameters,
        results=[State(), State()],
        time=None,
    )


def _grid(cells: int) -> DensityGrid:
    return DensityGrid.box(
        [-DENSITY_BOX, -DENSITY_BOX],
        [DENSITY_BOX, DENSITY_BOX],
        [cells, cells],
    )


def hopf_terms() -> list[list[Term]]:
    """``mu x - y - x r^2, x + mu y - y r^2`` as the page integrates it."""
    return [
        [
            Term(1.0, [1, 0], 1),
            Term(-1.0, [0, 1]),
            Term(-1.0, [3, 0]),
            Term(-1.0, [1, 2]),
        ],
        [
            Term(1.0, [1, 0]),
            Term(1.0, [0, 1], 1),
            Term(-1.0, [2, 1]),
            Term(-1.0, [0, 3]),
        ],
    ]


def noise_terms(sigma: float = SIGMA) -> list[list[list[Term]]]:
    """``sigma [[x, -y], [y, x]]``, one term list per component per driver."""
    return [
        [[Term(sigma, [1, 0])], [Term(sigma, [0, 1])]],
        [[Term(-sigma, [0, 1])], [Term(sigma, [1, 0])]],
    ]


def density_at(mu: float, grid: DensityGrid) -> StationaryFokkerPlanck | None:
    """The stationary problem at ``mu`` on ``grid``, or ``None`` where none exists.

    For ``mu <= 0`` the closed form is not normalisable: every path falls into
    the origin and the stationary measure is a point mass there. A grid asked
    for that density would return a grid-dependent spike, so it is not asked.
    """
    if mu <= 0.0:
        return None
    parameters = _parameters(mu)
    return StationaryFokkerPlanck(_drift(parameters), _noise(parameters), Ito(), grid)


def exponent_at(mu: float) -> float:
    """The origin's top Lyapunov exponent at ``mu``, by the library's estimator.

    The reference state is the origin itself, which the noise leaves exactly
    invariant, so only the tangent frame moves; what is estimated is the
    exponent whose sign change is the D-bifurcation, not the exponent of the
    stationary measure (which for this system is a different, always
    negative, number once ``mu > 0``).
    """

    def family(value: float) -> MatrixNoiseSystem:
        parameters = _parameters(value)
        return MatrixNoiseSystem(_drift(parameters), _noise(parameters), Ito())

    estimator = MeanTopExponent(
        family,
        np.zeros(2),
        EXPONENT_SETTINGS,
        EXPONENT_SEEDS,
    )
    return estimator.at(mu)


def exact_crest(mu: float) -> float | None:
    """``sqrt(mu - sigma^2)``: where the density's crest sits once it craters."""
    return math.sqrt(mu - P_EXACT) if mu > P_EXACT else None


def describe(frame: Frame) -> str | None:
    """Name the regime by what the density at this frame actually does."""
    density = frame.density
    if density is None or not density.values:
        return (
            "D-stable origin: the top exponent is negative, every path falls in, "
            "and the stationary measure is a point mass"
        )
    if density.crest is None:
        return (
            "Past D, before P: paths escape the origin yet the density still "
            "peaks there - an integrable spike, with excursions"
        )
    return (
        "Past P: the density craters, its crest at sqrt(mu - sigma^2), inside "
        "the deterministic cycle at sqrt(mu)"
    )


def _system() -> StageSystem:
    return StageSystem(
        "Noisy Hopf: two bifurcations",
        "The Hopf normal form driven by conformal multiplicative noise sigma z dB. "
        "At mu = 0 the origin's top Lyapunov exponent crosses zero and paths "
        "start to escape it: the dynamical (D) bifurcation. Not until "
        "mu = sigma^2 = 0.49 does the stationary density stop peaking at the "
        "origin and crater into a ring: the phenomenological (P) bifurcation. "
        "Two events, two parameter values, one axis.",
        "μ",
        equation=(
            r"dx = (\mu x - y - x r^2)\,dt + \sigma(x\,dW_1 - y\,dW_2),\quad "
            r"dy = (x + \mu y - y r^2)\,dt + \sigma(y\,dW_1 + x\,dW_2)"
        ),
        note=(
            "Written in the Ito reading; for this noise the Stratonovich "
            "correction (1/2) sum_k (dG_k/dx) G_k vanishes identically, so both "
            "readings are the same equation and neither threshold moves between "
            "them. Additive noise sigma dW was measured first and has no D "
            "threshold: the top exponent of the stationary motion stays negative "
            "for every mu (-1.23 at mu = -0.5 to -0.30 at mu = 1.5), so a "
            "dynamical bifurcation needs a noise that vanishes at the origin. "
            "The density is solved by StationaryFokkerPlanck on a 100 x 100 grid "
            "over [-2.5, 2.5]^2 for the panel, and the P threshold located by "
            "RadialCrater four cells from the origin on a 200 x 200 grid; the "
            "closed form is p = C r^(2 mu / sigma^2 - 2) exp(-r^2 / sigma^2) with "
            "its crest at sqrt(mu - sigma^2), so P is at mu = sigma^2 = 0.49 "
            "exactly, and the located value sits about 0.02 above it: the "
            "finite-volume solution overshoots the cells around the singular "
            "origin, and the probe reads a finite radius. The exponent is the "
            "Benettin estimate at the invariant origin over two paths of 200 time "
            "units with common random numbers, which is mu plus one shared "
            "Monte-Carlo offset of standard deviation 0.035; the exact exponent "
            "is mu, so D is at mu = 0 exactly, and the dashed closed forms show "
            "the gap. Sample paths are integrated in the browser by "
            "Euler-Maruyama from a generator seeded in the page."
        ),
    )


def noisy_hopf_stage() -> StageScene:
    """Build the stage: deterministic skeleton first, then everything stochastic."""
    eq = equation(Hopf, _parameters(P_LO))
    origin = continue_branch(eq, [0.0, 0.0], (P_LO, P_HI), ["hopf"])
    view = View(
        [Projection("x - y", _IDENTITY, ("x", "y"))],
        [BOX[0], BOX[1]],
        hopf_terms(),
    )
    continued = Continued(eq, 0, origin, cycle_branch())
    scene = stage_scene(
        continued,
        Film(
            np.linspace(P_LO, P_HI, FRAMES),
            Lattice(BOX, 36, "x", "y", view=view),
            _system(),
        ),
    )
    noise = NoiseModel(
        noise_terms(),
        _noise(eq.derivative.parameters),
        Ito(),
        seed=PATH_SEED,
        step=PATH_STEP,
    )
    diagnostics = Diagnostics(
        Measurements(
            density_at,
            exponent_at,
            EXPONENT_SAMPLES,
            ClosedForms(exponent=lambda mu: mu, crest=exact_crest),
        ),
        Thresholds(
            RadialCrater([0.0, 0.0], axis=0, offset=PROBE_OFFSET),
            P_BRACKET,
            D_BRACKET,
            Secant(
                RootFinderSettings(
                    value_tolerance=1.0e-6,
                    fraction_tolerance=1.0e-4,
                    max_iterations=8,
                ),
            ),
            grid=_grid(LOCATING_CELLS),
        ),
        _grid(DENSITY_CELLS),
    )
    return stochastic_stage(scene, continued, noise, diagnostics, describe)


STAGE_FILENAME = "noisy_hopf_stage.html"
STAGE_ENTRY = AtlasEntry(
    STAGE_FILENAME,
    "Noisy Hopf: two bifurcations",
    "The dynamical threshold at mu = 0 and the phenomenological one at "
    "mu = sigma^2, passing at different places on the same axis: sample paths, "
    "the stationary density, and the top Lyapunov exponent.",
    kind="stage",
)


def main(output_dir: str = "plots") -> None:
    """Write the stage page (and a one-card atlas) to ``output_dir``."""
    scene = noisy_hopf_stage()
    located = {
        point.kind: point.parameter
        for point in scene.timeline.special
        if point.kind in ("p_bifurcation", "d_bifurcation")
    }
    print(
        f"located P = {located['p_bifurcation']:.4f} (closed form {P_EXACT:.4f}), "
        f"D = {located['d_bifurcation']:.4f} (closed form {D_EXACT:.4f})",
    )
    path = publish(
        scene,
        STAGE_FILENAME,
        STAGE_ENTRY,
        output_dir,
        renderer=StochasticStageRenderer(),
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plots")
