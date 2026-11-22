"""The Lorenz manifold as a film: the stable manifold of the origin, ring by ring.

``x' = sigma (y - x), y' = x (rho - z) - y, z' = x y - beta z`` at the classical
``sigma = 10, rho = 28, beta = 8/3``. The origin is a saddle with one unstable
direction and a two-dimensional stable manifold, the *Lorenz manifold*: the set
of points whose orbits end at the origin rather than on the attractor. It
separates the orbits that next turn around ``C+`` from those that next turn
around ``C-``, so it is what organises the butterfly, and it is a surface, so
no integrated curve can show it.

It is grown here as Krauskopf and Osinga grew it, as geodesic level sets
(:class:`~discrecontinual_equations.continuation.geodesic_level_sets.GeodesicLevelSets`):
a ring on the Taylor chart of the manifold at distance 1 from the origin, then
rings at equal geodesic distance, each point of each ring the end of an orbit
segment from the ring before. The film's scrub bar is that distance. Near the
origin the manifold is nearly the plane of the stable eigenvectors and contains
the whole z-axis; as it grows the two halves either side of the axis wind
around it, and further out roll up into the two scrolls that wrap the
equilibria ``C+-`` - which is why the rings need so many stitches added as
they go. The crochet pattern is the table beneath the film.
"""

import math
import sys

import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.geodesic_level_sets import (
    GeodesicLevelSets,
    GeodesicSettings,
)
from discrecontinual_equations.continuation.manifold import StableManifold
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.webplot.manifold_film import (
    FilmCurve,
    FilmMarker,
    ManifoldFilm,
)
from discrecontinual_equations.webplot.manifold_film_renderer import (
    ManifoldFilmRenderer,
)
from discrecontinual_equations.webplot.report import AtlasEntry

try:  # python -m examples.continuation.stage_lorenz_manifold
    from examples.continuation.stage_support import Mu, equation, publish
except ImportError:  # run as a script path: only this directory is on sys.path
    from stage_support import Mu, equation, publish

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
# How far the manifold is grown, and how finely. The seed ring sits at
# distance 1 on the order-5 Taylor chart, where the chart's invariance defect
# is 6e-6. The spacing and the step are what set the page's size: the rings'
# length grows quickly once the scrolls form, and every point is a triangle or
# two the browser sorts on each redraw.
DISTANCE = 100.0
SETTINGS = GeodesicSettings(
    initial_radius=1.0,
    step=1.5,
    spacing=(1.5, 4.5),
)
# How long each branch of the one-dimensional unstable manifold is followed:
# enough to show it leaving towards both wings, not so long that it paints
# the attractor over the surface.
UNSTABLE_TIME = 1.6


class Lorenz(DeterministicFunction):
    """The Lorenz field, with rho the one parameter."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        rho = self.parameters[0].value
        x, y, z = point[0], point[1], point[2]
        return [SIGMA * (y - x), x * (rho - z) - y, x * y - BETA * z]


def _unstable_branches(function, jacobian: np.ndarray) -> list[FilmCurve]:
    """The origin's unstable manifold, a curve, both ways off the origin."""
    values, vectors = np.linalg.eig(jacobian)
    direction = np.real(vectors[:, int(np.argmax(values.real))])

    def rhs(_t, state):
        return function.eval(point=list(state), time=None)

    curves = []
    for sign in (1.0, -1.0):
        path = solve_ivp(
            rhs,
            (0.0, UNSTABLE_TIME),
            sign * 1.0e-6 * direction,
            rtol=1.0e-10,
            atol=1.0e-12,
            dense_output=True,
        )
        # Most of the time is spent leaving the origin; sample in arclength.
        times = np.linspace(0.0, UNSTABLE_TIME, 2000)
        states = path.sol(times).T
        steps = np.linalg.norm(np.diff(states, axis=0), axis=1)
        length = np.concatenate([[0.0], np.cumsum(steps)])
        keep = np.searchsorted(length, np.linspace(0.0, length[-1], 260))
        curves.append(FilmCurve("unstable manifold", "unstable", states[keep]))
    return curves


def lorenz_manifold_film() -> ManifoldFilm:
    """Grow the Lorenz manifold and frame it as a film."""
    eq = equation(Lorenz, [Mu(value=RHO)], count=3)
    function = eq.derivative
    origin = np.zeros(3)
    jacobian = AutomaticDifferentiation().jacobian(function, origin, 0.0)
    mesh = GeodesicLevelSets(SETTINGS).compute(
        function,
        origin,
        jacobian,
        StableManifold(),
        distance=DISTANCE,
    )
    pattern = mesh.crochet_pattern()
    film = ManifoldFilm(
        "The Lorenz manifold",
        "The stable manifold of the origin of the Lorenz system, grown as "
        "geodesic level sets: the surface separating orbits that next turn "
        "around C+ from those that next turn around C-. Scrub the distance to "
        "watch it grow ring by ring, and drag to turn it.",
        mesh,
        equation=(
            r"\dot x = \sigma(y - x),\ \dot y = x(\rho - z) - y,\ "
            r"\dot z = xy - \beta z"
        ),
        note=(
            f"sigma = 10, rho = 28, beta = 8/3. GeodesicLevelSets grew the "
            f"surface from a ring on the order-5 TaylorManifold chart at "
            f"geodesic distance 1 out to {mesh.ring_distances[-1]:.0f}: "
            f"{len(mesh.rings)} level sets, {len(mesh.vertices):,} vertices "
            f"and {len(mesh.triangles):,} triangles, each vertex the end of an "
            f"orbit segment solved as a two-point boundary-value problem to "
            f"{mesh.residual:.0e}, with {sum(r.increases for r in pattern):,} "
            f"stitches added and {sum(r.decreases for r in pattern):,} taken "
            f"away. Colour is geodesic distance, alternate rings a shade "
            f"apart; the red curves are the origin's one-dimensional unstable "
            f"manifold, which leaves towards the two wings. Krauskopf and "
            f"Osinga computed this surface much further out and crocheted it "
            f"(Nonlinearity 17, 2004; Math. Intelligencer 26, 2004); this "
            f"film stops where an SVG page can still turn it smoothly."
        ),
    )
    film.curves.extend(_unstable_branches(function, jacobian))
    film.markers.append(FilmMarker("origin", "saddle", origin))
    offset = math.sqrt(BETA * (RHO - 1.0))
    for name, sign in (("C+", 1.0), ("C-", -1.0)):
        film.markers.append(
            FilmMarker(name, "focus", [sign * offset, sign * offset, RHO - 1.0]),
        )
    return film


FILENAME = "lorenz_manifold.html"
ENTRY = AtlasEntry(
    FILENAME,
    "The Lorenz manifold",
    "The two-dimensional stable manifold of the Lorenz origin, grown as "
    "geodesic level sets and scrubbed ring by ring, coloured by distance.",
    kind="stage",
)


def main(output_dir: str = "plots") -> None:
    """Write the film (and a one-card atlas) to ``output_dir``."""
    film = lorenz_manifold_film()
    path = publish(film, FILENAME, ENTRY, output_dir, ManifoldFilmRenderer())
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plots")
