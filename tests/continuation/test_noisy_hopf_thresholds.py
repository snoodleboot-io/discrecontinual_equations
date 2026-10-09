"""The noisy Hopf film's thresholds, located as the film locates them.

The conformal noise ``s z dB`` on the Hopf drift gives ``D = s^2 r^2 I``, so the
planar density is ``C r^(2 mu / s^2 - 2) exp(-r^2 / s^2)`` and craters at
``mu = s^2`` exactly, while the origin's top exponent is ``mu`` itself. This is
the fine-grid check the fast suite cannot afford: the located P must agree with
the closed form to the few percent the film claims, and the crest the frames
draw must sit at ``sqrt(mu - s^2)``.
"""

import math
from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    StationaryFokkerPlanck,
)
from discrecontinual_equations.continuation.noise import NoiseMatrix
from discrecontinual_equations.continuation.root_finder import (
    RootFinderSettings,
    Secant,
)
from discrecontinual_equations.continuation.stochastic import Ito
from discrecontinual_equations.continuation.stochastic_threshold import (
    PhenomenologicalThreshold,
    RadialCrater,
)
from tests.continuation.fields import Alpha, ConformalColumn, HopfField, Sigma, State

_SIGMA = 0.7


def _problem(mu, grid):
    parameters = [Alpha(value=mu), Sigma(value=_SIGMA)]
    drift = HopfField(
        variables=[State(), State()],
        parameters=parameters,
        results=[State(), State()],
        time=None,
    )
    noise = NoiseMatrix(
        [
            ConformalColumn(index, [State(), State()], parameters, [State(), State()])
            for index in range(2)
        ],
    )
    return StationaryFokkerPlanck(drift, noise, Ito(), grid)


class TestNoisyHopfThresholds(TestCase):
    def test_the_located_p_agrees_with_the_variance_on_the_film_grid(self):
        # 200 cells over [-2.5, 2.5]^2, probe four cells out: the finite-volume
        # overshoot at the singular origin has decayed below a percent there
        # and the probe's O(r^2) bias is 0.01, so the located value sits a few
        # hundredths above s^2 and no more.
        grid = DensityGrid.box([-2.5, -2.5], [2.5, 2.5], [200, 200])
        located = PhenomenologicalThreshold(
            lambda mu: _problem(mu, grid),
            RadialCrater([0.0, 0.0], axis=0, offset=4),
            Secant(
                RootFinderSettings(
                    value_tolerance=1.0e-6,
                    fraction_tolerance=1.0e-4,
                    max_iterations=8,
                ),
            ),
        ).locate(0.15, 1.0)
        assert located is not None
        assert abs(located - _SIGMA**2) < 0.025
        assert located > _SIGMA**2

    def test_the_crest_sits_at_the_closed_form_radius(self):
        grid = DensityGrid.box([-2.5, -2.5], [2.5, 2.5], [100, 100])
        for mu in (0.8, 1.2, 1.5):
            density = _problem(mu, grid).solve()
            radii = [float(np.linalg.norm(m)) for m in density.interior_maxima()]
            ring = [r for r in radii if r > 1.5 * grid.spacings[0]]
            assert len(ring) > 8
            assert (
                abs(float(np.mean(ring)) - math.sqrt(mu - _SIGMA**2)) < grid.spacings[0]
            )
