"""The noiseless curve of the stochastic bifurcation page, in closed form.

Without noise ``dx = alpha x dt`` and the trivial solution grows as ``exp(alpha t)``,
so its exponent is ``alpha``. The page takes that from the closed form instead of
integrating 80,000 steps per point (DEQ-76); this checks the closed form against the
integration it replaced, at the page's own settings. The Euler-Maruyama estimate is
``log(1 + alpha dt) / dt`` exactly - the frame is multiplied by ``1 + alpha dt`` on
every step - so the two differ by that step's bias, ``alpha^2 dt / 2`` to leading
order, and by nothing else.
"""

import math
from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.lyapunov import (
    StochasticLyapunovSettings,
    StochasticSystem,
    stochastic_lyapunov,
)
from discrecontinual_equations.continuation.stochastic import Ito
from examples.continuation.pages_stochastic import bifurcation_exponent
from examples.continuation.systems import _stochastic_diffusion, _stochastic_drift

# The page's settings and parameter grid.
_PAGE = StochasticLyapunovSettings(dt=0.005, horizon=80000, transient=0, seed=7)
_ALPHAS = [-0.1 + 0.03 * step for step in range(13)]


def _integrated(alpha: float) -> float:
    system = StochasticSystem(
        _stochastic_drift(alpha),
        _stochastic_diffusion(0.0),
        Ito(),
    )
    return stochastic_lyapunov(system, np.array([0.0]), settings=_PAGE).top


class TestNoiselessExponent(TestCase):
    """The closed form at sigma = 0, against the integration it stands in for."""

    def test_the_closed_form_is_alpha(self):
        for alpha in _ALPHAS:
            assert abs(bifurcation_exponent(alpha, 0.0, _PAGE) - alpha) < 1.0e-9

    def test_the_integration_is_the_closed_form_less_the_euler_bias(self):
        dt = _PAGE.dt
        for alpha in (_ALPHAS[0], _ALPHAS[4], _ALPHAS[-1]):
            closed = bifurcation_exponent(alpha, 0.0, _PAGE)
            integrated = _integrated(alpha)
            # The integration is the per-step growth of the Euler map, exactly.
            assert abs(integrated - math.log1p(alpha * dt) / dt) < 1.0e-8
            # So it sits below the closed form by alpha^2 dt / 2, to O(alpha^3 dt^2).
            bias = alpha * alpha * dt / 2.0
            assert abs((closed - integrated) - bias) < 1.0e-6
            assert abs(closed - integrated) < 1.7e-4

    def test_with_noise_the_page_still_integrates(self):
        settings = StochasticLyapunovSettings(dt=0.005, horizon=2000, seed=7)
        system = StochasticSystem(
            _stochastic_drift(0.2),
            _stochastic_diffusion(0.6),
            Ito(),
        )
        expected = stochastic_lyapunov(system, np.array([0.0]), settings=settings).top
        assert bifurcation_exponent(0.2, 0.6, settings) == expected
