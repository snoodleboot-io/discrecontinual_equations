"""White-noise equations to colour, with the closed forms their tests rest on.

* The **multiplicative pitchfork** ``dx = (a x - x^3) dt + s x dW``. In the white
  Stratonovich reading its density is ``|x|^(2a/s^2 - 1) exp(-x^2 / s^2)``, so
  ``x^2 / s^2`` is Gamma-distributed with shape ``a / s^2``; the mode leaves the
  origin at ``a = s^2 / 2`` and the trivial solution's exponent is ``a``. Coloured
  (Hänggi and Jung's dye-laser intensity model, Sec. VII, with ``I = x^2``), two
  exact statements survive at every ``tau``: ``<x^2> = a`` (from the stationarity of
  ``log x^2``, whose drift is ``2 (a - x^2) + 2 s eta``), and the exponent of the
  trivial solution is ``a`` (its linearisation is ``(a + s eta) dx``, and ``eta``
  averages to zero).
* The **linear OU-driven system** ``dx = -k x dt + c dW``, coloured: Gaussian, with
  stationary covariance from the Lyapunov equation of the augmented pair, and
  ``var x = c^2 / (2 k (1 + k tau))`` (Hänggi and Jung eqs. 3.28-3.33).
"""

import numpy as np
from scipy import stats

from discrecontinual_equations.function.function import Function
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class Rate(Parameter, name="rate", abbreviation="a"):
    """The pitchfork parameter, or the OU decay rate."""


class Strength(Parameter, name="strength", abbreviation="s"):
    """The noise amplitude."""


class Pitchfork(StochasticFunction):
    """``dx = (a x - x^3) dt + s x dW``, with batch forms for the ensemble."""

    def eval(self, point, time=None):  # noqa: ARG002
        a = self.parameters[0].value
        return [a * point[0] - point[0] ** 3]

    def diffusion(self, point, time=None):  # noqa: ARG002
        return [self.parameters[1].value * point[0]]

    def eval_batch(self, points, time=None):  # noqa: ARG002
        a = self.parameters[0].value
        return a * points - points**3

    def diffusion_batch(self, points, time=None):  # noqa: ARG002
        return self.parameters[1].value * points


class LinearRelaxation(StochasticFunction):
    """``dx = -k x dt + c dW``."""

    def eval(self, point, time=None):  # noqa: ARG002
        return [-self.parameters[0].value * point[0]]

    def diffusion(self, point, time=None):  # noqa: ARG002
        return [self.parameters[1].value]

    def eval_batch(self, points, time=None):  # noqa: ARG002
        return -self.parameters[0].value * points

    def diffusion_batch(self, points, time=None):  # noqa: ARG002
        return np.full_like(points, self.parameters[1].value)


class PitchforkDrift(Function):
    """The white pitchfork's drift alone, for the closed-form 1-D tools."""

    def eval(self, point, time=None):  # noqa: ARG002
        a = self.parameters[0].value
        return [a * point[0] - point[0] ** 3]


class PitchforkNoise(Function):
    """The white pitchfork's amplitude ``s x`` alone."""

    def eval(self, point, time=None):  # noqa: ARG002
        return [self.parameters[1].value * point[0]]


class StateNoise(Function):
    """A constant column ``[e, 0]``: a little white noise on ``x`` of the pair."""

    def eval(self, point, time=None):  # noqa: ARG002
        return [self.parameters[0].value, 0.0]


def scalar(function_type, rate: float, strength: float):
    """A one-variable function with the two parameters above."""
    return function_type(
        variables=[Variable(name="x", discretization=[])],
        parameters=[Rate(value=rate), Strength(value=strength)],
        results=[Variable(name="dx/dt", discretization=[])],
        time=None,
    )


def state_noise(variables, size: float) -> StateNoise:
    """The regularising column on the augmented pair's ``x``."""
    return StateNoise(variables, [Rate(value=size)], variables, None)


def white_pitchfork_samples(
    rate: float,
    strength: float,
    uniforms: np.ndarray,
    signs: np.ndarray,
) -> np.ndarray:
    """Exact samples of the white Stratonovich pitchfork density, by inversion.

    ``x^2 / s^2 ~ Gamma(a / s^2)``. Fixed ``uniforms`` make the samples a continuous
    function of ``rate``: common random numbers for a threshold search.
    """
    return signs * strength * np.sqrt(stats.gamma.ppf(uniforms, rate / strength**2))
