"""Cubic vector fields with every coordinate reflection as a symmetry.

``x_i' = x_i (lam_i + sum_j A_ij x_j^2)`` commutes with each reflection
``x_k -> -x_k``, so every coordinate subspace is flow-invariant and every
equilibrium on an axis is a candidate vertex of a robust heteroclinic cycle: a
saddle-sink connection inside a coordinate plane persists under any perturbation
that keeps the reflections (Krupa and Melbourne 1995, Proposition 2.5).

Two named members of the family are the systems the heteroclinic-network oracles
are checked against:

* the Guckenheimer-Holmes cycle, in the normalisation of Krupa (1997, eq. 2.1):
  ``x_i' = x_i (lam + a1 x_i^2 + a2 x_{i+1}^2 + a3 x_{i+2}^2)`` with the indices
  cyclic, so ``A`` is circulant and the cyclic shift is a symmetry too;
* the Kirk-Silber network in four dimensions, which is this family with
  ``n = 4`` and the interaction chosen to place each saddle eigenvalue.

In ``u_i = x_i^2`` the family is Lotka-Volterra, ``u_i' = 2 u_i (lam_i + (A u)_i)``
(the form of Krupa 1997, eq. 7.1), so the Guckenheimer-Holmes cycle is the
May-Leonard cycle with ``alpha = a2/a1`` and ``beta = a3/a1``. The substitution
doubles every exponent and leaves every ratio of exponents - the only thing the
stability and residence-time results depend on - unchanged.

Parameters, in order: the ``n`` growth rates ``lam_i``, then ``A`` row-major. A
growth rate is the natural symmetry-breaking parameter: changing one of them keeps
every reflection (every plane stays invariant) but breaks the cyclic shift.
"""

import numpy as np

from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class Amplitude(Variable, name="amplitude", abbreviation="x"):
    """One mode amplitude of a reflection-symmetric cubic system."""


class GrowthRate(Parameter, name="growth", abbreviation="lam"):
    """The linear growth rate of one mode."""


class Interaction(Parameter, name="interaction", abbreviation="a"):
    """One cubic coupling coefficient ``A_ij``."""


class ReflectionCubic(DeterministicFunction):
    """``x_i' = x_i (lam_i + sum_j A_ij x_j^2)``; parameters ``lam`` then ``A``."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        """The derivative; written with ``*`` and ``+`` only so jets pass through."""
        size = len(point)
        values = [parameter.value for parameter in self.parameters]
        squares = [point[j] * point[j] for j in range(size)]
        derivative = []
        for i in range(size):
            rate = values[i]
            for j in range(size):
                rate = rate + values[size + i * size + j] * squares[j]
            derivative.append(point[i] * rate)
        return derivative


def reflection_cubic(growth: np.ndarray, interaction: np.ndarray) -> ReflectionCubic:
    """Build the field from growth rates ``lam`` and the interaction matrix ``A``."""
    size = len(growth)
    if interaction.shape != (size, size):
        message = (
            f"interaction must be {size}x{size} to match {size} growth rates, "
            f"got shape {interaction.shape}"
        )
        raise ValueError(message)
    parameters: list[Parameter] = [GrowthRate(value=float(rate)) for rate in growth]
    parameters.extend(
        Interaction(value=float(value)) for value in np.asarray(interaction).ravel()
    )
    return ReflectionCubic(
        variables=[Amplitude() for _ in range(size)],
        parameters=parameters,
        results=[Amplitude() for _ in range(size)],
        time=None,
    )


def guckenheimer_holmes(
    growth: float,
    self_coupling: float,
    forward: float,
    backward: float,
) -> ReflectionCubic:
    """The Guckenheimer-Holmes system in Krupa's (1997) eq. 2.1 normalisation.

    ``x_i' = x_i (lam + a1 x_i^2 + a2 x_{i+1}^2 + a3 x_{i+2}^2)`` with
    ``(growth, self_coupling, forward, backward) = (lam, a1, a2, a3)``. The cycle
    ``xi1 -> xi2 -> xi3 -> xi1`` along the positive axes exists iff ``a1 < 0`` and
    ``a2 < a1 < a3`` (Krupa 1997, Proposition 1), and is asymptotically stable iff
    ``2 a1 > a2 + a3`` (Theorem 1).

    Each growth rate is its own parameter (indices 0-2), so the cyclic symmetry can
    be broken by changing one of them.
    """
    row = np.array([self_coupling, forward, backward])
    interaction = np.array([np.roll(row, shift) for shift in range(3)])
    return reflection_cubic(np.full(3, growth), interaction)
