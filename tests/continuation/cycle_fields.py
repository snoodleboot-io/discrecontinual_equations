"""Systems with cycles whose manifolds and connections are known in closed form.

``SaddleCycle`` is the ticket's linear-cycle oracle made nonlinear in both
directions: ``r' = r (1 - r^2)``, ``phi' = omega``, ``z' = mu z + z^3``. The cycle
``r = 1, z = 0`` has period ``2 pi / omega`` and Floquet exponents ``-2``
(radial), ``0`` and ``mu`` (vertical). Each manifold solves its own scalar
equation, so the parameterizations are explicit:

    stable:   P = (1 - 2 sigma)^(-1/2) (cos w theta, sin w theta, 0)
    unstable: P = (cos w theta, sin w theta, sigma (1 - sigma^2 / mu)^(-1/2))

whose Taylor series in ``sigma`` never terminate. ``Conjugated`` pulls any field
back through the rational diffeomorphism ``h(x, y, z) = h2(h1(x, y, z))`` with
``h1 = (x, y, z + a x^2 / (1 + c y^2))`` and ``h2 = (x + b z^2, y, z)``: the
conjugated field's manifolds are ``h`` of the original's, with the same
exponents, so the oracle gains infinitely many Fourier modes in every component
without losing its closed form.

``TwistedCycle`` has a Mobius stable bundle: the normal plane ``(r - 1, z)``
turns half a revolution per period, and in the turning frame the dynamics are
``rho' = a rho``, ``zeta' = b zeta``. Its multipliers are ``-e^(a T)`` and
``-e^(b T)``, and the manifold of ``a`` is the Mobius band
``r = 1 + sigma cos(w theta / 2)``, ``z = sigma sin(w theta / 2)``.

``CylinderLadder`` has two saddle cycles on the invariant cylinder ``r = 1``,
at ``z = 0`` and ``z = 1``, with ``z' = z (1 - z)`` between them, so every orbit
on the cylinder between them is a cycle-to-cycle connection, known by the
rotational symmetry: ``z(t) = 1 / (1 + (1 / z0 - 1) e^-t)``.
"""

import math

import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import Alpha, State, Time

OMEGA = 1.3
MU = 0.7
SHEAR_A = 0.3
SHEAR_B = 0.2
SHEAR_C = 0.3


class SaddleCycle(DeterministicFunction):
    """r' = r (1 - r^2), phi' = omega, z' = mu z + z^3, in Cartesian form."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        x, y, z = point[0], point[1], point[2]
        growth = 1.0 - (x * x + y * y)
        return [growth * x - OMEGA * y, growth * y + OMEGA * x, MU * z + z * z * z]


class Conjugated(DeterministicFunction):
    """``h_* f``: the field ``f`` seen through the diffeomorphism ``h``."""

    def __init__(self, base: DeterministicFunction) -> None:
        super().__init__(variables=[], parameters=[], results=[])
        self._base = base

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        u, v, w = point[0], point[1], point[2]
        # Undo h2, then h1.
        x = u - SHEAR_B * w * w
        y = v
        damping = 1.0 / (1.0 + SHEAR_C * y * y)
        z = w - SHEAR_A * x * x * damping
        fx, fy, fz = self._base.eval([x, y, z], None)
        # Push forward through Dh1 at (x, y, z), then Dh2 at h1(x, y, z) = (x, y, w).
        gz = (
            fz
            + 2.0 * SHEAR_A * x * damping * fx
            - 2.0 * SHEAR_A * SHEAR_C * x * x * y * damping * damping * fy
        )
        return [fx + 2.0 * SHEAR_B * w * gz, fy, gz]


def conjugate(state: np.ndarray) -> np.ndarray:
    """``h(x, y, z)``."""
    x, y, z = state
    w = z + SHEAR_A * x * x / (1.0 + SHEAR_C * y * y)
    return np.array([x + SHEAR_B * w * w, y, w])


def saddle_stable(theta: float, sigma: float) -> np.ndarray:
    """The exact stable chart of ``SaddleCycle`` (before conjugation)."""
    radius = (1.0 - 2.0 * sigma) ** -0.5
    return np.array(
        [radius * math.cos(OMEGA * theta), radius * math.sin(OMEGA * theta), 0.0]
    )


def saddle_unstable(theta: float, sigma: float) -> np.ndarray:
    """The exact unstable chart of ``SaddleCycle`` (before conjugation)."""
    height = sigma * (1.0 - sigma * sigma / MU) ** -0.5
    return np.array([math.cos(OMEGA * theta), math.sin(OMEGA * theta), height])


TWIST_A = -1.0
TWIST_B = 0.5


class TwistedCycle(DeterministicFunction):
    """A cycle whose normal plane turns half a revolution per period."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        x, y, z = point[0], point[1], point[2]
        radius = (x * x + y * y) ** 0.5
        cosine = x / radius
        sine = y / radius
        s1 = radius - 1.0
        # e1 e1^T in (r, z) coordinates, e1 = (cos phi/2, sin phi/2).
        p11 = (1.0 + cosine) / 2.0
        p12 = sine / 2.0
        p22 = (1.0 - cosine) / 2.0
        a, b, half = TWIST_A, TWIST_B, OMEGA / 2.0
        rdot = (a * p11 + b * p22) * s1 + (a - b) * p12 * z - half * z
        zdot = (a - b) * p12 * s1 + (a * p22 + b * p11) * z + half * s1
        return [rdot * cosine - OMEGA * y, rdot * sine + OMEGA * x, zdot]


def twisted_stable(theta: float, sigma: float) -> np.ndarray:
    """The exact Mobius stable chart of ``TwistedCycle``."""
    radius = 1.0 + sigma * math.cos(OMEGA * theta / 2.0)
    return np.array(
        [
            radius * math.cos(OMEGA * theta),
            radius * math.sin(OMEGA * theta),
            sigma * math.sin(OMEGA * theta / 2.0),
        ],
    )


class CylinderLadder(DeterministicFunction):
    """Saddle cycles at z = 0 and z = 1 on the invariant cylinder r = 1.

    ``r' = r (1 - r^2) (1 - 2 z)``, ``phi' = omega``, ``z' = z (1 - z)``, plus an
    optional ``epsilon`` that tips the radial flow by ``z (1 - z) x``: it
    vanishes on both cycles, keeps them, and breaks the symmetry that makes the
    unstable manifold of the lower cycle coincide with the stable one of the
    upper.
    """

    def __init__(self, epsilon: float = 0.0) -> None:
        super().__init__(variables=[], parameters=[], results=[])
        self._epsilon = epsilon

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        x, y, z = point[0], point[1], point[2]
        squared = x * x + y * y
        radial = (1.0 - squared) * (1.0 - 2.0 * z) + self._epsilon * z * (1.0 - z) * x
        return [radial * x - OMEGA * y, radial * y + OMEGA * x, z * (1.0 - z)]


class FoldCycles(DeterministicFunction):
    """r' = (mu + r^2 - r^4) r, phi' = 1: the fold-of-cycles film's field."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        squared = x * x + y * y
        growth = mu + squared - squared * squared
        return [growth * x - y, x + growth * y]


def fold_equation(mu: float) -> DifferentialEquation:
    """The fold-of-cycles field as a planar equation in ``alpha = mu``."""
    parameters = [Alpha(value=mu)]
    return DifferentialEquation(
        variables=[State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=FoldCycles(
            variables=[State(), State()],
            parameters=parameters,
            results=[State(), State()],
            time=None,
        ),
    )


def circle(radius: float, nodes: int) -> np.ndarray:
    """``nodes + 1`` states round a circle, both ends at angle zero."""
    angle = 2.0 * math.pi * np.linspace(0.0, 1.0, nodes + 1)
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
