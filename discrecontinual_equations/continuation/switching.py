"""Seeding of secondary branches at detected branch points.

At a simple branch point ``y*`` two solution curves of ``F(y) = 0`` cross, and the
``n x (n + 1)`` Jacobian ``J = F_y(y*)`` has a two-dimensional kernel ``K`` holding
both tangents (Kuznetsov (1998), Section 10.2.4, eq. (10.57), Definition 10.3,
Lemma 10.1). The second tangent is fixed by the *algebraic branching equation*
``<phi, B(v, v)> = 0`` (Kuznetsov's (10.60)), which needs second derivatives.

:class:`NullSpaceBranchSwitcher` uses first derivatives only. It takes the right
singular vector of the smallest singular value of ``J`` - a vector of ``K`` - and
removes its component along the primary tangent ``q1``, leaving the unit vector
``q2`` of ``K`` orthogonal to ``q1``; the seed is ``y* + kick * q2``. This is the
"switching by perturbation" idea of Allgower and Georg (1990, Section 8.2): a seed
off the primary branch, close to the secondary one, from which the corrector and
continuation then find it. ``q2`` is Kuznetsov's basis vector, not the secondary
tangent itself, which is ``beta_1 q1 + beta_2 q2`` with ``(beta_1, beta_2)`` from
the branching equation; for a transcritical crossing the seed may therefore land
off both branches and rely on the seed refinement.

Where ``q2`` vanishes numerically (below ``singular_tolerance``) - the smallest
singular direction was ``q1`` itself, which happens when the localized point left a
one-dimensional kernel or the SVD picked ``q1`` out of ``K`` - the switcher falls
back to kicking state component 0 and the parameter by ``kick``. That suits the
pitchfork of a ``Z_2``-symmetric system whose symmetric branch is ``u = 0``; for any
other geometry it is a guess, and the user should check which branch was found.

The seed is then polished by Newton at its fixed parameter before continuation
starts, which can undo the switch. For ``x' = mu x - x^3`` the extended Jacobian at
the origin is zero, the SVD returns the primary tangent, and the fallback seed
``(kick, kick)`` is used: with ``kick = 0.05`` the polish returns to ``x = 0`` and
the "secondary" branch is the trivial one again; with ``kick = 0.2`` it lands on
``x = sqrt(mu)`` (measured). The kick is the configuration's ``initial_step``.

Limits: the SVD is of the extended Jacobian at the recorded point, which is only
as close to the branch point as the localizer put it; at an imperfectly located
branch point the kernel is one-dimensional and the smallest singular direction is
only approximately in ``K``.

References:
- Kuznetsov, Yu. A. "Elements of Applied Bifurcation Theory", 2nd ed., Applied
  Mathematical Sciences 112, Springer, 1998, doi:10.1007/b98848, Section 10.2.4,
  eqs. (10.57)-(10.60), Definition 10.3 and Lemma 10.1.
- Allgower, E. L. and Georg, K. "Numerical Continuation Methods: An Introduction",
  Springer, 1990, doi:10.1007/978-3-642-61257-2, Sections 8.2-8.3.
"""

from abc import ABC, abstractmethod

import numpy as np

from discrecontinual_equations.continuation.continuation_point import ContinuationPoint
from discrecontinual_equations.continuation.jacobian import JacobianProvider

_TRANSVERSE_COMPONENT = 0


class BranchSwitcher(ABC):
    """Produce a seed on the branch that crosses at a branch point."""

    @abstractmethod
    def secondary_seed(
        self,
        branch_point: ContinuationPoint,
    ) -> tuple[np.ndarray, float]:
        """Return a ``(state, parameter)`` seed on the bifurcating branch.

        Args:
            branch_point: The located branch point, with the primary tangent.

        Returns:
            A seed state and parameter near the secondary branch.
        """
        raise NotImplementedError


class NullSpaceBranchSwitcher(BranchSwitcher):
    """Switch along the second null direction of the extended Jacobian.

    Where the smallest singular direction is (numerically) the primary tangent
    itself, the transverse component vanishes and the switcher falls back to a
    fixed kick of state component 0 and the parameter. (The original note said
    this happens because a pitchfork's branches share a tangent; they do not -
    the trivial and the symmetric-broken branch cross orthogonally - it happens
    when the SVD returns ``q1`` from the kernel, or when an imprecisely located
    point leaves a one-dimensional kernel.)

    Args:
        jacobian: Source of the extended Jacobian at the branch point.
        kick: Distance of the seed from the branch point.
        singular_tolerance: Below this norm the orthogonal null direction is taken
            as absent, and the fallback kick is used.

    References:
    - Kuznetsov (1998), Section 10.2.4.
    - Allgower and Georg (1990), Section 8.2.
    """

    __slots__ = ["_jacobian", "_kick", "_singular_tolerance"]

    def __init__(
        self,
        jacobian: JacobianProvider,
        kick: float,
        singular_tolerance: float,
    ) -> None:
        self._jacobian = jacobian
        self._kick = kick
        self._singular_tolerance = singular_tolerance

    def secondary_seed(
        self,
        branch_point: ContinuationPoint,
    ) -> tuple[np.ndarray, float]:
        state = np.asarray(branch_point.state, dtype=float)
        parameter = float(branch_point.parameter)
        extended = self._jacobian.extended_jacobian(state, parameter)
        _left, _singular, right_vectors = np.linalg.svd(extended)
        primary = np.asarray(branch_point.tangent, dtype=float)
        primary = primary / np.linalg.norm(primary)
        candidate = right_vectors[-1]
        secondary = candidate - (candidate @ primary) * primary
        if np.linalg.norm(secondary) < self._singular_tolerance:
            return self._transverse_kick(state, parameter)
        secondary = secondary / np.linalg.norm(secondary)
        seed = np.concatenate([state, [parameter]]) + self._kick * secondary
        return seed[:-1], float(seed[-1])

    def _transverse_kick(
        self,
        state: np.ndarray,
        parameter: float,
    ) -> tuple[np.ndarray, float]:
        kicked = state.copy()
        kicked[_TRANSVERSE_COMPONENT] += self._kick
        return kicked, parameter + self._kick
