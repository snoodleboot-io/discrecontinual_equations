"""Switching onto the secondary branch at a detected simple branch point.

At a simple branch point ``y*`` of ``F(y) = 0``, ``F : R^(n+1) -> R^n``, two
solution curves cross transversally. The ``n x (n + 1)`` Jacobian ``J = F_y(y*)``
then has a two-dimensional kernel ``K`` that holds both tangents, and a
one-dimensional left kernel spanned by ``phi``. Differentiating ``F(y(s)) = 0``
twice along either branch and projecting onto ``phi`` gives the *algebraic
branching equation*: every branch tangent ``v`` in ``K`` satisfies
``<phi, B(v, v)> = 0``, with ``B`` the second derivative of ``F``. Writing
``v = xi1 q1 + xi2 q2`` for a basis ``q1, q2`` of ``K`` makes this a quadratic
in ``(xi1, xi2)``; at a simple branch point it has two distinct real root
directions, one the primary tangent and the other the secondary tangent.

:class:`NullSpaceBranchSwitcher` follows the four steps of Allgower and Georg
(1990), section 8.3: it takes the primary tangent recorded at the located point as
``q1``, finds ``q2`` and ``phi`` as minimal singular vectors, approximates the
2 x 2 coefficient matrix of the branching equation by second differences of
``phi . F``, and solves the quadratic for the secondary direction ``v2``. The seed
``y* + h v2`` is then corrected by Newton's method on ``F`` bordered by the
hyperplane through the seed normal to ``v2`` - not at fixed parameter, which at a
pitchfork falls back onto the trivial branch.

The corrected point is accepted only if it is demonstrably on another branch: it
must lie off the primary tangent line by a fixed fraction of the kick, and its
own tangent must be closer to ``v2`` than to the primary tangent. Otherwise the
switch is retried with larger kicks on both sides, and if none is accepted it
raises :class:`BranchSwitchError`. A switch never returns its starting branch.

References:
- Allgower, E. L. and Georg, K. "Introduction to Numerical Continuation
  Methods", Colorado State University lecture notes, 1990 (published by Springer
  as "Numerical Continuation Methods: An Introduction", 1990,
  doi:10.1007/978-3-642-61257-2), Definition (8.1.11), Lemma (8.1.13), and
  section 8.3, steps 2-4, eqs. (8.3.3)-(8.3.11).
- Kuznetsov, Yu. A. "Elements of Applied Bifurcation Theory", 2nd ed., Applied
  Mathematical Sciences 112, Springer, 1998, doi:10.1007/b98848, section 10.2.4
  "Branching points", eqs. (10.56)-(10.63), Definition 10.3 and Lemma 10.1.
"""

from abc import ABC, abstractmethod

import numpy as np

from discrecontinual_equations.continuation.continuation_point import ContinuationPoint
from discrecontinual_equations.continuation.corrector import NewtonSettings
from discrecontinual_equations.continuation.jacobian import JacobianProvider
from discrecontinual_equations.continuation.linear_solver import LinearSolver
from discrecontinual_equations.continuation.residual import ResidualFunction

# Second differences balance truncation O(e^2) against cancellation O(eps / e^2),
# so their step is the fourth root of machine precision.
_DIFFERENCE_STEP = float(np.finfo(float).eps) ** 0.25
# Kicks tried, as multiples of the configured kick, each on both sides of y*.
_KICK_MULTIPLES = (1.0, 2.0, 4.0, 8.0)
# A switched point must sit at least this fraction of the kick's transverse part
# away from the primary tangent line.
_TRANSVERSE_FRACTION = 0.5


class BranchSwitchError(ArithmeticError):
    """No point on a branch other than the primary one was found."""


class BranchSwitcher(ABC):
    """Produce a point on the branch that crosses at a branch point."""

    @abstractmethod
    def secondary_seed(
        self,
        branch_point: ContinuationPoint,
    ) -> tuple[np.ndarray, float]:
        """Return a ``(state, parameter)`` point on the bifurcating branch.

        Args:
            branch_point: The located branch point, with the primary tangent.

        Returns:
            A converged state and parameter on the secondary branch, from which
            continuation can start.

        Raises:
            BranchSwitchError: If no point off the primary branch was found.
        """
        raise NotImplementedError


class NullSpaceBranchSwitcher(BranchSwitcher):
    """Switch along the second root of the algebraic branching equation.

    The method, at the located point ``y*`` with primary unit tangent ``q1``:

    1. ``phi`` minimises ``|J^T phi|`` and ``q2`` minimises
       ``|J q2|^2 + (q1 . q2)^2`` over unit vectors (Allgower-Georg (8.3.3) and
       (8.3.5)); both are the last singular vectors of an SVD, which tolerates a
       point that is only approximately the branch point.
    2. ``alpha[i, j] = phi . F''(y*)[q_i, q_j]`` from second differences of
       ``g(xi1, xi2) = phi . F(y* + xi1 q1 + xi2 q2)`` (Allgower-Georg (8.3.8)-
       (8.3.9)). The step is ``eps^(1/4)``, not the ``eps^(1/3)`` the notes
       suggest: their justification, (10.3.24)-(10.3.27), is for a first
       difference.
    3. The two root directions of ``alpha11 xi1^2 + 2 alpha12 xi1 xi2 +
       alpha22 xi2^2 = 0`` (Allgower-Georg (8.3.11), Kuznetsov (10.60)) are
       computed without cancellation; the one farther from ``q1`` is the
       secondary tangent ``v2``. With ``alpha11 = 0`` this is Kuznetsov's
       ``xi1 = -alpha22 / (2 alpha12) xi2``. If the quadratic has no two
       distinct real roots (a degenerate, non-simple point), ``v2 = q2``.
    4. ``y* + h v2`` is corrected on ``F = 0`` bordered by ``v2``, and the result
       is checked to be off the primary branch, in point and in tangent.

    Limits: only simple branch points (``dim K = 2``) are handled. The primary
    tangent is the one stored on the located point; if the localizer left the
    point far from the true crossing (compared with the kick), the secondary
    direction is only approximate and the larger retry kicks are what find the
    branch.

    Args:
        residual: The residual ``F``.
        jacobian: Source of the extended Jacobian ``[F_x | F_lambda]``.
        linear_solver: Solver for the bordered Newton systems.
        kick: Distance from the branch point at which the secondary branch is
            sought (the configuration's ``initial_step``).
        singular_tolerance: Size below which the branching equation is taken as
            degenerate.
        settings: Newton tolerance and iteration cap for the correction.
        direction: The configured direction of travel. Where the secondary
            branch crosses the parameter transversally, the seed is placed on
            the side that continuation in this direction leaves the branch
            point by; the other side is tried next.

    References:
    - Allgower and Georg (1990), Definition (8.1.11), Lemma (8.1.13),
      eqs. (8.3.3)-(8.3.11).
    - Kuznetsov (1998), section 10.2.4, eqs. (10.57)-(10.63), Lemma 10.1.
    """

    __slots__ = [
        "_direction",
        "_jacobian",
        "_kick",
        "_linear_solver",
        "_residual",
        "_settings",
        "_singular_tolerance",
    ]

    def __init__(  # noqa: PLR0913 (injected collaborators and settings)
        self,
        residual: ResidualFunction,
        jacobian: JacobianProvider,
        linear_solver: LinearSolver,
        kick: float,
        singular_tolerance: float,
        settings: NewtonSettings,
        direction: int,
    ) -> None:
        self._residual = residual
        self._jacobian = jacobian
        self._linear_solver = linear_solver
        self._kick = kick
        self._singular_tolerance = singular_tolerance
        self._settings = settings
        self._direction = direction

    def secondary_seed(
        self,
        branch_point: ContinuationPoint,
    ) -> tuple[np.ndarray, float]:
        point = np.concatenate(
            [np.asarray(branch_point.state, dtype=float), [branch_point.parameter]],
        )
        primary = np.asarray(branch_point.tangent, dtype=float)
        primary = primary / np.linalg.norm(primary)
        direction = self.secondary_direction(point, primary)
        for multiple in _KICK_MULTIPLES:
            for side in self._sides(direction):
                kick = side * multiple * self._kick
                switched = self._switched_point(point, primary, direction, kick)
                if switched is not None:
                    return switched[:-1], float(switched[-1])
        message = (
            f"no branch other than the primary one was found near "
            f"{point.tolist()} with kicks up to {_KICK_MULTIPLES[-1] * self._kick}"
        )
        raise BranchSwitchError(message)

    def secondary_direction(
        self,
        point: np.ndarray,
        primary: np.ndarray,
    ) -> np.ndarray:
        """Unit tangent of the secondary branch at ``point`` (steps 1-3).

        Args:
            point: The branch point ``(state, parameter)``.
            primary: Unit tangent of the primary branch there.

        Returns:
            The unit secondary tangent ``v2`` in ``(state, parameter)`` space.
        """
        extended = self._extended(point)
        left_vectors, _singular, _right = np.linalg.svd(extended)
        phi = left_vectors[:, -1]
        _left, _singular, right_vectors = np.linalg.svd(
            np.vstack([extended, primary]),
        )
        complement = right_vectors[-1]
        alpha = self._branching_coefficients(point, phi, primary, complement)
        xi1, xi2 = self._secondary_root(alpha)
        direction = xi1 * primary + xi2 * complement
        return direction / np.linalg.norm(direction)

    def _sides(self, direction: np.ndarray) -> tuple[float, float]:
        # Seed first on the half-branch that continuation in the configured
        # direction leaves the branch point by, rather than runs back through.
        if direction[-1] * self._direction < 0.0:
            return -1.0, 1.0
        return 1.0, -1.0

    def _branching_coefficients(
        self,
        point: np.ndarray,
        phi: np.ndarray,
        first: np.ndarray,
        second: np.ndarray,
    ) -> np.ndarray:
        step = _DIFFERENCE_STEP * max(1.0, float(np.linalg.norm(point)))

        def g(xi1: float, xi2: float) -> float:
            shifted = point + xi1 * step * first + xi2 * step * second
            return float(phi @ self._residual.evaluate(shifted[:-1], shifted[-1]))

        centre = g(0.0, 0.0)
        alpha11 = (g(1.0, 0.0) - 2.0 * centre + g(-1.0, 0.0)) / step**2
        alpha22 = (g(0.0, 1.0) - 2.0 * centre + g(0.0, -1.0)) / step**2
        alpha12 = (g(1.0, 1.0) + g(-1.0, -1.0) - g(1.0, -1.0) - g(-1.0, 1.0)) / (
            4.0 * step**2
        )
        return np.array([[alpha11, alpha12], [alpha12, alpha22]])

    def _secondary_root(self, alpha: np.ndarray) -> tuple[float, float]:
        alpha11, alpha12, alpha22 = alpha[0, 0], alpha[0, 1], alpha[1, 1]
        scale = float(np.max(np.abs(alpha)))
        discriminant = alpha12 * alpha12 - alpha11 * alpha22
        if (
            scale < self._singular_tolerance
            or discriminant <= (self._singular_tolerance * scale) ** 2
        ):
            return 0.0, 1.0
        sign = 1.0 if alpha12 >= 0.0 else -1.0
        pivot = -(alpha12 + sign * np.sqrt(discriminant))
        # Roots xi1 / xi2 = pivot / alpha11 and alpha22 / pivot; |pivot| >= sqrt(D).
        roots = ((pivot, alpha11), (alpha22, pivot))
        farther = max(roots, key=lambda root: abs(root[1]) / np.hypot(*root))
        return float(farther[0]), float(farther[1])

    def _switched_point(
        self,
        point: np.ndarray,
        primary: np.ndarray,
        direction: np.ndarray,
        kick: float,
    ) -> np.ndarray | None:
        predicted = point + kick * direction
        corrected = self._correct(predicted, direction)
        if corrected is None:
            return None
        if np.linalg.norm(corrected - predicted) > abs(kick):
            return None
        displacement = corrected - point
        transverse = displacement - (displacement @ primary) * primary
        expected = abs(kick) * np.linalg.norm(
            direction - (direction @ primary) * primary,
        )
        if np.linalg.norm(transverse) < _TRANSVERSE_FRACTION * expected:
            return None
        _left, _singular, right_vectors = np.linalg.svd(self._extended(corrected))
        tangent = right_vectors[-1]
        if abs(tangent @ direction) <= abs(tangent @ primary):
            return None
        return corrected

    def _correct(
        self,
        predicted: np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray | None:
        current = predicted.copy()
        for _ in range(self._settings.max_iterations):
            augmented = self._augmented(current, predicted, direction)
            if np.linalg.norm(augmented) < self._settings.tolerance:
                return current
            bordered = np.vstack([self._extended(current), direction])
            current = current + self._linear_solver.solve(bordered, -augmented)
        augmented = self._augmented(current, predicted, direction)
        if np.linalg.norm(augmented) < self._settings.tolerance:
            return current
        return None

    def _augmented(
        self,
        current: np.ndarray,
        predicted: np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray:
        residual = self._residual.evaluate(current[:-1], float(current[-1]))
        constraint = direction @ (current - predicted)
        return np.concatenate([residual, [constraint]])

    def _extended(self, point: np.ndarray) -> np.ndarray:
        return self._jacobian.extended_jacobian(point[:-1], float(point[-1]))
