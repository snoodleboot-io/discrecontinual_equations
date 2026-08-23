"""Describe the qualitative dynamics in a region of parameter space.

A two-parameter bifurcation diagram partitions the plane into regions whose phase
portraits differ. This module labels a region from the field itself: it locates the
equilibria (Newton from seeds), classifies each by its Jacobian spectrum
(stable/unstable node or focus, or saddle), and - in the plane - detects a limit
cycle via the Poincare-Bendixson principle: a bounded trajectory that never settles
onto an equilibrium must wind onto a periodic orbit.
"""

import numpy as np

from discrecontinual_equations.function.function import Function

_NEWTON_TOLERANCE = 1.0e-11
_NEWTON_ITERATIONS = 80
_RESIDUAL_CEILING = 1.0e-8
_STEP = 1.0e-7
_DEDUP = 1.0e-5
_IMAGINARY = 1.0e-7
_CYCLE_HORIZON = 6000
_CYCLE_DT = 0.01
_SETTLE_FRACTION = 0.6
_CYCLE_FLOOR = 1.0e-2
_ESCAPE_CEILING = 1.0e3
_RETURN_FRACTION = 0.15
_MINIMUM_RETURNS = 3
_PERIOD_SCATTER = 0.2


def _evaluate(field: Function, state: np.ndarray) -> np.ndarray:
    return np.array(field.eval(point=list(state), time=None), dtype=float)


def _jacobian(field: Function, state: np.ndarray) -> np.ndarray:
    base = _evaluate(field, state)
    dimension = state.size
    columns = np.empty((dimension, dimension))
    for j in range(dimension):
        shifted = state.copy()
        shifted[j] += _STEP
        columns[:, j] = (_evaluate(field, shifted) - base) / _STEP
    return columns


def find_equilibria(
    field: Function,
    seeds: list[np.ndarray],
) -> list[np.ndarray]:
    """Locate distinct equilibria by Newton iteration from each seed."""
    found: list[np.ndarray] = []
    for seed in seeds:
        state = np.asarray(seed, dtype=float).copy()
        for _ in range(_NEWTON_ITERATIONS):
            residual = _evaluate(field, state)
            if np.linalg.norm(residual) < _NEWTON_TOLERANCE:
                break
            jacobian = _jacobian(field, state)
            try:
                step = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
            state = state + step
        if np.linalg.norm(_evaluate(field, state)) >= _RESIDUAL_CEILING:
            continue
        if not any(np.linalg.norm(state - other) < _DEDUP for other in found):
            found.append(state)
    return found


def classify_equilibrium(jacobian: np.ndarray) -> str:
    """Label an equilibrium from its Jacobian spectrum."""
    eigenvalues = np.linalg.eigvals(jacobian)
    real = eigenvalues.real
    oscillatory = bool(np.any(np.abs(eigenvalues.imag) > _IMAGINARY))
    shape = "focus" if oscillatory else "node"
    if np.all(real < -_IMAGINARY):
        return f"stable {shape}"
    if np.all(real > _IMAGINARY):
        return f"unstable {shape}"
    return "saddle"


def _integrate(field: Function, start: np.ndarray, direction: float) -> np.ndarray:
    state = np.asarray(start, dtype=float).copy()
    trajectory = np.empty((_CYCLE_HORIZON + 1, state.size))
    trajectory[0] = state
    step = direction * _CYCLE_DT
    for i in range(_CYCLE_HORIZON):
        k1 = _evaluate(field, state)
        k2 = _evaluate(field, state + 0.5 * step * k1)
        k3 = _evaluate(field, state + 0.5 * step * k2)
        k4 = _evaluate(field, state + step * k3)
        state = state + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(state)) or np.max(np.abs(state)) > _ESCAPE_CEILING:
            return trajectory[: i + 1]
        trajectory[i + 1] = state
    return trajectory


def _winds_onto_cycle(
    trajectory: np.ndarray,
    equilibria: list[np.ndarray],
) -> bool:
    if trajectory.shape[0] < _CYCLE_HORIZON:
        return False
    tail = trajectory[int(_CYCLE_HORIZON * _SETTLE_FRACTION) :]
    if not np.all(np.isfinite(tail)) or np.max(np.abs(tail)) > _ESCAPE_CEILING:
        return False
    span = float(np.max(np.linalg.norm(tail - tail.mean(axis=0), axis=1)))
    if span < _CYCLE_FLOOR:
        return False  # converged to a point
    nearest = min(
        (np.min(np.linalg.norm(tail - point, axis=1)) for point in equilibria),
        default=np.inf,
    )
    if nearest < _CYCLE_FLOOR:
        return False  # orbit passes through an equilibrium: not a cycle
    # Genuine periodicity: the settled trajectory returns to a reference point at a
    # consistent period. A homoclinic approach has period -> infinity and fails this.
    reference = tail[0]
    distance = np.linalg.norm(tail - reference, axis=1)
    threshold = _RETURN_FRACTION * span
    returns = [
        i
        for i in range(1, distance.size - 1)
        if distance[i] < threshold
        and distance[i] <= distance[i - 1]
        and distance[i] <= distance[i + 1]
    ]
    if len(returns) < _MINIMUM_RETURNS:
        return False
    gaps = np.diff(returns)
    return bool(np.std(gaps) < _PERIOD_SCATTER * np.mean(gaps))


def has_limit_cycle(
    field: Function,
    start: np.ndarray,
    equilibria: list[np.ndarray],
) -> bool:
    """Detect a planar limit cycle by the Poincare-Bendixson principle.

    A trajectory that stays bounded but never approaches an equilibrium must limit
    onto a periodic orbit. Forward integration reliably reveals stable (attracting)
    cycles; unstable cycles are not claimed, since forward trajectories cannot reach
    them and reverse-time detection is confounded by saddle connections.
    """
    return _winds_onto_cycle(_integrate(field, start, 1.0), equilibria)


def describe_region(
    field: Function,
    seeds: list[np.ndarray],
    cycle_start: np.ndarray | None = None,
) -> str:
    """Compose a short label of the dynamics: equilibria, stability, and cycles."""
    equilibria = find_equilibria(field, seeds)
    if not equilibria:
        return "no equilibria"
    labels = [classify_equilibrium(_jacobian(field, point)) for point in equilibria]
    labels.sort()
    summary = " + ".join(labels)
    if cycle_start is not None and has_limit_cycle(field, cycle_start, equilibria):
        summary += " + limit cycle"
    return summary
