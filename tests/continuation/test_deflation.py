"""Deflation: finding the roots a solver would otherwise keep rediscovering."""

from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.deflation import DeflatedSolver
from tests.continuation.fields import (
    CubicField,
    GridField,
    State,
)

_SADDLE_JACOBIAN = np.array([[0.0, 1.0], [1.0, 0.0]])


class TestDeflatedSolver(TestCase):
    """Deflated Newton finds distinct roots; matrix-free matches the direct solve."""

    def _cubic(self) -> CubicField:
        return CubicField(
            variables=[State()],
            parameters=[],
            results=[State()],
            time=None,
        )

    def _grid(self) -> GridField:
        return GridField(
            variables=[State(), State()],
            parameters=[],
            results=[State(), State()],
            time=None,
        )

    def test_finds_all_cubic_roots_from_one_seed(self):
        roots = DeflatedSolver(self._cubic()).find([np.array([0.5])])
        recovered = sorted(round(float(r[0]), 4) for r in roots)
        assert recovered == [-1.0, 0.0, 1.0]

    def test_finds_all_nine_grid_roots(self):
        seeds = [
            np.array([0.3, 0.4]),
            np.array([-0.6, 0.2]),
            np.array([0.2, -0.7]),
        ]
        roots = DeflatedSolver(self._grid()).find(seeds)
        assert len(roots) == 9
        field = self._grid()
        for root in roots:
            residual = np.linalg.norm(np.array(field.eval(point=list(root), time=None)))
            assert residual < 1.0e-6

    def test_matrix_free_matches_direct(self):
        seeds = [
            np.array([0.3, 0.4]),
            np.array([-0.6, 0.2]),
            np.array([0.2, -0.7]),
        ]
        direct = DeflatedSolver(self._grid()).find(seeds)
        krylov = DeflatedSolver(self._grid(), matrix_free=True).find(seeds)
        assert len(krylov) == len(direct) == 9

        def as_set(roots: list[np.ndarray]) -> set[tuple[float, float]]:
            return {(round(float(r[0])), round(float(r[1]))) for r in roots}

        assert as_set(krylov) == as_set(direct)
