"""Sparsity patterns of field Jacobians, and the column grouping that exploits them.

A field whose component ``i`` reads only a few states - a ring cell and its
neighbour, a lattice site and its four neighbours - has a Jacobian with ``O(n)``
non-zeros, yet a column-by-column difference or autodiff pass costs ``n`` field
evaluations whatever the sparsity. Curtis, Powell and Reid (IMA J. Appl. Math. 13,
1974) observed that columns with no non-zero in a common row - *structurally
orthogonal* columns - can be differenced together: perturbing the state along
``d = sum of e_j`` over such a group gives, in each row, the derivative with
respect to the one column of the group that row depends on. Coleman and Moré (SIAM
J. Numer. Anal. 20, 1983) recast the partition as colouring the *column
intersection graph*, whose vertices are columns joined when they share a non-zero
row; the fewest groups is that graph's chromatic number, bounded below by the
largest number of non-zeros in a row.

:class:`SparsityPattern` holds the boolean pattern, colours the columns once with
the Curtis-Powell-Reid greedy sweep in column order (optimal for banded and ring
patterns: two groups for an even ring, three for an odd one, against the row bound
of two), and scatters the compressed columns of one pass per group back into the
dense Jacobian. Because a row's value in a grouped difference reads only the states
that row structurally depends on, a declared pattern that is a superset of the
true one makes every entry bit-identical to the column-by-column one.
"""

from collections.abc import Callable

import numpy as np

_DETECTION_SEED = 20261122
_DETECTION_PROBES = 2
_DETECTION_SCALE = 1.0e-2


class SparsityPattern:
    """The non-zero structure of an ``(m, n)`` Jacobian and its column groups."""

    __slots__ = ["_groups", "_mask"]

    def __init__(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2:  # noqa: PLR2004 (a matrix)
            message = "a sparsity pattern is a two-dimensional boolean matrix"
            raise ValueError(message)
        self._mask = mask
        self._groups = _greedy_column_groups(mask)

    @classmethod
    def from_rows(cls, rows: list[list[int]], columns: int) -> "SparsityPattern":
        """Pattern from, for each row, the columns that row depends on."""
        mask = np.zeros((len(rows), columns), dtype=bool)
        for row, dependencies in enumerate(rows):
            mask[row, list(dependencies)] = True
        return cls(mask)

    @classmethod
    def detect(
        cls,
        jacobian: Callable[[np.ndarray], np.ndarray],
        state: np.ndarray,
    ) -> "SparsityPattern":
        """Pattern seen by ``jacobian`` at ``state`` and two nearby probe states.

        An entry is kept when it is non-zero at any of the three points, so a
        derivative that happens to vanish at ``state`` (a coupling through
        ``x_j * x_k`` at ``x_k = 0``) is still found. An entry that is zero at all
        three, or smaller than the difference step can resolve, is dropped: declare
        the pattern when it is known.
        """
        generator = np.random.default_rng(_DETECTION_SEED)
        mask = jacobian(state) != 0.0
        scale = _DETECTION_SCALE * (1.0 + np.abs(state))
        for _ in range(_DETECTION_PROBES):
            probe = state + scale * generator.uniform(-1.0, 1.0, state.shape)
            mask |= jacobian(probe) != 0.0
        return cls(mask)

    @property
    def mask(self) -> np.ndarray:
        """Boolean ``(m, n)`` array, ``True`` where the Jacobian may be non-zero."""
        return self._mask

    @property
    def groups(self) -> np.ndarray:
        """Group index of each column; columns sharing an index are orthogonal."""
        return self._groups

    @property
    def group_count(self) -> int:
        """Number of grouped passes one Jacobian costs."""
        return int(self._groups.max()) + 1 if self._groups.size else 0

    def directions(self) -> np.ndarray:
        """``(groups, n)`` array whose rows are the 0/1 seed directions."""
        columns = self._mask.shape[1]
        seeds = np.zeros((self.group_count, columns))
        seeds[self._groups, np.arange(columns)] = 1.0
        return seeds

    def expand(self, compressed: np.ndarray) -> np.ndarray:
        """Dense ``(m, n)`` Jacobian from the ``(m, groups)`` compressed columns."""
        rows, columns = np.nonzero(self._mask)
        dense = np.zeros(self._mask.shape, dtype=compressed.dtype)
        dense[rows, columns] = compressed[rows, self._groups[columns]]
        return dense


def _greedy_column_groups(mask: np.ndarray) -> np.ndarray:
    """Curtis-Powell-Reid: give each column, in order, the first group it fits.

    A column fits a group when no row holds a non-zero of both, i.e. when the
    column's rows are disjoint from the rows the group already covers. That is
    the greedy (sequential) colouring of the column intersection graph.
    """
    columns = mask.shape[1]
    groups = np.empty(columns, dtype=int)
    covered: list[np.ndarray] = []
    for column in range(columns):
        rows = mask[:, column]
        for index, taken in enumerate(covered):
            if not np.any(taken & rows):
                np.logical_or(taken, rows, out=taken)
                groups[column] = index
                break
        else:
            covered.append(rows.copy())
            groups[column] = len(covered) - 1
    return groups
