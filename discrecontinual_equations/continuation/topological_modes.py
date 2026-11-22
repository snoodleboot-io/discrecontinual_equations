"""Counting the modes of a density by its superlevel-set persistence.

:class:`~.stochastic.DensityModes` and
:meth:`~.fokker_planck.GridDensity.interior_maxima` count local maxima, and a local
maximum is a statement about three neighbouring cells. That is the right question for a
density known to round-off, and the wrong one for any other: a kernel density estimate
of a few thousand samples has dozens of local maxima in its tails, each one cell wide
and a sampling error tall, and a maximum count would report every one of them as a mode.
The phenomenological bifurcation is about the modes a density *has*, not the wiggles an
estimate of it shows, so the count needs a notion of how significant a maximum is.

Persistent homology supplies it, following Tanweer, Khasawneh, Munch and Tempelman,
"A topological framework for identifying phenomenological bifurcations in stochastic
dynamical systems" (arXiv:2305.03118). Sweep a level ``L`` down from the top of the
density and watch the superlevel set ``{p >= L}``: a component is *born* at the level
of each local maximum and *dies* at the level where it merges into a component born
higher (the elder rule). Each maximum is thereby paired with a saddle, and the
difference ``birth - death`` - its *persistence* - is exactly how far the density has
to be lowered before that peak stops being separate. The highest peak never merges;
its component is the essential class. The number of modes that matter is the number
of components whose persistence exceeds what the density's own uncertainty could
produce, and on a grid the zero-dimensional part of the paper's cubical filtration is
the union-find below, with cells connected when they share any face, corners
included (the paper's top-cell construction gives every lower face the larger of the
values around it, so two cells meeting at a corner join as soon as both are in).

Two sources of density, and one rule for each:

* **On a grid**, from :class:`~.stochastic.StationaryDensity` or
  :class:`~.fokker_planck.StationaryFokkerPlanck`, the density is exact to
  discretisation and a component counts when its persistence exceeds round-off,
  the same ``sqrt(eps)`` of the peak below which :class:`~.fokker_planck.GridDensity`
  already declines to call a cell a maximum. The count then agrees with the local
  maximum count wherever no two maxima are separated by less than round-off.
* **From samples**, through :class:`KernelDensity`, the density is an estimate with a
  known pointwise standard error, and a component counts when its persistence exceeds
  ``z`` standard errors of the difference between the estimate at its peak and at its
  saddle. This plays the role of the paper's ``epsilon`` (after Bobrowski, Mukherjee
  and Taylor, "Topological consistency via kernel estimation", Bernoulli 23, 2017),
  with the width chosen per component from the estimate's variance rather than by
  hand.

The sample path is the route to recorded data. Nothing in it needs a model: a device's
output, an experiment's record or an ensemble of simulated paths all arrive as an
array of states, and the same count and the same threshold locator apply. Two things
the caller owns, and the estimate cannot check: the samples must be (close to)
independent, or ``effective_samples`` must say how many independent ones they are
worth - a time series sampled faster than it decorrelates is worth its length divided
by twice its integrated autocorrelation time - and the kernel bandwidth moves a
threshold by a known amount, since what is counted is the mode structure of the
density smoothed by the kernel. For the multiplicative pitchfork that smoothing moves
the P-threshold from ``s^2 / 2`` to ``s^2 / 2 + h^2`` exactly (Gaussian kernel of
width ``h``), which is the size to expect elsewhere.
"""

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np
from scipy.signal import fftconvolve

from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    GridDensity,
)
from discrecontinual_equations.continuation.root_finder import ScalarRootFinder
from discrecontinual_equations.continuation.stochastic_threshold import (
    DensityProbe,
    ParameterThreshold,
)

# Persistence below this fraction of the peak is round-off; the same floor as
# GridDensity.interior_maxima.
_ROUND_OFF_FLOOR = float(np.sqrt(np.finfo(float).eps))
_DEFAULT_SIGNIFICANCE = 3.0
_KERNEL_REACH = 5.0
_BIMODAL = 2


class PersistencePair:
    """One component of the superlevel filtration: born at a peak, dead at a saddle.

    The essential component, born at the global maximum, never merges; it is reported
    as dying at the global minimum, so that its persistence is the full range of the
    density and it always counts.
    """

    __slots__ = ["_birth", "_birth_cell", "_death", "_death_cell", "_essential"]

    def __init__(
        self,
        birth: float,
        death: float,
        birth_cell: tuple[int, ...],
        death_cell: tuple[int, ...],
        *,
        essential: bool = False,
    ) -> None:
        self._birth = birth
        self._death = death
        self._birth_cell = birth_cell
        self._death_cell = death_cell
        self._essential = essential

    @property
    def birth(self) -> float:
        """Density at the peak that created the component."""
        return self._birth

    @property
    def death(self) -> float:
        """Density at the saddle where it merged into an older component."""
        return self._death

    @property
    def birth_cell(self) -> tuple[int, ...]:
        """Multi-index of the peak."""
        return self._birth_cell

    @property
    def death_cell(self) -> tuple[int, ...]:
        """Multi-index of the saddle."""
        return self._death_cell

    @property
    def essential(self) -> bool:
        """Whether this is the component of the global maximum."""
        return self._essential

    @property
    def persistence(self) -> float:
        """``birth - death``: how far the level must fall before the peak merges."""
        return self._birth - self._death


def _neighbour_strides(shape: tuple[int, ...]) -> np.ndarray:
    """Flat offsets of every cell sharing a face or corner, in a once-padded array."""
    padded = [size + 2 for size in shape]
    strides = [int(np.prod(padded[axis + 1 :])) for axis in range(len(shape))]
    origin = tuple(0 for _ in shape)
    return np.array(
        [
            int(np.dot(offset, strides))
            for offset in itertools.product((-1, 0, 1), repeat=len(shape))
            if offset != origin
        ],
        dtype=int,
    )


def superlevel_persistence(values: np.ndarray) -> list[PersistencePair]:
    """Zero-dimensional persistence of the superlevel sets of an array.

    Cells are added from the highest value down; a cell joins every component it
    touches, and when it touches several, all but the oldest die at its value. Ties
    are broken by the order of a stable sort, so a plateau contributes one component
    and zero-persistence pairs for the rest, which every significance rule drops. The
    pairs are returned most persistent first.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        message = "persistence needs at least one value"
        raise ValueError(message)
    shape = array.shape
    padded = np.pad(array, 1, constant_values=-np.inf)
    inside = np.pad(np.ones(shape, dtype=bool), 1, constant_values=False)
    flat = padded.ravel()
    strides = _neighbour_strides(shape)
    cells = np.flatnonzero(inside.ravel())
    order = cells[np.argsort(-flat[cells], kind="stable")]
    parent = np.full(flat.size, -1, dtype=int)
    rank = np.full(flat.size, -1, dtype=int)
    pairs: list[PersistencePair] = []

    def root(cell: int) -> int:
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = int(parent[cell])
        return cell

    for position, cell in enumerate(order):
        parent[cell] = cell
        rank[cell] = position
        touching = {
            root(int(cell + step)) for step in strides if parent[cell + step] >= 0
        }
        touching.discard(int(cell))
        if not touching:
            continue
        roots = sorted(touching, key=lambda r: rank[r])
        oldest = roots[0]
        parent[cell] = oldest
        for younger in roots[1:]:
            pairs.append(
                PersistencePair(
                    float(flat[younger]),
                    float(flat[cell]),
                    _unpad(younger, shape),
                    _unpad(int(cell), shape),
                ),
            )
            parent[younger] = oldest
    top, bottom = int(order[0]), int(order[-1])
    pairs.append(
        PersistencePair(
            float(flat[top]),
            float(flat[bottom]),
            _unpad(top, shape),
            _unpad(bottom, shape),
            essential=True,
        ),
    )
    return sorted(pairs, key=lambda pair: -pair.persistence)


def _unpad(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    padded = tuple(size + 2 for size in shape)
    return tuple(int(i) - 1 for i in np.unravel_index(flat_index, padded))


class TopologicalModes:
    """The significant components of a density's superlevel filtration.

    ``uncertainty`` is a pointwise standard error, shaped like ``values``; a component
    is significant when its persistence exceeds ``significance`` standard errors of
    the difference between the density at its peak and at its saddle. Without one the
    density is taken as exact, and a component is significant when its persistence
    exceeds round-off of the peak.
    """

    __slots__ = ["_pairs", "_significance", "_uncertainty", "_values"]

    def __init__(
        self,
        values: np.ndarray,
        uncertainty: np.ndarray | None = None,
        significance: float = _DEFAULT_SIGNIFICANCE,
    ) -> None:
        self._values = np.asarray(values, dtype=float)
        if uncertainty is not None and np.shape(uncertainty) != self._values.shape:
            message = "the uncertainty must have the shape of the density"
            raise ValueError(message)
        self._uncertainty = uncertainty
        self._significance = significance
        self._pairs = superlevel_persistence(self._values)

    @classmethod
    def of_density(cls, density: GridDensity) -> "TopologicalModes":
        """The exact-density reading of a solved grid density."""
        return cls(density.values)

    @property
    def pairs(self) -> list[PersistencePair]:
        """Every component, most persistent first; the essential one included."""
        return list(self._pairs)

    def bar(self, pair: PersistencePair) -> float:
        """The persistence a component must exceed to be counted.

        Never below round-off of the peak, so that a flat stretch - the empty tail of
        a kernel estimate, where the estimate and its error are both zero - cannot
        contribute a component of zero persistence that clears a zero bar.
        """
        floor = _ROUND_OFF_FLOOR * float(np.max(self._values))
        if self._uncertainty is None:
            return floor
        spread = math.hypot(
            float(self._uncertainty[pair.birth_cell]),
            float(self._uncertainty[pair.death_cell]),
        )
        return max(self._significance * spread, floor)

    def significant(self) -> list[PersistencePair]:
        """The components that count, the essential one first."""
        finite = [pair for pair in self._pairs if not pair.essential]
        kept = sorted(
            (pair for pair in finite if pair.persistence > self.bar(pair)),
            key=lambda pair: -(pair.persistence - self.bar(pair)),
        )
        essential = [pair for pair in self._pairs if pair.essential]
        return essential + kept

    def count(self) -> int:
        """Number of significant modes: the P-bifurcation's observable."""
        return len(self.significant())

    def locations(self, grid: DensityGrid) -> list[np.ndarray]:
        """Coordinates of the significant peaks on ``grid``."""
        return [
            np.array(
                [grid.axis(axis)[index] for axis, index in enumerate(pair.birth_cell)],
            )
            for pair in self.significant()
        ]

    def margin(self, modes: int = _BIMODAL) -> float:
        """How far the ``modes``-th component clears its bar; negative if it does not.

        The essential component is the first mode, so ``modes = 2`` reads the most
        significant finite component: its persistence minus its bar. The value is
        positive exactly when at least ``modes`` components count, and it is a
        continuous function of the density wherever the pairing does not change
        (persistence is stable under perturbation of the values), which makes it a
        test function a root finder can bracket. With fewer finite components than
        asked for it returns minus the bar at the global peak.
        """
        if modes < 1:
            message = "a density has at least one mode"
            raise ValueError(message)
        if modes == 1:
            return float(np.max(self._values))
        excesses = sorted(
            (
                pair.persistence - self.bar(pair)
                for pair in self._pairs
                if not pair.essential
            ),
            reverse=True,
        )
        if len(excesses) >= modes - 1:
            return excesses[modes - 2]
        essential = next(pair for pair in self._pairs if pair.essential)
        return -self.bar(essential)


class KernelDensity:
    """A Gaussian kernel density estimate of samples, on a :class:`DensityGrid`.

    ``samples`` is ``(n,)`` for one state variable or ``(n, d)``. The bandwidth is one
    standard deviation of the kernel per axis; left out, it is Scott's normal-reference
    rule ``h_i = std_i * n^(-1/(d+4))``, which over-smooths a multimodal density and is
    therefore a poor default for locating a bifurcation - pass one. The samples are
    linearly binned onto the grid and the bins convolved with the kernel by FFT
    (Wand and Jones, *Kernel Smoothing*, 1995, App. D), so the cost is the grid's and
    not ``n`` times the grid's; the binning error is ``O(spacing^2 / h^2)`` relative.
    Samples outside the grid are dropped from the bins but still counted in ``n``, so
    the grid should reach a few bandwidths beyond the data.

    The pointwise standard error is that of a mean of independent kernel values (see
    :meth:`standard_error`); its leading term is the textbook ``sqrt(p R(K) / (n h))``
    (Wand and Jones, Sec. 2.5). ``n`` there is the number of *independent* samples,
    which ``effective_samples`` overrides for correlated data.
    """

    __slots__ = ["_bandwidths", "_effective", "_grid", "_samples"]

    def __init__(
        self,
        samples: np.ndarray,
        grid: DensityGrid,
        bandwidth: float | Sequence[float] | None = None,
        effective_samples: float | None = None,
    ) -> None:
        points = np.asarray(samples, dtype=float)
        if points.ndim == 1:
            points = points[:, None]
        if points.ndim != _BIMODAL or points.shape[1] != grid.dimension:
            message = "samples must be (n,) or (n, d) with d the grid's dimension"
            raise ValueError(message)
        if points.shape[0] < _BIMODAL:
            message = "a density estimate needs at least two samples"
            raise ValueError(message)
        self._samples = points
        self._grid = grid
        self._bandwidths = self._resolve_bandwidths(bandwidth)
        self._effective = (
            float(points.shape[0]) if effective_samples is None else effective_samples
        )

    def _resolve_bandwidths(
        self,
        bandwidth: float | Sequence[float] | None,
    ) -> np.ndarray:
        count, dimension = self._samples.shape
        if bandwidth is None:
            spread = np.std(self._samples, axis=0, ddof=1)
            return spread * count ** (-1.0 / (dimension + 4))
        widths = np.broadcast_to(
            np.asarray(bandwidth, dtype=float),
            (dimension,),
        ).copy()
        if np.min(widths) <= 0.0:
            message = "kernel bandwidths must be positive"
            raise ValueError(message)
        return widths

    @property
    def bandwidths(self) -> np.ndarray:
        """Kernel standard deviation along each axis."""
        return self._bandwidths.copy()

    def density(self) -> GridDensity:
        """The estimate at every cell centre."""
        grid = self._grid
        counts = self._binned()
        kernel = self._kernel()
        smoothed = fftconvolve(counts, kernel, mode="same")
        # FFT round-off can leave values of order 1e-17 below zero; a density is not.
        estimate = np.clip(smoothed, 0.0, None) / self._samples.shape[0]
        return GridDensity(grid, estimate)

    def standard_error(self) -> np.ndarray:
        """Pointwise standard error of :meth:`density`, grid-shaped.

        The estimate at ``x`` is the mean of ``n`` independent values ``K_h(x - X_i)``,
        so its variance is ``(E[K_h^2] - p^2) / n``, and both moments are estimated
        from the same samples: the second by binning against the squared kernel. The
        familiar ``p R(K) / (n h)`` is the leading term of this; at the bandwidths a
        bifurcation needs, the ``p^2 / n`` it drops is a fifth of the variance.
        """
        counts = self._binned()
        count = self._samples.shape[0]
        first = np.clip(fftconvolve(counts, self._kernel(), mode="same"), 0.0, None)
        second = np.clip(
            fftconvolve(counts, self._kernel() ** 2, mode="same"),
            0.0,
            None,
        )
        variance = np.clip(second / count - (first / count) ** 2, 0.0, None)
        return np.sqrt(variance / self._effective)

    def modes(self, significance: float = _DEFAULT_SIGNIFICANCE) -> TopologicalModes:
        """The significant modes of the estimate."""
        return TopologicalModes(
            self.density().values,
            uncertainty=self.standard_error(),
            significance=significance,
        )

    def _binned(self) -> np.ndarray:
        """Linear binning: each sample split between the ``2^d`` nodes around it."""
        grid = self._grid
        origins = np.array([grid.axis(axis)[0] for axis in range(grid.dimension)])
        scaled = (self._samples - origins) / grid.spacings
        lower = np.floor(scaled).astype(int)
        fraction = scaled - lower
        counts = np.zeros(grid.shape)
        shape = np.array(grid.shape)
        for corner in itertools.product((0, 1), repeat=grid.dimension):
            step = np.array(corner)
            index = lower + step
            weight = np.prod(np.where(step == 1, fraction, 1.0 - fraction), axis=1)
            valid = np.all((index >= 0) & (index < shape), axis=1)
            np.add.at(counts, tuple(index[valid].T), weight[valid])
        return counts

    def _kernel(self) -> np.ndarray:
        """The Gaussian kernel sampled at grid offsets, as a density per cell volume."""
        grid = self._grid
        axes = []
        for axis in range(grid.dimension):
            spacing = float(grid.spacings[axis])
            width = float(self._bandwidths[axis])
            reach = min(
                math.ceil(_KERNEL_REACH * width / spacing),
                grid.shape[axis] - 1,
            )
            offsets = np.arange(-reach, reach + 1) * spacing
            axes.append(
                np.exp(-0.5 * (offsets / width) ** 2)
                / (width * math.sqrt(2 * math.pi)),
            )
        kernel = axes[0]
        for factor in axes[1:]:
            kernel = np.multiply.outer(kernel, factor)
        return kernel


class PersistenceProbe(DensityProbe):
    """:meth:`TopologicalModes.margin` of a solved grid density, for any locator.

    Positive once the density has at least ``modes`` components above round-off, so
    with :class:`~.stochastic_threshold.PhenomenologicalThreshold` it locates where a
    second mode appears without being told where to look for it - unlike
    :class:`~.stochastic_threshold.RadialCrater`, which must be given the centre the
    peak leaves.
    """

    __slots__ = ["_modes"]

    def __init__(self, modes: int = _BIMODAL) -> None:
        self._modes = modes

    def value(self, density: GridDensity) -> float:
        """The margin of the ``modes``-th component over round-off."""
        return TopologicalModes.of_density(density).margin(self._modes)


class ModeCountThreshold(ParameterThreshold):
    """Where the significant mode count of a density first reaches ``modes``.

    ``modes_at`` maps a parameter to the modes of whatever density is at hand there:
    a grid density read exactly, or a kernel estimate of samples. For a sampled
    density the test function is a different random function at every parameter
    unless the samples are drawn with common random numbers, for the reason
    :mod:`.stochastic_threshold` gives for the dynamical threshold; with them it is
    a deterministic function of the parameter and bisection applies.
    """

    __slots__ = ["_modes", "_modes_at"]

    def __init__(
        self,
        modes_at: Callable[[float], TopologicalModes],
        root_finder: ScalarRootFinder,
        modes: int = _BIMODAL,
    ) -> None:
        super().__init__(root_finder)
        self._modes_at = modes_at
        self._modes = modes

    def test_value(self, parameter: float) -> float:
        """The margin of the ``modes``-th component at this parameter."""
        return self._modes_at(parameter).margin(self._modes)
