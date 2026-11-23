"""The Wiener increments a stochastic Runge-Kutta step consumes.

A stochastic Runge-Kutta scheme is not an ODE Runge-Kutta scheme with a random
number added at the end. Its derivation (an Ito-Taylor expansion truncated at
some order, with derivatives replaced by supporting values) tells it exactly
which functionals of the driving Brownian motion it needs over one step, and the
stage structure only has the order it was derived for when those functionals
are supplied with the joint distribution the derivation assumed. Drawing one
fresh normal per stage, or using ``sqrt(dt)`` where an increment belongs, gives
a different scheme whose order nobody has worked out. This module is where the
schemes get their increments, so that the correlation is in one place and can
be checked once.

Two functionals cover every scheme in this package:

* ``delta_w = W(t + h) - W(t)``, the increment, which is ``N(0, h)``.
* ``delta_z = integral_t^{t+h} (W(s) - W(t)) ds``, the time integral of the
  increment (Kloeden and Platen's ``I_(1,0)``), which the order 1.5 strong
  scheme needs. It is Gaussian with ``Var delta_z = h^3 / 3`` and
  ``Cov(delta_w, delta_z) = h^2 / 2``, so with two independent standard normals
  ``u1`` and ``u2`` the pair is exactly
  ``delta_w = sqrt(h) u1``, ``delta_z = (h^{3/2} / 2) (u1 + u2 / sqrt(3))``.

Both are per noise channel, and the channels are independent. How many there are
is the equation's business: a vector diffusion has one channel per state
component (diagonal noise, component ``i`` driven by its own ``W_i``), a matrix
diffusion ``n x m`` has ``m`` (DEQ-28), and a solver asks for the count
:meth:`~.coefficients.ItoCoefficients.drivers` reports.

A third functional couples the channels: the Levy areas (DEQ-36),

* ``A_(j,k) = (1/2) (I_(j,k) - I_(k,j))`` with ``I_(j,k) = int_t^{t+h}
  int_t^s dW_j(u) dW_k(s)``, an antisymmetric ``m x m`` matrix, from which every
  twofold iterated Ito integral follows as ``I_(j,k) = (1/2) (dW_j dW_k -
  delta_jk h) + A_(j,k)``. This is what a Milstein step needs for noise that does
  not commute. ``A`` has mean zero, variance ``h^2 / 4`` for each pair, is
  uncorrelated with every increment and with the area of every other pair, and
  given the increments has variance ``h^2 / 12 + h (dW_j^2 + dW_k^2) / 12``
  (Levy's formula); its law is not Gaussian and has no closed-form sampler.

:class:`LevyAreaSampler` draws it by the algorithm of Mrongowius and Rossler
(arXiv:2101.09542): the truncated Fourier series of the Brownian bridge
(Milstein; Kloeden, Platen and Wright), with the part of the remainder that is
linear in ``dW`` simulated exactly and the rest replaced by a Gaussian of the
right (diagonal, ``dW``-independent) covariance. Its first two moments are exact
at any truncation and its pathwise L2 error falls like ``h / n`` in the number
``n`` of terms;
:meth:`LevyAreaSampler.terms` picks ``n`` so that the error per step is at most
``h^(3/2)``, which is what a scheme of strong order 1.0 needs (Kastner and
Rossler, arXiv:2201.08424, (64)-(67)). Areas are drawn only when a scheme asks
(``levy_area=True``), and from a stream of their own, so asking for them never
changes the increments a seeded source gives.

Why the increments come from a source object rather than from a generator inside
each solver: a strong-order measurement is the expected pathwise distance
between the scheme and the exact solution, driven by the same Brownian path. It
is only meaningful if every step size sees the same path, which means the path
has to exist before the solver runs. :class:`BrownianPath` is that pre-generated
path, reading each step's ``delta_w``, ``delta_z`` and Levy areas off a fine grid;
:class:`GaussianWienerSource` is the ordinary case, drawing fresh increments from
a generator the solver owns - never ``np.random.seed``, which seeds the
process-wide stream and lets one solver alter another's results.

References:
- Mrongowius, J. and Rossler, A. "On the approximation and simulation of iterated
  stochastic integrals and the corresponding Levy areas in terms of a
  multidimensional Brownian motion", arXiv:2101.09542 (2021).
- Kastner, F. and Rossler, A. "An analysis of approximation algorithms for iterated
  stochastic integrals and a Julia and Matlab simulation toolbox",
  arXiv:2201.08424 (2022).
- Levy's stochastic area and its characteristic functions, as quoted in
  https://en.wikipedia.org/wiki/L%C3%A9vy%27s_stochastic_area.
"""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# How many normals of each kind a chunk of cells draws at a time, when a path's
# every cell is given its areas (DEQ-70): 2 MB a block, so that the working
# memory is a few such blocks however fine the grid.
_CHUNK_NORMALS = 2**18


@dataclass(frozen=True)
class WienerIncrements:
    """What one step of Brownian motion gives a scheme, one entry per channel."""

    delta_w: np.ndarray
    """``W(t + h) - W(t)``."""

    delta_z: np.ndarray
    """``integral_t^{t+h} (W(s) - W(t)) ds``, Kloeden and Platen's ``I_(1,0)``."""

    levy_area: np.ndarray | None = None
    """``A_(j,k)``, ``m x m`` and antisymmetric; ``None`` unless a scheme asked."""

    def iterated_integrals(self, step: float) -> np.ndarray:
        """``I_(j,k) = (1/2) (dW_j dW_k - delta_jk h) + A_(j,k)``, inner index ``j``.

        ``I_(j,k) = int_t^{t+h} int_t^s dW_j(u) dW_k(s)``: the twofold Ito
        integrals, with the convention of Kloeden and Platen and of Mrongowius and
        Rossler (their (7)) that the first index is the inner integral. The
        Stratonovich ``J_(j,k)`` is this plus ``(1/2) delta_jk h``.
        """
        if self.levy_area is None:
            message = (
                "these increments carry no Levy area; ask the source for "
                "increments(..., levy_area=True)"
            )
            raise ValueError(message)
        dimension = self.delta_w.size
        return (
            0.5 * (np.outer(self.delta_w, self.delta_w) - step * np.eye(dimension))
            + self.levy_area
        )


class LevyAreaSampler:
    """Levy areas given the increments, by Mrongowius and Rossler's algorithm.

    With ``dW`` the increments over a step ``h`` of ``m`` independent Brownian
    motions and ``X_k, Y_k ~ N(0, I_m)``, ``k = 1..n``, ``Psi_1 ~ N(0, I_m)`` and
    ``Psi_2`` one standard normal per pair, all independent, the area of the pair
    ``j < l`` is (Mrongowius and Rossler, arXiv:2101.09542, Section 5.1, steps 2-4)::

        A_(j,l) = h / (2 pi) sum_{k=1}^n (1/k) [X_jk (Y_lk - sqrt(2/h) dW_l)
                                              - X_lk (Y_jk - sqrt(2/h) dW_j)]
                + sqrt(h) / (sqrt(2) pi) sqrt(tail) (dW_j Psi_1l - dW_l Psi_1j)
                + h / (sqrt(2) pi) sqrt(tail) Psi_2,(j,l)

    with ``tail = pi^2 / 6 - sum_{k=1}^n 1 / k^2 = sum_{k>n} 1 / k^2``. The first
    line is the truncated Fourier series of the Brownian bridge (Kloeden,
    Platen and Wright 1992, Milstein 1995), with ``X_k`` and ``Y_k`` the scaled
    cosine and sine coefficients; the second is its remainder's part linear in
    ``dW``, which is exactly Gaussian given ``dW`` and so simulated exactly; the
    third replaces the rest of the remainder, conditionally Gaussian with a
    covariance depending on the discarded ``X_k``, by a Gaussian with that
    covariance's mean, ``h^2 tail / (2 pi^2)`` times the identity. The
    approximation is unbiased, its variance is exactly ``h^2 / 4``, and its L2
    error is at most ``h sqrt(m) / (sqrt(12) pi n)`` (their Corollary 4.5).

    The sampler is scale-free in the sense that ``A(h) = h A(1)`` given ``dW /
    sqrt(h)``; it takes ``dW`` with any leading batch axes, and
    :meth:`sample_cells` draws a fine path's every cell in bounded memory.
    """

    __slots__ = ["_accuracy", "_generator"]

    def __init__(self, generator: np.random.Generator, accuracy: float = 1.0) -> None:
        if not accuracy > 0.0:
            message = f"accuracy must be positive, got {accuracy}"
            raise ValueError(message)
        self._generator = generator
        self._accuracy = float(accuracy)

    @staticmethod
    def terms(step: float, dimension: int, accuracy: float = 1.0) -> int:
        """The truncation ``n`` that keeps the L2 error per step below ``c1 h^(3/2)``.

        From Mrongowius and Rossler's (36), ``max E|I^(n) - I|^2 <= m h^2 / (12
        pi^2 n^2)``; asking for ``c1 h^(3/2)`` (Kastner and Rossler,
        arXiv:2201.08424, (64) with ``gamma = 1``) gives ``n >= sqrt(m / (12
        pi^2)) / (c1 sqrt(h))`` (their (67)), which is ``O(h^(-1/2))`` - against
        ``O(h^(-1))`` for the Fourier series without the remainder terms.

        ``accuracy`` is ``c1``. Any constant gives strong order 1.0 as ``h -> 0``;
        the default 1 (Kastner and Rossler's ``epsilon = h^(3/2)``) keeps ``n``
        at 1 for ``h >= 2^-5`` with two channels, and the sampler's error falls
        more slowly than ``1 / n`` for small ``n``, so sampler-driven Milstein
        measures a slope of about 0.75 over ``h = 2^-3 ... 2^-10`` (within the
        bound: at most ``0.4 h``).
        ``0.05`` measures 0.96 to 0.98 there, for 8 to 17 times the terms
        (DEQ-67).
        """
        bound = np.sqrt(dimension / (12.0 * np.pi**2)) / (accuracy * np.sqrt(step))
        return max(1, int(np.ceil(bound)))

    @staticmethod
    def assemble(  # noqa: PLR0913 (the increments, the step and four normals)
        delta_w: np.ndarray,
        step: float,
        x: np.ndarray,
        y: np.ndarray,
        psi1: np.ndarray,
        psi2: np.ndarray,
    ) -> np.ndarray:
        """The areas from given standard normals, as ``(..., m, m)``.

        ``delta_w`` is ``(..., m)``; ``x`` and ``y`` are ``(..., n, m)``; ``psi1``
        is ``(..., m)``; ``psi2`` is ``(..., m, m)`` of which only the entries
        above the diagonal are read. Public so that a pathwise error can be
        measured against a reference coupled to the same normals.
        """
        count = x.shape[-2]
        k = np.arange(1, count + 1, dtype=float)
        tail = np.pi**2 / 6.0 - float(np.sum(1.0 / k**2))
        shifted = y - np.sqrt(2.0 / step) * delta_w[..., np.newaxis, :]
        # series[j, l] = sum_k X_jk (Y_lk - sqrt(2/h) dW_l) / k
        series = np.einsum("...kj,...kl->...jl", x / k[:, np.newaxis], shifted)
        fourier = step / (2.0 * np.pi) * (series - np.swapaxes(series, -1, -2))
        linear = delta_w[..., :, np.newaxis] * psi1[..., np.newaxis, :]
        first = (
            np.sqrt(step * tail)
            / (np.sqrt(2.0) * np.pi)
            * (linear - np.swapaxes(linear, -1, -2))
        )
        upper = np.triu(psi2, 1)
        second = (
            step
            * np.sqrt(tail)
            / (np.sqrt(2.0) * np.pi)
            * (upper - np.swapaxes(upper, -1, -2))
        )
        return fourier + first + second

    def sample(
        self,
        delta_w: np.ndarray,
        step: float,
        terms: int | None = None,
    ) -> np.ndarray:
        """Draw the areas for increments ``delta_w`` of shape ``(..., m)``.

        ``terms`` defaults to :meth:`terms` for the step at this sampler's
        accuracy; the ``2 m n + m + m (m - 1) / 2`` normals are drawn in the order
        ``X``, ``Y``, ``Psi_1``, ``Psi_2``.
        """
        dimension = delta_w.shape[-1]
        count = self._count(step, dimension, terms)
        streams = [self._generator] * 4
        return self._draw(streams, delta_w, step, count)

    def sample_cells(
        self,
        delta_w: np.ndarray,
        step: float,
        terms: int | None = None,
        chunk_normals: int | None = None,
    ) -> np.ndarray:
        """What :meth:`sample` gives for ``delta_w`` ``(cells, m)``, in chunks.

        The same numbers bit for bit, and the generator left where
        :meth:`sample` leaves it, but drawn a chunk of cells at a time, so that
        the working memory is a few chunks rather than ``cells x n x m x 2``
        normals - several GB for a path of ``2^20`` cells (DEQ-70). ``sample``
        draws all the ``X`` before all the ``Y``, then ``Psi_1``, then
        ``Psi_2``; a chunk needs its cells' share of each, so a first pass walks
        the stream in chunks, discarding the draws and keeping the generator's
        state where each block begins, and a second draws every block's chunk
        from a copy of the generator set to that state. The normals are drawn
        twice; that is the price of an unchanged stream, since a normal takes a
        varying number of the generator's outputs and a block's end cannot be
        jumped to. ``chunk_normals`` (default ``_CHUNK_NORMALS``) bounds the
        normals of one kind a chunk draws.
        """
        cells, dimension = delta_w.shape
        count = self._count(step, dimension, terms)
        budget = _CHUNK_NORMALS if chunk_normals is None else chunk_normals
        chunk = max(1, budget // (count * dimension))
        pairs = dimension * (dimension - 1) // 2
        blocks = [(count, dimension), (count, dimension), (dimension,), (pairs,)]
        starts = []
        for shape in blocks:
            starts.append(self._generator.bit_generator.state)
            for begin in range(0, cells, chunk):
                self._generator.standard_normal((min(chunk, cells - begin), *shape))
        streams = []
        for state in starts:
            stream = copy.deepcopy(self._generator)
            stream.bit_generator.state = state
            streams.append(stream)
        areas = np.empty((cells, dimension, dimension))
        for begin in range(0, cells, chunk):
            cut = slice(begin, begin + chunk)
            areas[cut] = self._draw(streams, delta_w[cut], step, count)
        return areas

    def _count(self, step: float, dimension: int, terms: int | None) -> int:
        """``terms`` if given, else the rule's at this sampler's accuracy."""
        if terms is not None:
            return terms
        return self.terms(step, dimension, self._accuracy)

    def _draw(
        self,
        streams: list[np.random.Generator],
        delta_w: np.ndarray,
        step: float,
        count: int,
    ) -> np.ndarray:
        """The areas with ``X``, ``Y``, ``Psi_1``, ``Psi_2`` from the four streams."""
        batch = delta_w.shape[:-1]
        dimension = delta_w.shape[-1]
        x = streams[0].standard_normal((*batch, count, dimension))
        y = streams[1].standard_normal((*batch, count, dimension))
        psi1 = streams[2].standard_normal((*batch, dimension))
        # One Psi_2 per pair j < l, in the row-major order of the upper triangle.
        upper = np.triu_indices(dimension, 1)
        psi2 = np.zeros((*batch, dimension, dimension))
        psi2[..., upper[0], upper[1]] = streams[3].standard_normal(
            (*batch, len(upper[0])),
        )
        return self.assemble(delta_w, step, x, y, psi1, psi2)


def _area_stream(sequence: np.random.SeedSequence) -> LevyAreaSampler:
    """A sampler on a stream spawned from ``sequence``, independent of its draws."""
    return LevyAreaSampler(np.random.default_rng(sequence.spawn(1)[0]))


class WienerSource(ABC):
    """Supplies the Wiener functionals for one step of a scheme."""

    @abstractmethod
    def increments(
        self,
        time: float,
        step: float,
        dimension: int,
        *,
        levy_area: bool = False,
    ) -> WienerIncrements:
        """The increments over ``[time, time + step]`` for ``dimension`` channels.

        With ``levy_area`` the result carries the Levy areas between channels too.
        """
        raise NotImplementedError


class GaussianWienerSource(WienerSource):
    """Fresh increments from a generator this source owns.

    ``seed=None`` means fresh entropy, as ``np.random.default_rng`` does. The
    Levy areas come from a second stream spawned from the same seed, so a seeded
    source gives the same increments whether or not areas are asked for.
    """

    __slots__ = ["_areas", "_generator"]

    def __init__(self, seed: int | None = None) -> None:
        sequence = np.random.SeedSequence(seed)
        # default_rng(SeedSequence(seed)) is the stream default_rng(seed) gives.
        self._generator = np.random.default_rng(sequence)
        self._areas = _area_stream(sequence)

    def increments(
        self,
        time: float,  # noqa: ARG002
        step: float,
        dimension: int,
        *,
        levy_area: bool = False,
    ) -> WienerIncrements:
        u1 = self._generator.standard_normal(dimension)
        u2 = self._generator.standard_normal(dimension)
        root = np.sqrt(step)
        delta_w = root * u1
        delta_z = 0.5 * step * root * (u1 + u2 / np.sqrt(3.0))
        area = self._areas.sample(delta_w, step) if levy_area else None
        return WienerIncrements(delta_w=delta_w, delta_z=delta_z, levy_area=area)


class BrownianPath(WienerSource):
    """A Brownian path fixed in advance, read at whatever step a scheme uses.

    The path is held on a uniform grid of spacing ``resolution``. A step must be
    a whole number of grid cells, so that every step size that divides the
    horizon sees exactly the same path: ``delta_w`` is then the exact difference
    of grid values, and ``delta_z`` is the trapezoidal integral of ``W - W(t)``
    over the cells inside the step, whose error is of order ``resolution^2`` per
    step and so well below any scheme error at the step sizes a convergence
    study uses.

    The Levy area over a step is the trapezoidal double integral of the path -
    the area of the polygon through the grid nodes, ``(1/2) sum (W_j(s) - W_j(t))
    dW_l(s) - (W_l(s) - W_l(t)) dW_j(s)`` over the cells - plus the areas inside
    the cells, which a polygon cannot see. That sum is Chen's relation, so the
    reading is consistent between step sizes: the areas of two adjacent steps and
    the cross term of their increments add up to the area of their union. A path
    from :meth:`sample` draws the cell areas with :class:`LevyAreaSampler`, on a
    stream of its own, only when first asked and a chunk of cells at a time
    (:meth:`LevyAreaSampler.sample_cells`), and its reading has the law of
    the Levy area at any resolution. A path built from ``values`` alone has no
    cell areas unless given ``areas``; its reading then lacks variance ``h
    resolution / 4`` of the ``h^2 / 4`` and converges to the law as the grid is
    refined.
    """

    __slots__ = ["_area_sampler", "_areas", "_resolution", "_start", "_values"]

    def __init__(
        self,
        start: float,
        resolution: float,
        values: np.ndarray,
        areas: np.ndarray | None = None,
    ) -> None:
        if values.ndim != 2:  # noqa: PLR2004 (a matrix: grid nodes by channels)
            message = "values must have shape (nodes, channels)"
            raise ValueError(message)
        self._start = float(start)
        self._resolution = float(resolution)
        self._values = np.asarray(values, dtype=float)
        cells, dimension = len(self._values) - 1, self._values.shape[1]
        if areas is not None and areas.shape != (cells, dimension, dimension):
            message = "areas must have shape (cells, channels, channels)"
            raise ValueError(message)
        self._areas = areas
        self._area_sampler: LevyAreaSampler | None = None

    @classmethod
    def sample(
        cls,
        start: float,
        end: float,
        resolution: float,
        dimension: int,
        seed: int | None = None,
    ) -> "BrownianPath":
        """Draw a path on ``[start, end]`` with ``W(start) = 0`` on every channel."""
        sequence = np.random.SeedSequence(seed)
        generator = np.random.default_rng(sequence)
        cells = round((end - start) / resolution)
        if cells < 1 or not np.isclose(start + cells * resolution, end):
            message = "the horizon must be a whole number of resolution cells"
            raise ValueError(message)
        increments = generator.standard_normal((cells, dimension)) * np.sqrt(resolution)
        values = np.vstack([np.zeros((1, dimension)), np.cumsum(increments, axis=0)])
        path = cls(start=start, resolution=resolution, values=values)
        path._area_sampler = _area_stream(sequence)
        return path

    @property
    def dimension(self) -> int:
        return self._values.shape[1]

    @property
    def end(self) -> float:
        return self._start + (self._values.shape[0] - 1) * self._resolution

    @property
    def values(self) -> np.ndarray:
        """``W`` at the grid nodes, ``(nodes, channels)``; a copy."""
        return self._values.copy()

    def _node(self, time: float) -> int:
        offset = (time - self._start) / self._resolution
        node = round(offset)
        if (
            not np.isclose(offset, node, atol=1.0e-6)
            or node < 0
            or node >= len(self._values)
        ):
            message = f"time {time} is not a grid node of this path"
            raise ValueError(message)
        return node

    def value(self, time: float) -> np.ndarray:
        """``W(time)`` on every channel; ``time`` must be a grid node."""
        return self._values[self._node(time)]

    def _cell_areas(self) -> np.ndarray | None:
        """The areas inside the grid cells, drawn on first use for a sampled path."""
        if self._areas is None and self._area_sampler is not None:
            cells = np.diff(self._values, axis=0)
            self._areas = self._area_sampler.sample_cells(cells, self._resolution)
        return self._areas

    def increments(
        self,
        time: float,
        step: float,
        dimension: int,
        *,
        levy_area: bool = False,
    ) -> WienerIncrements:
        if dimension != self.dimension:
            message = (
                f"path has {self.dimension} channels, scheme asked for {dimension}"
            )
            raise ValueError(message)
        first = self._node(time)
        last = self._node(time + step)
        segment = self._values[first : last + 1] - self._values[first]
        delta_w = segment[-1].copy()
        delta_z = np.trapezoid(segment, dx=self._resolution, axis=0)
        area = None
        if levy_area:
            # (1/2) sum over cells of (W(s) - W(t)) dW^T - its transpose: the
            # polygon's area; the trapezoid's extra (1/2) dW dW^T is symmetric and
            # cancels in the antisymmetric part.
            products = segment[:-1].T @ np.diff(segment, axis=0)
            area = 0.5 * (products - products.T)
            cell_areas = self._cell_areas()
            if cell_areas is not None:
                area = area + np.sum(cell_areas[first:last], axis=0)
        return WienerIncrements(delta_w=delta_w, delta_z=delta_z, levy_area=area)
