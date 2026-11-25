"""The multiple Ito integrals up to multiplicity three, drawn jointly (DEQ-66).

A strong order 1.5 scheme for an equation driven by ``m`` Wiener processes
consumes, over every step ``[t, t + h]``, the Ito-Taylor expansion's integrals
(Kloeden and Platen (10.4.6); Kuznetsov, arXiv:1802.00888, (5)):

* ``I_(j) = dW_j`` and ``I_(j,0) = int (W_j(s) - W_j(t)) ds``, Gaussian, which
  :mod:`~.wiener` already draws as ``delta_w`` and ``delta_z``;
* ``I_(j,k) = int int dW_j dW_k``, the twofold integrals, whose antisymmetric part
  is the Levy area;
* ``I_(j,k,l) = int int int dW_j dW_k dW_l``, the threefold ones,

with the first index the innermost integral throughout, the convention of
Kloeden and Platen, of Kuznetsov and of
:meth:`~.wiener.WienerIncrements.iterated_integrals`.
They are functionals of one Brownian path, and a scheme only has its order when
they are drawn as such: drawing the areas and the triple integrals from separate
streams gives each the right law and the pair the wrong one.

**Method.** Kuznetsov's multiple Fourier-Legendre series (arXiv:1802.00888,
Theorem 1 and Section 5). With ``phi_p`` the Legendre polynomials orthonormal on
``[t, t + h]``, the coefficients ``zeta_p^(i) = int phi_p dW_i`` are independent
standard normals for every ``p`` and channel ``i``, and

* ``dW = sqrt(h) zeta_0`` and ``delta_z = (h^(3/2) / 2)(zeta_0 - zeta_1 /
  sqrt(3))`` exactly (his (42), (43) and (7); his ``I_(1)`` is ``int (t - s) dW``,
  so ``delta_z = h I_(0) + I_(1)``). These are exactly the pair the Gaussian source
  draws, so ``zeta_0`` and ``zeta_1`` are read back from it and only
  ``zeta_2, zeta_3, ...`` are new: the integrals are drawn *given* the increments
  a source already gave, which keeps every single-channel run as it was.
* ``I_(j,k) = (h / 2)(zeta_0^j zeta_0^k + sum_{i=1}^{q} (zeta_(i-1)^j zeta_i^k
  - zeta_i^j zeta_(i-1)^k) / sqrt(4 i^2 - 1) - delta_jk)`` (his (44)), with
  mean-square error ``(h^2 / 2)(1/2 - sum_{i=1}^q 1 / (4 i^2 - 1)) = h^2 / (4 (2 q
  + 1))`` for ``j != k`` (his (52)) and none for ``j = k``.
* ``I_(j,k,l) = sum_{p1,p2,p3=0}^{q} C_(p3,p2,p1) (zeta_p1^j zeta_p2^k
  zeta_p3^l - 1{j=k} 1{p1=p2} zeta_p3^l - 1{k=l} 1{p2=p3} zeta_p1^j - 1{j=l}
  1{p1=p3} zeta_p2^k)`` (his (55)), with ``C_(p3,p2,p1) = int phi_p3 int phi_p2
  int phi_p1`` ``= h^(3/2) sqrt((2p1+1)(2p2+1)(2p3+1)) / 8 * Cbar_(p3,p2,p1)`` and
  ``Cbar`` the same integral of the plain Legendre polynomials over ``[-1, 1]``
  (his (56), (57)); for ``j = k = l`` the exact ``(dW^3 - 3 h dW) / 6`` (his (58)
  and Remark 2). Its mean-square error is given exactly for every pattern of
  repeated channels by his (49) and (59)-(61): ``h^3 / 6 - sum C^2`` less a
  pattern-dependent cross sum; it falls like ``h^3 / (8 q)``.

``Cbar`` is computed here, not tabulated: with ``Q_p(x) = int_{-1}^x P_p``,
integrating by parts once gives ``Cbar_(p3,p2,p1) = 2 delta_(p3,0) int P_p2 Q_p1
- int Q_p3 P_p2 Q_p1``, a polynomial integral of degree at most ``3 q + 2`` that
Gauss-Legendre quadrature with ``(3 q + 4) / 2 + 2`` nodes evaluates exactly. It
reproduces Kuznetsov's quoted error for pairwise different channels at ``q = 6``,
``0.01956 h^3`` (his (63)).

**Erratum in the source.** Kuznetsov's (65), the fourfold analogue of (63) at
``q = 2``, prints ``0.0236084 h^4``. The exact value from (49) with the exact
coefficients is ``1/24 - sum C^2 = 234761/10245312 = 0.0229140 h^4``, so the
printed constant is a summation slip (DEQ-66 verification, DEQ-114). Nothing
here uses (65): the fourfold integrals are not drawn, since order 1.5 does not
need them.

**Truncation.** Order 1.5 needs a mean-square error of at most ``C h^4`` per
integral (Kuznetsov (8) with ``r = 4``, (66), (67); Kastner and Rossler,
arXiv:2201.08424, (64) with ``gamma = 3/2``). With ``C = 1``, the constant
:meth:`~.wiener.LevyAreaSampler.terms` uses, the twofold integrals need ``q >=
(1 / (4 h^2) - 1) / 2`` coefficients - ``O(h^-2)``, as for any series without a
remainder correction - and the threefold ones the least ``q`` whose exact error,
worst over the patterns, is at most ``h^4``: about ``0.13 / h``. Both use the same
``zeta``, the triple integrals the first ``q_3 + 1`` of them (Kuznetsov's
Algorithm 1, ``q << q_1``). Per step and channel that is ``q_1 - 1`` normals, and
per step ``O(m^3 q_3^3)`` operations for the ``m^3`` triple integrals.

Two sources supply them. :class:`GaussianIteratedIntegralSource` draws fresh ones,
with increments identical to :class:`~.wiener.GaussianWienerSource` for the same
seed. :class:`IteratedIntegralPath` is a path fixed in advance, holding every
integral of every grid cell and reading any step's by Chen's relation, so that a
convergence study sees one path at every step size.

References:
- Kuznetsov, D. F. "To numerical modeling with strong orders 1.0, 1.5, and 2.0 of
  convergence for multidimensional dynamical systems with random disturbances",
  arXiv:1802.00888v6 (2022), Sections 2, 3, 5 and 6; Automation and Remote
  Control 79(7), 1240-1254, 2018, DOI 10.1134/S0005117918070056. Theorem 1;
  eqs. (5), (42)-(44), (49), (52), (55)-(61), (63), (66), (67). The printed
  constant of (65) is a slip (see above).
- Kastner, F. and Rossler, A. "An analysis of approximation algorithms for iterated
  stochastic integrals and a Julia and Matlab simulation toolbox",
  arXiv:2201.08424 (2022), (64)-(67); Numerical Algorithms (2022),
  DOI 10.1007/s11075-022-01401-z.
- Kloeden, P. E. and Platen, E. "Numerical Solution of Stochastic Differential
  Equations", Springer, 1992, Sections 5.2, 5.8 and 10.4.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial import legendre

from discrecontinual_equations.solver.stochastic.wiener import (
    GaussianWienerSource,
    WienerIncrements,
    WienerSource,
)


@dataclass(frozen=True)
class IteratedIntegrals:
    """The multiple Ito integrals of one step over ``m`` channels, inner index first.

    References:
        Kuznetsov, D. F. To numerical modeling with strong orders 1.0, 1.5, and
        2.0 of convergence for multidimensional dynamical systems with random
        disturbances. arXiv:1802.00888v6 (2022), eq. (5) and its notation.
    """

    delta_w: np.ndarray
    """``I_(j) = W_j(t + h) - W_j(t)``, ``(m,)``."""

    delta_z: np.ndarray
    """``I_(j,0) = int_t^{t+h} (W_j(s) - W_j(t)) ds``, ``(m,)``."""

    double: np.ndarray
    """``I_(j,k) = int_t^{t+h} int_t^s dW_j(u) dW_k(s)``, ``(m, m)``."""

    triple: np.ndarray
    """``I_(j,k,l)``, ``(m, m, m)``: ``dW_j`` innermost, ``dW_l`` outermost."""


@lru_cache(maxsize=64)
def _legendre_triple(terms: int) -> np.ndarray:
    """``C_(p3,p2,p1)`` for a unit step, ``p <= terms``, indexed ``[p3, p2, p1]``."""
    nodes, weights = legendre.leggauss((3 * terms + 4) // 2 + 2)
    basis = np.eye(terms + 2)
    values = np.array([legendre.legval(nodes, basis[p]) for p in range(terms + 2)])
    # Q_p = int_{-1}^x P_p = (P_(p+1) - P_(p-1)) / (2 p + 1); Q_0 = x + 1.
    antiderivatives = np.empty((terms + 1, nodes.size))
    antiderivatives[0] = nodes + 1.0
    for p in range(1, terms + 1):
        antiderivatives[p] = (values[p + 1] - values[p - 1]) / (2 * p + 1)
    plain = values[: terms + 1]
    # bar[a, b, c] = -sum_n w_n Q_a(x_n) P_b(x_n) Q_c(x_n), as one product.
    products = antiderivatives[:, np.newaxis, :] * plain[np.newaxis, :, :]
    bar = -(products @ (weights * antiderivatives).T)
    bar[0] += 2.0 * np.einsum("n,bn,cn->bc", weights, plain, antiderivatives)
    scale = np.sqrt(2.0 * np.arange(terms + 1) + 1.0)
    coefficients = bar * np.einsum("a,b,c->abc", scale, scale, scale) / 8.0
    coefficients.setflags(write=False)
    return coefficients


class IteratedIntegralSampler:
    """Multiple Ito integrals given the increments, by Kuznetsov's Legendre series.

    See the module docstring for the formulas. :meth:`complete` takes the
    ``delta_w`` and ``delta_z`` a source drew and returns the integrals of the same
    path: ``zeta_0`` and ``zeta_1`` are recovered from them, the higher
    coefficients drawn here. Every method takes leading batch axes, so a fine
    path's cells can be drawn at once.

    References:
        Kuznetsov, D. F. To numerical modeling with strong orders 1.0, 1.5, and
        2.0 of convergence for multidimensional dynamical systems with random
        disturbances. arXiv:1802.00888v6 (2022), Theorem 1 and eqs. (42)-(44), (49),
        (52), (55)-(61).

        Kastner, F. and Rossler, A. An analysis of approximation algorithms for
        iterated stochastic integrals and a Julia and MATLAB simulation toolbox.
        Numerical Algorithms, 2022; arXiv:2201.08424, eq. (64).
    """

    __slots__ = ["_generator"]

    def __init__(self, generator: np.random.Generator) -> None:
        """Set up the sampler on a generator of the caller's.

        Args:
            generator: The stream the coefficients ``zeta_2, zeta_3, ...`` are
                drawn from.
        """
        self._generator = generator

    @staticmethod
    def coefficients(terms: int) -> np.ndarray:
        """``C_(p3,p2,p1)`` for ``h = 1`` and ``p <= terms``, indexed ``[p3, p2, p1]``.

        For a step ``h`` multiply by ``h^(3/2)``. Read-only and cached.

        Args:
            terms: The largest degree ``q``.

        Returns:
            The ``(q + 1, q + 1, q + 1)`` array of coefficients.

        References:
            Kuznetsov (2022), eqs. (56) and (57).
        """
        return _legendre_triple(terms)

    @staticmethod
    def double_error(terms: int) -> float:
        """Kuznetsov's (52) for ``h = 1``: ``E (I_(j,k) - I^q_(j,k))^2``, ``j != k``.

        Args:
            terms: The truncation ``q``.

        Returns:
            ``1 / (4 (2 q + 1))``; multiply by ``h^2`` for a step ``h``.
        """
        return 0.25 / (2 * terms + 1)

    @staticmethod
    @lru_cache(maxsize=256)
    def triple_errors(terms: int) -> dict[str, float]:
        """Kuznetsov's exact (49), (59)-(61) for ``h = 1``, by pattern of channels.

        Keys ``"distinct"``, ``"inner"`` (``j = k != l``), ``"outer"`` (``j != k =
        l``) and ``"ends"`` (``j = l != k``); ``j = k = l`` is exact.

        Args:
            terms: The truncation ``q``.

        Returns:
            The mean-square error by pattern; multiply by ``h^3`` for a step
            ``h``.
        """
        c = _legendre_triple(terms)
        rest = 1.0 / 6.0 - float(np.sum(c**2))
        return {
            "distinct": rest,
            "inner": rest - float(np.sum(c * c.transpose(0, 2, 1))),
            "outer": rest - float(np.sum(c * c.transpose(1, 0, 2))),
            "ends": rest - float(np.sum(c * c.transpose(2, 1, 0))),
        }

    @staticmethod
    def double_terms(step: float) -> int:
        """The least ``q >= 1`` with ``h^2 / (4 (2 q + 1)) <= h^4``: order 1.5.

        Args:
            step: The step ``h``.

        Returns:
            The truncation ``q_1`` of the twofold series.

        References:
            Kuznetsov (2022), eqs. (52) and (66).
        """
        bound = (0.25 / step**2 - 1.0) / 2.0
        return max(1, int(np.ceil(bound)))

    @staticmethod
    @lru_cache(maxsize=64)
    def triple_terms(step: float) -> int:
        """The least ``q`` whose worst exact error (49), (59)-(61) is ``<= h^4``.

        Args:
            step: The step ``h``.

        Returns:
            The truncation ``q_3`` of the threefold series.

        References:
            Kuznetsov (2022), eqs. (49), (59)-(61) and (67).
        """

        def worst(terms: int) -> float:
            return max(IteratedIntegralSampler.triple_errors(terms).values())

        # The worst error is about 1 / (8 q) for large q and below it for small
        # q, so the answer is near 0.13 / h: start there and walk to it.
        terms = max(0, int(0.12 / step) - 1)
        while worst(terms) > step:
            terms += 1
        while terms > 0 and worst(terms - 1) <= step:
            terms -= 1
        return terms

    @staticmethod
    def leading(delta_w: np.ndarray, delta_z: np.ndarray, step: float) -> np.ndarray:
        """``zeta_0`` and ``zeta_1`` from the increments, as ``(..., 2, m)``.

        Args:
            delta_w: The increments ``I_(j)``, ``(..., m)``.
            delta_z: The time integrals ``I_(j,0)``, ``(..., m)``.
            step: The step ``h``.

        Returns:
            The first two Legendre coefficients of each channel.

        References:
            Kuznetsov (2022), eqs. (42) and (43).
        """
        zeta0 = delta_w / np.sqrt(step)
        zeta1 = np.sqrt(3.0) * (zeta0 - 2.0 * delta_z / step**1.5)
        return np.stack([zeta0, zeta1], axis=-2)

    @staticmethod
    def assemble(
        zeta: np.ndarray,
        step: float,
        double_terms: int,
        triple_terms: int,
    ) -> IteratedIntegrals:
        """The integrals from given Legendre coefficients ``zeta``, ``(..., p, m)``.

        Needs ``p > max(double_terms, triple_terms, 1)``. Public so that a pathwise
        error can be measured against a reference built from the same ``zeta``.

        Args:
            zeta: The Legendre coefficients, ``(..., p, m)``.
            step: The step ``h``.
            double_terms: The truncation ``q_1`` of the twofold series.
            triple_terms: The truncation ``q_3`` of the threefold series.

        Returns:
            The integrals of the step.

        References:
            Kuznetsov (2022), eqs. (44), (55) and (58).
        """
        dimension = zeta.shape[-1]
        eye = np.eye(dimension)
        root = np.sqrt(step)
        zeta0 = zeta[..., 0, :]
        delta_w = root * zeta0
        delta_z = 0.5 * step * root * (zeta0 - zeta[..., 1, :] / np.sqrt(3.0))

        # Kuznetsov (44): the antisymmetric part pairs neighbouring coefficients.
        i = np.arange(1, double_terms + 1, dtype=float)
        weight = 1.0 / np.sqrt(4.0 * i**2 - 1.0)
        lower = zeta[..., :double_terms, :] * weight[:, np.newaxis]
        upper = zeta[..., 1 : double_terms + 1, :]
        series = np.einsum("...pj,...pk->...jk", lower, upper)
        double = (
            0.5
            * step
            * (
                np.einsum("...j,...k->...jk", zeta0, zeta0)
                + series
                - np.swapaxes(series, -1, -2)
                - eye
            )
        )

        # Kuznetsov (55): the full product less the three pairings of equal
        # channels at equal degrees, the Ito correction of a multiple series.
        c = _legendre_triple(triple_terms)
        x = zeta[..., : triple_terms + 1, :]
        count = triple_terms + 1
        # One index at a time, innermost first: O(q^3 m) per step, then O(q^2 m^2).
        first = np.matmul(c.reshape(count * count, count), x).reshape(
            (*x.shape[:-2], count, count, dimension),
        )
        second = np.matmul(np.swapaxes(first, -1, -2), x[..., np.newaxis, :, :])
        full = np.einsum("...cij,...ck->...ijk", second, x)
        inner = np.einsum("c,...ck->...k", np.einsum("cbb->c", c), x)
        outer = np.einsum("a,...ai->...i", np.einsum("bba->a", c), x)
        ends = np.einsum("b,...bj->...j", np.einsum("aba->b", c), x)
        triple = (
            full
            - eye[:, :, np.newaxis] * inner[..., np.newaxis, np.newaxis, :]
            - outer[..., :, np.newaxis, np.newaxis] * eye[np.newaxis, :, :]
            - eye[:, np.newaxis, :] * ends[..., np.newaxis, :, np.newaxis]
        )
        # Kuznetsov (58): one channel thrice is exact.
        diagonal = np.arange(dimension)
        triple[..., diagonal, diagonal, diagonal] = (zeta0**3 - 3.0 * zeta0) / 6.0
        return IteratedIntegrals(
            delta_w=delta_w,
            delta_z=delta_z,
            double=double,
            triple=step**1.5 * triple,
        )

    def complete(
        self,
        delta_w: np.ndarray,
        delta_z: np.ndarray,
        step: float,
        terms: tuple[int, int] | None = None,
    ) -> IteratedIntegrals:
        """The integrals of the path whose increments are ``delta_w``, ``delta_z``.

        ``terms = (q_1, q_3)`` defaults to :meth:`double_terms` and
        :meth:`triple_terms` for the step; ``max(q_1, q_3, 1) - 1`` new normals are
        drawn per channel. ``delta_w`` and ``delta_z`` come back unchanged.

        Args:
            delta_w: The increments, ``(..., m)``.
            delta_z: The time integrals ``I_(j,0)``, ``(..., m)``.
            step: The step ``h``.
            terms: ``(q_1, q_3)``, or ``None`` for the order 1.5 truncation.

        Returns:
            The integrals of the path those increments belong to.
        """
        q1, q3 = (
            (self.double_terms(step), self.triple_terms(step))
            if terms is None
            else terms
        )
        count = max(q1, q3, 1) + 1
        batch, dimension = delta_w.shape[:-1], delta_w.shape[-1]
        fresh = self._generator.standard_normal((*batch, count - 2, dimension))
        zeta = np.concatenate([self.leading(delta_w, delta_z, step), fresh], axis=-2)
        integrals = self.assemble(zeta, step, q1, q3)
        return IteratedIntegrals(
            delta_w=np.array(delta_w, dtype=float),
            delta_z=np.array(delta_z, dtype=float),
            double=integrals.double,
            triple=integrals.triple,
        )


class IteratedIntegralSource(ABC):
    """Supplies the multiple integrals up to multiplicity three for one step."""

    @abstractmethod
    def iterated_integrals(
        self,
        time: float,
        step: float,
        dimension: int,
    ) -> IteratedIntegrals:
        """The integrals over ``[time, time + step]`` for ``dimension`` channels.

        Args:
            time: The start of the step.
            step: The step ``h``.
            dimension: The number of channels.

        Returns:
            The step's integrals.
        """
        raise NotImplementedError


class GaussianIteratedIntegralSource(WienerSource, IteratedIntegralSource):
    """Fresh integrals, with the increments :class:`GaussianWienerSource` gives.

    :meth:`increments` is a :class:`~.wiener.GaussianWienerSource` of the same
    seed, number for number, so a solver that switches to this source changes
    nothing it drew before. :meth:`iterated_integrals` draws the step's increments
    from that same stream and completes them with an
    :class:`IteratedIntegralSampler` on a stream of its own (the second child of
    the seed; the Levy areas of the Gaussian source use the first). The Levy
    areas :meth:`increments` gives on request are the Gaussian source's, drawn
    independently of :meth:`iterated_integrals`; a scheme uses one or the other.

    References:
        Kuznetsov, D. F. To numerical modeling with strong orders 1.0, 1.5, and
        2.0 of convergence for multidimensional dynamical systems with random
        disturbances. arXiv:1802.00888v6 (2022), Section 5 and Algorithm 1 (Section 6).
    """

    __slots__ = ["_increments", "_sampler"]

    def __init__(self, seed: int | None = None) -> None:
        """Set up the increment stream and the sampler's stream.

        Args:
            seed: The seed of both; ``None`` for fresh entropy.
        """
        self._increments = GaussianWienerSource(seed)
        child = np.random.SeedSequence(seed).spawn(2)[1]
        self._sampler = IteratedIntegralSampler(np.random.default_rng(child))

    def increments(
        self,
        time: float,
        step: float,
        dimension: int,
        *,
        levy_area: bool = False,
    ) -> WienerIncrements:
        return self._increments.increments(time, step, dimension, levy_area=levy_area)

    def iterated_integrals(
        self,
        time: float,
        step: float,
        dimension: int,
    ) -> IteratedIntegrals:
        increments = self._increments.increments(time, step, dimension)
        return self._sampler.complete(increments.delta_w, increments.delta_z, step)


class IteratedIntegralPath(WienerSource, IteratedIntegralSource):
    """A path fixed in advance by every integral of every grid cell.

    The cells of width ``resolution`` each carry ``I_(j)``, ``I_(j,0)``,
    ``I_(j,k)`` and ``I_(j,k,l)``. A step of whole cells is read by Chen's
    relation, which holds for Ito integrals with the time index as well
    (Kloeden and Platen, Section 5.2): over ``[s, u]`` split at ``r``,
    ``I_a[s,u] = sum over the ways of cutting the multi-index a = (a', a'') of
    I_a'[s,r] I_a''[r,u]``. The reading is exact algebra on the cells, so the
    readings of two adjacent steps compose to the reading of their union, and every
    step size that divides the horizon sees the same path.

    :meth:`sample` draws the cells with :class:`IteratedIntegralSampler` at a
    fixed truncation ``cell_terms``, not the order-1.5 one for the cell width,
    which at a fine grid would be ``O(resolution^-2)`` normals per cell. The cells'
    truncation errors are independent and of mean zero, so over a step of ``N``
    cells they add up to ``N`` times a cell's: a mean-square error of ``h
    resolution / (4 (2 q + 1))`` in a twofold integral and at most ``h resolution^2
    / (8 q)`` in a threefold one, which is within the ``h^4`` that order 1.5 asks
    at step ``h`` while ``resolution <= 4 (2 q + 1) h^3``.

    References:
        Kloeden, P. E. and Platen, E. Numerical Solution of Stochastic
        Differential Equations. Springer, 1992, Section 5.2 (Chen's relation,
        as cited since DEQ-66).

        Kuznetsov, D. F. To numerical modeling with strong orders 1.0, 1.5, and
        2.0 of convergence for multidimensional dynamical systems with random
        disturbances. arXiv:1802.00888v6 (2022), eqs. (52) and (59)-(61) (the cells'
        truncation errors).
    """

    __slots__ = ["_delta_w", "_delta_z", "_double", "_resolution", "_start", "_triple"]

    def __init__(  # noqa: PLR0913 (where and how fine, and the four integrals)
        self,
        start: float,
        resolution: float,
        delta_w: np.ndarray,
        delta_z: np.ndarray,
        double: np.ndarray,
        triple: np.ndarray,
    ) -> None:
        """Hold a path given by its cells' integrals.

        Args:
            start: The time of the first node.
            resolution: The cell width.
            delta_w: ``I_(j)`` of each cell, ``(cells, m)``.
            delta_z: ``I_(j,0)`` of each cell, ``(cells, m)``.
            double: ``I_(j,k)`` of each cell, ``(cells, m, m)``.
            triple: ``I_(j,k,l)`` of each cell, ``(cells, m, m, m)``.

        Raises:
            ValueError: When the shapes do not agree.
        """
        cells, dimension = delta_w.shape
        m = dimension
        if (
            delta_z.shape != (cells, m)
            or double.shape != (cells, m, m)
            or triple.shape != (cells, m, m, m)
        ):
            message = (
                "cell integrals must have shapes (cells, m), (cells, m), "
                "(cells, m, m) and (cells, m, m, m)"
            )
            raise ValueError(message)
        self._start = float(start)
        self._resolution = float(resolution)
        self._delta_w = np.asarray(delta_w, dtype=float)
        self._delta_z = np.asarray(delta_z, dtype=float)
        self._double = np.asarray(double, dtype=float)
        self._triple = np.asarray(triple, dtype=float)

    @classmethod
    def sample(  # noqa: PLR0913 (the horizon, the grid, the channels and the draw)
        cls,
        start: float,
        end: float,
        resolution: float,
        dimension: int,
        seed: int | None = None,
        cell_terms: int = 4,
    ) -> "IteratedIntegralPath":
        """Draw a path on ``[start, end]`` with ``cell_terms`` Legendre terms a cell.

        Args:
            start: The start of the horizon.
            end: The end of the horizon.
            resolution: The cell width, which must divide the horizon.
            dimension: The number of channels.
            seed: The seed; ``None`` for fresh entropy.
            cell_terms: The truncation of both series in each cell.

        Returns:
            The path.

        Raises:
            ValueError: When the horizon is not a whole number of cells.
        """
        cells = round((end - start) / resolution)
        if cells < 1 or not np.isclose(start + cells * resolution, end):
            message = "the horizon must be a whole number of resolution cells"
            raise ValueError(message)
        generator = np.random.default_rng(seed)
        zeta = generator.standard_normal((cells, max(cell_terms, 1) + 1, dimension))
        integrals = IteratedIntegralSampler.assemble(
            zeta,
            resolution,
            cell_terms,
            cell_terms,
        )
        return cls(
            start=start,
            resolution=resolution,
            delta_w=integrals.delta_w,
            delta_z=integrals.delta_z,
            double=integrals.double,
            triple=integrals.triple,
        )

    @property
    def dimension(self) -> int:
        return self._delta_w.shape[1]

    @property
    def values(self) -> np.ndarray:
        """``W - W(start)`` at the grid nodes, ``(nodes, channels)``."""
        return np.vstack(
            [np.zeros((1, self.dimension)), np.cumsum(self._delta_w, axis=0)],
        )

    def _cell(self, time: float) -> int:
        offset = (time - self._start) / self._resolution
        cell = round(offset)
        if (
            not np.isclose(offset, cell, atol=1.0e-6)
            or cell < 0
            or cell > len(self._delta_w)
        ):
            message = f"time {time} is not a grid node of this path"
            raise ValueError(message)
        return cell

    def iterated_integrals(
        self,
        time: float,
        step: float,
        dimension: int,
    ) -> IteratedIntegrals:
        if dimension != self.dimension:
            message = (
                f"path has {self.dimension} channels, scheme asked for {dimension}"
            )
            raise ValueError(message)
        first, last = self._cell(time), self._cell(time + step)
        dw = self._delta_w[first:last]
        # Chen's relation cell by cell, as sums: the running W - W(time) and the
        # running twofold integral are each taken at the start of the cell.
        before = np.cumsum(dw, axis=0) - dw
        cell_double = self._double[first:last] + np.einsum("cj,ck->cjk", before, dw)
        double_before = np.cumsum(cell_double, axis=0) - cell_double
        triple = np.sum(
            self._triple[first:last]
            + np.einsum("cjk,cl->cjkl", double_before, dw)
            + np.einsum("cj,ckl->cjkl", before, self._double[first:last]),
            axis=0,
        )
        return IteratedIntegrals(
            delta_w=np.sum(dw, axis=0),
            delta_z=np.sum(
                self._delta_z[first:last] + before * self._resolution, axis=0
            ),
            double=np.sum(cell_double, axis=0),
            triple=triple,
        )

    def increments(
        self,
        time: float,
        step: float,
        dimension: int,
        *,
        levy_area: bool = False,
    ) -> WienerIncrements:
        integrals = self.iterated_integrals(time, step, dimension)
        area = None
        if levy_area:
            area = 0.5 * (integrals.double - integrals.double.T)
        return WienerIncrements(
            delta_w=integrals.delta_w,
            delta_z=integrals.delta_z,
            levy_area=area,
        )
