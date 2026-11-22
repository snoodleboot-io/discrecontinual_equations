"""Stability indices of heteroclinic cycles, and which cycle of a network wins.

A cycle in a network is never asymptotically stable: at a node shared with another
cycle, part of the unstable manifold leaves along the other cycle. What remains is
how much of a small neighbourhood of each connection the cycle still attracts. The
**local stability index** of Podvigina and Ashwin (2011, Definition 5 and section
2.3) measures it: with ``Sigma_eps`` the fraction of an ``eps``-ball about a point
of the connection whose orbits stay near the cycle and converge to it,

* ``sigma > 0`` when ``1 - Sigma_eps ~ eps^sigma``: almost all of a small ball is
  attracted, more so the smaller the ball;
* ``sigma < 0`` when ``Sigma_eps ~ eps^|sigma|``: a thin cusp of the ball is;
* ``+inf`` for a full neighbourhood and ``-inf`` for a set of measure zero.

The index is constant along a connection (their Theorem 2.2), so a cycle of ``m``
connections has ``m`` of them, and the cycle attracts a set of positive measure
(it is *fragmentarily asymptotically stable*) iff any is above ``-inf``, and a set
of full density (it is *predominantly asymptotically stable*) iff all are positive
(Lohse 2015; Castro and Lohse 2014, Theorem 2.6).

The indices are computed as in Garrido-da-Silva and Castro (2019) for
*quasi-simple* cycles - every connection one-dimensional, in a flow-invariant
subspace ``P_j``, all ``P_j`` of one dimension. Near ``xi_j`` the flow takes the
incoming section to the outgoing one by ``(w, z) -> (w^{c/e}, z_s w^{-t_s/e})``
(their eq. 3); in ``eta = ln`` coordinates that is linear, and the global map along
the connection only permutes the coordinates (their Assumption A), so the passage
from one incoming section to the next is a **basic transition matrix** (eq. 7)

    M_j = A_j [[a_j, 0, ..., 0], [b_{j,1}, 1, ..., 0], ..., [b_{j,nt}, 0, ..., 1]],

with ``a_j = c_j / e_j``, ``b_{j,s} = -t_{j,s} / e_j`` and ``A_j`` a permutation.
Whether ``M = M_{j-1} ... M_{j+1} M_j`` sends a positive-measure set of
``eta -> -inf`` is decided by its leading eigenvalue and eigenvector (their Lemma
3.2), and the index is the minimum of the function ``F^index`` over a few rows of
partial products (Theorems 3.4 and 3.10).

``A_j`` is found geometrically here: the directions leaving ``xi_j`` normal to
``P_j`` (the contracting line, then the transverse eigenvectors) are matched with
the directions arriving at ``xi_{j+1}`` normal to ``P_j`` (the expanding line,
then its transverse eigenvectors). For a cycle of type B they match in order; for
type C the contracting line arrives as a transverse one, which is the permutation
of Podvigina and Ashwin's ``[[b, 1], [a, 0]]``. A cycle whose directions do not
match one-to-one fails Assumption A and is refused rather than guessed at.

Directions are compared through orthogonal complements, which is exact when the
subspaces' complements are orthogonal - fixed-point subspaces of an orthogonal
symmetry group, i.e. every simple cycle in its natural coordinates. In oblique
coordinates a type B cycle still comes out unchanged (its contracting and
expanding lines coincide, and the rest is their complement), but a type C cycle
is refused, as is a node whose repeated transverse eigenvalue leaves its
eigenvectors undetermined.
"""

import math

import numpy as np
from scipy.linalg import null_space, orth

from discrecontinual_equations.continuation.heteroclinic_network import (
    HeteroclinicNetwork,
)

# Singular values below this fraction of the largest are treated as zero.
_RANK_TOLERANCE = 1.0e-10
# Two unit directions are the same line when |cos| is within this of one.
_ALIGNMENT_TOLERANCE = 1.0e-8
# A transverse eigenvalue is real when its imaginary part is below this, relative
# to the size of the Jacobian.
_REAL_TOLERANCE = 1.0e-9
# Entries of a transition matrix and components of an eigenvector are treated as
# zero below this, relative to the largest one.
_ZERO_TOLERANCE = 1.0e-12


def f_index(alpha: np.ndarray) -> float:
    """Stability index of ``{x : prod |x_i|^alpha_i < 1}`` at the origin.

    In ``eta = ln |x|`` coordinates the set is the half-space ``alpha . eta < 0``;
    this is its local stability index, ``F^+(alpha) - F^+(-alpha)`` with
    ``F^+ = +inf`` if every ``alpha_i >= 0``, ``0`` if ``sum alpha <= 0``, and
    ``-sum alpha / min alpha`` otherwise (Garrido-da-Silva and Castro 2019, Lemma
    3.9 and Appendix A). For two components it is Podvigina and Ashwin's
    ``f^index(alpha, beta)`` (2011, p. 905).
    """
    values = np.asarray(alpha, dtype=float)
    if not np.any(values):
        message = "the index of alpha = 0 is undefined: the set is empty"
        raise ValueError(message)
    return _f_plus(values) - _f_plus(-values)


def _f_plus(alpha: np.ndarray) -> float:
    lowest = float(np.min(alpha))
    total = float(np.sum(alpha))
    if lowest >= 0.0:
        return math.inf
    if total <= 0.0:
        return 0.0
    return -total / lowest


def stability_indices(matrices: list[np.ndarray]) -> np.ndarray:
    """Local stability index along each connection, from the transition matrices.

    ``matrices[j]`` takes the incoming section at node ``j`` to the one at node
    ``j + 1``; entry ``j`` of the result is the index along the connection *into*
    node ``j``. Garrido-da-Silva and Castro (2019):

    * all entries non-negative (no positive transverse eigenvalue): every index is
      ``+inf`` if the return matrix has an eigenvalue of modulus above one and
      ``-inf`` otherwise (Theorem 3.4);
    * otherwise, every index is ``-inf`` unless every return matrix ``M^(j)`` has a
      real leading eigenvalue above one with an eigenvector of one sign (Lemma 3.2,
      Lemma 3.5, Theorem 3.10(a)); if they all do,
      ``sigma_j = min(F(v^max_j), min F(rows of M_l ... M_j))`` over the nodes ``l``
      whose matrix has a negative entry, ``v^max_j`` being the row of the
      eigenvector matrix's inverse belonging to the leading eigenvalue (Theorem
      3.10(b)).
    """
    count = len(matrices)
    negative = [
        index for index, matrix in enumerate(matrices) if _has_negative_entry(matrix)
    ]
    returns = [_partial_product(matrices, (j - 1) % count, j) for j in range(count)]
    if not negative:
        attracting = float(np.max(np.abs(np.linalg.eigvals(returns[0])))) > 1.0
        return np.full(count, math.inf if attracting else -math.inf)
    leading = [pair for pair in (_leading_pair(matrix) for matrix in returns) if pair]
    if len(leading) < count:
        return np.full(count, -math.inf)
    indices = []
    for j, (_, row) in enumerate(leading):
        candidates = [f_index(_rounded(row))]
        for node in negative:
            product = _partial_product(matrices, node, j)
            candidates.extend(f_index(_rounded(row)) for row in product)
        indices.append(min(candidates))
    return np.array(indices)


def transition_matrices(
    network: HeteroclinicNetwork,
    cycle: tuple[str, ...],
) -> list[np.ndarray]:
    """The basic transition matrix of each node of ``cycle`` (GdS-C 2019, eq. 7).

    Rows and columns are ordered ``(w, z_1, ..., z_nt)`` at each node: the
    expanding direction, then the transverse eigenvectors in the order their
    eigenvalues are listed by :func:`transverse_eigenvalues`.
    """
    count = len(cycle)
    planes = _planes(network, cycle)
    spectrum = network.spectrum(cycle)
    pairs = _transverse(network, cycle, planes)
    values = [pair[0] for pair in pairs]
    transverse = [pair[1] for pair in pairs]
    matrices = []
    for j in range(count):
        following = (j + 1) % count
        normal = null_space(planes[j].T, rcond=_RANK_TOLERANCE)
        leaving = [_line_within(planes[j - 1], normal), *transverse[j]]
        arriving = [_line_within(planes[following], normal), *transverse[following]]
        permutation = _permutation(leaving, arriving, cycle, j)
        local = np.eye(len(leaving))
        local[0, 0] = spectrum.contracting[j] / spectrum.expanding[j]
        local[1:, 0] = -values[j] / spectrum.expanding[j]
        matrices.append(permutation @ local)
    return matrices


def transverse_eigenvalues(
    network: HeteroclinicNetwork,
    cycle: tuple[str, ...],
) -> list[np.ndarray]:
    """Every transverse eigenvalue ``t_{j,s}`` at each node of ``cycle``.

    These are the eigenvalues of the Jacobian induced on the complement of
    ``P_{j-1} + P_j``; :class:`CycleSpectrum` keeps only the largest of them. A
    quasi-simple cycle has the same number ``nt`` at every node.
    """
    pairs = _transverse(network, cycle, _planes(network, cycle))
    return [pair[0] for pair in pairs]


class CycleStability:
    """The transition matrices of one cycle and the stability index they imply.

    ``indices[j]`` is the local stability index along the connection into node
    ``j``, i.e. ``nodes[j - 1] -> nodes[j]``.
    """

    __slots__ = ["indices", "matrices", "nodes"]

    def __init__(self, nodes: tuple[str, ...], matrices: list[np.ndarray]) -> None:
        self.nodes = nodes
        self.matrices = matrices
        self.indices = stability_indices(matrices)

    @classmethod
    def of(
        cls,
        network: HeteroclinicNetwork,
        cycle: tuple[str, ...],
    ) -> "CycleStability":
        """The stability of ``cycle``, a cycle of ``network``."""
        return cls(tuple(cycle), transition_matrices(network, cycle))

    def index(self, source: str, target: str) -> float:
        """The local stability index along the connection ``source -> target``."""
        count = len(self.nodes)
        for j in range(count):
            if (self.nodes[j - 1], self.nodes[j]) == (source, target):
                return float(self.indices[j])
        message = f"{source} -> {target} is not a connection of cycle {self.nodes}"
        raise KeyError(message)

    def return_matrix(self, node: int) -> np.ndarray:
        """``M^(j) = M_{j-1} ... M_{j+1} M_j``: once round from node ``j``'s section."""
        count = len(self.matrices)
        return _partial_product(self.matrices, (node - 1) % count, node % count)

    @property
    def is_attractor(self) -> bool:
        """Some index above ``-inf``: a positive-measure set converges to the cycle.

        Garrido-da-Silva and Castro (2019) call this fragmentary asymptotic
        stability; with every index ``-inf`` the cycle attracts a null set.
        """
        return bool(np.any(self.indices > -math.inf))

    @property
    def is_predominantly_stable(self) -> bool:
        """Every index positive: the cycle attracts almost all of a neighbourhood.

        Predominant asymptotic stability, which for a cycle whose indices exist and
        are non-zero is equivalent to positive indices along every connection
        (Lohse 2015; Castro and Lohse 2014, Theorem 2.6).
        """
        return bool(np.all(self.indices > 0.0))

    @property
    def is_asymptotically_stable(self) -> bool:
        """Every index ``+inf``: a full neighbourhood is attracted."""
        return bool(np.all(self.indices == math.inf))

    @property
    def delta(self) -> float:
        """Kirk and Silber's ``delta``: the sign that decides a B-type cycle's fate.

        For a cycle with one transverse direction and one node ``q`` whose
        transverse eigenvalue is positive, the return matrix from the next node is
        ``[[rho, 0], [delta, 1]]`` and its leading eigenvector is
        ``(rho - 1, delta)``; with ``rho > 1`` the cycle attracts a set of positive
        measure iff ``delta > 0``. For the Kirk-Silber network this is Castro and
        Lohse's (2014) eq. (3), ``delta = b1 a2 a3 + b3 a2 + b2`` (Lemma 5.1).
        """
        negative = [
            index
            for index, matrix in enumerate(self.matrices)
            if _has_negative_entry(matrix)
        ]
        if self.matrices[0].shape != (2, 2) or len(negative) != 1:
            message = (
                "delta is defined for a cycle with one transverse direction and "
                f"exactly one node with a positive transverse eigenvalue; this one "
                f"has {self.matrices[0].shape[0] - 1} and {len(negative)}"
            )
            raise ValueError(message)
        following = self.return_matrix(negative[0] + 1)
        if following[0, 1] != 0.0:
            message = "delta is defined for type B cycles, whose return is triangular"
            raise ValueError(message)
        return float(following[1, 0])


class NetworkCompetition:
    """Every cycle of a network, and which of them attract anything at all.

    ``stabilities`` maps each elementary cycle to its :class:`CycleStability`. A
    cycle whose indices are all ``-inf`` attracts a null set and loses the
    competition; the others are ``survivors``. Kirk and Silber (1994) and Castro
    and Lohse (2014, Proposition 5.3) give networks in which exactly one cycle
    survives although neither cycle is asymptotically stable.
    """

    __slots__ = ["stabilities"]

    def __init__(self, network: HeteroclinicNetwork) -> None:
        self.stabilities = {
            cycle: CycleStability.of(network, cycle) for cycle in network.cycles()
        }

    @property
    def survivors(self) -> list[tuple[str, ...]]:
        """The cycles that attract a set of positive measure."""
        return [
            cycle
            for cycle, stability in self.stabilities.items()
            if stability.is_attractor
        ]

    @property
    def winner(self) -> tuple[str, ...] | None:
        """The only surviving cycle, or ``None`` when none or several survive."""
        survivors = self.survivors
        return survivors[0] if len(survivors) == 1 else None


def _planes(network: HeteroclinicNetwork, cycle: tuple[str, ...]) -> list[np.ndarray]:
    by_key = {connection.key: connection for connection in network.connections}
    planes = []
    for j, source in enumerate(cycle):
        key = (source, cycle[(j + 1) % len(cycle)])
        if key not in by_key:
            message = f"no connection {key[0]} -> {key[1]} in the network"
            raise KeyError(message)
        planes.append(by_key[key].basis)
    dimensions = {plane.shape[1] for plane in planes}
    if len(dimensions) != 1:
        message = (
            f"cycle {cycle} is not quasi-simple: its connections lie in subspaces "
            f"of different dimensions {sorted(dimensions)}"
        )
        raise ValueError(message)
    return planes


def _transverse(
    network: HeteroclinicNetwork,
    cycle: tuple[str, ...],
    planes: list[np.ndarray],
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Transverse eigenvalues and directions at every node.

    With every ``P_j`` of one dimension ``d`` and consecutive ones differing by one
    direction (checked in :func:`_line_within`), each node has ``n - d - 1``.
    """
    return [_transverse_pairs(network, cycle, planes, j) for j in range(len(cycle))]


def _transverse_pairs(
    network: HeteroclinicNetwork,
    cycle: tuple[str, ...],
    planes: list[np.ndarray],
    node: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Eigenvalues and unit eigenvectors of the map induced off ``P_{j-1} + P_j``."""
    jacobian = network.saddle(cycle[node]).jacobian
    spanned = orth(np.hstack([planes[node - 1], planes[node]]), rcond=_RANK_TOLERANCE)
    complement = null_space(spanned.T, rcond=_RANK_TOLERANCE)
    if complement.shape[1] == 0:
        return np.zeros(0), []
    values, vectors = np.linalg.eig(complement.T @ jacobian @ complement)
    scale = max(1.0, float(np.linalg.norm(jacobian)))
    if np.any(np.abs(values.imag) > _REAL_TOLERANCE * scale):
        message = (
            f"complex transverse eigenvalues {values.tolist()} at {cycle[node]!r}: "
            f"the transition matrices need real ones"
        )
        raise ValueError(message)
    directions = [complement @ vectors[:, s].real for s in range(len(values))]
    return values.real, [d / np.linalg.norm(d) for d in directions]


def _line_within(plane: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """The one direction of ``plane`` outside the plane ``normal`` complements."""
    projected = orth(normal @ (normal.T @ plane), rcond=_RANK_TOLERANCE)
    if projected.shape[1] != 1:
        message = (
            "not quasi-simple: consecutive connection subspaces must differ by "
            f"exactly one direction, but these differ by {projected.shape[1]}"
        )
        raise ValueError(message)
    return projected[:, 0]


def _permutation(
    leaving: list[np.ndarray],
    arriving: list[np.ndarray],
    cycle: tuple[str, ...],
    node: int,
) -> np.ndarray:
    """``A_j``: which arriving direction each leaving direction becomes."""
    cosines = np.abs(np.array(arriving) @ np.array(leaving).T)
    permutation = (cosines > 1.0 - _ALIGNMENT_TOLERANCE).astype(float)
    if np.all(permutation.sum(axis=0) == 1.0) and np.all(
        permutation.sum(axis=1) == 1.0
    ):
        return permutation
    following = cycle[(node + 1) % len(cycle)]
    message = (
        f"along {cycle[node]} -> {following} the directions normal to the connection "
        f"at the two saddles are not a permutation of each other, so the global map "
        f"is not a rescaled permutation (Assumption A of Garrido-da-Silva and Castro "
        f"2019) - or a repeated transverse eigenvalue left its eigenvectors "
        f"undetermined"
    )
    raise ValueError(message)


def _partial_product(matrices: list[np.ndarray], last: int, first: int) -> np.ndarray:
    """``M_last ... M_first``, cyclically: ``first`` applied first."""
    count = len(matrices)
    product = np.eye(matrices[0].shape[0])
    for step in range((last - first) % count + 1):
        product = matrices[(first + step) % count] @ product
    return product


def _has_negative_entry(matrix: np.ndarray) -> bool:
    return bool(np.any(matrix < -_ZERO_TOLERANCE * np.max(np.abs(matrix))))


def _rounded(vector: np.ndarray) -> np.ndarray:
    """``vector`` with components below rounding level, relative to its largest, zeroed.

    ``f_index`` depends on the signs of the components, and an exact zero computed
    as ``-1e-17`` would turn ``+inf`` into a meaningless large finite index.
    """
    scale = float(np.max(np.abs(vector)))
    return np.where(np.abs(vector) > _ZERO_TOLERANCE * scale, vector, 0.0)


def _leading_pair(matrix: np.ndarray) -> tuple[float, np.ndarray] | None:
    """``(lambda_max, v_max)`` if Lemma 3.2's conditions hold, else ``None``.

    ``lambda_max`` is the eigenvalue of largest modulus; it must be real and above
    one, with an eigenvector ``w`` whose components share a sign. (Lemma 3.2 takes
    the largest modulus other than one; when the largest is one or below, the
    eigenvalue it would take is below one too, and the conditions fail either
    way.) ``v_max`` is the left eigenvector scaled so ``v . w = 1`` with
    ``w > 0``: the row of the inverse eigenvector matrix that gives ``w``'s
    coefficient.
    """
    values, vectors = np.linalg.eig(matrix)
    top = int(np.argmax(np.abs(values)))
    value = values[top]
    if abs(value.imag) > _ZERO_TOLERANCE * abs(value) or value.real <= 1.0:
        return None
    right = vectors[:, top].real
    right = right / right[np.argmax(np.abs(right))]
    if np.any(right <= _ZERO_TOLERANCE):
        return None
    left_values, left_vectors = np.linalg.eig(matrix.T)
    left = left_vectors[:, np.argmin(np.abs(left_values - value))].real
    return float(value.real), left / float(left @ right)
