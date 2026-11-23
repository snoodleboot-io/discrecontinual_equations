"""Heteroclinic networks: saddles, the connections between them, and their cycles.

A robust heteroclinic cycle joins saddles ``xi_1 -> ... -> xi_m -> xi_1`` by
connections that each lie in a flow-invariant subspace ``P_j`` in which
``xi_{j+1}`` is a sink, so every connection is a saddle-sink connection and
survives any perturbation that keeps the subspaces invariant (Krupa and Melbourne
1995, hypothesis (H1) and Proposition 2.5). Several cycles sharing saddles or
connections form a network.

This module is the bookkeeping layer over the projection boundary-value solver in
``connecting_orbit.py``:

* **saddles** are refined by Newton's method and carry their Jacobians;
* **connections** name a source, a target and the invariant subspace between
  them. Each is solved as a :class:`HeteroclinicOrbit` *restricted to the
  subspace* - in an orthonormal basis of it, then lifted back - where it is a
  saddle-sink connection and isolated, as Beyn's projection method needs (IMA
  J. Numer. Anal. 10, 1990). In the full space it need not be: when the source's
  unstable manifold is 2-D the full-space problem has a family of solutions, and
  in non-orthogonal coordinates the full-space solve does not find the one in the
  plane. The full-space solve is still run, uninformed of the subspace, as a
  check; and every returned path must satisfy the full field's collocation
  equations, which is what refuses a "connection" in a plane the flow leaves;
* **cycles** are the elementary cycles of the directed graph of connections;
* each cycle's **spectrum** classifies the eigenvalues at every node by the role
  they play *for that cycle* - radial, contracting, expanding, transverse - which
  is what the stability of the cycle and the growth of its passage times depend
  on. A saddle shared by two cycles can play different roles in each.

The classification follows Krupa and Melbourne (1995), section 2.3: with
``L_j = P_{j-1} cap P_j``, ``-r_j`` is the largest real part of an eigenvalue of
``Df(xi_j)`` on ``L_j``, ``-c_j`` the largest of the remaining eigenvalues on
``P_{j-1}``, ``e_j`` the largest of the remaining ones on ``P_j``, and ``t_j`` the
largest on the complement of ``P_{j-1} + P_j`` (``-inf`` when that is empty). Each
set is read off the Jacobian *restricted to an invariant subspace* (or induced on
the quotient by one) rather than by sorting eigenvectors, so a repeated eigenvalue
whose eigenvectors span several roles is still assigned correctly.
"""

import math

import numpy as np
from scipy.linalg import null_space, orth

from discrecontinual_equations.continuation.connecting_orbit import (
    CollocationResidual,
    HeteroclinicOrbit,
    MeshSpec,
    OrbitResolution,
    OrbitSolution,
    Terminus,
)
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.function.function import Function
from discrecontinual_equations.variable import Variable

_AUTODIFF = AutomaticDifferentiation()
_NEWTON_TOLERANCE = 1.0e-13
_NEWTON_ITERATIONS = 50
# A subspace is invariant for a Jacobian when J maps it into itself; this is the
# largest out-of-subspace component allowed, relative to the Jacobian's size.
_INVARIANCE_TOLERANCE = 1.0e-8
# Singular values below this fraction of the largest are treated as zero when
# intersecting and complementing subspaces.
_RANK_TOLERANCE = 1.0e-10
# The default seed's sigmoid rate. The connection itself sets the true rate; this
# only has to put the seed's transition inside the interval.
_SEED_RATE = 0.5


def coordinate_plane(dimension: int, indices: tuple[int, ...]) -> np.ndarray:
    """Orthonormal basis (as columns) of the subspace spanned by coordinate axes.

    The fixed-point subspace of a set of coordinate reflections is of this form,
    which makes it the invariant subspace of every reflection-symmetric system.
    """
    return np.eye(dimension)[:, list(indices)]


class NetworkSaddle:
    """A vertex of the network: its name, position, and Jacobian there."""

    __slots__ = ["equilibrium", "jacobian", "name"]

    def __init__(
        self,
        name: str,
        equilibrium: np.ndarray,
        jacobian: np.ndarray,
    ) -> None:
        self.name = name
        self.equilibrium = equilibrium
        self.jacobian = jacobian


class NetworkConnection:
    """An edge of the network: ``source -> target`` inside an invariant subspace."""

    __slots__ = ["basis", "source", "target"]

    def __init__(self, source: str, target: str, basis: np.ndarray) -> None:
        self.source = source
        self.target = target
        self.basis = orth(basis)

    @property
    def key(self) -> tuple[str, str]:
        """``(source, target)``, unique within a network."""
        return (self.source, self.target)


class FullSpaceCheck:
    """The same connection solved in the full space, not told about the subspace.

    ``subspace_defect`` is its largest distance from the subspace and
    ``residual`` its collocation residual under the full field. Where the
    connection is isolated in the full space this lands on the subspace solve;
    where it is not, or where the subspace is not invariant along the orbit, the
    two disagree, and these say how.
    """

    __slots__ = ["residual", "solution", "subspace_defect"]

    def __init__(
        self,
        solution: OrbitSolution,
        subspace_defect: float,
        residual: CollocationResidual,
    ) -> None:
        self.solution = solution
        self.subspace_defect = subspace_defect
        self.residual = residual


class ConnectionOrbit:
    """A solved connection with the numbers that say whether to trust it.

    ``solution`` is the connection solved inside its subspace and lifted to the
    full space; ``subspace_defect`` is the largest distance of any of its nodes
    from the subspace, which is rounding by construction. ``residual`` is its
    collocation residual under the *full* field (see :class:`CollocationResidual`):
    the subspace solve sees only the field's component in the subspace, so this is
    what says the path is a trajectory of the actual flow. ``resolution`` is the
    solver's own error estimate in mesh spacing and interval length (see
    :class:`OrbitResolution`), and ``full_space`` the check solve.
    """

    __slots__ = [
        "connection",
        "full_space",
        "residual",
        "resolution",
        "solution",
        "subspace_defect",
    ]

    def __init__(  # noqa: PLR0913 - the path and each of its separate checks
        self,
        connection: NetworkConnection,
        solution: OrbitSolution,
        subspace_defect: float,
        resolution: OrbitResolution,
        *,
        residual: CollocationResidual,
        full_space: FullSpaceCheck,
    ) -> None:
        self.connection = connection
        self.solution = solution
        self.subspace_defect = subspace_defect
        self.resolution = resolution
        self.residual = residual
        self.full_space = full_space


class NotATrajectoryError(ValueError):
    """A solved path whose collocation residual its mesh cannot explain.

    ``orbit`` is the refused :class:`ConnectionOrbit`, with every number that was
    measured on it, for diagnosis.
    """

    def __init__(self, message: str, orbit: ConnectionOrbit) -> None:
        super().__init__(message)
        self.orbit = orbit


class _Restricted(DeterministicFunction):
    """``g(z) = Q^T f(Q z)``: the field on an invariant subspace, in its basis ``Q``.

    ``Q`` has orthonormal columns. On a subspace the flow leaves, ``g`` drops the
    component of ``f`` that leaves, which is why a path solved with it must be
    checked against ``f`` itself. Written with ``float * x`` and ``+`` only, so
    Taylor jets pass through and automatic differentiation stays exact.
    """

    __slots__ = ["_basis", "_field"]

    def __init__(self, field: Function, basis: np.ndarray) -> None:
        size = basis.shape[1]
        super().__init__(
            variables=[Variable() for _ in range(size)],
            parameters=field.parameters,
            results=[Variable() for _ in range(size)],
            time=None,
        )
        self._field = field
        self._basis = basis

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        """``Q^T f(Q z)``."""
        rows, columns = self._basis.shape
        state = [_combine(self._basis[row], point) for row in range(rows)]
        rate = self._field.eval(state, time)
        return [_combine(self._basis[:, column], rate) for column in range(columns)]


class _ReducedOrbit(HeteroclinicOrbit):
    """A heteroclinic orbit solved in subspace coordinates, measured in full ones.

    The resolution estimate compares paths component by component, and a
    component of the reduced coordinates is not one of the full space; lifting
    first keeps the estimate in the coordinates the caller reads the orbit in.
    """

    __slots__ = ["_basis"]

    def __init__(
        self,
        function: Function,
        source: Terminus,
        target: Terminus,
        mesh: MeshSpec,
        basis: np.ndarray,
    ) -> None:
        super().__init__(function, source, target, mesh)
        self._basis = basis

    def _shift(self, moved: np.ndarray, original: np.ndarray) -> float:
        return float(np.max(np.abs((moved - original) @ self._basis.T)))


class CycleSpectrum:
    """The eigenvalue roles at every node of one cycle, and what they imply.

    Arrays are indexed by node in the order the cycle was given. ``radial``,
    ``contracting`` and ``expanding`` are the positive magnitudes ``r_j``, ``c_j``
    and ``e_j``; ``transverse`` is the signed ``t_j`` (``-inf`` when the cycle's two
    subspaces at a node fill the whole space). Notation of Krupa and Melbourne
    (1995), section 2.3.
    """

    __slots__ = ["contracting", "expanding", "nodes", "radial", "transverse"]

    def __init__(
        self,
        nodes: tuple[str, ...],
        roles: dict[str, np.ndarray],
    ) -> None:
        self.nodes = nodes
        self.radial = roles["radial"]
        self.contracting = roles["contracting"]
        self.expanding = roles["expanding"]
        self.transverse = roles["transverse"]

    @property
    def stability_ratio(self) -> float:
        """``rho = prod c_j / prod e_j``: the return map contracts like ``w^rho``.

        For a cycle in ``R^3`` (no transverse directions) ``rho > 1`` is necessary
        and sufficient for asymptotic stability, generically (Krupa and Melbourne
        1995, Theorem 2.9 and Remark 2.10(b)).
        """
        return float(np.prod(self.contracting) / np.prod(self.expanding))

    @property
    def is_asymptotically_stable(self) -> bool:
        """Krupa and Melbourne (1995) Theorem 2.7, with its standing ``t_j < 0``.

        Asymptotically stable if ``prod min(c_j, e_j - t_j) > prod e_j``. A positive
        transverse eigenvalue means part of a node's unstable manifold leaves the
        cycle - as it does at every shared node of a network - and the cycle cannot
        be asymptotically stable, so that case is ``False``. The condition is
        sufficient; it is also necessary for the cycles of Theorem 2.9, which
        include every cycle in ``R^3``.
        """
        if np.any(self.transverse >= 0.0):
            return False
        weakest = np.minimum(self.contracting, self.expanding - self.transverse)
        return bool(np.prod(weakest) > np.prod(self.expanding))

    @property
    def residence_ratios(self) -> np.ndarray:
        """Predicted growth of the passage time from each node to the next.

        Entry ``j`` is ``c_j / e_{j+1}``: the time spent near node ``j + 1`` divided
        by the time spent near node ``j``, in the limit of a trajectory converging
        to the cycle (Gaunersdorfer 1992, Lemma 1). Their product is
        ``stability_ratio``, the growth per revolution (Remark 2 there).
        """
        return self.contracting / np.roll(self.expanding, -1)


class HeteroclinicNetwork:
    """Saddles and connections of one vector field, and the cycles they form."""

    __slots__ = ["_connections", "_function", "_saddles"]

    def __init__(self, function: Function) -> None:
        self._function = function
        self._saddles: dict[str, NetworkSaddle] = {}
        self._connections: dict[tuple[str, str], NetworkConnection] = {}

    @property
    def function(self) -> Function:
        """The vector field whose saddles and connections these are."""
        return self._function

    @property
    def saddles(self) -> list[NetworkSaddle]:
        """Every saddle, in the order added."""
        return list(self._saddles.values())

    @property
    def connections(self) -> list[NetworkConnection]:
        """Every connection, in the order added."""
        return list(self._connections.values())

    def saddle(self, name: str) -> NetworkSaddle:
        """The saddle called ``name``."""
        if name not in self._saddles:
            message = f"no saddle named {name!r}; known: {sorted(self._saddles)}"
            raise KeyError(message)
        return self._saddles[name]

    def add_saddle(self, name: str, guess: np.ndarray) -> NetworkSaddle:
        """Refine ``guess`` to an equilibrium by Newton's method and add it."""
        if name in self._saddles:
            message = f"a saddle named {name!r} is already in the network"
            raise ValueError(message)
        saddle = self._refine(name, np.asarray(guess, dtype=float))
        self._saddles[name] = saddle
        return saddle

    def add_connection(
        self,
        source: str,
        target: str,
        basis: np.ndarray,
    ) -> NetworkConnection:
        """Add ``source -> target`` inside the subspace spanned by ``basis`` columns.

        The subspace must be invariant for the Jacobian at both ends, which it is
        whenever it is invariant for the flow; a subspace that is not is rejected
        here rather than producing a meaningless spectrum later.
        """
        for name in (source, target):
            self.saddle(name)
        connection = NetworkConnection(source, target, basis)
        if connection.key in self._connections:
            message = f"the network already has a connection {source} -> {target}"
            raise ValueError(message)
        for name in (source, target):
            _require_invariant(self.saddle(name), connection.basis)
        self._connections[connection.key] = connection
        return connection

    def refresh(self) -> None:
        """Re-solve every saddle from where it was, after the field has changed."""
        for name, saddle in self._saddles.items():
            self._saddles[name] = self._refine(name, saddle.equilibrium)

    def cycles(self) -> list[tuple[str, ...]]:
        """Every elementary cycle, each starting at its earliest-added saddle.

        A depth-first search from each saddle that only visits later-added saddles
        finds each cycle exactly once. Networks here have a handful of saddles, so
        the exponential worst case of enumerating cycles is not a concern.
        """
        order = {name: index for index, name in enumerate(self._saddles)}
        found: list[tuple[str, ...]] = []
        for start in self._saddles:
            self._extend([start], order, found)
        return sorted(found, key=lambda cycle: [order[name] for name in cycle])

    def spectrum(self, cycle: tuple[str, ...]) -> CycleSpectrum:
        """Classify the eigenvalues at each node of ``cycle`` by their role in it."""
        count = len(cycle)
        planes = [
            self._connection(cycle[j], cycle[(j + 1) % count]).basis
            for j in range(count)
        ]
        roles: dict[str, list[float]] = {
            "radial": [],
            "contracting": [],
            "expanding": [],
            "transverse": [],
        }
        for j, name in enumerate(cycle):
            jacobian = self.saddle(name).jacobian
            incoming, outgoing = planes[j - 1], planes[j]
            line = _intersection(incoming, outgoing)
            roles["radial"].append(-_leading(jacobian, line))
            roles["contracting"].append(
                -_leading(jacobian, _complement_within(incoming, line)),
            )
            roles["expanding"].append(
                _leading(jacobian, _complement_within(outgoing, line)),
            )
            spanned = orth(np.hstack([incoming, outgoing]))
            roles["transverse"].append(_leading(jacobian, null_space(spanned.T)))
        return CycleSpectrum(
            tuple(cycle),
            {role: np.array(values) for role, values in roles.items()},
        )

    def connection_margins(self, connection: NetworkConnection) -> tuple[float, float]:
        """``(expansion at the source, contraction at the target)`` in the subspace.

        The connection is a robust saddle-sink connection while both are positive:
        the source must have an unstable direction inside the subspace and the
        target must attract within it (Krupa and Melbourne 1995, (H1)). Either
        reaching zero is where the connection can break.
        """
        basis = connection.basis
        source = self.saddle(connection.source).jacobian
        target = self.saddle(connection.target).jacobian
        return (_leading(source, basis), -_leading(target, basis))

    def persistence_margin(self, connection: NetworkConnection) -> float:
        """The smaller of :meth:`connection_margins`; negative means no connection."""
        return min(self.connection_margins(connection))

    def mesh(
        self,
        connection: NetworkConnection,
        half_length: float,
        intervals: int,
    ) -> MeshSpec:
        """A mesh whose phase condition pins the orbit half-way between its ends.

        The phase coordinate is the one that changes most from source to target, and
        its value at the centre node is the midpoint of that change.
        """
        source = self.saddle(connection.source).equilibrium
        target = self.saddle(connection.target).equilibrium
        index = int(np.argmax(np.abs(target - source)))
        return MeshSpec(
            half_length=half_length,
            intervals=intervals,
            phase_index=index,
            phase_value=float(0.5 * (source[index] + target[index])),
        )

    def seed(self, connection: NetworkConnection, mesh: MeshSpec) -> np.ndarray:
        """A sigmoid from source to target along the chord; crude but in the plane."""
        source = self.saddle(connection.source).equilibrium
        target = self.saddle(connection.target).equilibrium
        times = np.linspace(-mesh.half_length, mesh.half_length, mesh.intervals + 1)
        weight = 0.5 * (1.0 + np.tanh(_SEED_RATE * times))
        return np.outer(1.0 - weight, source) + np.outer(weight, target)

    def solve(
        self,
        connection: NetworkConnection,
        mesh: MeshSpec,
        seed: np.ndarray | None = None,
    ) -> ConnectionOrbit:
        """Solve the connection inside its subspace, and check it in the full space.

        The boundary-value problem is posed in an orthonormal basis of the subspace
        whose first vector is the phase coordinate's projection onto it, so the
        full space's phase condition ``u_c[i] = v`` is ``z_c[0] = v / |B^T e_i|``
        there and the lifted orbit satisfies the same one. In the subspace the
        departure end is projected out of the source's stable eigenspace and the
        arrival end out of the target's unstable one (Beyn 1990; the conditions
        ``L^-(u(tau_-) - xi) = 0`` of Doedel et al. 2008, eq. 5), and the target is
        a sink, so the connection is isolated even where, in the full space, the
        source's unstable manifold is 2-D and the connections leaving it are not.

        The full-space solve from the same seed is kept as ``full_space``. The
        resolution estimate is the subspace solve's, measured in full coordinates.

        A connection whose :meth:`persistence_margin` is not positive does not exist
        in its subspace, and is refused: the least-squares solve would otherwise
        return its best non-trajectory, which can sit on both saddles, in the
        plane, with a small ``resolution`` estimate. A solved path whose collocation
        residual under the full field exceeds the mesh's truncation error is not a
        trajectory either - the subspace is not invariant along it - and is refused
        with :class:`NotATrajectoryError`, which carries it.
        """
        margin = self.persistence_margin(connection)
        if margin <= 0.0:
            message = (
                f"no connection {connection.source} -> {connection.target} in its "
                f"subspace: persistence margin {margin:.6g} is not positive (the "
                f"source must expand into it and the target attract within it)"
            )
            raise ValueError(message)
        source = self.saddle(connection.source)
        target = self.saddle(connection.target)
        basis, weight = _phase_basis(connection.basis, mesh.phase_index)
        reduced = _ReducedOrbit(
            _Restricted(self._function, basis),
            Terminus(basis.T @ source.equilibrium, basis.T @ source.jacobian @ basis),
            Terminus(basis.T @ target.equilibrium, basis.T @ target.jacobian @ basis),
            MeshSpec(mesh.half_length, mesh.intervals, 0, mesh.phase_value / weight),
            basis,
        )
        full = HeteroclinicOrbit(
            self._function,
            Terminus(source.equilibrium, source.jacobian),
            Terminus(target.equilibrium, target.jacobian),
            mesh,
        )
        start = self.seed(connection, mesh) if seed is None else seed
        inside = reduced.solve(start @ basis)
        lifted = OrbitSolution(inside.times, inside.states @ basis.T)
        check = full.solve(start)
        orbit = ConnectionOrbit(
            connection,
            lifted,
            _subspace_defect(lifted.states, connection.basis),
            reduced.estimate_orbit_error(inside),
            residual=full.collocation_residual(lifted),
            full_space=FullSpaceCheck(
                check,
                _subspace_defect(check.states, connection.basis),
                full.collocation_residual(check),
            ),
        )
        if not orbit.residual.is_trajectory:
            message = (
                f"{connection.source} -> {connection.target} is not a trajectory: "
                f"its collocation residual {orbit.residual.largest:.3g} exceeds the "
                f"mesh's truncation error {orbit.residual.tolerance:.3g}, so the flow "
                f"leaves the subspace along it and no connection lies there"
            )
            raise NotATrajectoryError(message, orbit)
        return orbit

    def _connection(self, source: str, target: str) -> NetworkConnection:
        key = (source, target)
        if key not in self._connections:
            message = f"no connection {source} -> {target} in the network"
            raise KeyError(message)
        return self._connections[key]

    def _extend(
        self,
        path: list[str],
        order: dict[str, int],
        found: list[tuple[str, ...]],
    ) -> None:
        """Depth-first extension of ``path``; record it when it closes on its start."""
        for source, target in self._connections:
            if source != path[-1]:
                continue
            if target == path[0]:
                found.append(tuple(path))
            elif order[target] > order[path[0]] and target not in path:
                self._extend([*path, target], order, found)

    def _refine(self, name: str, guess: np.ndarray) -> NetworkSaddle:
        state = guess.copy()
        for _ in range(_NEWTON_ITERATIONS):
            residual = np.array(self._function.eval(list(state), None), dtype=float)
            if float(np.linalg.norm(residual)) < _NEWTON_TOLERANCE:
                jacobian = _AUTODIFF.jacobian(self._function, state, 0.0)
                return NetworkSaddle(name, state, np.asarray(jacobian, dtype=float))
            jacobian = _AUTODIFF.jacobian(self._function, state, 0.0)
            state = state - np.linalg.solve(jacobian, residual)
        message = (
            f"Newton did not converge to an equilibrium for saddle {name!r} from "
            f"{guess.tolist()} within {_NEWTON_ITERATIONS} iterations"
        )
        raise ValueError(message)


def _require_invariant(saddle: NetworkSaddle, basis: np.ndarray) -> None:
    """Reject a subspace the Jacobian at ``saddle`` does not map into itself."""
    image = saddle.jacobian @ basis
    outside = image - basis @ (basis.T @ image)
    scale = max(1.0, float(np.linalg.norm(saddle.jacobian)))
    if float(np.linalg.norm(outside)) > _INVARIANCE_TOLERANCE * scale:
        message = (
            f"the subspace is not invariant for the Jacobian at {saddle.name!r}; a "
            f"robust connection needs a flow-invariant subspace"
        )
        raise ValueError(message)


def _phase_basis(basis: np.ndarray, index: int) -> tuple[np.ndarray, float]:
    """An orthonormal basis of the subspace led by the phase coordinate's projection.

    Returns the basis and ``|B^T e_index|``, the length of that projection. The
    phase coordinate of a point ``Q z`` of the subspace is then that length times
    ``z[0]``. A coordinate orthogonal to the subspace cannot pin an orbit in it.
    """
    weight = float(np.linalg.norm(basis[index]))
    if weight < _RANK_TOLERANCE:
        message = (
            f"the phase coordinate {index} is orthogonal to the connection's "
            f"subspace, so it cannot pin an orbit there; choose another"
        )
        raise ValueError(message)
    first = (basis @ basis[index] / weight)[:, None]
    return np.hstack([first, _complement_within(basis, first)]), weight


def _subspace_defect(states: np.ndarray, basis: np.ndarray) -> float:
    """Largest distance of any state from the subspace spanned by ``basis``."""
    outside = states - (states @ basis) @ basis.T
    return float(np.max(np.linalg.norm(outside, axis=1)))


def _combine(weights: np.ndarray, values: list[float]) -> float:
    """``sum_j weights[j] values[j]``; at run time the values may be Taylor jets."""
    total = 0.0
    for weight, value in zip(weights, values, strict=True):
        total = total + float(weight) * value
    return total


def _intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the intersection of two column spaces."""
    coefficients = null_space(np.hstack([first, -second]), rcond=_RANK_TOLERANCE)
    if coefficients.size == 0:
        return np.zeros((first.shape[0], 0))
    return orth(first @ coefficients[: first.shape[1]], rcond=_RANK_TOLERANCE)


def _complement_within(space: np.ndarray, part: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the orthogonal complement of ``part`` inside ``space``."""
    if part.shape[1] == 0:
        return space
    residual = space - part @ (part.T @ space)
    return orth(residual, rcond=_RANK_TOLERANCE)


def _leading(jacobian: np.ndarray, basis: np.ndarray) -> float:
    """Largest real part of the eigenvalues the Jacobian carries on ``basis``.

    When ``basis`` spans an invariant subspace, ``B^T J B`` is the restriction; when
    it spans the orthogonal complement of one, it is the map induced on the
    quotient, whose eigenvalues are the remaining ones. Either way it is the
    eigenvalues of that role, without matching eigenvectors. An empty basis has
    no eigenvalues and returns ``-inf``.
    """
    if basis.shape[1] == 0:
        return -math.inf
    return float(np.max(np.linalg.eigvals(basis.T @ jacobian @ basis).real))
