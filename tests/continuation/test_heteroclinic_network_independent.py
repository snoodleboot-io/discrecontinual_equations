"""Independent checks of the heteroclinic network layer.

Each oracle here is built without the library's own field evaluation or visit
measurement wherever that is possible:

* a linear change of coordinates ``y = M x`` with a non-orthogonal ``M`` makes
  every saddle's Jacobian non-normal. Diagonal Jacobians - all the coordinate
  systems have - cannot tell left eigenvectors from right ones, nor a restriction
  to an invariant subspace from an orthogonal projection; this one can;
* the residence-time growth is measured on a trajectory of the log-coordinate
  Lotka-Volterra flow ``y_i' = lam_i + sum_j A_ij exp(2 y_j)``, written here, with
  residence defined by a coordinate threshold rather than a ball, and compared
  with ``c_j / e_{j+1}`` derived by hand from the eigenvalues at the axial
  equilibria (Gaunersdorfer 1992);
* the Kirk-Silber competition is checked on both sides of ``delta~ = 0``
  (Castro and Lohse 2014, Table 1 and Proposition 5.3): the same start near the
  xi3-cycle stays there for ``delta~ = +0.1`` and leaves for the xi4-cycle at
  ``delta~ = -0.1``.
"""

import itertools
import math
from typing import ClassVar
from unittest import TestCase

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from discrecontinual_equations.continuation.connecting_orbit import (
    HeteroclinicOrbit,
    MeshSpec,
    Terminus,
)
from discrecontinual_equations.continuation.heteroclinic_network import (
    HeteroclinicNetwork,
    coordinate_plane,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.systems.reflection_cubic import (
    Amplitude,
    guckenheimer_holmes,
    reflection_cubic,
)

_NAMES = ("xi1", "xi2", "xi3")
# Condition number 366: far enough from orthogonal that left and right
# eigenvectors, and restriction and projection, differ at order one.
_SHEAR = np.array([[1.0, 0.7, -0.3], [0.2, 1.0, 0.9], [-0.5, 0.4, 1.0]])


class _Sheared(DeterministicFunction):
    """``g(y) = M f(M^-1 y)``: the same flow in non-orthogonal coordinates.

    Written with ``float * x`` and ``+`` only, so Taylor jets pass through and the
    network's automatic-differentiation Jacobians are exact.
    """

    inner: DeterministicFunction
    forward: np.ndarray
    inverse: np.ndarray

    def eval(
        self,
        point: list[float],
        time: float | None = None,
    ) -> list[float]:
        size = len(point)
        original = [_combine(self.inverse[row], point, size) for row in range(size)]
        field = self.inner.eval(original, time)
        return [_combine(self.forward[row], field, size) for row in range(size)]


def _combine(row: np.ndarray, values: list, size: int):
    total = 0.0
    for column in range(size):
        total = total + float(row[column]) * values[column]
    return total


def _sheared(inner: DeterministicFunction, matrix: np.ndarray) -> _Sheared:
    size = matrix.shape[0]
    field = _Sheared(
        variables=[Amplitude() for _ in range(size)],
        parameters=inner.parameters,
        results=[Amplitude() for _ in range(size)],
        time=None,
    )
    field.inner = inner
    field.forward = matrix
    field.inverse = np.linalg.inv(matrix)
    return field


def _sheared_gh_network(a2: float, a3: float) -> HeteroclinicNetwork:
    """GH with ``lam = 1``, ``a1 = -1`` (unit saddles), in ``y = M x``."""
    field = _sheared(guckenheimer_holmes(1.0, -1.0, a2, a3), _SHEAR)
    network = HeteroclinicNetwork(field)
    for index, name in enumerate(_NAMES):
        network.add_saddle(name, _SHEAR @ np.eye(3)[index])
    for index in range(3):
        following = (index + 1) % 3
        network.add_connection(
            _NAMES[index],
            _NAMES[following],
            _SHEAR @ coordinate_plane(3, (index, following)),
        )
    return network


class TestNonNormalCoordinates(TestCase):
    """The roles and the projection BVP are coordinate-free; diagonal tests are not."""

    def test_roles_are_the_closed_form_in_sheared_coordinates(self):
        """Eigenvalues are similarity invariants: ``r = 2``, ``c = -a2 - 1``,
        ``e = 1 + a3``, with ``lam = 1`` and ``a1 = -1``.

        Reading roles by an orthogonal projection of a non-normal Jacobian onto a
        non-invariant complement would mix them; the quotient construction must not.
        """
        for a2, a3 in ((-1.5, -0.6), (-3.0, -0.2), (-1.2, -0.9)):
            spectrum = _sheared_gh_network(a2, a3).spectrum(_NAMES)
            assert np.allclose(spectrum.radial, 2.0, atol=1.0e-10)
            assert np.allclose(spectrum.contracting, -a2 - 1.0, atol=1.0e-10)
            assert np.allclose(spectrum.expanding, 1.0 + a3, atol=1.0e-10)
            assert np.all(np.isneginf(spectrum.transverse))
            assert spectrum.is_asymptotically_stable == (a2 + a3 < -2.0)

    def test_projection_bvp_converges_to_the_exact_sheared_connection(self):
        """At ``a2 + a3 = 2 a1`` the connection is ``u = 1 / (1 + exp(0.8 t))``.

        In ``y = M x`` the Jacobians are non-normal, so projection conditions
        built from right eigenvectors instead of left ones put the ends in the
        wrong subspaces and the orbit misses the exact one at order one. Built
        correctly the trapezoidal BVP is second order: doubling the mesh divides
        the error by four.
        """
        network = _sheared_gh_network(-1.4, -0.6)
        source, target = network.saddle("xi1"), network.saddle("xi2")
        index = int(np.argmax(np.abs(target.equilibrium - source.equilibrium)))
        phase = float(0.5 * (source.equilibrium[index] + target.equilibrium[index]))

        def exact(time: float) -> np.ndarray:
            u = 1.0 / (1.0 + math.exp(0.8 * time))
            return _SHEAR @ np.array([math.sqrt(u), math.sqrt(1.0 - u), 0.0])

        centre = brentq(lambda shift: exact(shift)[index] - phase, -10.0, 10.0)
        errors = []
        for intervals in (100, 200):
            mesh = MeshSpec(16.0, intervals, index, phase)
            orbit = HeteroclinicOrbit(
                network.function,
                Terminus(source.equilibrium, source.jacobian),
                Terminus(target.equilibrium, target.jacobian),
                mesh,
            )
            seed = network.seed(network.connections[0], mesh)
            solution = orbit.solve(seed)
            truth = np.array([exact(time + centre) for time in solution.times])
            errors.append(float(np.max(np.abs(solution.states - truth))))
        assert errors[0] < 5.0e-4
        assert 3.9 < errors[0] / errors[1] < 4.1


def _cubic_rates(growth: np.ndarray, interaction: np.ndarray):
    """``c_j`` and ``e_j`` of the axial cycle ``x1 -> x2 -> x3``, by hand.

    At the saddle on axis ``j``, ``x_j^2 = -lam_j / A_jj`` and the eigenvalue in
    direction ``k`` is ``lam_k + A_kj x_j^2``; the cycle arrives from axis
    ``j - 1`` (contracting) and leaves for axis ``j + 1`` (expanding).
    """
    squares = -growth / np.diag(interaction)
    contracting = np.array(
        [-(growth[j - 1] + interaction[j - 1, j] * squares[j]) for j in range(3)],
    )
    expanding = np.array(
        [
            growth[(j + 1) % 3] + interaction[(j + 1) % 3, j] * squares[j]
            for j in range(3)
        ],
    )
    return squares, contracting, expanding


class TestResidenceIndependently(TestCase):
    """Gaunersdorfer's ratio, measured with only the spectrum from the library."""

    def test_step_growth_is_c_j_over_e_next_with_a_threshold_residence(self):
        growth = np.array([1.0, 0.8, 1.0])
        row = np.array([-1.0, -1.5, -0.6])
        interaction = np.array([np.roll(row, shift) for shift in range(3)])
        squares, contracting, expanding = _cubic_rates(growth, interaction)
        hand = contracting / np.roll(expanding, -1)
        # The three ratios differ by a factor of seven, so any index slip shows.
        assert np.ptp(hand) > 2.5

        field = guckenheimer_holmes(1.0, -1.0, -1.5, -0.6)
        field.parameters[1].value = 0.8
        network = HeteroclinicNetwork(field)
        for index, name in enumerate(_NAMES):
            network.add_saddle(name, math.sqrt(squares[index]) * np.eye(3)[index])
        for index in range(3):
            following = (index + 1) % 3
            network.add_connection(
                _NAMES[index],
                _NAMES[following],
                coordinate_plane(3, (index, following)),
            )
        library = network.spectrum(_NAMES).residence_ratios
        assert np.allclose(library, hand, rtol=1.0e-10)

        horizon = 3000.0
        solution = solve_ivp(
            lambda _t, logs: growth + interaction @ np.exp(2.0 * logs),
            (0.0, horizon),
            np.log([0.9, 0.3, 0.1]),
            method="DOP853",
            rtol=1.0e-11,
            atol=1.0e-12,
            dense_output=True,
        )
        times = np.linspace(0.0, horizon, 600001)
        logs = solution.sol(times).T
        inside = logs > 0.5 * np.log(squares) - 0.05
        visits = []
        for axis in range(3):
            changes = np.diff(inside[:, axis].astype(int))
            entries, exits = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
            for entry in entries:
                later = exits[exits > entry]
                if later.size:
                    visits.append((times[entry], axis, times[later[0]] - times[entry]))
        visits.sort()
        order = [axis for _, axis, _ in visits]
        durations = np.array([duration for _, _, duration in visits])
        revolutions = durations[3:] - durations[:-3]
        ratios = revolutions[1:] / revolutions[:-1]
        measured = np.full(3, np.nan)
        for k in range(len(ratios) - 1, -1, -1):
            if np.isnan(measured[order[k]]):
                measured[order[k]] = ratios[k]
        assert np.max(np.abs(measured / hand - 1.0)) < 1.0e-3


def _kirk_silber_interaction(eigenvalues: dict[str, float]) -> np.ndarray:
    """Unit saddles; ``1 + A_kj`` is the eigenvalue at xi_j in direction k."""
    e = eigenvalues
    mu = np.zeros((4, 4))
    mu[1, 0], mu[2, 0], mu[3, 0] = e["e12"], -e["c13"], -e["c14"]
    mu[0, 1], mu[2, 1], mu[3, 1] = -e["c21"], e["e23"], e["e24"]
    mu[0, 2], mu[1, 2], mu[3, 2] = e["e31"], -e["c32"], -e["c34"]
    mu[0, 3], mu[1, 3], mu[2, 3] = e["e41"], -e["c42"], -e["c43"]
    interaction = mu - 1.0
    np.fill_diagonal(interaction, -1.0)
    return interaction


class TestKirkSilberSignBoundary(TestCase):
    """Castro and Lohse (2014): the xi3-cycle loses every index exactly at delta~ = 0.

    ``delta~ = c34/e31 + c14 c32/(e12 e31) - e24 c13 c32/(e12 e31 e23)``, which
    is ``c34 - 0.8`` for the eigenvalues below, while ``delta = 10/3`` throughout.
    Starting near the xi3-cycle, the trajectory must stay on it for
    ``delta~ = +0.1`` and switch to the xi4-cycle for ``delta~ = -0.1``.
    Measured, as the node visited after each xi2: ``3333333`` and ``33333444``
    by ``t = 1500``. (At ``-0.05`` the switch comes too, after six revolutions;
    the escape slows as ``delta~ -> 0^-``.)
    """

    _BASE: ClassVar[dict[str, float]] = {
        "e12": 1.0, "c13": 2.0, "c14": 1.0, "c21": 1.0, "e23": 1.0, "e24": 0.9,
        "e31": 1.0, "c32": 1.0, "e41": 1.0, "c42": 1.5, "c43": 2.0,
    }  # fmt: skip

    def test_the_winner_flips_with_the_sign_of_delta_tilde(self):
        for c34, winner in ((0.9, 2), (0.7, 3)):
            eigenvalues = dict(self._BASE, c34=c34)
            interaction = _kirk_silber_interaction(eigenvalues)
            network = HeteroclinicNetwork(reflection_cubic(np.ones(4), interaction))
            for index in range(4):
                network.add_saddle(f"xi{index + 1}", np.eye(4)[index])
            for source, target in ((1, 2), (2, 3), (3, 1), (2, 4), (4, 1)):
                network.add_connection(
                    f"xi{source}",
                    f"xi{target}",
                    coordinate_plane(4, (source - 1, target - 1)),
                )
            three = network.spectrum(("xi1", "xi2", "xi3"))
            # Table 1, xi3-cycle: a~2 = c32/e31, a~3 = c13/e12, b~ = -t / e.
            delta_tilde = (
                (-three.transverse[1] / three.expanding[1])
                * (three.contracting[2] / three.expanding[2])
                * (three.contracting[0] / three.expanding[0])
                - three.transverse[0]
                / three.expanding[0]
                * (three.contracting[2] / three.expanding[2])
                - three.transverse[2] / three.expanding[2]
            )
            assert abs(delta_tilde - (c34 - 0.8)) < 1.0e-10
            assert (delta_tilde > 0.0) == (winner == 2)

            horizon = 1500.0
            solution = solve_ivp(
                lambda _t, logs, a=interaction: 1.0 + a @ np.exp(2.0 * logs),
                (0.0, horizon),
                np.log([0.9, 0.3, 0.1, 1.0e-4]),
                method="DOP853",
                rtol=1.0e-10,
                atol=1.0e-10,
                dense_output=True,
            )
            dominant = np.argmax(solution.sol(np.linspace(0.0, horizon, 75001)), 0)
            switches = [int(dominant[0])] + [
                int(b) for a, b in itertools.pairwise(dominant) if a != b
            ]
            after_xi2 = [
                switches[k + 1] for k in range(len(switches) - 1) if switches[k] == 1
            ]
            assert len(after_xi2) >= 6
            assert after_xi2[:3] == [2, 2, 2]
            assert after_xi2[-2:] == [winner, winner]
            if winner == 2:
                assert set(after_xi2) == {2}


class _Leak(DeterministicFunction):
    """GH plus ``eps x1^2 x2^2`` in ``x3'``: the x1x2 plane is no longer invariant.

    The term and its first derivatives vanish on every axis, so the Jacobians at
    the saddles - all ``add_connection`` can check - are exactly those of GH.
    """

    inner: DeterministicFunction
    strength: float

    def eval(
        self,
        point: list[float],
        time: float | None = None,
    ) -> list[float]:
        derivative = self.inner.eval(point, time)
        leak = point[0] * point[0] * point[1] * point[1]
        derivative[2] = derivative[2] + self.strength * leak
        return derivative


class TestSubspaceChecks(TestCase):
    """The plane is checked, not assumed: at the saddles and along the orbit."""

    def test_a_plane_that_is_not_invariant_at_a_saddle_is_rejected(self):
        network = HeteroclinicNetwork(guckenheimer_holmes(1.0, -1.0, -1.5, -0.6))
        for index, name in enumerate(_NAMES):
            network.add_saddle(name, np.eye(3)[index])
        tilted = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.3]])
        with pytest.raises(ValueError, match="not invariant"):
            network.add_connection("xi1", "xi2", tilted)

    def test_subspace_defect_sees_an_orbit_leave_a_plane_the_flow_leaves(self):
        """Measured: defect 0 at ``eps = 0`` and 5.0e-3 at ``eps = 0.05``."""
        inner = guckenheimer_holmes(1.0, -1.0, -1.5, -0.6)
        defects = []
        for strength in (0.0, 0.05):
            field = _Leak(
                variables=[Amplitude() for _ in range(3)],
                parameters=inner.parameters,
                results=[Amplitude() for _ in range(3)],
                time=None,
            )
            field.inner, field.strength = inner, strength
            network = HeteroclinicNetwork(field)
            for index, name in enumerate(_NAMES):
                network.add_saddle(name, np.eye(3)[index])
            connection = network.add_connection(
                "xi1",
                "xi2",
                coordinate_plane(3, (0, 1)),
            )
            mesh = network.mesh(connection, half_length=20.0, intervals=200)
            defects.append(network.solve(connection, mesh).subspace_defect)
        assert defects[0] < 1.0e-12
        assert defects[1] > 1.0e-3


class TestNoConnectionIsRefused(TestCase):
    """A connection the eigenvalues rule out must not come back as an orbit.

    In the stable GH system xi1 attracts in the x1x3 plane (``-c = -0.5`` there)
    and xi3 repels in it (``e = 0.4``), so there is no orbit xi1 -> xi3. Measured
    before the refusal existed: at 20/400 the solver returned a path within 2e-5
    of both saddles, ``subspace_defect`` 0 and ``resolution.estimate`` 3.4e-4 -
    every number the network offers as a reason to trust an orbit - while its
    trapezoidal residual was 4e-2: not a trajectory at all.
    """

    def test_solve_refuses_when_a_margin_is_not_positive(self):
        network = HeteroclinicNetwork(guckenheimer_holmes(1.0, -1.0, -1.5, -0.6))
        for index, name in enumerate(_NAMES):
            network.add_saddle(name, np.eye(3)[index])
        backwards = network.add_connection("xi1", "xi3", coordinate_plane(3, (0, 2)))
        assert max(network.connection_margins(backwards)) < 0.0
        mesh = network.mesh(backwards, half_length=20.0, intervals=400)
        with pytest.raises(ValueError, match="margin"):
            network.solve(backwards, mesh)
