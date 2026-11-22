"""Cycle-to-cycle connecting orbits from parameterized manifolds.

Castelli, Lessard and Mireles James (SIAM J. Appl. Dyn. Syst. 14, 2015,
Section 4) project the boundary conditions of a connecting orbit onto
high-order charts of the two manifolds. With ``P`` a chart of the unstable
manifold of the source cycle and ``Q`` one of the stable manifold of the
target, a point ``P(theta_0, sigma_0)`` whose orbit reaches
``Q(theta_1, sigma_1)`` after a time ``L`` is a connection, and the conjugacies
supply the two infinite tails for free. Fixing ``|sigma_0| = R_0`` and
``|sigma_1| = R_1`` (their phase conditions) leaves, for one-dimensional
manifold coordinates, their equation (35)

    F(theta_0, L, theta_1) = phi_L(P(theta_0, R_0)) - Q(theta_1, R_1) = 0,

three unknowns for a field in three dimensions. It is solved here by
Gauss-Newton with the flow and its variational equation integrated by DOP853.
Where the two manifolds coincide - by a symmetry, say - the connections form a
family and the Jacobian is rank-deficient; the least-squares step then picks
the nearest member of the family rather than failing.

Seeds come from the manifolds' intersection: :meth:`CycleConnectingOrbit.seeds`
flows a ring of points ``P(theta, R_0)`` and reports where each trajectory
passes closest to the ring ``Q(theta, R_1)``. This is a new formulation beside
the equilibrium connecting-orbit collocation solver, which it does not touch.
"""

import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.cycle_manifold import CycleManifoldChart
from discrecontinual_equations.continuation.fourier_taylor import field_jacobians
from discrecontinual_equations.function.function import Function

_TOLERANCE = 1.0e-10
_ITERATIONS = 40
_BACKTRACK_STEPS = 20
_BACKTRACK_FACTOR = 0.5
_INTEGRATION_TOLERANCE = 1.0e-12
_SEGMENT_SAMPLES = 400
# A ring is sampled this many times per Fourier mode it can hold: dense enough
# that the nearest sample is a good seed.
_RING_SAMPLES = 240
# Times along each trajectory at which its distance to the target ring is read.
_SCAN_TIMES = 600
# A seed only has to fall in Newton's basin; the scan's trajectories need not be
# integrated to the connection's own accuracy.
_SCAN_TOLERANCE = 1.0e-8


class CycleConnection:
    """A computed cycle-to-cycle connection.

    The orbit leaves the source cycle's unstable chart at
    ``(source_phase, source_sigma)``, flows for ``time``, and arrives on the
    target's stable chart at ``(target_phase, target_sigma)``. ``times`` and
    ``states`` sample that finite segment; :meth:`orbit` adds the two tails the
    charts' conjugacies give, which run back to the source cycle and on to
    the target.
    """

    __slots__ = [
        "_source",
        "_target",
        "residual",
        "source_phase",
        "source_sigma",
        "states",
        "target_phase",
        "target_sigma",
        "time",
        "times",
    ]

    def __init__(  # noqa: PLR0913 (the connection's defining data)
        self,
        source: CycleManifoldChart,
        target: CycleManifoldChart,
        unknowns: np.ndarray,
        sigmas: tuple[float, float],
        residual: float,
        segment: tuple[np.ndarray, np.ndarray],
    ) -> None:
        self._source = source
        self._target = target
        # The charts are 2T-periodic in phase, and Newton is free to wander a
        # whole number of those.
        self.source_phase = float(unknowns[0]) % (2.0 * source.period)
        self.time = float(unknowns[1])
        self.target_phase = float(unknowns[2]) % (2.0 * target.period)
        self.source_sigma, self.target_sigma = sigmas
        self.residual = residual
        self.times, self.states = segment

    def orbit(self, tail: float, samples: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Times and states from ``-tail`` before the segment to ``tail`` after it.

        Time zero is where the orbit leaves the source chart. Before it the
        state is ``P(theta_0 + t, e^(lambda_u t) R_0)``, after the segment
        ``Q(theta_1 + s, e^(lambda_s s) R_1)``: exact to the charts' accuracy.
        """
        before = np.linspace(-tail, 0.0, samples, endpoint=False)
        after = np.linspace(0.0, tail, samples + 1)[1:]
        head = np.array(
            [
                self._source.flow_image(self.source_phase, self.source_sigma, t)
                for t in before
            ],
        )
        end = np.array(
            [
                self._target.flow_image(self.target_phase, self.target_sigma, s)
                for s in after
            ],
        )
        times = np.concatenate([before, self.times, self.time + after])
        return times, np.vstack([head, self.states, end])


class CycleConnectingOrbit:
    """Find orbits from one cycle's unstable chart to another's stable chart."""

    __slots__ = ["_function", "_sigmas", "_source", "_target"]

    def __init__(
        self,
        function: Function,
        source: CycleManifoldChart,
        target: CycleManifoldChart,
        source_sigma: float,
        target_sigma: float,
    ) -> None:
        if source.exponent <= 0.0:
            message = "the source chart must be of an unstable manifold"
            raise ValueError(message)
        if target.exponent >= 0.0:
            message = "the target chart must be of a stable manifold"
            raise ValueError(message)
        self._function = function
        self._source = source
        self._target = target
        self._sigmas = (float(source_sigma), float(target_sigma))

    def _field(self, state: np.ndarray) -> np.ndarray:
        return np.array(self._function.eval(point=list(state), time=None), dtype=float)

    def _flow(
        self,
        start: np.ndarray,
        time: float,
        samples: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The state after ``time``, its monodromy, and states at ``samples``."""
        dimension = start.size

        def rhs(_t: float, packed: np.ndarray) -> np.ndarray:
            state = packed[:dimension]
            matrix = packed[dimension:].reshape(dimension, dimension)
            jacobian = field_jacobians(self._function, state[None, :])[0]
            return np.concatenate([self._field(state), (jacobian @ matrix).ravel()])

        packed = np.concatenate([start, np.eye(dimension).ravel()])
        solution = solve_ivp(
            rhs,
            (0.0, time),
            packed,
            method="DOP853",
            rtol=_INTEGRATION_TOLERANCE,
            atol=_INTEGRATION_TOLERANCE,
            t_eval=samples,
        )
        # With ``samples`` the last sample is ``time`` itself, so the last
        # column is the end state either way.
        end = solution.y[:, -1]
        return (
            end[:dimension],
            end[dimension:].reshape(dimension, dimension),
            solution.y[:dimension].T,
        )

    def _trajectory(self, start: np.ndarray, times: np.ndarray) -> np.ndarray:
        """States at ``times`` along the orbit of ``start``, to seed accuracy."""
        solution = solve_ivp(
            lambda _t, state: self._field(state),
            (0.0, float(times[-1])),
            start,
            method="DOP853",
            rtol=_SCAN_TOLERANCE,
            atol=_SCAN_TOLERANCE,
            t_eval=times,
        )
        return solution.y.T

    def residual(self, unknowns: np.ndarray) -> np.ndarray:
        """``phi_L(P(theta_0, R_0)) - Q(theta_1, R_1)``: equation (35)."""
        start = self._source.point(float(unknowns[0]), self._sigmas[0])
        end, _, _ = self._flow(start, float(unknowns[1]))
        return end - self._target.point(float(unknowns[2]), self._sigmas[1])

    def _linearise(self, unknowns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The residual (35) and its Jacobian in ``(theta_0, L, theta_1)``."""
        phase, time, arrival = (float(value) for value in unknowns)
        start = self._source.point(phase, self._sigmas[0])
        end, monodromy, _ = self._flow(start, time)
        residual = end - self._target.point(arrival, self._sigmas[1])
        jacobian = np.column_stack(
            [
                monodromy @ self._source.tangent(phase, self._sigmas[0]),
                self._field(end),
                -self._target.tangent(arrival, self._sigmas[1]),
            ],
        )
        return residual, jacobian

    def solve(self, seed: np.ndarray) -> CycleConnection:
        """Damped Gauss-Newton from ``seed = (theta_0, L, theta_1)``.

        A full step taken where the flow is still far from the target ring can
        throw the end point out along the target cycle's unstable direction,
        so each step is halved until the residual falls.
        """
        unknowns = np.asarray(seed, dtype=float).copy()
        residual, jacobian = self._linearise(unknowns)
        for _ in range(_ITERATIONS):
            size = float(np.linalg.norm(residual))
            if size < _TOLERANCE:
                break
            step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
            for _ in range(_BACKTRACK_STEPS):
                trial = unknowns + step
                if trial[1] > 0.0:
                    trial_residual, trial_jacobian = self._linearise(trial)
                    if np.linalg.norm(trial_residual) < size:
                        break
                step = _BACKTRACK_FACTOR * step
            else:
                break
            unknowns, residual, jacobian = trial, trial_residual, trial_jacobian
        if np.linalg.norm(residual) >= _TOLERANCE:
            message = "the connecting orbit's Newton iteration did not converge"
            raise ValueError(message)
        samples = np.linspace(0.0, float(unknowns[1]), _SEGMENT_SAMPLES)
        start = self._source.point(float(unknowns[0]), self._sigmas[0])
        _, _, states = self._flow(start, float(unknowns[1]), samples)
        return CycleConnection(
            self._source,
            self._target,
            unknowns,
            self._sigmas,
            float(np.linalg.norm(residual)),
            (samples, states),
        )

    def seeds(self, horizon: float, phases: int = 24) -> list[np.ndarray]:
        """Seeds from the manifolds' intersection, nearest first.

        ``phases`` points of the source ring ``P(theta, R_0)``, over the chart's
        full ``2T`` period, are each flowed for ``horizon``; along each
        trajectory the nearest point of the target ring ``Q(theta, R_1)`` is
        found. Every phase whose nearest approach is a local minimum over the
        ring of phases becomes a seed ``(theta_0, L, theta_1)``.
        """
        source_period = 2.0 * self._source.period
        target_period = 2.0 * self._target.period
        ring_phases = np.linspace(0.0, target_period, _RING_SAMPLES, endpoint=False)
        ring = np.array(
            [self._target.point(phase, self._sigmas[1]) for phase in ring_phases],
        )
        times = np.linspace(0.0, horizon, _SCAN_TIMES)
        found: list[tuple[float, np.ndarray]] = []
        starts = np.linspace(0.0, source_period, phases, endpoint=False)
        for phase in starts:
            start = self._source.point(float(phase), self._sigmas[0])
            states = self._trajectory(start, times)
            gaps = np.linalg.norm(states[:, None, :] - ring[None, :, :], axis=2)
            at, nearest = np.unravel_index(np.argmin(gaps), gaps.shape)
            found.append(
                (
                    float(gaps[at, nearest]),
                    np.array([phase, times[at], ring_phases[nearest]]),
                ),
            )
        minima = [
            found[i]
            for i in range(len(found))
            if found[i][0] <= found[i - 1][0]
            and found[i][0] <= found[(i + 1) % len(found)][0]
        ]
        return [seed for _, seed in sorted(minima, key=lambda pair: pair[0])]
