"""Cycle branches, and what it takes to resolve one.

Split out of webplot_examples, which assembles the atlas from these.
"""

import math

import numpy as np

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CycleSeed,
)
from discrecontinual_equations.continuation.deflation import DeflatedSolver
from discrecontinual_equations.continuation.periodic_orbit import (
    AdaptivePeriodicOrbit,
    HermiteSimpsonOrbit,
    PeriodicOrbit,
)
from discrecontinual_equations.webplot.scene import Scene
from discrecontinual_equations.webplot.scene_builder import (
    curve_scene,
    with_regions,
)

try:  # python -m examples.continuation.pages_cycles
    from examples.continuation.systems import (
        First,
        FoldCycles,
        GridSystem,
        Hopf,
        SnicSystem,
        VanDerPol,
        _equation,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from systems import (
        First,
        FoldCycles,
        GridSystem,
        Hopf,
        SnicSystem,
        VanDerPol,
        _equation,
    )


MU = "\u03bc"
ALPHA = "\u03b1"
LAMBDA = "\u03bb"
_FOLD_WINDOW = 0.02
_LOOP_ESCAPE = 50.0
_LOOP_DEPARTURE = 0.6
_LOOP_RETURN = 0.25
_IMAG_TOLERANCE = 1.0e-9
_MODE_IMAG = 1.0e-6
_PERIOD_TOLERANCE = 1.0e-2
BETA1, BETA2, BETA3 = "\u03b2\u2081", "\u03b2\u2082", "\u03b2\u2083"
NORM = "\u2016x\u2016"
_IMAGINARY_TOLERANCE = 1.0e-9

MU = "\u03bc"
_FOLD_WINDOW = 0.02
_PERIOD_TOLERANCE = 1.0e-2
_VDP_STEP = 0.001
_VDP_SETTLE = 50.0


def _limit_cycle_branch() -> Scene:
    equation = _equation(Hopf, 2, [First(value=0.0)])
    field = equation.derivative
    intervals = 60
    nodes = np.linspace(0.0, 1.0, intervals + 1)
    cycle: list[tuple[float, float]] = [(0.0, 0.0)]
    seed = None
    period = 2.0 * math.pi
    for step in range(1, 26):
        mu = 0.024 * step
        field.parameters[0].value = mu
        if seed is None:
            seed = np.column_stack(
                [
                    math.sqrt(mu) * np.cos(2.0 * math.pi * nodes),
                    math.sqrt(mu) * np.sin(2.0 * math.pi * nodes),
                ],
            )
        orbit = PeriodicOrbit(field, intervals, phase_index=1, phase_value=0.0)
        solution = orbit.solve(seed, period)
        if solution is None:
            continue
        amplitude = float(np.max(np.linalg.norm(solution.states, axis=1)))
        cycle.append((mu, amplitude))
        seed = solution.states
        period = solution.period
    stable = [(mu / 100.0 - 0.6, 0.0) for mu in range(61)]
    unstable = [(0.6 * mu / 60.0, 0.0) for mu in range(61)]
    lines = [
        ("stable", "stable equilibrium", stable, None),
        ("unstable", "unstable equilibrium", unstable, None),
        ("stable", "limit cycle", cycle, None),
    ]
    markers = [("hopf", "Hopf", [(0.0, 0.0)], None)]
    return curve_scene(
        lines,
        markers,
        (MU, "amplitude"),
        "Supercritical Hopf: limit-cycle branch",
        "The periodic orbit born at the Hopf grows as amplitude = sqrt(mu); each "
        "point is a limit cycle solved by the periodic boundary-value problem.",
    )


def _fold_of_cycles_branch() -> Scene:
    equation = _equation(FoldCycles, 2, [First(value=-0.1)])
    intervals = 50
    grid = np.linspace(0.0, 1.0, intervals + 1)
    radius = math.sqrt((1.0 + math.sqrt(1.0 + 4.0 * -0.1)) / 2.0)
    seed = CycleSeed(
        np.column_stack(
            [
                radius * np.cos(2.0 * math.pi * grid),
                radius * np.sin(2.0 * math.pi * grid),
            ],
        ),
        2.0 * math.pi,
        -0.1,
    )
    continuation = CycleContinuation(
        equation,
        0,
        intervals,
        phase_index=1,
        phase_value=0.0,
    )
    points, bifurcations = continuation.trace(seed, 0.03, 130, direction=-1.0)
    stable: list[tuple[float, float]] = []
    unstable: list[tuple[float, float]] = []
    for point in points:
        order = np.argsort(np.abs(point.multipliers - 1.0))
        nontrivial = point.multipliers[order[1:]]
        dominant = float(np.max(np.abs(nontrivial)))
        target = stable if dominant < 1.0 else unstable
        target.append((point.parameter, point.amplitude))
    lines = [
        ("stable", "stable cycle", stable, None),
        ("unstable", "unstable cycle", unstable, None),
    ]
    fold = next((b for b in bifurcations if b.kind == "fold_of_cycles"), None)
    markers = []
    if fold is not None:
        amplitude = min(
            (
                p.amplitude
                for p in points
                if abs(p.parameter - fold.parameter) < _FOLD_WINDOW
            ),
            default=radius,
        )
        markers = [("fold", "fold of cycles", [(fold.parameter, amplitude)], None)]
    return curve_scene(
        lines,
        markers,
        (MU, "amplitude"),
        "Fold of cycles",
        "Two limit cycles - one stable, one unstable - collide and annihilate as a "
        "Floquet multiplier passes +1; the branch turns at mu = -1/4.",
    )


def _snic_period_divergence() -> Scene:
    equation = _equation(SnicSystem, 2, [First(value=1.3)])
    field = equation.derivative
    intervals = 200
    grid = np.linspace(0.0, 1.0, intervals + 1)
    seed = np.column_stack([np.cos(2.0 * math.pi * grid), np.sin(2.0 * math.pi * grid)])
    curve: list[tuple[float, float]] = []
    for step in range(18):
        mu = 1.3 + 0.1 * step
        field.parameters[0].value = mu
        oracle = 2.0 * math.pi / math.sqrt(mu * mu - 1.0)
        solution = PeriodicOrbit(
            field,
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed, oracle * 1.02)
        if (
            solution is not None
            and abs(solution.period - oracle) / oracle < _PERIOD_TOLERANCE
        ):
            curve.append((mu, solution.period))
    lines = [("stable", "limit cycle period", curve, None)]
    scene = curve_scene(
        lines,
        [],
        (MU, "period"),
        "SNIC: infinite-period bifurcation",
        "Each point is a limit cycle solved by the periodic boundary-value problem; "
        "the period diverges as mu approaches the SNIC at mu = 1, where a saddle-node "
        "forms on the invariant circle.",
    )
    return with_regions(
        scene,
        [("SNIC: period to infinity at mu=1", 1.35, curve[0][1])],
    )


def _van_der_pol_cycle(mu: float, intervals: int):
    def rhs(state: np.ndarray) -> np.ndarray:
        x, y = state
        return np.array([y, mu * (1.0 - x * x) * y - x])

    def step(state: np.ndarray) -> np.ndarray:
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * _VDP_STEP * k1)
        k3 = rhs(state + 0.5 * _VDP_STEP * k2)
        k4 = rhs(state + _VDP_STEP * k3)
        return state + _VDP_STEP / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    state = np.array([2.0, 0.0])
    for _ in range(int(_VDP_SETTLE / _VDP_STEP)):
        state = step(state)
    while True:
        nxt = step(state)
        if state[1] < 0.0 <= nxt[1]:
            state = nxt
            break
        state = nxt
    times = [0.0]
    trajectory = [state.copy()]
    current = state.copy()
    clock = 0.0
    while True:
        current = step(current)
        clock += _VDP_STEP
        trajectory.append(current.copy())
        times.append(clock)
        if trajectory[-2][1] < 0.0 <= current[1] and clock > 1.0:
            break
    trajectory = np.array(trajectory)
    times = np.array(times)
    period = times[-1]
    sample = np.linspace(0.0, period, intervals + 1)
    seed = np.column_stack(
        [
            np.interp(sample, times, trajectory[:, 0]),
            np.interp(sample, times, trajectory[:, 1]),
        ],
    )
    return period, seed, trajectory


def _adaptive_mesh() -> Scene:
    mu, intervals = 5.0, 120
    field = _equation(VanDerPol, 2, [First(value=mu)]).derivative
    period, seed, trajectory = _van_der_pol_cycle(mu, intervals)
    solution = AdaptivePeriodicOrbit(
        field,
        intervals,
        phase_index=1,
        phase_value=0.0,
    ).solve(seed, period)
    cycle = [(float(p[0]), float(p[1])) for p in trajectory[::20]]
    nodes = [(float(s[0]), float(s[1])) for s in solution.states]
    lines = [("stable", "limit cycle", cycle, None)]
    markers = [("equilibrium", "adapted nodes", nodes, None)]
    return curve_scene(
        lines,
        markers,
        ("x", "y"),
        "Adaptive mesh on a van der Pol relaxation oscillation",
        "The limit cycle of van der Pol at mu = 5, with the solver's adapted nodes. "
        "Equidistributing a curvature monitor concentrates nodes on the fast jumps "
        "and spares the slow branches - the mesh a uniform time grid cannot supply.",
    )


def _collocation_convergence() -> Scene:
    mu = 0.5

    def period_error(solver_type, intervals: int) -> float:
        field = _equation(Hopf, 2, [First(value=mu)]).derivative
        nodes = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [
                math.sqrt(mu) * np.cos(2.0 * np.pi * nodes),
                math.sqrt(mu) * np.sin(2.0 * np.pi * nodes),
            ],
        )
        solution = solver_type(
            field,
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed, 6.0)
        return abs(solution.period - 2.0 * np.pi)

    counts = [20, 40, 80, 160]
    lines = []
    for solver_type, kind, label in (
        (PeriodicOrbit, "unstable", "trapezoidal (order 2)"),
        (HermiteSimpsonOrbit, "stable", "Hermite-Simpson (order 4)"),
    ):
        points = [
            (math.log10(n), math.log10(period_error(solver_type, n))) for n in counts
        ]
        lines.append((kind, label, points, None))
    return curve_scene(
        lines,
        [],
        ("log10 intervals", "log10 period error"),
        "Collocation order: trapezoidal vs Hermite-Simpson",
        "Period error of the exact Hopf cycle (T = 2 pi) against node count, on log "
        "axes. The slopes are -2 and -4: Hermite-Simpson gains two orders, reaching "
        "the same accuracy with far fewer nodes.",
    )


def _deflated_roots() -> Scene:
    field = _equation(GridSystem, 2, []).derivative
    seeds = [
        np.array([0.3, 0.4]),
        np.array([-0.6, 0.2]),
        np.array([0.2, -0.7]),
    ]
    roots = DeflatedSolver(field).find(seeds)
    markers = [
        (
            "equilibrium",
            "roots found",
            [(float(r[0]), float(r[1])) for r in roots],
            None,
        ),
        ("saddle", "seeds", [(float(s[0]), float(s[1])) for s in seeds], None),
    ]
    return curve_scene(
        [],
        markers,
        ("x", "y"),
        "Deflated Newton: distinct roots from a few seeds",
        "Deflation repels Newton from roots already found, so a handful of seeds "
        "uncover all nine equilibria of the decoupled quartic gradient - "
        "disconnected branches an ordinary solver would miss.",
    )
