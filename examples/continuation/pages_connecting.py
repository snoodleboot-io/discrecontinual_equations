"""Connecting orbits and the manifolds that carry them.

Split out of webplot_examples, which assembles the atlas from these.
"""

import math

import numpy as np

from discrecontinual_equations.continuation.connecting_orbit import (
    HeteroclinicOrbit,
    HomoclinicOrbit,
    MeshSpec,
    Terminus,
)
from discrecontinual_equations.continuation.homoclinic_shooting import (
    Departure,
    HomoclinicShooting,
    ReturnSettings,
)
from discrecontinual_equations.continuation.manifold import (
    StableManifold,
    TaylorManifold,
    UnstableManifold,
)
from discrecontinual_equations.continuation.symmetry import fourier_reduce
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.systems import (
    FerroelectricRing,
)
from discrecontinual_equations.webplot.scene import Scene
from discrecontinual_equations.webplot.scene_builder import (
    curve_scene,
    with_regions,
)

try:  # python -m examples.continuation.pages_connecting
    from examples.continuation.systems import (
        DampedWell,
        DoubleWell,
        First,
        Jerk,
        MelnikovSystem,
        Saddle,
        Second,
        State,
        Third,
        TiltedHomoclinic,
        _equation,
        _homoclinic_tilt,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from systems import (
        DampedWell,
        DoubleWell,
        First,
        Jerk,
        MelnikovSystem,
        Saddle,
        Second,
        State,
        Third,
        TiltedHomoclinic,
        _equation,
        _homoclinic_tilt,
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
LAMBDA = "\u03bb"
_LOOP_ESCAPE = 50.0
_LOOP_DEPARTURE = 0.6
_LOOP_RETURN = 0.25
_IMAG_TOLERANCE = 1.0e-9
_MODE_IMAG = 1.0e-6
_SADDLE_JACOBIAN = np.array([[0.0, 1.0], [1.0, 0.0]])
_PHASE_WINDOW = 2.2


def _saddle() -> DeterministicFunction:
    return _equation(Saddle, 2, [First(value=0.0)]).derivative


def _chart_branches(selection, colour):
    chart = TaylorManifold(order=7).compute(
        _saddle(),
        np.zeros(2),
        _SADDLE_JACOBIAN,
        selection,
    )
    lines = []
    for sign in (1.0, -1.0):
        points = []
        for theta in np.linspace(0.0, 0.9 * sign, 60):
            state = chart.point([float(theta)])
            if abs(state[0]) < _PHASE_WINDOW and abs(state[1]) < _PHASE_WINDOW:
                points.append((float(state[0]), float(state[1])))
        kind = "manifold_unstable" if colour == "unstable" else "manifold_stable"
        lines.append((kind, "", points, None))
    return lines


def _manifolds() -> Scene:
    lines = _chart_branches(UnstableManifold(), "unstable")
    lines += _chart_branches(StableManifold(), "stable")
    markers = [("equilibrium", "saddle", [(0.0, 0.0)], None)]
    return curve_scene(
        lines,
        markers,
        ("x", "y"),
        "Stable and unstable manifolds of a saddle",
        "Taylor charts of W^u (red) and W^s (green); their union is the homoclinic.",
    )


def _homoclinic_orbit():
    half, intervals = 13.0, 200
    times = np.linspace(-half, half, intervals + 1)
    seed = np.zeros((intervals + 1, 2))
    for k, t in enumerate(times):
        seed[k, 0] = 1.2 / math.cosh(0.45 * t) ** 2
        seed[k, 1] = -0.9 * math.tanh(0.45 * t) / math.cosh(0.45 * t) ** 2
    mesh = MeshSpec(half, intervals, phase_index=1, phase_value=0.0)
    return HomoclinicOrbit(_saddle(), np.zeros(2), _SADDLE_JACOBIAN, mesh).solve(seed)


def _homoclinic_loop(orbit) -> Scene:
    loop = [(float(x), float(y)) for x, y in orbit.states]
    lines = [("orbit", "homoclinic orbit", loop, None)]
    lines += _chart_branches(UnstableManifold(), "unstable")
    markers = [("equilibrium", "saddle", [(0.0, 0.0)], None)]
    return curve_scene(
        lines,
        markers,
        ("x", "y"),
        "Homoclinic orbit (phase plane)",
        "The connecting orbit leaves and returns to the saddle along its manifolds.",
    )


def _homoclinic_pulse(orbit) -> Scene:
    pulse = [
        (float(t), float(x))
        for t, x in zip(orbit.times, orbit.component(0), strict=True)
    ]
    return curve_scene(
        [("orbit", "x(t)", pulse, None)],
        [],
        ("t", "x"),
        "Homoclinic orbit (time series)",
        "The pulse matches the exact 3/2 sech^2(t/2) to trapezoidal order.",
    )


def _connecting_orbit_entries() -> list[tuple[str, Scene]]:
    orbit = _homoclinic_orbit()
    return [
        ("homoclinic_loop.html", _homoclinic_loop(orbit)),
        ("homoclinic_pulse.html", _homoclinic_pulse(orbit)),
    ]


def _homoclinic_curve_shooting() -> Scene:
    field = _equation(
        MelnikovSystem,
        2,
        [First(value=0.0), Second(value=0.0)],
    ).derivative
    shooter = HomoclinicShooting(
        field,
        0,
        Departure(np.array([0.05, 0.05]), np.array([1.0, 1.0])),
        ReturnSettings(dt=0.003, horizon=70.0, departure_radius=0.6),
    )
    upward = shooter.trace(1, [0.05 * k for k in range(3)], 0.0, 0.06)
    downward = shooter.trace(1, [-0.05, -0.10], 0.0, 0.06)
    points = sorted(
        [(nu, mu) for nu, mu in downward] + [(nu, mu) for nu, mu in upward],
    )
    lines = [("stable", "homoclinic curve", points, None)]
    markers = [("bifurcation", "unperturbed loop", [(0.0, 0.0)], None)]
    return curve_scene(
        lines,
        markers,
        ("nu", MU),
        "Homoclinic curve traced by shooting",
        "The locus of saddle homoclinic orbits in the two-parameter plane, each "
        "point located by shooting the unstable manifold. It follows the Melnikov "
        "line mu I1 + nu I2 = 0 through the unperturbed loop at the origin.",
    )


def _symmetry_modes() -> Scene:
    gain, size = -0.3, 3

    def jacobian(coupling: float) -> np.ndarray:
        ring = FerroelectricRing(
            variables=[State() for _ in range(size)],
            parameters=[
                First(value=coupling),
                Second(value=gain),
                Third(value=0.0),
            ],
            results=[State() for _ in range(size)],
            time=None,
        )
        base = np.array(ring.eval(point=[0.0] * size, time=None))
        matrix = np.zeros((size, size))
        for j in range(size):
            shifted = [0.0] * size
            shifted[j] = 1.0e-7
            matrix[:, j] = (
                np.array(ring.eval(point=shifted, time=None)) - base
            ) / 1.0e-7
        return matrix

    steady: list[tuple[float, float]] = []
    oscillatory: list[tuple[float, float]] = []
    for step in range(29):
        coupling = -0.6 + 0.05 * step
        reduction = fourier_reduce(jacobian(coupling))
        for value in reduction.eigenvalues:
            if abs(value.imag) < _MODE_IMAG:
                steady.append((coupling, float(value.real)))
            else:
                oscillatory.append((coupling, float(value.real)))
                break
    lines = [
        ("stable", "steady mode (k = 0)", steady, None),
        ("unstable", "wave mode (k = 1, 2)", oscillatory, None),
    ]
    markers = [
        ("bifurcation", "symmetry breaking", [(gain, 0.0)], None),
        ("bifurcation", "Hopf (rotating wave)", [(-2.0 * gain, 0.0)], None),
    ]
    return curve_scene(
        lines,
        markers,
        (LAMBDA, "mode eigenvalue real part"),
        "Symmetry reduction of a ring: Fourier-mode stability",
        "The circulant linearisation of the three-cell ferroelectric ring, "
        "block-diagonalised by the Fourier basis. The real k=0 mode gives steady "
        "symmetry breaking at lambda = a; the complex k=1,2 modes give a rotating "
        "wave through a Hopf bifurcation at lambda = -2a.",
    )


def _shilnikov_loop() -> Scene:
    field = _equation(Jerk, 3, [First(value=0.48)]).derivative
    jacobian = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -1.0, -0.48]])
    values, vectors = np.linalg.eig(jacobian)
    unstable = vectors[:, int(np.argmax(values.real))].real
    unstable = unstable / np.linalg.norm(unstable)
    if unstable[0] < 0.0:
        unstable = -unstable
    state = 1.0e-6 * unstable
    dt = 0.005
    path: list[tuple[float, float]] = [(float(state[0]), float(state[2]))]
    left = False
    for _ in range(40000):
        values_k1 = np.array(field.eval(point=list(state), time=None))
        mid = state + 0.5 * dt * values_k1
        values_k2 = np.array(field.eval(point=list(mid), time=None))
        mid2 = state + 0.5 * dt * values_k2
        values_k3 = np.array(field.eval(point=list(mid2), time=None))
        end = state + dt * values_k3
        values_k4 = np.array(field.eval(point=list(end), time=None))
        state = state + dt / 6.0 * (
            values_k1 + 2.0 * values_k2 + 2.0 * values_k3 + values_k4
        )
        distance = float(np.linalg.norm(state))
        if distance > _LOOP_ESCAPE:
            break
        if distance > _LOOP_DEPARTURE:
            left = True
        path.append((float(state[0]), float(state[2])))
        if left and distance < _LOOP_RETURN:
            break
    lines = [("unstable", "unstable manifold", path, None)]
    markers = [("saddle", "saddle-focus", [(0.0, 0.0)], None)]
    return curve_scene(
        lines,
        markers,
        ("x", "z"),
        "Shilnikov saddle-focus homoclinic loop",
        "The one-dimensional unstable manifold of the saddle-focus, located by "
        "shooting at a = 0.48, leaves along a line and spirals back toward the "
        "origin along the two-dimensional stable manifold. Its saddle index is below "
        "one, so nearby dynamics are chaotic.",
    )


def _homoclinic_return() -> Scene:
    field = _equation(DampedWell, 2, [First(value=0.0)]).derivative
    departure = Departure(np.array([0.05, 0.05]), np.array([1.0, 1.0]))
    shooter = HomoclinicShooting(field, 0, departure)
    points: list[tuple[float, float]] = []
    for step in range(11):
        mu = -0.05 + 0.01 * step
        gap = shooter.gap(mu)
        if not math.isnan(gap):  # skip NaN (diverged before returning)
            points.append((mu, gap))
    lines = [("stable", "return gap", points, None)]
    markers = [("bifurcation", "homoclinic at mu = 0", [(0.0, 0.0)], None)]
    return curve_scene(
        lines,
        markers,
        (MU, "return gap"),
        "Locating a homoclinic orbit by shooting",
        "The unstable manifold is integrated and its signed miss on return to the "
        "saddle is recorded. The gap crosses zero at mu = 0, where the homoclinic "
        "orbit exists; a bracketing search converges to it.",
    )


def _three_dimensional_homoclinic() -> Scene:
    tilt = _homoclinic_tilt()
    half_length, intervals = 12.0, 240
    times = np.linspace(-half_length, half_length, intervals + 1)
    base = np.column_stack(
        [
            1.5 / np.cosh(times / 2.0) ** 2,
            -1.5 / np.cosh(times / 2.0) ** 2 * np.tanh(times / 2.0),
            np.zeros_like(times),
        ],
    )
    exact = base @ tilt.T
    seed = (
        np.column_stack(
            [
                1.2 / np.cosh(times / 2.5) ** 2,
                np.zeros_like(times),
                np.zeros_like(times),
            ],
        )
        @ tilt.T
    )
    jacobian = (
        tilt @ np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -2.0]]) @ tilt.T
    )
    field = _equation(TiltedHomoclinic, 3, []).derivative
    mesh = MeshSpec(
        half_length=half_length,
        intervals=intervals,
        phase_index=0,
        phase_value=float(exact[intervals // 2, 0]),
    )
    solution = HomoclinicOrbit(field, np.zeros(3), jacobian, mesh).solve(seed)
    path = list(zip(solution.component(0), solution.component(2), strict=True))
    lines = [("unstable", "homoclinic loop", path, None)]
    markers = [("saddle", "saddle", [(0.0, 0.0)], None)]
    return curve_scene(
        lines,
        markers,
        ("x", "z"),
        "Three-dimensional homoclinic orbit",
        "A homoclinic loop to a saddle with a two-dimensional stable eigenspace, "
        "solved by the projection boundary-value problem and shown projected onto the "
        "x-z plane. The same machinery projects onto complex (spiral) eigenspaces.",
    )


def _heteroclinic_cycle() -> Scene:
    field = _equation(DoubleWell, 2, []).derivative
    half_length, intervals = 8.0, 200
    times = np.linspace(-half_length, half_length, intervals + 1)
    mesh = MeshSpec(half_length=half_length, intervals=intervals)

    def jacobian(x: float) -> np.ndarray:
        return np.array([[0.0, 1.0], [-1.0 + 3.0 * x * x, 0.0]])

    left = Terminus(np.array([-1.0, 0.0]), jacobian(-1.0))
    right = Terminus(np.array([1.0, 0.0]), jacobian(1.0))
    upper = HeteroclinicOrbit(field, left, right, mesh).solve(
        np.column_stack([np.tanh(times / 2.0), np.zeros_like(times)]),
    )
    lower = HeteroclinicOrbit(field, right, left, mesh).solve(
        np.column_stack([-np.tanh(times / 2.0), np.zeros_like(times)]),
    )
    lines = [
        (
            "unstable",
            "connection -1 to +1",
            list(zip(upper.component(0), upper.component(1), strict=True)),
            None,
        ),
        (
            "stable",
            "connection +1 to -1",
            list(zip(lower.component(0), lower.component(1), strict=True)),
            None,
        ),
    ]
    markers = [("saddle", "saddles", [(-1.0, 0.0), (1.0, 0.0)], None)]
    return curve_scene(
        lines,
        markers,
        ("x", "y"),
        "Heteroclinic cycle between two saddles",
        "Two saddle-to-saddle connections, each solved by a projection "
        "boundary-value problem, close into the double-well separatrix loop joining "
        "the saddles at (-1, 0) and (+1, 0).",
    )


def _saddle_focus_onset() -> Scene:
    sigma, beta = 10.0, 8.0 / 3.0
    stable: list[tuple[float, float]] = []
    focus: list[tuple[float, float]] = []
    for step in range(28):
        rho = 20.0 + 0.5 * step
        coordinate = math.sqrt(beta * (rho - 1.0))
        jacobian = np.array(
            [
                [-sigma, sigma, 0.0],
                [rho - (rho - 1.0), -1.0, -coordinate],
                [coordinate, coordinate, -beta],
            ],
        )
        eigenvalues = np.linalg.eigvals(jacobian)
        spiral = max(
            value.real for value in eigenvalues if abs(value.imag) > _IMAG_TOLERANCE
        )
        (focus if spiral > 0.0 else stable).append((rho, spiral))
    lines = [
        ("stable", "stable focus (Re < 0)", stable, None),
        ("unstable", "saddle-focus (Re > 0)", focus, None),
    ]
    hopf = sigma * (sigma + beta + 3.0) / (sigma - beta - 1.0)
    markers = [("bifurcation", "Hopf: saddle-focus onset", [(hopf, 0.0)], None)]
    scene = curve_scene(
        lines,
        markers,
        ("rho", "spiral eigenvalue real part"),
        "Shilnikov saddle-focus onset in the Lorenz equilibria",
        "Real part of the complex eigenvalue pair at the Lorenz C-plus equilibrium. "
        "It crosses zero at the subcritical Hopf point rho = 24.74, beyond which the "
        "equilibrium is a saddle-focus whose homoclinic loops drive chaos.",
    )
    return with_regions(
        scene,
        [("saddle-focus: chaos region", 25.5, focus[-1][1])],
    )
