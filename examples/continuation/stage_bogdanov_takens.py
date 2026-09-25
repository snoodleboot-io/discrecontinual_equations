"""Data for the Stage + Timeline prototype: a Bogdanov-Takens slice as a film.

Fix b2 and scrub b1 through the whole story - stable focus, Hopf, a cycle that
grows until the homoclinic kills it, then the fold. Every number here comes from
the library's own continuation: the equilibrium branch with its eigenvalues, the
cycle branch with its Floquet multipliers. The vector field is sampled on a grid
per frame so the browser can advect particles through it, and the saddle's
manifolds are integrated so the homoclinic closing is visible.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CycleSeed,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable
from sweet_tea.registry import Registry

import discrecontinual_equations

Registry.fill_registry(
    path=str(Path(discrecontinual_equations.__file__).parent),
    module="discrecontinual_equations",
    exclude=["*.tests", "*.examples", "*.plot"],
)

B2 = 0.5
P_LO, P_HI, FRAMES = -0.6, 0.1, 57
BOX_X, BOX_Y = (-1.25, 0.9), (-0.85, 0.85)
NX, NY = 36, 36


class First(Parameter, name="First", abbreviation="p1"):
    pass


class Second(Parameter, name="Second", abbreviation="p2"):
    pass


class State(Variable, name="State", abbreviation="v"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


class BogdanovTakensFamily(DeterministicFunction):
    """x' = y, y' = b1 + b2 y + x^2 + x y."""

    def eval(self, point, time=None):  # noqa: ARG002
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        x, y = point[0], point[1]
        return [y, b1 + b2 * y + x * x + x * y]


def equation(b1: float) -> DifferentialEquation:
    parameters = [First(value=b1), Second(value=B2)]
    return DifferentialEquation(
        variables=[State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=BogdanovTakensFamily(
            variables=[State(), State()],
            parameters=parameters,
            results=[State(), State()],
            time=None,
        ),
    )


def field_at(eq, b1, x, y):
    eq.derivative.parameters[0].value = b1
    return np.array(eq.derivative.eval(point=[x, y], time=None), dtype=float)


def r(v, d=5):
    return round(float(v), d)


# ---- equilibrium branch, from the library ---------------------------------
eq = equation(P_LO)
config = ContinuationConfig(
    continuation_parameter_index=0,
    detectors=["fold", "hopf"],
    initial_parameter=P_LO,
    direction=1,
    measure="component",
    parameter_lower_bound=P_LO - 0.02,
    parameter_upper_bound=P_HI + 0.02,
    maximum_points=600,
    maximum_step=0.015,
)
branch = ContinuerBuilder.build(config, eq).solve(eq, [-math.sqrt(-P_LO), 0.0])
branch_points = [
    {
        "p": r(pt.parameter), "x": r(pt.state[0]), "y": r(pt.state[1]),
        "stability": pt.stability, "kind": pt.kind,
        "eig": [[r(a), r(b)] for a, b in pt.eigenvalues],
    }
    for pt in branch.points
]
special = [
    {"p": r(pt.parameter), "x": r(pt.state[0]), "kind": pt.kind,
     "frequency": None if pt.frequency is None else r(pt.frequency)}
    for pt in branch.special_points
]
print(f"branch: {len(branch_points)} points, special: {[(s['kind'], s['p']) for s in special]}")

# ---- cycle branch, from the library, seeded honestly --------------------
# The Hopf here is subcritical (integrating past it escapes; before it decays), so
# the cycle is unstable and lives on the stable-focus side. An unstable cycle is an
# attractor in reversed time, so integrate backwards from near the focus to land on
# it, cut one period at the y = 0 section, and hand that to the continuation.
hopf = next((s for s in special if s["kind"] == "hopf"), None)
p_hopf = hopf["p"] if hopf else -(B2 * B2)
intervals = 80
grid = np.linspace(0.0, 1.0, intervals + 1)


def integrated_cycle(b1):
    xf = -math.sqrt(-b1)

    def rhs(_t, s):
        return -field_at(eq, b1, s[0], s[1])  # reversed time

    settle = solve_ivp(rhs, [0.0, 400.0], [xf + 0.05, 0.0], rtol=1e-10, atol=1e-12,
                       max_step=0.05).y[:, -1]

    def section(_t, s):
        return s[1]
    section.direction = -1.0  # reversed time: forward-time upward crossing
    run = solve_ivp(rhs, [0.0, 200.0], settle, rtol=1e-10, atol=1e-12, events=section,
                    dense_output=True, max_step=0.05)
    crossings = run.t_events[0]
    period = float(crossings[1] - crossings[0])
    times = crossings[0] + np.linspace(0.0, period, intervals + 1)
    states = run.sol(times).T[::-1]  # reverse into forward-time order
    return states, period


def trace_from(b1, direction):
    states, period = integrated_cycle(b1)
    seed = CycleSeed(states, period, b1)
    cont = CycleContinuation(equation(b1), 0, intervals, phase_index=1, phase_value=0.0)
    pts, bifs = cont.trace(seed, 0.02, 400, direction=direction)
    good = []
    for c in pts:
        if c.amplitude < 1.0e-3 or not (2.0 < c.solution.period < 400.0):
            break
        good.append(c)
    return good, bifs


P_SEED = p_hopf - 0.05
down, bifs_down = trace_from(P_SEED, -1.0)   # toward the homoclinic
up, bifs_up = trace_from(P_SEED, +1.0)       # back toward the Hopf
cycles_pts = sorted(up[1:] + down, key=lambda c: c.parameter)
cycle_bifs = bifs_down + bifs_up
print(f"cycle branch: {len(cycles_pts)} points, "
      f"p in [{min(c.parameter for c in cycles_pts):.4f}, {max(c.parameter for c in cycles_pts):.4f}], "
      f"amplitude [{min(c.amplitude for c in cycles_pts):.3f}, {max(c.amplitude for c in cycles_pts):.3f}], "
      f"period max {max(c.solution.period for c in cycles_pts):.2f}, "
      f"bifurcations {[(b.kind, round(float(b.parameter), 4)) for b in cycle_bifs]}")
cycles = [
    {
        "p": r(c.parameter), "period": r(c.solution.period, 4), "amplitude": r(c.amplitude),
        "multipliers": [[r(m.real), r(m.imag)] for m in c.multipliers],
        "states": [[r(a, 4), r(b, 4)] for a, b in c.solution.states],
    }
    for c in cycles_pts
]

# ---- per-frame: field grid, equilibria, saddle manifolds, nearest cycle ---
xs = np.linspace(*BOX_X, NX)
ys = np.linspace(*BOX_Y, NY)
frames = []
params = np.linspace(P_LO, P_HI, FRAMES)
frame_step = params[1] - params[0]


def nearest_branch(b1, x):
    best = min(branch.points, key=lambda pt: abs(pt.parameter - b1) * 4 + abs(pt.state[0] - x))
    return best


def manifold(b1, origin, direction, forward, kind):
    def rhs(_t, s):
        return field_at(eq, b1, s[0], s[1]) * (1.0 if forward else -1.0)

    def leave(_t, s):
        return min(s[0] - (BOX_X[0] - 0.2), (BOX_X[1] + 0.2) - s[0],
                   s[1] - (BOX_Y[0] - 0.2), (BOX_Y[1] + 0.2) - s[1])
    leave.terminal = True
    start = origin + 1.0e-4 * direction
    sol = solve_ivp(rhs, [0.0, 60.0], start, method="RK45", rtol=1e-8, atol=1e-10,
                    events=leave, max_step=0.05, dense_output=True)
    t_end = sol.t[-1]
    ts = np.linspace(0.0, t_end, 360)
    pts = sol.sol(ts).T
    return {"kind": kind, "points": [[r(a, 4), r(b, 4)] for a, b in pts]}


for b1 in params:
    field = []
    for y in ys:
        for x in xs:
            u, v = field_at(eq, b1, x, y)
            field.extend([r(u, 4), r(v, 4)])
    equilibria, manifolds = [], []
    if b1 < 0:
        for x in (-math.sqrt(-b1), math.sqrt(-b1)):
            pt = nearest_branch(b1, x)
            equilibria.append({"x": r(x), "y": 0.0, "stability": pt.stability,
                               "eig": [[r(a), r(b)] for a, b in pt.eigenvalues]})
        sx = math.sqrt(-b1)
        J = np.array([[0.0, 1.0], [2.0 * sx, B2 + sx]])
        vals, vecs = np.linalg.eig(J)
        for i in range(2):
            v = np.real(vecs[:, i]); v /= np.linalg.norm(v)
            unstable = vals[i].real > 0
            for s in (+1.0, -1.0):
                manifolds.append(manifold(b1, np.array([sx, 0.0]), s * v, unstable,
                                          "unstable" if unstable else "stable"))
    ci = None
    if cycles:
        j = min(range(len(cycles)), key=lambda k: abs(cycles[k]["p"] - b1))
        if abs(cycles[j]["p"] - b1) <= 0.6 * frame_step:
            ci = j
    frames.append({"p": r(b1), "field": field, "equilibria": equilibria,
                   "manifolds": manifolds, "cycle": ci})

data = {
    "system": {
        "title": "Bogdanov–Takens, one slice",
        "equation": "\\dot x = y,\\quad \\dot y = \\beta_1 + \\beta_2 y + x^2 + x y",
        "b2": B2, "parameter": "β₁", "x_label": "x", "y_label": "y",
    },
    "box": {"x": list(BOX_X), "y": list(BOX_Y)}, "grid": {"nx": NX, "ny": NY},
    "branch": {"points": branch_points, "special": special},
    "cycles": cycles,
    "cycle_bifurcations": [{"kind": b.kind, "p": r(b.parameter)} for b in cycle_bifs],
    "frames": frames,
}
out = sys.argv[1]
with open(out, "w") as f:
    json.dump(data, f, separators=(",", ":"))
print(f"wrote {out}: frames {len(frames)}, cycles covered "
      f"{sum(1 for fr in frames if fr['cycle'] is not None)} frames")
