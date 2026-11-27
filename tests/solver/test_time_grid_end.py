"""A fixed-step grid ends at ``end_time``, and a span that is not a whole number
of steps is refused (DEQ-129).

``FixedStepConfig.times`` was ``np.arange(start, end + h, h)``. When the span is
not a multiple of ``h`` that puts a node past ``end_time`` (0.8 with ``h`` =
0.015 ended at 0.81), and when it is a multiple, rounding in ``arange`` can add
a node anyway: ``(0, 10, 0.005)`` ended at 10.005 and ``(0, 20, 0.01)`` at
20.01, a step past the end. The grid is now ``start + h * arange(n + 1)`` with
``n`` the rounded step count and its last node set to ``end_time``, and a span
that is not ``n h`` to rounding is refused rather than clipped: every
fixed-step solver steps with the constant ``dt``, the multistep coefficients
assume it, and a :class:`BrownianPath` is read a whole number of cells at a
time, so a shorter last step would be a different scheme, not a shorter grid.

The draws below are decimal numbers, as a caller writes them, with ``end``
the decimal sum ``start + n h`` - a span that is a multiple of ``h`` exactly on
paper and only to rounding in binary.
"""

from decimal import Decimal

import numpy as np
import pytest

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.simple_ode_function import SimpleODEFunction
from discrecontinual_equations.solver.deterministic.adams_bashforth2.adams_bashforth2_config import (  # noqa: E501
    AdamsBashforth2Config,
)
from discrecontinual_equations.solver.deterministic.adams_bashforth2.adams_bashforth2_solver import (  # noqa: E501
    AdamsBashforth2Solver,
)
from discrecontinual_equations.solver.deterministic.adams_bashforth3.adams_bashforth3_config import (  # noqa: E501
    AdamsBashforth3Config,
)
from discrecontinual_equations.solver.deterministic.adams_bashforth3.adams_bashforth3_solver import (  # noqa: E501
    AdamsBashforth3Solver,
)
from discrecontinual_equations.solver.deterministic.euler.euler_config import (
    EulerConfig,
)
from discrecontinual_equations.solver.deterministic.euler.euler_solver import (
    EulerSolver,
)
from discrecontinual_equations.solver.solver_config import FixedStepConfig
from discrecontinual_equations.solver.stochastic.ensemble import StochasticEnsemble
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config import (  # noqa: E501
    EulerMaruyamaConfig,
)
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_solver import (  # noqa: E501
    EulerMaruyamaSolver,
)
from discrecontinual_equations.solver.stochastic.gottwald_melbourne.gottwald_melbourne_config import (  # noqa: E501
    GottwaldMelbourneConfig,
)
from discrecontinual_equations.solver.stochastic.gottwald_melbourne.gottwald_melbourne_solver import (  # noqa: E501
    GottwaldMelbourneSolver,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_config import (
    MilsteinConfig,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_solver import (
    MilsteinSolver,
)
from discrecontinual_equations.solver.stochastic.srk2.srk2_config import SRK2Config
from discrecontinual_equations.solver.stochastic.srk2.srk2_solver import SRK2Solver
from discrecontinual_equations.solver.stochastic.srk3.srk3_config import SRK3Config
from discrecontinual_equations.solver.stochastic.srk3.srk3_solver import SRK3Solver
from discrecontinual_equations.solver.stochastic.srk4.srk4_config import SRK4Config
from discrecontinual_equations.solver.stochastic.srk4.srk4_solver import SRK4Solver
from discrecontinual_equations.solver.stochastic.srk5.srk5_config import SRK5Config
from discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_config import (  # noqa: E501
    ZhanDuanLiLiConfig,
)
from discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_solver import (  # noqa: E501
    ZhanDuanLiLiSolver,
)
from discrecontinual_equations.variable import Variable
from tests.solver.geometric_brownian_motion import GeometricBrownianMotionCase

_DRAWS = 3000
_MOST_STEPS = 5000
# A node of the grid is start + k h in floating point: a few ulps of the
# larger of |start| and |end| away from the decimal value.
_ULPS = 8


def _decimal_draw(generator: np.random.Generator) -> tuple[Decimal, Decimal, int]:
    """A decimal start and step as a caller writes them, and a step count."""
    digits = int(generator.integers(1, 6))
    step = Decimal(int(generator.integers(1, 10**digits))) / Decimal(10**digits)
    start = Decimal(int(generator.integers(-5000, 5000))) / Decimal(1000)
    steps = int(generator.integers(1, _MOST_STEPS))
    return start, step, steps


def _multiples() -> list[tuple[float, float, float, int]]:
    """``(start, end, h, n)`` with ``end = start + n h`` exactly in decimal."""
    generator = np.random.default_rng(129)
    draws = []
    for _ in range(_DRAWS):
        start, step, steps = _decimal_draw(generator)
        end = start + steps * step
        draws.append((float(start), float(end), float(step), steps))
    return draws


def _scale(start: float, end: float) -> float:
    return _ULPS * float(np.spacing(max(abs(start), abs(end), 1.0)))


def test_the_last_node_is_end_time_for_every_whole_span():
    # Arrange: decimal draws whose binary spans are multiples of h only to rounding.
    draws = _multiples()

    for start, end, step, steps in draws:
        # Act
        config = FixedStepConfig(start_time=start, end_time=end, step_size=step)
        times = config.times

        # Assert: n + 1 nodes from start to end, each start + k h to rounding.
        assert config.n_steps == steps, (start, end, step)
        assert times.shape == (steps + 1,), (start, end, step)
        assert times[0] == start
        assert times[-1] == end
        assert np.all(np.diff(times) > 0)
        nodes = [float(Decimal(start) + k * Decimal(step)) for k in (1, steps // 2)]
        assert np.allclose(
            times[[1, steps // 2]],
            nodes,
            rtol=0.0,
            atol=_scale(start, end) * max(1, steps),
        )


def test_the_draws_include_spans_arange_got_wrong():
    # Arrange: the old grid, to show the draws reach the rounding cases.
    draws = _multiples()

    # Act
    wrong = [
        (start, end, step)
        for start, end, step, steps in draws
        if len(np.arange(start, end + step, step)) != steps + 1
    ]

    # Assert: a good share of the draws are cases the old grid miscounted.
    assert len(wrong) > _DRAWS // 20


def test_a_span_that_is_not_a_whole_number_of_steps_is_refused():
    # Arrange: decimal draws with end a fraction of a step past start + n h.
    generator = np.random.default_rng(130)

    for _ in range(_DRAWS):
        start, step, steps = _decimal_draw(generator)
        fraction = Decimal(int(generator.integers(1, 100))) / Decimal(100)
        end = float(start + (steps + fraction) * step)
        config = FixedStepConfig(
            start_time=float(start),
            end_time=end,
            step_size=float(step),
        )

        # Act / Assert
        with pytest.raises(ValueError, match="whole number of steps"):
            _ = config.times
        with pytest.raises(ValueError, match="whole number of steps"):
            _ = config.n_steps


@pytest.mark.parametrize(
    ("end", "step", "nodes"),
    [(0.8, 0.01, 81), (10.0, 0.005, 2001), (20.0, 0.01, 2001), (5.0, 0.1, 51)],
)
def test_the_reported_grids_end_at_end_time(end, step, nodes):
    # Act
    times = FixedStepConfig(start_time=0.0, end_time=end, step_size=step).times

    # Assert: (10, 0.005) and (20, 0.01) had a node past the end before.
    assert times.shape == (nodes,)
    assert times[-1] == end


def test_the_reported_overshoot_is_refused():
    # Arrange: 0.8 is 53 1/3 steps of 0.015; the old grid ended at 0.81.
    config = FixedStepConfig(start_time=0.0, end_time=0.8, step_size=0.015)

    # Act / Assert
    with pytest.raises(ValueError, match="whole number of steps"):
        _ = config.times


@pytest.mark.parametrize(
    ("start", "end", "step"),
    [
        (0.0, 1.0, 0.0),
        (0.0, 1.0, -0.1),
        (1.0, 0.0, 0.1),
        (0.0, 1.0, float("nan")),
        (0.0, float("inf"), 0.1),
        (0.0, float("nan"), 0.1),
    ],
)
def test_a_step_that_cannot_reach_end_time_is_refused(start, end, step):
    # Arrange
    config = FixedStepConfig(start_time=start, end_time=end, step_size=step)

    # Act / Assert
    with pytest.raises(ValueError, match="step"):
        _ = config.times


def test_a_short_span_far_from_zero_is_a_whole_number_of_steps():
    # Arrange: decimal draws with |start| up to 1e4 and a span of a few steps,
    # where subtracting start loses more than the span's own rounding.
    generator = np.random.default_rng(131)

    for _ in range(_DRAWS):
        start = Decimal(int(generator.integers(-(10**7), 10**7))) / Decimal(1000)
        step = Decimal(int(generator.integers(1, 1000))) / Decimal(1000)
        steps = int(generator.integers(1, 10))
        end = float(start + steps * step)

        # Act
        config = FixedStepConfig(
            start_time=float(start),
            end_time=end,
            step_size=float(step),
        )

        # Assert
        assert config.n_steps == steps, (start, end, step)
        assert config.times[-1] == end


def test_the_tolerance_scales_with_the_endpoints_not_the_span():
    # Arrange: at t = 1e9 a node is only known to about 1e-7, so three steps of
    # 0.001 miss their span by 1e-7 - a third of a millionth of a step only to
    # the span, rounding to the endpoints.
    start = float(Decimal("1000000000.001"))
    end = float(Decimal("1000000000.001") + 3 * Decimal("0.001"))
    config = FixedStepConfig(start_time=start, end_time=end, step_size=0.001)

    # Act
    times = config.times

    # Assert
    assert abs((end - start) - 3 * 0.001) > 1e-9
    assert times.shape == (4,)
    assert times[-1] == end


@pytest.mark.parametrize(
    ("start", "step"),
    [(1.0e6, 1.0e-3), (6.0e4, 1.0e-5), (1.0e7, 1.0e-2), (1.0e9, 0.1)],
)
def test_a_half_step_span_far_from_zero_is_refused(start, step):
    # Arrange: 1e-9 of |t| is a step or more here; half a step is still not
    # rounding, and accepting it made the last step 1.5 (or 0.5) times the rest.
    end = start + 1000.5 * step
    config = FixedStepConfig(start_time=start, end_time=end, step_size=step)

    # Act / Assert
    with pytest.raises(ValueError, match="whole number of steps"):
        _ = config.times


@pytest.mark.parametrize(
    ("step", "accepted"),
    [("0.333333333333", True), ("0.333333", False)],
)
def test_a_step_rounded_from_one_third_is_accepted_only_to_rounding(step, accepted):
    # Arrange: three steps of 1/3 span 1; 12 digits miss it by 1e-12, which is
    # rounding, and 6 digits by 1e-6, which is a step that does not divide it.
    config = FixedStepConfig(start_time=0.0, end_time=1.0, step_size=float(step))

    # Act / Assert
    if accepted:
        assert config.n_steps == 3
        assert config.times[-1] == 1.0
    else:
        with pytest.raises(ValueError, match="whole number of steps"):
            _ = config.times


def test_an_empty_span_is_the_start_alone():
    # Act: start == end has always given the one-node grid; it still does.
    config = FixedStepConfig(start_time=0.5, end_time=0.5, step_size=0.1)

    # Assert
    assert config.times.tolist() == [0.5]
    assert config.n_steps == 0


def test_a_grid_arange_counted_right_is_unchanged_to_rounding():
    # Arrange: the old grid on the draws it counted right.
    draws = [
        (start, end, step)
        for start, end, step, steps in _multiples()
        if len(np.arange(start, end + step, step)) == steps + 1
    ]

    for start, end, step in draws:
        # Act
        old = np.arange(start, end + step, step)
        new = FixedStepConfig(start_time=start, end_time=end, step_size=step).times

        # Assert: the same nodes, a few ulps apart at most.
        assert np.allclose(new, old, rtol=0.0, atol=_scale(start, end) * len(old))


_CONFIGS = [
    EulerConfig,
    AdamsBashforth2Config,
    AdamsBashforth3Config,
    EulerMaruyamaConfig,
    MilsteinConfig,
    GottwaldMelbourneConfig,
    ZhanDuanLiLiConfig,
    SRK2Config,
    SRK3Config,
    SRK4Config,
    SRK5Config,
]
# Gottwald-Melbourne refuses a fast step epsilon longer than the step.
_EXTRA = {GottwaldMelbourneConfig: {"epsilon": 0.001}}


@pytest.mark.parametrize("config_class", _CONFIGS, ids=lambda c: c.__name__)
def test_every_fixed_step_config_ends_at_end_time(config_class):
    # Act: (0, 10, 0.005) is a multiple that arange miscounted.
    config = config_class(
        start_time=0.0,
        end_time=10.0,
        step_size=0.005,
        **_EXTRA.get(config_class, {}),
    )

    # Assert
    assert config.times.shape == (2001,)
    assert config.times[0] == 0.0
    assert config.times[-1] == 10.0


@pytest.mark.parametrize("config_class", _CONFIGS, ids=lambda c: c.__name__)
def test_every_fixed_step_config_refuses_a_partial_last_step(config_class):
    # Arrange
    config = config_class(
        start_time=0.0,
        end_time=0.8,
        step_size=0.015,
        **_EXTRA.get(config_class, {}),
    )

    # Act / Assert
    with pytest.raises(ValueError, match="whole number of steps"):
        _ = config.times


def _ordinary() -> DifferentialEquation:
    time_variable = Variable(name="t")
    variables = [Variable(name="y")]
    parameters = [Parameter(name="k", value=1.0)]
    return DifferentialEquation(
        variables=variables,
        time=time_variable,
        parameters=parameters,
        derivative=SimpleODEFunction(
            variables,
            parameters,
            [Variable(name="dy/dt")],
            time_variable,
        ),
    )


_SEED = {"random_seed": 1}


def _stochastic() -> DifferentialEquation:
    return GeometricBrownianMotionCase().equation()


_SOLVERS = [
    ("euler", EulerSolver, EulerConfig, _ordinary, {}),
    ("adams_bashforth2", AdamsBashforth2Solver, AdamsBashforth2Config, _ordinary, {}),
    ("adams_bashforth3", AdamsBashforth3Solver, AdamsBashforth3Config, _ordinary, {}),
    (
        "euler_maruyama",
        EulerMaruyamaSolver,
        EulerMaruyamaConfig,
        _stochastic,
        _SEED,
    ),
    ("milstein", MilsteinSolver, MilsteinConfig, _stochastic, _SEED),
    (
        "gottwald_melbourne",
        GottwaldMelbourneSolver,
        GottwaldMelbourneConfig,
        _stochastic,
        {"epsilon": 0.01, **_SEED},
    ),
    (
        "zhan_duan_li_li",
        ZhanDuanLiLiSolver,
        ZhanDuanLiLiConfig,
        _stochastic,
        _SEED,
    ),
    ("srk2", SRK2Solver, SRK2Config, _stochastic, _SEED),
    ("srk3", SRK3Solver, SRK3Config, _stochastic, _SEED),
    ("srk4", SRK4Solver, SRK4Config, _stochastic, _SEED),
]


@pytest.mark.parametrize(
    ("solver_class", "config_class", "make_equation", "extra"),
    [entry[1:] for entry in _SOLVERS],
    ids=[entry[0] for entry in _SOLVERS],
)
def test_a_solve_stops_at_end_time(solver_class, config_class, make_equation, extra):
    # Arrange: 2.3 = 230 steps of 0.01; arange's last node was an ulp short of
    # it, 2.2999999999999998.
    config = config_class(start_time=0.0, end_time=2.3, step_size=0.01, **extra)

    # Act
    curve = solver_class(config).solve(make_equation(), [1.0])

    # Assert: the last state is the state at end_time, after 230 steps.
    assert len(curve.time.discretization) == 231
    assert curve.time.discretization[-1] == 2.3


def test_the_ensemble_stops_at_end_time():
    # Arrange
    config = EulerMaruyamaConfig(
        start_time=0.0,
        end_time=2.3,
        step_size=0.01,
        random_seed=1,
    )

    # Act
    result = StochasticEnsemble(config).solve(
        _stochastic(),
        [1.0],
        paths=3,
        keep_every=1,
    )

    # Assert
    assert result.times.shape == (231,)
    assert result.times[-1] == 2.3


def test_adams_bashforth3_without_a_step_or_a_count_is_refused():
    # Arrange: neither field given, where np.arange(start, end, None) raised a
    # TypeError.
    config = AdamsBashforth3Config(start_time=0.0, end_time=1.0)

    # Act / Assert
    with pytest.raises(ValueError, match="step_size or a number_of_steps"):
        _ = config.times
