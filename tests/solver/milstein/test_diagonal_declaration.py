"""A wrong ``diagonal_noise`` declaration is caught, not stepped through (DEQ-48).

``StochasticFunction.diagonal_noise = True`` promises that ``b_i`` reads ``x_i``
alone, and the solvers then read every ``d b_i / d x_i`` from two evaluations that
move all components at once. A function that declares it but couples components
was stepped silently with the wrong slopes: ``CrossVectorNoise`` declared diagonal
fell to strong order 0.5 with ten times the error. Once per solve, at the first
state the shortcut would be used (the initial state), the solvers now read the
full Jacobian of ``b`` - ``2 n`` evaluations - and when an off-diagonal entry
exceeds the commutativity tolerance relative to the largest entry they warn,
naming the function, and solve as if nothing had been declared, bit for bit.

The check sees only the initial state: a coupling that vanishes there is not
caught.

A second guard here: the ensemble's commutativity verdict treated a non-finite
product as no product at all, so one path gone to NaN hid the coupling of every
other path and non-commuting noise was stepped as commuting. NaN is now
non-commuting, as it is in the per-path check, and the other paths take their
Levy areas (the ensemble refused such noise before DEQ-65).
"""

import warnings

import numpy as np
import pytest

from discrecontinual_equations.solver.stochastic.ensemble import StochasticEnsemble
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config import (  # noqa: E501
    EulerMaruyamaConfig,
)
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_solver import (  # noqa: E501
    EulerMaruyamaSolver,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_config import (
    MilsteinConfig,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_solver import (
    MilsteinSolver,
)
from discrecontinual_equations.solver.stochastic.srk2.srk2_config import SRK2Config
from discrecontinual_equations.solver.stochastic.srk2.srk2_solver import SRK2Solver
from discrecontinual_equations.solver.stochastic.wiener import BrownianPath
from tests.solver.milstein.test_diagonal_milstein import (
    CrossVectorNoise,
    DeclaredDiagonalNoise,
    DiagonalMatrixNoise,
    DiagonalNoise,
    _equation,
)
from tests.solver.test_ensemble_levy_area import (
    EachPathItsOwn,
    Replay,
    ensemble_draws,
)

_STEP = 2.0**-5
_RESOLUTION = 2.0**-8
_CROSS_START = [0.3, -0.2]
_DIAGONAL_START = [0.5, 1.0, 1.5]


class DeclaredCrossVectorNoise(CrossVectorNoise):
    """:class:`CrossVectorNoise`, declaring diagonal noise it does not have."""

    diagonal_noise = True


def _per_path(solver_type, config_type, function_type, start, calculus):
    """Final state and the warnings of one solve, driven by a fixed path."""
    config = config_type(
        start_time=0.0,
        end_time=0.5,
        step_size=_STEP,
        calculus=calculus,
    )
    path = BrownianPath.sample(0.0, 0.5, _RESOLUTION, len(start), seed=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        curve = solver_type(config, wiener=path).solve(
            _equation(function_type, len(start)),
            start,
        )
    final = [float(result.discretization[-1]).hex() for result in curve.results]
    return final, [str(warning.message) for warning in caught]


def _declaration_warnings(messages):
    return [message for message in messages if "diagonal_noise" in message]


_PER_PATH = [
    (MilsteinSolver, MilsteinConfig, "ito"),
    (MilsteinSolver, MilsteinConfig, "stratonovich"),
    (EulerMaruyamaSolver, EulerMaruyamaConfig, "stratonovich"),
    (SRK2Solver, SRK2Config, "stratonovich"),
]


@pytest.mark.parametrize(
    ("solver_type", "config_type", "calculus"),
    _PER_PATH,
    ids=["milstein-ito", "milstein-stratonovich", "em-stratonovich", "srk2-strat"],
)
def test_a_wrong_declaration_warns_and_solves_as_undeclared(
    solver_type,
    config_type,
    calculus,
):
    # Act
    found, _ = _per_path(
        solver_type,
        config_type,
        CrossVectorNoise,
        _CROSS_START,
        calculus,
    )
    declared, messages = _per_path(
        solver_type,
        config_type,
        DeclaredCrossVectorNoise,
        _CROSS_START,
        calculus,
    )

    # Assert: one warning naming the function, and the undeclared step's bits.
    warned = _declaration_warnings(messages)
    assert len(warned) == 1
    assert "DeclaredCrossVectorNoise" in warned[0]
    assert declared == found


@pytest.mark.parametrize(
    ("solver_type", "config_type", "calculus"),
    _PER_PATH,
    ids=["milstein-ito", "milstein-stratonovich", "em-stratonovich", "srk2-strat"],
)
def test_a_true_declaration_is_kept_without_a_warning(
    solver_type,
    config_type,
    calculus,
):
    # Act
    found, _ = _per_path(
        solver_type,
        config_type,
        DiagonalNoise,
        _DIAGONAL_START,
        calculus,
    )
    declared, messages = _per_path(
        solver_type,
        config_type,
        DeclaredDiagonalNoise,
        _DIAGONAL_START,
        calculus,
    )

    # Assert
    assert _declaration_warnings(messages) == []
    assert declared == found


def test_the_check_costs_two_n_evaluations_once_per_solve():
    # Arrange: 16 components; the declared step evaluates b three times a step.
    dimension, steps = 16, 4
    config = MilsteinConfig(
        start_time=0.0,
        end_time=steps * _STEP,
        step_size=_STEP,
        random_seed=1,
    )
    DeclaredDiagonalNoise.evaluations = 0

    # Act
    MilsteinSolver(config).solve(
        _equation(DeclaredDiagonalNoise, dimension),
        np.linspace(0.5, 1.5, dimension).tolist(),
    )

    # Assert: the drivers, then the check's 1 + 2 n, then three a step.
    assert DeclaredDiagonalNoise.evaluations == 1 + (1 + 2 * dimension) + 3 * steps


def _ensemble(function_type, start, scheme, config_type, calculus):
    config = config_type(
        start_time=0.0,
        end_time=0.5,
        step_size=_STEP,
        random_seed=6,
        calculus=calculus,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = StochasticEnsemble(config, scheme=scheme).solve(
            _equation(function_type, len(start)),
            np.array(start),
            paths=8,
        )
    return result.final, [str(warning.message) for warning in caught]


def test_an_ensemble_with_a_wrong_declaration_warns_and_solves_as_undeclared():
    # Act
    found, _ = _ensemble(
        CrossVectorNoise,
        _CROSS_START,
        "euler_maruyama",
        EulerMaruyamaConfig,
        "stratonovich",
    )
    declared, messages = _ensemble(
        DeclaredCrossVectorNoise,
        _CROSS_START,
        "euler_maruyama",
        EulerMaruyamaConfig,
        "stratonovich",
    )

    # Assert
    warned = _declaration_warnings(messages)
    assert len(warned) == 1
    assert "DeclaredCrossVectorNoise" in warned[0]
    assert np.array_equal(declared, found)


def test_a_milstein_ensemble_with_a_wrong_declaration_solves_as_undeclared():
    # The undeclared noise does not commute, which the ensemble steps with Levy
    # areas (DEQ-65; it refused before); declared, it was stepped as diagonal. It
    # must warn, then take the undeclared step, bit for bit.
    found, _ = _ensemble(
        CrossVectorNoise,
        _CROSS_START,
        "milstein",
        MilsteinConfig,
        "ito",
    )
    declared, messages = _ensemble(
        DeclaredCrossVectorNoise,
        _CROSS_START,
        "milstein",
        MilsteinConfig,
        "ito",
    )

    # Assert
    warned = _declaration_warnings(messages)
    assert len(warned) == 1
    assert "DeclaredCrossVectorNoise" in warned[0]
    assert np.array_equal(declared, found)


def test_an_ensemble_with_a_true_declaration_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ensemble(
            DeclaredDiagonalNoise,
            _DIAGONAL_START,
            "milstein",
            MilsteinConfig,
            "ito",
        )


# One path started at NaN, the others with noise that does not commute.
_WITH_NAN = np.array([[np.nan, -0.2], [0.3, -0.2], [0.1, 0.4]])


@pytest.mark.parametrize("source", ["generator", "paths"])
@pytest.mark.parametrize(
    "function_type",
    [CrossVectorNoise, DiagonalMatrixNoise],
    ids=["vector", "matrix"],
)
def test_a_nan_path_does_not_hide_the_others_from_the_commutativity_check(
    function_type,
    source,
):
    # Arrange: the matrix form is diag(b) on three components, which commutes; it
    # stands for the general check, which must not wave NaN through either. A
    # NaN hiding the vector noise's coupling would step the other paths without
    # their areas, which the per-path solver, judging each alone, gives them.
    config = MilsteinConfig(
        start_time=0.0,
        end_time=4 * _STEP,
        step_size=_STEP,
        random_seed=6,
    )
    start = _WITH_NAN
    if function_type is DiagonalMatrixNoise:
        start = np.array([[np.nan, 1.0, 1.5], [0.5, 1.0, 1.5]])
    count, channels = start.shape
    if source == "generator":
        increments, areas = ensemble_draws(config, count, channels)
        sources = [
            Replay([step[p] for step in increments], [step[p] for step in areas])
            for p in range(count)
        ]
        wiener = None
    else:
        sources = [
            BrownianPath.sample(0.0, 4 * _STEP, _RESOLUTION, channels, seed=p)
            for p in range(count)
        ]
        wiener = EachPathItsOwn(
            [
                BrownianPath.sample(0.0, 4 * _STEP, _RESOLUTION, channels, seed=p)
                for p in range(count)
            ],
        )

    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = StochasticEnsemble(config, scheme="milstein", wiener=wiener).solve(
            _equation(function_type, channels),
            start,
        )
        expected = [
            MilsteinSolver(config, wiener=sources[p]).solve(
                _equation(function_type, channels),
                list(start[p]),
            )
            for p in range(1, count)
        ]

    # Assert
    assert np.all(np.isnan(found.final[0]))
    for p, curve in enumerate(expected, start=1):
        np.testing.assert_allclose(
            found.final[p],
            [result.discretization[-1] for result in curve.results],
            rtol=1e-12,
        )
