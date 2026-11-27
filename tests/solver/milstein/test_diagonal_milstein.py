"""Diagonal-noise Milstein in O(n^2) numbers, not an (n, n, n) array (DEQ-48).

With a vector diffusion ``G = diag(b)`` the products ``(dG_k/dx) G_j`` the
general Milstein step builds are ``b_j (d b_k / d x_j) e_k``: one nonzero entry
each, so the ``(n, n, n)`` array cost ``O(n^4)`` time and ``O(n^3)`` memory a
step to hold ``n^2`` numbers, and the ensemble paid that per path, as did its
Stratonovich drift correction. The oracles here are:

* the diagonal step against the general one, which is still what a matrix
  ``diag(b)`` takes - to rounding, per path and in the ensemble;
* the diagonal step against Kloeden and Platen's diagonal-noise Milstein scheme
  with the exact derivative, written out in this file (Section 10.3; restated
  for the Stratonovich form as eq. (17) of Garcia-Alvarez, arXiv:1102.4401, and
  for commuting Ito noise as eq. (1.10) of Wang and Gan, arXiv:1102.0662);
* values recorded before the change (3cecacb), to 1e-10 relative, for commuting
  diagonal noise and for vector noise that does not commute, whose Levy-area
  step must not move;
* the memory a step holds, which was cubic in ``n``.
"""

import tracemalloc
import warnings

import numpy as np
import pytest

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.solver.stochastic.ensemble import (
    BatchedCoefficients,
    StochasticEnsemble,
)
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config import (  # noqa: E501
    EulerMaruyamaConfig,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_config import (
    MilsteinConfig,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_solver import (
    MilsteinSolver,
)
from discrecontinual_equations.solver.stochastic.wiener import (
    BrownianPath,
    GaussianWienerSource,
)
from discrecontinual_equations.variable import Variable

_STEP = 2.0**-5
_RESOLUTION = 2.0**-8
_ROUNDING = 1.0e-13
_WEAK = 1.0e-8
# Central differences against the exact derivative: about 1e-10 relative a step.
_DIFFERENCES = 1.0e-8


def _b(x):
    return 0.3 * np.sin(x) + 0.2 * x


def _db(x):
    return 0.3 * np.cos(x) + 0.2


class DiagonalNoise(StochasticFunction):
    """``dx_i = -x_i dt + (0.3 sin x_i + 0.2 x_i) dW_i``: each b_i reads x_i only."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[float]:
        return [-value for value in point]

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[float]:
        return [float(_b(value)) for value in point]

    def eval_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        return -points

    def diffusion_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        return _b(points)


class DiagonalMatrixNoise(DiagonalNoise):
    """The same equation with the diffusion handed over as the matrix diag(b)."""

    def diffusion(  # type: ignore[override]
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[list[float]]:
        return np.diag(_b(np.asarray(point))).tolist()

    def diffusion_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        return np.einsum("pi,ij->pij", _b(points), np.eye(points.shape[1]))


class CrossVectorNoise(StochasticFunction):
    """``dx = -x/2 dt + cos(y) dW_1, dy = -y/2 dt + cos(x) dW_2``, as a vector.

    Diagonal in shape, but each b_i reads the other component: the noise does
    not commute, and the step must draw Levy areas as it did before DEQ-48.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[float]:
        return [-0.5 * point[0], -0.5 * point[1]]

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[float]:
        return [float(np.cos(point[1])), float(np.cos(point[0]))]

    def diffusion_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        return np.cos(points[:, ::-1])


class WeaklyCoupledNoise(DiagonalNoise):
    """``b_0`` also reads ``x_1``, by 1e-8: below the commutativity tolerance.

    The general step keeps the cross product ``b_1 d b_0 / d x_1`` that this
    admits, so the diagonal one must keep it too, the right way round.
    """

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[float]:
        values = [float(_b(value)) for value in point]
        values[0] += _WEAK * point[1]
        return values

    def diffusion_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        values = _b(points)
        values[:, 0] += _WEAK * points[:, 1]
        return values


class WeaklyCoupledMatrixNoise(WeaklyCoupledNoise):
    """:class:`WeaklyCoupledNoise` as the matrix diag(b), for the general step."""

    def diffusion(  # type: ignore[override]
        self,
        point: list[float],
        time: float | None = None,
    ) -> list[list[float]]:
        return np.diag(super().diffusion(point, time)).tolist()

    def diffusion_batch(
        self,
        points: np.ndarray,
        time: float | None = None,
    ) -> np.ndarray:
        values = super().diffusion_batch(points, time)
        return np.einsum("pi,ij->pij", values, np.eye(points.shape[1]))


def _equation(function_type, dimension: int) -> DifferentialEquation:
    time = Variable(name="t")
    variables = [Variable(name=f"x{i}") for i in range(dimension)]
    results = [Variable(name=f"r{i}") for i in range(dimension)]
    return DifferentialEquation(
        variables=variables,
        time=time,
        parameters=[],
        derivative=function_type(variables, [], results, time),
    )


def _final(  # noqa: PLR0913 (a problem, its calculus, and how it is driven)
    function_type,
    start,
    calculus,
    *,
    wiener=None,
    seed=None,
    end=1.0,
):
    config = MilsteinConfig(
        start_time=0.0,
        end_time=end,
        step_size=_STEP,
        random_seed=seed,
        calculus=calculus,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curve = MilsteinSolver(config, wiener=wiener).solve(
            _equation(function_type, len(start)),
            start,
        )
    return np.array([result.discretization[-1] for result in curve.results])


_START = [0.5, 1.0, 1.5]


# (vector form, matrix form, horizon). The weak coupling is compared over one
# step: its cross derivative is a difference quotient of a 1e-8 term, so a one-ulp
# difference in the state moves it by ulp / slope_step, about 1e-10, and two
# trajectories that agree to rounding at every step part by 1e-13 within a few.
_FORMS = [
    (DiagonalNoise, DiagonalMatrixNoise, 1.0),
    (WeaklyCoupledNoise, WeaklyCoupledMatrixNoise, _STEP),
]


@pytest.mark.parametrize(
    ("vector", "matrix", "end"),
    _FORMS,
    ids=["diagonal", "weak"],
)
@pytest.mark.parametrize("calculus", ["ito", "stratonovich"])
class TestAgainstTheGeneralStep:
    def test_per_path_matches_the_matrix_form_to_rounding(
        self,
        calculus,
        vector,
        matrix,
        end,
    ):
        # Arrange: one Brownian path for both forms of one equation.
        path = BrownianPath.sample(0.0, 1.0, _RESOLUTION, 3, seed=4)

        # Act
        diagonal = _final(vector, _START, calculus, wiener=path, end=end)
        general = _final(matrix, _START, calculus, wiener=path, end=end)

        # Assert
        np.testing.assert_allclose(diagonal, general, rtol=_ROUNDING, atol=0.0)

    def test_ensemble_matches_the_matrix_form_to_rounding(
        self,
        calculus,
        vector,
        matrix,
        end,
    ):
        # Arrange: the same seed draws the same increments for both forms.
        config = MilsteinConfig(
            start_time=0.0,
            end_time=end,
            step_size=_STEP,
            random_seed=8,
            calculus=calculus,
        )
        start = np.array(_START)

        # Act
        diagonal = StochasticEnsemble(config, scheme="milstein").solve(
            _equation(vector, 3),
            start,
            paths=16,
        )
        general = StochasticEnsemble(config, scheme="milstein").solve(
            _equation(matrix, 3),
            start,
            paths=16,
        )

        # Assert
        np.testing.assert_allclose(
            diagonal.final,
            general.final,
            rtol=_ROUNDING,
            atol=0.0,
        )


def _kloeden_platen(start, calculus, path):
    """Kloeden and Platen's diagonal-noise Milstein scheme, by hand.

    Ito: ``x + a h + b dW + (1/2) b b' (dW^2 - h)``; Stratonovich, with the
    drift as written: ``x + a h + b dW + (1/2) b b' dW^2``. ``b'`` exact.
    """
    x = np.array(start)
    for step in range(round(1.0 / _STEP)):
        dw = path.increments(step * _STEP, _STEP, x.size).delta_w
        square = dw * dw - (_STEP if calculus == "ito" else 0.0)
        x = x - x * _STEP + _b(x) * dw + 0.5 * _b(x) * _db(x) * square
    return x


@pytest.mark.parametrize("calculus", ["ito", "stratonovich"])
def test_the_step_is_kloeden_and_platens_diagonal_scheme(calculus):
    # Arrange
    path = BrownianPath.sample(0.0, 1.0, _RESOLUTION, 3, seed=6)

    # Act
    solved = _final(DiagonalNoise, _START, calculus, wiener=path)

    # Assert
    expected = _kloeden_platen(_START, calculus, path)
    np.testing.assert_allclose(solved, expected, rtol=_DIFFERENCES, atol=0.0)


# Recorded at 3cecacb, before DEQ-48: the general step (written out from its
# float.hex pins). b is 0.3 sin x + 0.2 x or cos of the other component, both
# vectorised ufuncs that AVX-512 runners round an ulp differently, and the step's
# central difference turns an ulp of b into about 1e-11 of the slope; so the
# values are held to 1e-10 relative, not to their bits (DEQ-160). Any change to
# the step itself moves them by far more (outputs/verification/deq-160).
_RECORDED = 1.0e-10
_BEFORE = {
    ("ito", "generator", "diagonal"): [
        0.24376757977694102,
        0.2213127018429813,
        0.6216643282091252,
    ],
    ("ito", "path", "diagonal"): [
        0.19609775912621272,
        0.4947142460158116,
        0.41715714128452286,
    ],
    ("stratonovich", "gaussian", "diagonal"): [
        0.44375415042829064,
        0.28682881074302036,
        0.7241890822829439,
    ],
    ("ito", "generator", "cross"): [0.45374360587941986, -0.6983875823963551],
    ("ito", "path", "cross"): [0.25581798358859636, 0.33340893132393445],
    ("stratonovich", "gaussian", "cross"): [
        0.5843524309232905,
        -0.4281220724413468,
    ],
}


@pytest.mark.parametrize(("calculus", "source", "noise"), list(_BEFORE))
def test_the_values_of_before_are_kept(calculus, source, noise):
    # Arrange
    function_type, start = {
        "diagonal": (DiagonalNoise, [0.5, 1.0, 1.5]),
        "cross": (CrossVectorNoise, [0.3, -0.2]),
    }[noise]
    wiener = {
        "generator": None,
        "path": BrownianPath.sample(0.0, 1.0, _RESOLUTION, len(start), seed=12),
        "gaussian": GaussianWienerSource(13),
    }[source]

    # Act
    final = _final(
        function_type,
        start,
        calculus,
        wiener=wiener,
        seed=11 if source == "generator" else None,
    )

    # Assert
    np.testing.assert_allclose(
        final,
        _BEFORE[(calculus, source, noise)],
        rtol=_RECORDED,
        atol=0.0,
    )


def test_vector_noise_that_does_not_commute_still_warns_and_draws_areas():
    # Arrange
    config = MilsteinConfig(start_time=0.0, end_time=0.25, step_size=_STEP)

    # Act / Assert
    with pytest.warns(UserWarning, match="does not commute"):
        MilsteinSolver(config).solve(_equation(CrossVectorNoise, 2), [0.3, -0.2])


def test_the_ensemble_steps_vector_noise_that_does_not_commute_with_areas():
    # Arrange: refused before DEQ-65. One member, one seed: the per-path
    # solver's increments and areas (see tests/solver/test_ensemble_levy_area.py).
    config = MilsteinConfig(
        start_time=0.0,
        end_time=0.25,
        step_size=_STEP,
        random_seed=4,
    )

    # Act
    with pytest.warns(UserWarning, match="Levy areas"):
        found = StochasticEnsemble(config, scheme="milstein").solve(
            _equation(CrossVectorNoise, 2),
            np.array([0.3, -0.2]),
        )
    expected = _final(CrossVectorNoise, [0.3, -0.2], "ito", seed=4, end=0.25)

    # Assert
    np.testing.assert_allclose(found.final[0], expected, rtol=_ROUNDING)


def test_batched_diagonal_slopes_are_the_jacobians_diagonal():
    # Arrange
    coefficients = BatchedCoefficients(_equation(DiagonalNoise, 3), "stratonovich")
    points = np.array([[0.5, 1.0, 1.5], [-0.2, 0.0, 2.0]])

    # Act
    slopes = coefficients.diagonal_slopes(points, 0.0)

    # Assert: the same differences, value for value.
    jacobians = coefficients.column_jacobians(points, 0.0)
    index = np.arange(3)
    assert np.array_equal(slopes, jacobians[:, index, index, index])


# The memory one step holds. Before DEQ-48 these were cubic in n: 6.5 MB for one
# path at n = 64, and 13 MB per (n, n, n) array for 50 paths at n = 32.
_PATH_DIMENSION = 64
_ENSEMBLE_DIMENSION = 32
_PATHS = 50
_MOST_PATH_BYTES = 1_000_000
_MOST_ENSEMBLE_BYTES = 2_000_000


def _peak(run) -> int:
    tracemalloc.start()
    try:
        run()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize(
    "function_type",
    [DiagonalNoise, WeaklyCoupledNoise],
    ids=["diagonal", "weak"],
)
def test_one_path_holds_no_cubic_array(function_type):
    # Arrange: a coupling below the tolerance commutes and stays on the fast path.
    start = np.linspace(0.5, 1.5, _PATH_DIMENSION).tolist()
    config = MilsteinConfig(start_time=0.0, end_time=2 * _STEP, step_size=_STEP)
    equation = _equation(function_type, _PATH_DIMENSION)

    # Act
    peak = _peak(lambda: MilsteinSolver(config).solve(equation, start))

    # Assert
    assert peak < _MOST_PATH_BYTES


@pytest.mark.parametrize(
    ("scheme", "config_type", "calculus"),
    [
        ("milstein", MilsteinConfig, "ito"),
        ("euler_maruyama", EulerMaruyamaConfig, "stratonovich"),
    ],
)
def test_an_ensemble_holds_no_cubic_array(scheme, config_type, calculus):
    # Arrange: Euler-Maruyama in Stratonovich exercises the drift correction.
    start = np.linspace(0.5, 1.5, _ENSEMBLE_DIMENSION)
    config = config_type(
        start_time=0.0,
        end_time=2 * _STEP,
        step_size=_STEP,
        random_seed=1,
        calculus=calculus,
    )
    equation = _equation(DiagonalNoise, _ENSEMBLE_DIMENSION)
    ensemble = StochasticEnsemble(config, scheme=scheme)

    # Act
    peak = _peak(lambda: ensemble.solve(equation, start, paths=_PATHS))

    # Assert
    assert peak < _MOST_ENSEMBLE_BYTES


class DeclaredDiagonalNoise(DiagonalNoise):
    """:class:`DiagonalNoise`, declaring what it is: each b_i reads x_i alone."""

    diagonal_noise = True
    evaluations = 0

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,
    ) -> list[float]:
        type(self).evaluations += 1
        return super().diffusion(point, time)


@pytest.mark.parametrize("calculus", ["ito", "stratonovich"])
@pytest.mark.parametrize("source", ["generator", "path"])
def test_declared_diagonal_noise_takes_the_same_step_bit_for_bit(calculus, source):
    # Arrange
    def wiener():
        if source == "generator":
            return None
        return BrownianPath.sample(0.0, 1.0, _RESOLUTION, 3, seed=2)

    seed = 5 if source == "generator" else None

    # Act
    found = _final(DiagonalNoise, _START, calculus, wiener=wiener(), seed=seed)
    declared = _final(
        DeclaredDiagonalNoise,
        _START,
        calculus,
        wiener=wiener(),
        seed=seed,
    )

    # Assert: the two moved points give every b_i the same input, hence the bits.
    # Exact is fair here: both runs are in this process, on this CPU (DEQ-160).
    assert [float(v).hex() for v in declared] == [float(v).hex() for v in found]


@pytest.mark.parametrize(
    ("scheme", "config_type", "calculus"),
    [
        ("milstein", MilsteinConfig, "ito"),
        ("milstein", MilsteinConfig, "stratonovich"),
        ("euler_maruyama", EulerMaruyamaConfig, "stratonovich"),
    ],
)
def test_declared_diagonal_noise_in_an_ensemble_bit_for_bit(
    scheme,
    config_type,
    calculus,
):
    # Arrange
    config = config_type(
        start_time=0.0,
        end_time=0.5,
        step_size=_STEP,
        random_seed=9,
        calculus=calculus,
    )

    # Act
    found, declared = (
        StochasticEnsemble(config, scheme=scheme).solve(
            _equation(function_type, 3),
            np.array(_START),
            paths=8,
        )
        for function_type in (DiagonalNoise, DeclaredDiagonalNoise)
    )

    # Assert
    assert np.array_equal(found.final, declared.final)


def test_declared_diagonal_noise_costs_a_fixed_number_of_evaluations_a_step():
    # Arrange: 16 components, whose undeclared step evaluates b 2 n + 1 = 33 times.
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

    # Assert: one for the drivers, the declaration's check once (1 + 2 n), then
    # b and its two moved copies a step.
    assert DeclaredDiagonalNoise.evaluations == 1 + (1 + 2 * dimension) + 3 * steps
