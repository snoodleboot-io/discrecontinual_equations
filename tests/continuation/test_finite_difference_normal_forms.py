"""The finite-difference derivative provider against automatic differentiation.

DEQ-167: ``FiniteDifferenceDerivatives`` took one step, ``1e-6``, for every
derivative order, so its third differences were rounding noise (van der Pol's
``l1`` came back ``+21`` against an exact ``-0.93``), and it cast complex
directions to real with a ``ComplexWarning``. Each order now takes
``eps^(1/(m+2))`` and a complex direction is split into real ones.

Oracle: every normal-form coefficient the library computes with the provider
(the ``classify`` coefficients, the full normal-form records of
:mod:`~discrecontinual_equations.continuation.codim2_normal_form`, and the
``cusp_coefficient``, ``cubic_fold_coefficient``, ``first_lyapunov`` and
``HopfCoefficients`` of :mod:`~discrecontinual_equations.continuation.normal_form`)
against the same computation with automatic differentiation, which is exact to
rounding. The systems are the codim-2 scenarios of ``test_codim2`` at their
exact codim-2 points, van der Pol at its Hopf point, and the embedded normal
forms of ``normal_form_systems`` hidden behind random similarity transforms,
so that every direction the forms are polarised along is generic. The six
scenarios also run end to end through ``Codim2Config(derivative_provider=
"finite_difference_derivatives")``.

Tolerance: second-order central stencils at their balancing steps leave about
``eps^(2/(m+2))`` relative, ``5e-7`` at ``m = 3`` and ``6e-6`` at ``m = 4``;
the provider-dependent coefficients here use orders one to three. Measured
worst over every comparison below: see ``_AGREEMENT``.
"""

import dataclasses
import warnings
from fractions import Fraction

import numpy as np
import pytest
from sweet_tea.abstract_factory import AbstractFactory

from discrecontinual_equations.continuation.codim2_builder import (
    Codim2Driver,
    CurveSeed,
)
from discrecontinual_equations.continuation.codim2_config import Codim2Config
from discrecontinual_equations.continuation.codim2_normal_form import (
    ResonanceError,
    bogdanov_takens_normal_form,
    classify,
    cusp_normal_form,
    hopf_hopf_normal_form,
    hopf_normal_form,
    zero_hopf_normal_form,
)
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
    DerivativeProvider,
    FiniteDifferenceDerivatives,
)
from discrecontinual_equations.continuation.normal_form import (
    HopfCoefficients,
    cubic_fold_coefficient,
    cusp_coefficient,
    first_lyapunov,
)
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.finite_difference import ORDER_STEPS, order_step
from tests.continuation import test_codim2 as scenarios
from tests.continuation.normal_form_systems import (
    EmbeddedBogdanovTakens,
    EmbeddedCusp,
    EmbeddedHopfHopf,
    EmbeddedZeroHopf,
    build,
    random_transform,
)


class _BogdanovTakens(EmbeddedBogdanovTakens):
    transform = random_transform(4, 11)


class _ZeroHopf(EmbeddedZeroHopf):
    transform = random_transform(5, 5)


class _HopfHopf(EmbeddedHopfHopf):
    transform = random_transform(5, 17)


class _Cusp(EmbeddedCusp):
    transform = random_transform(3, 23)


# Largest disagreement allowed between the two providers, relative to the
# largest coefficient of the same record, so that a coefficient that is exactly
# zero is held to the record's scale. Measured worst 2.4e-7 (van der Pol's
# HopfCoefficients, whose third differences dominate); the next is 5.1e-9. The
# Done-when bound is 1e-5; this is four times the worst measured.
_AGREEMENT = 1.0e-6
_VAN_DER_POL_EPSILON = 0.2


def _problem(
    equation: DifferentialEquation,
    provider: DerivativeProvider,
) -> TwoParameterProblem:
    return TwoParameterProblem(equation, 0, 1, 0.0, provider)


def _scenario(function_type, count: int, first: float, second: float):
    return scenarios._equation(function_type, count, first, second)  # noqa: SLF001


def _numbers(record: object) -> dict[str, np.ndarray]:
    """Every numeric field of a normal-form record, flattened, by name."""
    if isinstance(record, dict):
        items = record.items()
    elif dataclasses.is_dataclass(record):
        items = ((f.name, getattr(record, f.name)) for f in dataclasses.fields(record))
    else:
        return {"value": np.atleast_1d(np.asarray(record, dtype=complex))}
    numbers: dict[str, np.ndarray] = {}
    for name, value in items:
        if dataclasses.is_dataclass(value):
            for inner, array in _numbers(value).items():
                numbers[f"{name}.{inner}"] = array
        elif isinstance(value, (int, float, complex, np.ndarray, np.number)):
            numbers[name] = np.atleast_1d(np.asarray(value, dtype=complex))
    return numbers


def _assert_agree(exact: object, approximate: object) -> None:
    expected = _numbers(exact)
    computed = _numbers(approximate)
    assert expected.keys() == computed.keys()
    assert expected, "nothing to compare"
    scale = max(float(np.max(np.abs(values))) for values in expected.values())
    for name, values in expected.items():
        error = float(np.max(np.abs(computed[name] - values)))
        assert error <= _AGREEMENT * scale, (name, error, scale)


def _both(build_equation, compute):
    """``compute(problem)`` with autodiff and with finite differences.

    The finite-difference run turns every warning into an error, so a
    ``ComplexWarning`` fails it.
    """
    exact = compute(_problem(build_equation(), AutomaticDifferentiation()))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        approximate = compute(_problem(build_equation(), FiniteDifferenceDerivatives()))
    return exact, approximate


# (name, equation builder, computation) at an exact codim-2 or Hopf point.
_ORIGIN2 = np.zeros(2)
_CASES = [
    (
        "bt scenario: bogdanov_takens_normal_form",
        lambda: _scenario(scenarios.BogdanovTakensNormalForm, 2, 0.0, 0.0),
        lambda problem: bogdanov_takens_normal_form(problem, _ORIGIN2, 0.0, 0.0),
    ),
    (
        "bt scenario: classify",
        lambda: _scenario(scenarios.BogdanovTakensNormalForm, 2, 0.0, 0.0),
        lambda problem: classify("bogdanov_takens", problem, _ORIGIN2, 0.0, 0.0)[1],
    ),
    (
        "zero-hopf scenario: zero_hopf_normal_form",
        lambda: _scenario(scenarios.ZeroHopfSystem, 3, 0.0, 0.0),
        lambda problem: zero_hopf_normal_form(problem, np.zeros(3), 0.0, 0.0),
    ),
    (
        "zero-hopf scenario: classify",
        lambda: _scenario(scenarios.ZeroHopfSystem, 3, 0.0, 0.0),
        lambda problem: classify("zero_hopf", problem, np.zeros(3), 0.0, 0.0)[1],
    ),
    (
        # The first pair's h20 solve meets the second pair at 2 i: the 1:2
        # resonance makes it singular for either provider.
        "hopf-hopf scenario: l1 of the second pair",
        lambda: _scenario(scenarios.HopfHopfSystem, 4, 0.0, 0.0),
        lambda problem: first_lyapunov(problem, np.zeros(4), 0.0, 0.0, 2.0),
    ),
    (
        "cusp scenario: cusp_normal_form",
        lambda: _scenario(scenarios.CuspNormalForm, 1, 0.0, 0.0),
        lambda problem: cusp_normal_form(problem, np.zeros(1), 0.0, 0.0),
    ),
    (
        "cusp scenario: classify",
        lambda: _scenario(scenarios.CuspNormalForm, 1, 0.0, 0.0),
        lambda problem: classify("cusp", problem, np.zeros(1), 0.0, 0.0)[1],
    ),
    (
        "cusp scenario: cubic_fold_coefficient",
        lambda: _scenario(scenarios.CuspNormalForm, 1, 0.0, 0.0),
        lambda problem: cubic_fold_coefficient(problem, np.zeros(1), 0.0, 0.0),
    ),
    (
        "cusp scenario: cusp_coefficient on the fold at x = 0.3",
        lambda: _scenario(scenarios.CuspNormalForm, 1, -0.054, 0.27),
        lambda problem: cusp_coefficient(problem, np.array([0.3]), -0.054, 0.27),
    ),
    (
        "bautin scenario: hopf_normal_form",
        lambda: _scenario(scenarios.BautinSystem, 2, 0.0, 0.5),
        lambda problem: hopf_normal_form(problem, _ORIGIN2, 0.0, 0.5),
    ),
    (
        "bautin scenario: HopfCoefficients",
        lambda: _scenario(scenarios.BautinSystem, 2, 0.0, 0.5),
        lambda problem: _hopf_record(problem, _ORIGIN2, (0.0, 0.5), 1.0),
    ),
    (
        "van der Pol: HopfCoefficients",
        lambda: _scenario(scenarios.VanDerPolFastSlow, 2, 1.0, _VAN_DER_POL_EPSILON),
        lambda problem: _hopf_record(
            problem,
            np.array([1.0, -2.0 / 3.0]),
            (1.0, _VAN_DER_POL_EPSILON),
            np.sqrt(_VAN_DER_POL_EPSILON),
        ),
    ),
    (
        "embedded bogdanov-takens: bogdanov_takens_normal_form",
        lambda: build(_BogdanovTakens, 4),
        lambda problem: bogdanov_takens_normal_form(problem, np.zeros(4), 0.0, 0.0),
    ),
    (
        "embedded zero-hopf: zero_hopf_normal_form",
        lambda: build(_ZeroHopf, 5),
        lambda problem: zero_hopf_normal_form(problem, np.zeros(5), 0.0, 0.0),
    ),
    (
        # Not at the zero-Hopf point: there A is singular and h11 = -A^-1 B is
        # not defined for either provider.
        "embedded hopf-hopf: HopfCoefficients of the first pair",
        lambda: build(_HopfHopf, 5),
        lambda problem: _hopf_record(
            problem,
            np.zeros(5),
            (0.0, 0.0),
            EmbeddedHopfHopf.omega1,
        ),
    ),
    (
        "embedded hopf-hopf: hopf_hopf_normal_form",
        lambda: build(_HopfHopf, 5),
        lambda problem: hopf_hopf_normal_form(problem, np.zeros(5), 0.0, 0.0),
    ),
    (
        "embedded hopf-hopf: HopfCoefficients of the second pair",
        lambda: build(_HopfHopf, 5),
        lambda problem: _hopf_record(
            problem,
            np.zeros(5),
            (0.0, 0.0),
            EmbeddedHopfHopf.omega2,
        ),
    ),
    (
        "embedded cusp: cusp_normal_form",
        lambda: build(_Cusp, 3),
        lambda problem: cusp_normal_form(problem, np.zeros(3), 0.0, 0.0),
    ),
    (
        "embedded cusp: cubic_fold_coefficient",
        lambda: build(_Cusp, 3),
        lambda problem: cubic_fold_coefficient(problem, np.zeros(3), 0.0, 0.0),
    ),
]


def _hopf_record(
    problem: TwoParameterProblem,
    state: np.ndarray,
    parameters: tuple[float, float],
    omega: float,
) -> dict[str, object]:
    coefficients = HopfCoefficients(problem, state, *parameters, omega)
    return {
        "g21": coefficients.g21,
        "h11": coefficients.h11,
        "h20": coefficients.h20,
        "l1": coefficients.first_lyapunov,
    }


@pytest.mark.parametrize(
    ("build_equation", "compute"),
    [case[1:] for case in _CASES],
    ids=[case[0] for case in _CASES],
)
def test_every_coefficient_matches_automatic_differentiation(build_equation, compute):
    exact, approximate = _both(build_equation, compute)
    _assert_agree(exact, approximate)


def test_the_resonant_double_hopf_is_refused_by_both():
    # The 1:2 scenario has no coefficients; both providers must say so.
    def compute(problem):
        with pytest.raises(ResonanceError):
            hopf_hopf_normal_form(problem, np.zeros(4), 0.0, 0.0)
        return classify("hopf_hopf", problem, np.zeros(4), 0.0, 0.0)

    exact, approximate = _both(
        lambda: _scenario(scenarios.HopfHopfSystem, 4, 0.0, 0.0),
        compute,
    )
    assert exact == approximate == ("resonant 1:2", {})


class TestVanDerPol:
    """The Done-when case: ``l1`` of van der Pol to ``1e-5`` relative."""

    exact = -1.0 / (2.0 * np.sqrt(_VAN_DER_POL_EPSILON) * (1.0 + _VAN_DER_POL_EPSILON))

    def _l1(self, provider: DerivativeProvider) -> float:
        equation = _scenario(
            scenarios.VanDerPolFastSlow,
            2,
            1.0,
            _VAN_DER_POL_EPSILON,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            return first_lyapunov(
                _problem(equation, provider),
                np.array([1.0, -2.0 / 3.0]),
                1.0,
                _VAN_DER_POL_EPSILON,
                np.sqrt(_VAN_DER_POL_EPSILON),
            )

    def test_the_default_provider(self):
        l1 = self._l1(FiniteDifferenceDerivatives())
        assert abs(l1 - self.exact) <= 1.0e-5 * abs(self.exact)

    def test_the_provider_the_registry_builds(self):
        # Codim2Config's opt-in key builds the provider with no configuration.
        provider = AbstractFactory[DerivativeProvider].create(
            "finite_difference_derivatives",
        )
        l1 = self._l1(provider)
        assert abs(l1 - self.exact) <= 1.0e-5 * abs(self.exact)


# The six codim-2 scenarios of test_codim2, through the config path.
_SCENARIOS = {
    "bt_hopf": (
        (scenarios.BogdanovTakensNormalForm, 2, -0.25, -0.5),
        {"state": [-0.5, 0.0], "parameter_a": -0.25, "frequency": 1.0},
        {
            "continuation_parameter_index": 0,
            "second_parameter_index": 1,
            "curve": "hopf",
            "codim2_detectors": ["bogdanov_takens"],
            "initial_parameter": -0.5,
            "direction": 1,
            "parameter_lower_bound": -0.6,
            "parameter_upper_bound": 0.05,
        },
    ),
    "bt_fold": (
        (scenarios.BogdanovTakensNormalForm, 2, 0.0, -0.5),
        {"state": [0.0, 0.0], "parameter_a": 0.0},
        {
            "continuation_parameter_index": 0,
            "second_parameter_index": 1,
            "curve": "fold",
            "codim2_detectors": ["fold_bogdanov_takens"],
            "initial_parameter": -0.5,
            "direction": 1,
            "parameter_lower_bound": -0.6,
            "parameter_upper_bound": 0.5,
        },
    ),
    "zero_hopf": (
        (scenarios.ZeroHopfSystem, 3, 0.5, 0.0),
        {"state": [0.5**0.5, 0.0, 0.0], "parameter_a": 0.0, "frequency": 1.0},
        {
            "continuation_parameter_index": 1,
            "second_parameter_index": 0,
            "curve": "hopf",
            "codim2_detectors": ["zero_hopf"],
            "initial_parameter": 0.5,
            "direction": -1,
            "parameter_lower_bound": -0.1,
            "parameter_upper_bound": 0.6,
        },
    ),
    "hopf_hopf": (
        (scenarios.HopfHopfSystem, 4, 0.0, 0.5),
        {"state": [0.0] * 4, "parameter_a": 0.0, "frequency": 1.0},
        {
            "continuation_parameter_index": 0,
            "second_parameter_index": 1,
            "curve": "hopf",
            "codim2_detectors": ["hopf_hopf"],
            "initial_parameter": 0.5,
            "direction": -1,
            "parameter_lower_bound": -0.3,
            "parameter_upper_bound": 0.6,
        },
    ),
    "cusp": (
        (scenarios.CuspNormalForm, 1, -0.25, 0.75),
        {"state": [0.5], "parameter_a": 0.75},
        {
            "continuation_parameter_index": 1,
            "second_parameter_index": 0,
            "curve": "fold",
            "codim2_detectors": ["cusp"],
            "initial_parameter": -0.25,
            "direction": 1,
            "parameter_lower_bound": -0.35,
            "parameter_upper_bound": 0.35,
        },
    ),
    "bautin": (
        (scenarios.BautinSystem, 2, 0.0, 0.5),
        {"state": [0.0, 0.0], "parameter_a": 0.0, "frequency": 1.0},
        {
            "continuation_parameter_index": 0,
            "second_parameter_index": 1,
            "curve": "hopf",
            "codim2_detectors": ["generalized_hopf"],
            "initial_parameter": 0.5,
            "direction": -1,
            "parameter_lower_bound": -0.6,
            "parameter_upper_bound": 0.6,
        },
    ),
}
# Where a curve crosses a codim-2 point, the two providers' curves differ by the
# finite-difference error of the augmented Jacobian Newton converged with, and
# the located point by its interpolation along that curve: measured 1.1e-10.
_LOCATION = 1.0e-8


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_the_codim2_scenarios_through_the_config(name):
    equation_arguments, seed, settings = _SCENARIOS[name]

    def run(provider: str):
        config = Codim2Config(derivative_provider=provider, **settings)
        return Codim2Driver.run(
            config,
            _scenario(*equation_arguments),
            CurveSeed(**seed),
        )

    exact = run("automatic_differentiation")
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        approximate = run("finite_difference_derivatives")
    assert [p.kind for p in approximate.points] == [p.kind for p in exact.points]
    assert exact.points
    for expected, computed in zip(exact.points, approximate.points, strict=True):
        assert computed.classification == expected.classification
        assert abs(computed.parameter_a - expected.parameter_a) <= _LOCATION
        assert abs(computed.parameter_b - expected.parameter_b) <= _LOCATION
        if expected.coefficients:
            _assert_agree(expected.coefficients, computed.coefficients)
        else:
            assert computed.coefficients == {}


class TestOrderSteps:
    """``eps^(1/(m+2))`` per order, correctly rounded, and its edges."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_each_step_is_the_correctly_rounded_root_of_eps(self, order):
        # Exact rational check, independent of any pow kernel: the halfway
        # points either side of the literal bracket eps^(1/(m+2)).
        step = Fraction(order_step(order))
        below = Fraction(np.nextafter(order_step(order), 0.0))
        above = Fraction(np.nextafter(order_step(order), 1.0))
        eps = Fraction(float(np.finfo(float).eps))
        power = order + 2
        assert ((step + below) / 2) ** power <= eps < ((step + above) / 2) ** power

    @pytest.mark.parametrize("order", [0, 5])
    def test_orders_outside_one_to_four_are_refused(self, order):
        with pytest.raises(ValueError, match="orders 1 to 4"):
            order_step(order)

    def test_the_provider_takes_them_by_default(self):
        provider = FiniteDifferenceDerivatives()
        assert [provider.relative_step(m) for m in range(1, 5)] == list(ORDER_STEPS)
        # s = (2, 1): max |d_j| / s_j = 0.25, so t = r_m 0.5 / 0.25 = 2 r_m.
        point = np.array([2.0, -0.5])
        direction = np.array([0.5, 0.25])
        for order in range(1, 5):
            step = provider.step(point, direction, order)
            assert step == pytest.approx(2.0 * ORDER_STEPS[order - 1], rel=1.0e-15)

    def test_a_complex_point_is_refused(self):
        provider = FiniteDifferenceDerivatives()
        function = _scenario(scenarios.CuspNormalForm, 1, 0.0, 0.0).derivative
        with pytest.raises(ValueError, match="real points"):
            provider.taylor(function, np.array([1.0 + 1.0e-3j]), np.ones(1), 1, 0.0)
        # A complex array with a zero imaginary part is a real point.
        real = provider.taylor(function, np.array([0.5 + 0.0j]), np.ones(1), 1, 0.0)
        assert real[1][0] == pytest.approx(-0.75, rel=1.0e-8)

    def test_a_complex_direction_of_a_cubic_is_exact_to_rounding(self):
        # f(x, y) = (x^3, x y^2): P_3 along d = u + i v is (d_x^3, d_x d_y^2),
        # complex, written out; only the real stencils at real points are used.
        class Cubic(scenarios.CuspNormalForm):
            def eval(self, point, time=None):  # noqa: ARG002 (base signature)
                x, y = point[0], point[1]
                return [x * x * x, x * y * y]

        function = _scenario(Cubic, 2, 0.0, 0.0).derivative
        direction = np.array([0.6 + 0.3j, -0.2 + 0.7j])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            computed = FiniteDifferenceDerivatives().taylor(
                function,
                np.zeros(2),
                direction,
                3,
                0.0,
            )
        dx, dy = direction
        expected = np.array([dx**3, dx * dy * dy])
        assert np.max(np.abs(computed[3] - expected)) <= 1.0e-8 * np.max(
            np.abs(expected),
        )
        assert np.max(np.abs(computed[1])) <= 1.0e-10
