"""Invariant manifolds and the charts that parameterise them."""

from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.manifold import (
    StableManifold,
    TaylorManifold,
    UnstableManifold,
)
from discrecontinual_equations.continuation.region_analysis import (
    _jacobian,
    describe_region,
    find_equilibria,
)
from discrecontinual_equations.continuation.shilnikov import classify_saddle_focus
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import (
    Alpha,
    BetaOne,
    BetaTwo,
    BogdanovTakensField,
    HopfField,
    LorenzField,
    State,
    Time,
    _integrate,
    _saddle,
)

_SADDLE_JACOBIAN = np.array([[0.0, 1.0], [1.0, 0.0]])


_SADDLE_JACOBIAN = np.array([[0.0, 1.0], [1.0, 0.0]])


class TestInvariantManifold(TestCase):
    def test_unstable_manifold_is_flow_invariant(self):
        chart = TaylorManifold(order=6).compute(
            _saddle(),
            np.zeros(2),
            _SADDLE_JACOBIAN,
            UnstableManifold(),
        )
        start = chart.point([0.05])
        flowed = chart.flow_image([0.05], 0.5)
        integrated = _integrate(_saddle(), start, 0.001, 500)
        assert np.linalg.norm(integrated - flowed) < 1.0e-8

    def test_unstable_manifold_lies_on_homoclinic_level_set(self):
        chart = TaylorManifold(order=6).compute(
            _saddle(),
            np.zeros(2),
            _SADDLE_JACOBIAN,
            UnstableManifold(),
        )
        for theta in np.linspace(0.0, 0.5, 6):
            x, y = chart.point([float(theta)])
            energy = 0.5 * y * y - 0.5 * x * x + x**3 / 3.0
            assert abs(energy) < 1.0e-6

    def test_stable_manifold_has_negative_eigenvalue(self):
        chart = TaylorManifold(order=4).compute(
            _saddle(),
            np.zeros(2),
            _SADDLE_JACOBIAN,
            StableManifold(),
        )
        assert chart.eigenvalues[0] < 0.0


class TestManifoldChartOffTheOrigin(TestCase):
    """The chart's constant term is the equilibrium; ``point`` must not add it twice.

    Every other chart test sits its saddle at the origin, where adding the
    equilibrium a second time changes nothing. At (1, 0) it moved every chart
    point by (1, 0), so a manifold started from the chart began off the manifold.
    """

    def test_chart_starts_at_the_equilibrium(self):
        class Shifted(DeterministicFunction):
            def eval(self, point, time=None):  # noqa: ARG002 (base signature)
                return [point[0] - 1.0, -point[1]]

        function = Shifted(
            variables=[State(), State()],
            parameters=[Alpha(value=0.0)],
            results=[State(), State()],
            time=None,
        )
        equilibrium = np.array([1.0, 0.0])
        jacobian = np.array([[1.0, 0.0], [0.0, -1.0]])
        for selection, direction in (
            (UnstableManifold(), np.array([1.0, 0.0])),
            (StableManifold(), np.array([0.0, 1.0])),
        ):
            chart = TaylorManifold(order=4).compute(
                function,
                equilibrium,
                jacobian,
                selection,
            )
            assert np.allclose(chart.point([0.0]), equilibrium)
            step = chart.point([1.0e-3]) - equilibrium
            assert np.allclose(np.abs(step), 1.0e-3 * direction, atol=1.0e-9)


class TestRegionAnalysis(TestCase):
    """Region labels from equilibria, stability, and Poincare-Bendixson cycles."""

    def _hopf(self, mu: float) -> DifferentialEquation:
        parameters = [Alpha(value=mu)]
        return DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=parameters,
            derivative=HopfField(
                variables=[State(), State()],
                parameters=parameters,
                results=[State(), State()],
                time=None,
            ),
        )

    def test_supercritical_hopf_reports_limit_cycle(self):
        label = describe_region(
            self._hopf(0.25).derivative,
            [np.zeros(2)],
            cycle_start=np.array([0.05, 0.0]),
        )
        assert label == "unstable focus + limit cycle"

    def test_stable_focus_has_no_cycle(self):
        label = describe_region(
            self._hopf(-0.25).derivative,
            [np.zeros(2)],
            cycle_start=np.array([0.5, 0.0]),
        )
        assert label == "stable focus"

    def _bt(self, first: float, second: float) -> DifferentialEquation:
        parameters = [BetaOne(value=first), BetaTwo(value=second)]
        return DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=parameters,
            derivative=BogdanovTakensField(
                variables=[State(), State()],
                parameters=parameters,
                results=[State(), State()],
                time=None,
            ),
        )

    def test_bogdanov_takens_regions(self):
        seeds = [np.array([0.4, 0.0]), np.array([-0.4, 0.0])]
        top = describe_region(self._bt(0.05, 0.16).derivative, seeds)
        above = describe_region(self._bt(-0.012, 0.16).derivative, seeds)
        below = describe_region(self._bt(-0.05, 0.16).derivative, seeds)
        assert top == "no equilibria"
        assert above == "saddle + unstable focus"
        assert below == "saddle + stable focus"


class TestSaddleFocus(TestCase):
    """Saddle-focus detection and the Shilnikov saddle index."""

    def _block(self, spiral_real: float, frequency: float, real: float) -> np.ndarray:
        return np.array(
            [
                [spiral_real, -frequency, 0.0],
                [frequency, spiral_real, 0.0],
                [0.0, 0.0, real],
            ],
        )

    def test_saddle_index_matches_prescribed(self):
        chaotic = classify_saddle_focus(self._block(-0.5, 3.0, 1.0))
        assert chaotic is not None
        assert abs(chaotic.saddle_index - 0.5) < 1.0e-9
        assert chaotic.satisfies_shilnikov_criterion
        tame = classify_saddle_focus(self._block(-2.0, 3.0, 1.0))
        assert tame is not None
        assert abs(tame.saddle_index - 2.0) < 1.0e-9
        assert not tame.satisfies_shilnikov_criterion

    def test_saddle_index_is_similarity_invariant(self):
        generator = np.random.default_rng(0)
        change = generator.standard_normal((3, 3))
        transformed = change @ self._block(-0.5, 3.0, 1.0) @ np.linalg.inv(change)
        result = classify_saddle_focus(transformed)
        assert result is not None
        assert abs(result.saddle_index - 0.5) < 1.0e-8

    def test_rejects_real_saddle(self):
        assert classify_saddle_focus(np.diag([1.0, -1.0, -2.0])) is None

    def test_lorenz_equilibria_are_shilnikov_saddle_foci(self):
        field = LorenzField(
            variables=[State(), State(), State()],
            parameters=[],
            results=[State(), State(), State()],
            time=None,
        )
        seeds = [np.array([8.5, 8.5, 27.0]), np.array([-8.5, -8.5, 27.0])]
        equilibria = find_equilibria(field, seeds)
        assert len(equilibria) == 2
        for point in equilibria:
            focus = classify_saddle_focus(_jacobian(field, point))
            assert focus is not None
            assert focus.satisfies_shilnikov_criterion
