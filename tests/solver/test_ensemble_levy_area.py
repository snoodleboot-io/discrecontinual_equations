"""The Milstein ensemble with Levy areas: non-commuting noise, all paths at once.

Three yardsticks, DEQ-65's.

* Path by path against :class:`MilsteinSolver` given the same increments and
  areas, to rounding: each member driven through a source of its own Brownian
  path, and the ensemble's own draws replayed into the per-path solver - the
  increments from the ensemble's generator, the areas from a
  :class:`LevyAreaSampler` on the stream spawned from the seed, batched over the
  paths in path order.
* Strong order 1.0 on ``dx = -x/2 dt + cos(y) dW_1``, ``dy = -y/2 dt + cos(x)
  dW_2`` over six step sizes, against Milstein with analytic derivatives written
  out here on the path's ``2^-12`` grid, whose cell areas come from a ``2^-15``
  polygon computed here too (the per-path solver's independent test, batched).
* Commuting noise: the bits of every commuting ensemble are those recorded
  before the change (d71dd72), for matrix, vector and declared-diagonal noise,
  in both calculi and from each kind of source.
"""

import hashlib
import warnings

import numpy as np
import pytest

from discrecontinual_equations.solver.stochastic.ensemble import StochasticEnsemble
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
    LevyAreaSampler,
    WienerIncrements,
    WienerSource,
)
from tests.solver.conformal_geometric_brownian_motion import ConformalCase
from tests.solver.geometric_brownian_motion import slope
from tests.solver.milstein.test_diagonal_milstein import (
    CrossVectorNoise,
    DeclaredDiagonalNoise,
    DiagonalMatrixNoise,
    DiagonalNoise,
    _equation,
)
from tests.solver.milstein.test_milstein_levy_area import (
    CrossCosineNoise,
    IteratedIntegralNoise,
)
from tests.solver.milstein.test_milstein_levy_area import (
    _equation as _planar_equation,
)

_STEP = 2.0**-5
_RESOLUTION = 2.0**-8
_START = [0.3, -0.2]
# Relative: einsum and the per-path products may round once differently a step.
_ROUNDING = 1.0e-12


class EachPathItsOwn(WienerSource):
    """One source per member: the ensemble asks once per path a step, in order."""

    def __init__(self, sources: list[WienerSource]) -> None:
        self._sources = sources
        self._next = 0

    def increments(
        self,
        time: float,
        step: float,
        dimension: int,
        *,
        levy_area: bool = False,
    ) -> WienerIncrements:
        source = self._sources[self._next]
        self._next = (self._next + 1) % len(self._sources)
        return source.increments(time, step, dimension, levy_area=levy_area)


class Replay(WienerSource):
    """Hands out given increments and areas, one step after another."""

    def __init__(self, increments: list[np.ndarray], areas: list[np.ndarray]) -> None:
        self._steps = iter(zip(increments, areas, strict=True))

    def increments(
        self,
        time: float,  # noqa: ARG002
        step: float,  # noqa: ARG002
        dimension: int,
        *,
        levy_area: bool = False,
    ) -> WienerIncrements:
        delta_w, area = next(self._steps)
        assert delta_w.size == dimension
        return WienerIncrements(
            delta_w=delta_w,
            delta_z=np.zeros(dimension),
            levy_area=area if levy_area else None,
        )


def ensemble_draws(config, count, channels):
    """What a seeded Milstein ensemble draws when no step commutes, rebuilt here.

    Its increments come from ``default_rng(seed)`` as ``(paths, m)`` normals times
    ``sqrt(dt)``; its areas from a sampler on the stream the per-path solver
    spawns from the seed, every path at once.
    """
    generator = np.random.default_rng(config.random_seed)
    sampler = LevyAreaSampler(
        np.random.default_rng(np.random.SeedSequence(config.random_seed).spawn(1)[0]),
    )
    increments, areas = [], []
    for _ in range(config.n_steps):
        delta_w = generator.standard_normal((count, channels)) * np.sqrt(config.dt)
        increments.append(delta_w)
        areas.append(sampler.sample(delta_w, config.dt))
    return increments, areas


def _per_path(equation, start, config, wiener):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curve = MilsteinSolver(config, wiener=wiener).solve(equation, list(start))
    return np.array([result.discretization[-1] for result in curve.results])


def _ensemble(equation, start, config, wiener=None, paths=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            StochasticEnsemble(config, scheme="milstein", wiener=wiener)
            .solve(equation, np.array(start), paths=paths)
            .final
        )


_NON_COMMUTING = {
    "cross-cosine": (lambda: _planar_equation(CrossCosineNoise), 2),
    "iterated-integral": (lambda: _planar_equation(IteratedIntegralNoise), 2),
    "cross-vector": (lambda: _equation(CrossVectorNoise, 2), 2),
}


class TestPathByPathAgainstThePerPathSolver:
    @pytest.mark.parametrize("calculus", ["ito", "stratonovich"])
    @pytest.mark.parametrize("noise", list(_NON_COMMUTING))
    def test_each_member_driven_by_its_own_path(self, noise, calculus):
        # Arrange: five members, five Brownian paths carrying their cell areas.
        make, channels = _NON_COMMUTING[noise]
        config = MilsteinConfig(
            start_time=0.0,
            end_time=0.5,
            step_size=_STEP,
            calculus=calculus,
        )
        starts = np.array([[0.3, -0.2], [0.1, 0.4], [-0.5, 0.2], [0.0, 1.0], [1.2, 0]])

        def paths():
            return [
                BrownianPath.sample(0.0, 0.5, _RESOLUTION, channels, seed=40 + index)
                for index in range(len(starts))
            ]

        # Act
        found = _ensemble(make(), starts, config, wiener=EachPathItsOwn(paths()))
        expected = np.array(
            [
                _per_path(make(), start, config, path)
                for start, path in zip(starts, paths(), strict=True)
            ],
        )

        # Assert
        np.testing.assert_allclose(found, expected, rtol=_ROUNDING, atol=1e-15)

    @pytest.mark.parametrize("calculus", ["ito", "stratonovich"])
    @pytest.mark.parametrize("noise", list(_NON_COMMUTING))
    def test_the_ensembles_own_draws_replayed_into_the_per_path_solver(
        self,
        noise,
        calculus,
    ):
        # Arrange
        make, channels = _NON_COMMUTING[noise]
        count = 6
        config = MilsteinConfig(
            start_time=0.0,
            end_time=0.5,
            step_size=_STEP,
            calculus=calculus,
            random_seed=17,
        )
        increments, areas = ensemble_draws(config, count, channels)

        # Act
        found = _ensemble(make(), _START, config, paths=count)
        expected = np.array(
            [
                _per_path(
                    make(),
                    _START,
                    config,
                    Replay(
                        [step[path] for step in increments],
                        [step[path] for step in areas],
                    ),
                )
                for path in range(count)
            ],
        )

        # Assert
        np.testing.assert_allclose(found, expected, rtol=_ROUNDING, atol=1e-15)

    def test_the_areas_matter(self):
        # Arrange: the same replay without its areas must miss by far more.
        config = MilsteinConfig(
            start_time=0.0,
            end_time=0.5,
            step_size=_STEP,
            random_seed=17,
        )
        generator = np.random.default_rng(17)
        increments = [
            generator.standard_normal(2) * np.sqrt(_STEP) for _ in range(config.n_steps)
        ]

        # Act
        found = _ensemble(_planar_equation(CrossCosineNoise), _START, config)[0]
        without = _per_path(
            _planar_equation(CrossCosineNoise),
            _START,
            config,
            Replay(increments, [np.zeros((2, 2))] * config.n_steps),
        )

        # Assert
        assert np.max(np.abs(found - without)) > 1e-4

    def test_non_commuting_noise_warns_about_the_cost_once(self):
        # Arrange
        config = MilsteinConfig(start_time=0.0, end_time=0.25, step_size=_STEP)

        # Act
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            StochasticEnsemble(config, scheme="milstein").solve(
                _planar_equation(CrossCosineNoise),
                np.array(_START),
                paths=3,
            )

        # Assert
        messages = [str(w.message) for w in caught if "Levy areas" in str(w.message)]
        assert len(messages) == 1
        assert "euler_maruyama" in messages[0]


_FINE = 15
_NODES = 12
_SUB = 2 ** (_FINE - _NODES)
_STEPS = [2.0**-k for k in range(2, 8)]


def _fine_paths(count: int, seed: int) -> tuple[list[BrownianPath], np.ndarray]:
    """``count`` paths on ``2^-12`` with ``2^-15`` polygon cell areas, and their
    cross-cosine reference at ``(0.3, -0.2)``, computed here."""
    generator = np.random.default_rng(seed)
    steps = generator.standard_normal((count, 2**_FINE, 2)) * np.sqrt(2.0**-_FINE)
    values = np.concatenate(
        [np.zeros((count, 1, 2)), np.cumsum(steps, axis=1)],
        axis=1,
    )
    cells = steps.reshape(count, 2**_NODES, _SUB, 2)
    relative = np.cumsum(cells, axis=2) - cells
    cell_areas = 0.5 * np.sum(
        relative[..., 0] * cells[..., 1] - relative[..., 1] * cells[..., 0],
        axis=2,
    )
    nodes = values[:, ::_SUB]
    paths = []
    for index in range(count):
        matrix = np.zeros((2**_NODES, 2, 2))
        matrix[:, 0, 1] = cell_areas[index]
        matrix[:, 1, 0] = -cell_areas[index]
        paths.append(BrownianPath(0.0, 2.0**-_NODES, nodes[index], areas=matrix))
    # Milstein with L^1 G_2 = (0, -sin x cos y), L^2 G_1 = (-sin y cos x, 0):
    # the step adds L^1 G_2 I_(1,2) + L^2 G_1 I_(2,1), every path at once.
    step = 2.0**-_NODES
    x = np.full(count, _START[0])
    y = np.full(count, _START[1])
    increments = np.diff(nodes, axis=1)
    for n in range(2**_NODES):
        dw = increments[:, n]
        i12 = 0.5 * dw[:, 0] * dw[:, 1] + cell_areas[:, n]
        i21 = 0.5 * dw[:, 0] * dw[:, 1] - cell_areas[:, n]
        x, y = (
            x - 0.5 * x * step + np.cos(y) * dw[:, 0] - np.sin(y) * np.cos(x) * i21,
            y - 0.5 * y * step + np.cos(x) * dw[:, 1] - np.sin(x) * np.cos(y) * i12,
        )
    return paths, np.stack([x, y], axis=1)


class TestStrongOrderOnNonCommutingNoise:
    def test_cross_cosine_noise_converges_at_order_one(self):
        # Arrange
        count = 40
        paths, reference = _fine_paths(count, seed=65)
        milstein, euler = {}, {}

        # Act
        for step in _STEPS:
            final = _ensemble(
                _planar_equation(CrossCosineNoise),
                _START,
                MilsteinConfig(start_time=0.0, end_time=1.0, step_size=step),
                wiener=EachPathItsOwn(paths),
                paths=count,
            )
            milstein[step] = float(
                np.mean(np.linalg.norm(final - reference, axis=1)),
            )
            config = EulerMaruyamaConfig(start_time=0.0, end_time=1.0, step_size=step)
            final = StochasticEnsemble(config, wiener=EachPathItsOwn(paths)).solve(
                _planar_equation(CrossCosineNoise),
                np.array(_START),
                paths=count,
            )
            euler[step] = float(
                np.mean(np.linalg.norm(final.final - reference, axis=1)),
            )

        # Assert
        assert 0.85 < slope(milstein) < 1.15, milstein
        assert slope(euler) < 0.7, euler
        # The reference's own error is O(2^-12): far below the finest error.
        assert milstein[_STEPS[-1]] < 6e-3, milstein
        assert milstein[_STEPS[-1]] < 0.3 * euler[_STEPS[-1]], (milstein, euler)


# sha256 of the final (4, n) states of a commuting Milstein ensemble, recorded
# on d71dd72 before the areas were added.
_COMMUTING_BEFORE = {
    ("conformal", "ito", "generator"): (
        "457c659f2096fdd1765e2e107e5d451d9af7fdaa966715ae18ceb2535cff7c2d"
    ),
    ("conformal", "ito", "path"): (
        "fb087cce24f9dda2fbd95c4e9be3a029c26eee0336ac5df8c011ec32f93b8f22"
    ),
    ("conformal", "ito", "gaussian"): (
        "87268570465bf68c5a9d7f8663bda9f69dba5beb7edf4ff0ac7bae42b7079394"
    ),
    ("conformal", "stratonovich", "generator"): (
        "bf40c81da26ded4ceaa7ceba50a48e5ab25f947e34a5b2e0de96a47483f2cfd0"
    ),
    ("conformal", "stratonovich", "path"): (
        "c87c84f039795f49bc2b24c7ba9205a957538fd4a918dbdeb6345949547148cf"
    ),
    ("conformal", "stratonovich", "gaussian"): (
        "14f4842dccc319892ad0928162e5d18af22b2e98a63add9cbad097847f3617be"
    ),
    ("diagonal", "ito", "generator"): (
        "148111d27c814d659ac77635408fc814258a076967578e5ea05db213e953ff84"
    ),
    ("diagonal", "ito", "path"): (
        "620043187bd66316f511756f7b7edf48e6decaf23d3e5e2927d5f778825b4da1"
    ),
    ("diagonal", "ito", "gaussian"): (
        "034d78789f15876893677a16c887ddecf277bc80bc47c785277455a5d9354c6a"
    ),
    ("diagonal", "stratonovich", "generator"): (
        "d20f0ed66f4317dfc850caa4365c143385838c56bc59c98daaef4ef2cab26ff9"
    ),
    ("diagonal", "stratonovich", "path"): (
        "42144fe18d42da388b6d3c8e06086ddd7b0e912af92f97485c16e7e4a79ed28f"
    ),
    ("diagonal", "stratonovich", "gaussian"): (
        "695ad4404e9e7dbab983d5a836f6779245645487bf183bede698ddc3c1c92efe"
    ),
}


def _commuting(noise):
    if noise == "conformal":
        return ConformalCase().equation(), [1.0, 0.5], 2
    function_type = {
        "diagonal": DiagonalNoise,
        "matrix": DiagonalMatrixNoise,
        "declared": DeclaredDiagonalNoise,
    }[noise]
    return _equation(function_type, 3), [0.5, 1.0, 1.5], 3


@pytest.mark.parametrize("source", ["generator", "path", "gaussian"])
@pytest.mark.parametrize("calculus", ["ito", "stratonovich"])
@pytest.mark.parametrize("noise", ["conformal", "diagonal", "matrix", "declared"])
def test_commuting_ensembles_keep_their_bits(noise, calculus, source):
    # Arrange: diagonal noise as a vector, as the matrix diag(b), and declared
    # all recorded the same bits; each is held to them.
    equation, start, channels = _commuting(noise)
    wiener = {
        "generator": None,
        "path": BrownianPath.sample(0.0, 0.5, _RESOLUTION, channels, seed=21),
        "gaussian": GaussianWienerSource(22),
    }[source]
    config = MilsteinConfig(
        start_time=0.0,
        end_time=0.5,
        step_size=_STEP,
        random_seed=23,
        calculus=calculus,
    )

    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        final = (
            StochasticEnsemble(config, scheme="milstein", wiener=wiener)
            .solve(equation, np.array(start), paths=4)
            .final
        )

    # Assert
    recorded = "conformal" if noise == "conformal" else "diagonal"
    digest = hashlib.sha256(final.tobytes()).hexdigest()
    assert digest == _COMMUTING_BEFORE[(recorded, calculus, source)]
