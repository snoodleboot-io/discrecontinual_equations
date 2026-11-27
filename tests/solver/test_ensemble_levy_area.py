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
    _ENSEMBLE_DIMENSION,
    _MOST_ENSEMBLE_BYTES,
    _PATHS,
    CrossVectorNoise,
    DeclaredDiagonalNoise,
    DiagonalMatrixNoise,
    DiagonalNoise,
    _equation,
    _peak,
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
# How far a value recorded on one machine may be from the same run on another
# whose vectorised transcendentals round differently (DEQ-118 verification).
_RECORDED = 1.0e-10
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

    def test_non_commuting_noise_warns_about_the_cost_once_a_solve(self):
        # Arrange
        config = MilsteinConfig(start_time=0.0, end_time=0.25, step_size=_STEP)
        ensemble = StochasticEnsemble(config, scheme="milstein")

        # Act: two solves of one ensemble.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(2):
                ensemble.solve(
                    _planar_equation(CrossCosineNoise),
                    np.array(_START),
                    paths=3,
                )

        # Assert
        messages = [str(w.message) for w in caught if "Levy areas" in str(w.message)]
        assert len(messages) == 2
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


# The final (4, n) states of a commuting Milstein ensemble, recorded again at
# DEQ-118, whose relative slope step moved every state by at most 1.3e-11
# relative (against _COMMUTING_BEFORE_DEQ118 below). Held to _RECORDED, not to
# the bit: the diagonal cases difference 0.3 sin x + 0.2 x, and a vectorised
# sin that rounds one ulp differently on another CPU moves a slope by about
# eps |b| / h, some 4e-11 relative.
_COMMUTING_BEFORE = {
    ("conformal", "ito", "generator"): [
        "0x1.ee5cb6142b7c2p+0",
        "0x1.0237cadd813cbp+0",
        "0x1.99e441cecc7bep+0",
        "0x1.44ed15775396fp-1",
        "0x1.50a096ee9e6f0p+0",
        "0x1.8138a361781e2p-2",
        "0x1.5f40cd452969bp-1",
        "-0x1.a7cd283aa9f7fp-3",
    ],
    ("conformal", "ito", "path"): [
        "0x1.c0705aeb52c20p-1",
        "0x1.641452325f262p-3",
        "0x1.c0705aeb52c20p-1",
        "0x1.641452325f262p-3",
        "0x1.c0705aeb52c20p-1",
        "0x1.641452325f262p-3",
        "0x1.c0705aeb52c20p-1",
        "0x1.641452325f262p-3",
    ],
    ("conformal", "ito", "gaussian"): [
        "0x1.179fbf2176ff6p+0",
        "0x1.0a08333358630p-1",
        "0x1.89b08d71362e8p+0",
        "0x1.ba391f3de43bbp-2",
        "0x1.9bfa90d434a18p-1",
        "0x1.3dec179a47dbcp-1",
        "0x1.d2ffb148f420ep+0",
        "0x1.d634de4a4bd0cp-1",
    ],
    ("conformal", "stratonovich", "generator"): [
        "0x1.ee5cb6142b7c2p+0",
        "0x1.0237cadd813cbp+0",
        "0x1.99e441cecc7bep+0",
        "0x1.44ed15775396fp-1",
        "0x1.50a096ee9e6f0p+0",
        "0x1.8138a361781e2p-2",
        "0x1.5f40cd452969bp-1",
        "-0x1.a7cd283aa9f7fp-3",
    ],
    ("conformal", "stratonovich", "path"): [
        "0x1.c0705aeb52c20p-1",
        "0x1.641452325f262p-3",
        "0x1.c0705aeb52c20p-1",
        "0x1.641452325f262p-3",
        "0x1.c0705aeb52c20p-1",
        "0x1.641452325f262p-3",
        "0x1.c0705aeb52c20p-1",
        "0x1.641452325f262p-3",
    ],
    ("conformal", "stratonovich", "gaussian"): [
        "0x1.179fbf2176ff6p+0",
        "0x1.0a08333358630p-1",
        "0x1.89b08d71362e8p+0",
        "0x1.ba391f3de43bbp-2",
        "0x1.9bfa90d434a18p-1",
        "0x1.3dec179a47dbcp-1",
        "0x1.d2ffb148f420ep+0",
        "0x1.d634de4a4bd0cp-1",
    ],
    ("diagonal", "ito", "generator"): [
        "0x1.cfa8d8ab3bb55p-2",
        "0x1.cf1f451818b00p-2",
        "0x1.58a82b6904815p-1",
        "0x1.cd09b1dedef75p-3",
        "0x1.395b8501f1379p-1",
        "0x1.6670015e09b17p-1",
        "0x1.8dfe3eb419c8ap-2",
        "0x1.25e64159b25a9p-1",
        "0x1.0b1b7483906ecp+0",
        "0x1.0ccf8baa05c45p-2",
        "0x1.e744e810ddc66p-3",
        "0x1.ae9fb39edeec4p-1",
    ],
    ("diagonal", "ito", "path"): [
        "0x1.de98adc71e732p-3",
        "0x1.0b70159a0dd49p-1",
        "0x1.e8f54307fa79ep-1",
        "0x1.de98adc71e732p-3",
        "0x1.0b70159a0dd49p-1",
        "0x1.e8f54307fa79ep-1",
        "0x1.de98adc71e732p-3",
        "0x1.0b70159a0dd49p-1",
        "0x1.e8f54307fa79ep-1",
        "0x1.de98adc71e732p-3",
        "0x1.0b70159a0dd49p-1",
        "0x1.e8f54307fa79ep-1",
    ],
    ("diagonal", "ito", "gaussian"): [
        "0x1.74d309351e44fp-3",
        "0x1.0c642bf37aa6dp-1",
        "0x1.0a204e0c26d6fp+0",
        "0x1.4c7f9500c43e6p-2",
        "0x1.1c8e856529860p-1",
        "0x1.e266b94d22529p-1",
        "0x1.fa230489e75bep-2",
        "0x1.3dd35f10c7cebp-1",
        "0x1.e3c7f2bb5f319p-2",
        "0x1.3dfb8da4066c6p-2",
        "0x1.c750129159961p-2",
        "0x1.f7b841c0f53f3p-1",
    ],
    ("diagonal", "stratonovich", "generator"): [
        "0x1.eb16040e0f143p-2",
        "0x1.e7b51049efa1bp-2",
        "0x1.67370be597d0ep-1",
        "0x1.eb531e7676420p-3",
        "0x1.48d80eadbcef1p-1",
        "0x1.75a1cc3cc294cp-1",
        "0x1.a612aa0f96120p-2",
        "0x1.3507da3b7ba54p-1",
        "0x1.13bcaf79de9bfp+0",
        "0x1.1d6f8e6ba21bap-2",
        "0x1.0232cffd54ebfp-2",
        "0x1.c0b54e03924cep-1",
    ],
    ("diagonal", "stratonovich", "path"): [
        "0x1.fd74a5b8b1e6cp-3",
        "0x1.195b25f3df834p-1",
        "0x1.fbb221e8e3bbbp-1",
        "0x1.fd74a5b8b1e6cp-3",
        "0x1.195b25f3df834p-1",
        "0x1.fbb221e8e3bbbp-1",
        "0x1.fd74a5b8b1e6cp-3",
        "0x1.195b25f3df834p-1",
        "0x1.fbb221e8e3bbbp-1",
        "0x1.fd74a5b8b1e6cp-3",
        "0x1.195b25f3df834p-1",
        "0x1.fbb221e8e3bbbp-1",
    ],
    ("diagonal", "stratonovich", "gaussian"): [
        "0x1.8d4fdf5affbf0p-3",
        "0x1.1a8d7aeb7f12cp-1",
        "0x1.1243d34d70c46p+0",
        "0x1.611ffd20b8666p-2",
        "0x1.298901489a4f8p-1",
        "0x1.f3a1b725a80bcp-1",
        "0x1.0b9aa8da8d212p-1",
        "0x1.4e129b10d9727p-1",
        "0x1.f9753aee63a88p-2",
        "0x1.51e8094067c8ep-2",
        "0x1.dfa324aac20f0p-2",
        "0x1.03887804ff274p+0",
    ],
}


# The same final states as recorded on d71dd72, before the areas were added, with
# the absolute slope step of 1e-6 that DEQ-118 replaced: each new state is held
# to 1e-10 relative of its value here.
_COMMUTING_BEFORE_DEQ118 = {
    ("conformal", "ito", "generator"): [
        "0x1.ee5cb6142b72dp+0",
        "0x1.0237cadd8135ep+0",
        "0x1.99e441ced4b9fp+0",
        "0x1.44ed1577560afp-1",
        "0x1.50a096eea0ae7p+0",
        "0x1.8138a3617a898p-2",
        "0x1.5f40cd452d428p-1",
        "-0x1.a7cd283aae90cp-3",
    ],
    ("conformal", "ito", "path"): [
        "0x1.c0705aeb5333ap-1",
        "0x1.6414523265fa1p-3",
        "0x1.c0705aeb5333ap-1",
        "0x1.6414523265fa1p-3",
        "0x1.c0705aeb5333ap-1",
        "0x1.6414523265fa1p-3",
        "0x1.c0705aeb5333ap-1",
        "0x1.6414523265fa1p-3",
    ],
    ("conformal", "ito", "gaussian"): [
        "0x1.179fbf21784b4p+0",
        "0x1.0a083333573d4p-1",
        "0x1.89b08d71398d6p+0",
        "0x1.ba391f3de440cp-2",
        "0x1.9bfa90d4384e1p-1",
        "0x1.3dec179a48278p-1",
        "0x1.d2ffb148f23c5p+0",
        "0x1.d634de4a4940bp-1",
    ],
    ("conformal", "stratonovich", "generator"): [
        "0x1.ee5cb61420c94p+0",
        "0x1.0237cadd8699ep+0",
        "0x1.99e441cecee9ap+0",
        "0x1.44ed1577566fcp-1",
        "0x1.50a096ee9b57fp+0",
        "0x1.8138a3618d1b5p-2",
        "0x1.5f40cd452dfeap-1",
        "-0x1.a7cd283aa7412p-3",
    ],
    ("conformal", "stratonovich", "path"): [
        "0x1.c0705aeb54881p-1",
        "0x1.641452326324dp-3",
        "0x1.c0705aeb54881p-1",
        "0x1.641452326324dp-3",
        "0x1.c0705aeb54881p-1",
        "0x1.641452326324dp-3",
        "0x1.c0705aeb54881p-1",
        "0x1.641452326324dp-3",
    ],
    ("conformal", "stratonovich", "gaussian"): [
        "0x1.179fbf2175be0p+0",
        "0x1.0a0833335933ap-1",
        "0x1.89b08d7131d7dp+0",
        "0x1.ba391f3def1d6p-2",
        "0x1.9bfa90d433e79p-1",
        "0x1.3dec179a485e5p-1",
        "0x1.d2ffb148f19f6p+0",
        "0x1.d634de4a4d579p-1",
    ],
    ("diagonal", "ito", "generator"): [
        "0x1.cfa8d8ab3a9d6p-2",
        "0x1.cf1f451819a9dp-2",
        "0x1.58a82b69014e3p-1",
        "0x1.cd09b1dee084fp-3",
        "0x1.395b8501f2515p-1",
        "0x1.6670015e07145p-1",
        "0x1.8dfe3eb4186f9p-2",
        "0x1.25e64159b1ee4p-1",
        "0x1.0b1b748391e82p+0",
        "0x1.0ccf8baa0371fp-2",
        "0x1.e744e810d788ep-3",
        "0x1.ae9fb39edcac3p-1",
    ],
    ("diagonal", "ito", "path"): [
        "0x1.de98adc71c8cfp-3",
        "0x1.0b70159a0c254p-1",
        "0x1.e8f54307f7f08p-1",
        "0x1.de98adc71c8cfp-3",
        "0x1.0b70159a0c254p-1",
        "0x1.e8f54307f7f08p-1",
        "0x1.de98adc71c8cfp-3",
        "0x1.0b70159a0c254p-1",
        "0x1.e8f54307f7f08p-1",
        "0x1.de98adc71c8cfp-3",
        "0x1.0b70159a0c254p-1",
        "0x1.e8f54307f7f08p-1",
    ],
    ("diagonal", "ito", "gaussian"): [
        "0x1.74d309351f075p-3",
        "0x1.0c642bf37a1bdp-1",
        "0x1.0a204e0c24457p+0",
        "0x1.4c7f9500c5cc3p-2",
        "0x1.1c8e85652a9fdp-1",
        "0x1.e266b94d247efp-1",
        "0x1.fa230489e7135p-2",
        "0x1.3dd35f10c7d1ap-1",
        "0x1.e3c7f2bb5e54ep-2",
        "0x1.3dfb8da407190p-2",
        "0x1.c75012915d60dp-2",
        "0x1.f7b841c0f6938p-1",
    ],
    ("diagonal", "stratonovich", "generator"): [
        "0x1.eb16040e0b273p-2",
        "0x1.e7b51049f07cfp-2",
        "0x1.67370be59371fp-1",
        "0x1.eb531e76759f9p-3",
        "0x1.48d80eadc0e4fp-1",
        "0x1.75a1cc3cc3c5cp-1",
        "0x1.a612aa0f92e83p-2",
        "0x1.3507da3b7aee9p-1",
        "0x1.13bcaf79deb87p+0",
        "0x1.1d6f8e6ba2f9ap-2",
        "0x1.0232cffd58458p-2",
        "0x1.c0b54e039411fp-1",
    ],
    ("diagonal", "stratonovich", "path"): [
        "0x1.fd74a5b8af817p-3",
        "0x1.195b25f3dfdf6p-1",
        "0x1.fbb221e8e2d15p-1",
        "0x1.fd74a5b8af817p-3",
        "0x1.195b25f3dfdf6p-1",
        "0x1.fbb221e8e2d15p-1",
        "0x1.fd74a5b8af817p-3",
        "0x1.195b25f3dfdf6p-1",
        "0x1.fbb221e8e2d15p-1",
        "0x1.fd74a5b8af817p-3",
        "0x1.195b25f3dfdf6p-1",
        "0x1.fbb221e8e2d15p-1",
    ],
    ("diagonal", "stratonovich", "gaussian"): [
        "0x1.8d4fdf5afe54dp-3",
        "0x1.1a8d7aeb7f7bcp-1",
        "0x1.1243d34d6b45ap+0",
        "0x1.611ffd20b93c4p-2",
        "0x1.298901489aa5ap-1",
        "0x1.f3a1b725a4a91p-1",
        "0x1.0b9aa8da8db4cp-1",
        "0x1.4e129b10db097p-1",
        "0x1.f9753aee5b25fp-2",
        "0x1.51e80940664c2p-2",
        "0x1.dfa324aac5ae1p-2",
        "0x1.03887804fe2e1p+0",
    ],
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
    # all recorded the same values; each is held to them, to _RECORDED.
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
    recorded_values = [
        float.fromhex(h) for h in _COMMUTING_BEFORE[(recorded, calculus, source)]
    ]
    np.testing.assert_allclose(final.ravel(), recorded_values, rtol=_RECORDED, atol=0.0)
    before = [
        float.fromhex(h) for h in _COMMUTING_BEFORE_DEQ118[(recorded, calculus, source)]
    ]
    np.testing.assert_allclose(final.ravel(), before, rtol=1.0e-10, atol=0.0)


class _NoAreas(WienerSource):
    """A source that fails a scheme asking it for areas, counting what it gives."""

    def __init__(self, channels: int) -> None:
        self._source = GaussianWienerSource(31)
        self._channels = channels

    def increments(
        self,
        time: float,
        step: float,
        dimension: int,
        *,
        levy_area: bool = False,
    ) -> WienerIncrements:
        assert not levy_area, "a commuting step asked its source for areas"
        return self._source.increments(time, step, dimension)


@pytest.mark.parametrize("calculus", ["ito", "stratonovich"])
@pytest.mark.parametrize("noise", ["conformal", "diagonal", "matrix", "declared"])
def test_commuting_ensembles_never_ask_their_source_for_areas(noise, calculus):
    # Arrange: the per-path solver asks only when the noise does not commute.
    equation, start, channels = _commuting(noise)
    config = MilsteinConfig(
        start_time=0.0,
        end_time=0.25,
        step_size=_STEP,
        calculus=calculus,
    )

    # Act
    final = (
        StochasticEnsemble(config, scheme="milstein", wiener=_NoAreas(channels))
        .solve(equation, np.array(start), paths=3)
        .final
    )

    # Assert
    assert np.all(np.isfinite(final))


class _CountedDeclaredNoise(DeclaredDiagonalNoise):
    """Declared diagonal noise counting its batched diffusion evaluations."""

    batches = 0

    def diffusion_batch(
        self,
        points: np.ndarray,
        time: float | None = None,
    ) -> np.ndarray:
        type(self).batches += 1
        return super().diffusion_batch(points, time)


@pytest.mark.parametrize("dimension", [3, 12])
def test_declared_noise_with_a_source_costs_what_it_does_without(dimension):
    # Arrange: deciding whether to ask a source for areas must not read the
    # 2 n slopes a declaration spares.
    config = MilsteinConfig(
        start_time=0.0,
        end_time=4 * _STEP,
        step_size=_STEP,
        random_seed=3,
    )
    counts = []

    # Act
    for wiener in (None, GaussianWienerSource(3)):
        _CountedDeclaredNoise.batches = 0
        StochasticEnsemble(config, scheme="milstein", wiener=wiener).solve(
            _equation(_CountedDeclaredNoise, dimension),
            np.linspace(0.5, 1.5, dimension),
            paths=3,
        )
        counts.append(_CountedDeclaredNoise.batches)

    # Assert
    assert counts[1] == counts[0]


def test_a_commuting_vector_ensemble_with_a_source_holds_no_cubic_array():
    # Arrange: asking a source whether to give areas is decided by the
    # O(paths n) verdict, not the (paths, n, n, n) products of the general step.
    config = MilsteinConfig(
        start_time=0.0,
        end_time=2 * _STEP,
        step_size=_STEP,
    )
    equation = _equation(DiagonalNoise, _ENSEMBLE_DIMENSION)
    start = np.linspace(0.5, 1.5, _ENSEMBLE_DIMENSION)
    ensemble = StochasticEnsemble(
        config,
        scheme="milstein",
        wiener=GaussianWienerSource(5),
    )

    # Act
    peak = _peak(lambda: ensemble.solve(equation, start, paths=_PATHS))

    # Assert
    assert peak < _MOST_ENSEMBLE_BYTES


class _OwnAreas(WienerSource):
    """A source that makes its areas itself, never by a LevyAreaSampler."""

    def __init__(self) -> None:
        self._source = GaussianWienerSource(13)
        self._generator = np.random.default_rng(13)

    def increments(
        self,
        time: float,
        step: float,
        dimension: int,
        *,
        levy_area: bool = False,
    ) -> WienerIncrements:
        draw = self._source.increments(time, step, dimension)
        area = None
        if levy_area:
            upper = np.triu(self._generator.standard_normal((dimension, dimension)), 1)
            area = 0.5 * step * (upper - upper.T)
        return WienerIncrements(
            delta_w=draw.delta_w,
            delta_z=draw.delta_z,
            levy_area=area,
        )


@pytest.mark.parametrize("nan", [False, True], ids=["finite", "nan-path"])
@pytest.mark.parametrize("noise", [DiagonalNoise, CrossVectorNoise])
def test_with_a_source_the_ensemble_never_draws_areas_of_its_own(
    noise,
    nan,
    monkeypatch,
):
    # Arrange: with a source, every area is the source's. The vector verdict
    # taken before asking it must be the one the O(paths n) step then reaches -
    # NaN included - or that step would fall through to the generator's sampler.
    def refuse(*_args, **_kwargs):
        message = "the ensemble drew areas of its own while given a source"
        raise AssertionError(message)

    monkeypatch.setattr(LevyAreaSampler, "sample", refuse)
    config = MilsteinConfig(start_time=0.0, end_time=4 * _STEP, step_size=_STEP)
    start = np.array([[0.3, -0.2], [0.1, 0.4], [-0.5, 0.2]])
    if nan:
        start[0, 0] = np.nan

    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final = (
            StochasticEnsemble(config, scheme="milstein", wiener=_OwnAreas())
            .solve(_equation(noise, 2), start)
            .final
        )

    # Assert
    assert np.all(np.isfinite(final[1:]))
