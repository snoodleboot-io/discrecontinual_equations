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


# The final (4, n) states of a commuting Milstein ensemble, recorded on d71dd72
# before the areas were added, held two ways (DEQ-160).
#
# Conformal noise, by sha256: every operation on the way is IEEE-exact - the
# drift a x and diffusion sigma [[x, -y], [y, x]] are linear, their central
# differences are linear, the step and the einsum contractions only add and
# multiply, and the increments are the RNG's normals scaled by sqrt(dt) - so
# any conforming CPU gives these bits, and a digest is a fair pin.
_CONFORMAL_BEFORE = {
    ("ito", "generator"): (
        "457c659f2096fdd1765e2e107e5d451d9af7fdaa966715ae18ceb2535cff7c2d"
    ),
    ("ito", "path"): (
        "fb087cce24f9dda2fbd95c4e9be3a029c26eee0336ac5df8c011ec32f93b8f22"
    ),
    ("ito", "gaussian"): (
        "87268570465bf68c5a9d7f8663bda9f69dba5beb7edf4ff0ac7bae42b7079394"
    ),
    ("stratonovich", "generator"): (
        "bf40c81da26ded4ceaa7ceba50a48e5ab25f947e34a5b2e0de96a47483f2cfd0"
    ),
    ("stratonovich", "path"): (
        "c87c84f039795f49bc2b24c7ba9205a957538fd4a918dbdeb6345949547148cf"
    ),
    ("stratonovich", "gaussian"): (
        "14f4842dccc319892ad0928162e5d18af22b2e98a63add9cbad097847f3617be"
    ),
}

# Diagonal noise b = 0.3 sin x + 0.2 x, as values: sin is vectorised (SVML on
# AVX-512 runners rounds it an ulp differently), and the step's central
# difference of b amplifies an ulp in b to about 1e-11 of the slope, so these are
# held to 1e-10 relative rather than to their bits. The decimals are the bits
# the sha256 pins recorded (outputs/verification/deq-160).
_DIAGONAL_BEFORE = {
    ("ito", "generator"): [
        [0.45279253525828034, 0.4522677226981758, 0.6731580320053151],
        [0.22511614762703622, 0.6120263638378299, 0.700073282936082],
        [0.3886651799603978, 0.5740223333350198, 1.0433876820769972],
        [0.262510473491476, 0.23792439748189448, 0.8410621768401473],
    ],
    ("ito", "path"): [
        [0.23368964923753885, 0.5223395109697528, 0.9549961993434133],
        [0.23368964923753885, 0.5223395109697528, 0.9549961993434133],
        [0.23368964923753885, 0.5223395109697528, 0.9549961993434133],
        [0.23368964923753885, 0.5223395109697528, 0.9549961993434133],
    ],
    ("ito", "gaussian"): [
        [0.18204314416277936, 0.5242017492974395, 1.0395554332392487],
        [0.3247054368707262, 0.5557748495096458, 0.9421899706524054],
        [0.49427420703593167, 0.6207532604202328, 0.4724424292431494],
        [0.3105299121250704, 0.44464138996349406, 0.9838276432085769],
    ],
    ("stratonovich", "generator"): [
        [0.4795761712619317, 0.4762766404274971, 0.701591846250924],
        [0.23990463065156556, 0.6422733866788323, 0.7297500442672447],
        [0.4121805736547019, 0.6035755345924062, 1.0770978615316567],
        [0.27874586612664987, 0.2521469591502572, 0.8763832454078829],
    ],
    ("stratonovich", "path"): [
        [0.24875764341226222, 0.5495235309790008, 0.9915934178147007],
        [0.24875764341226222, 0.5495235309790008, 0.9915934178147007],
        [0.24875764341226222, 0.5495235309790008, 0.9915934178147007],
        [0.24875764341226222, 0.5495235309790008, 0.9915934178147007],
    ],
    ("stratonovich", "gaussian"): [
        [0.19400000091972416, 0.5518606579278109, 1.071347433472758],
        [0.3448485900137792, 0.5811233903053321, 0.975843165726774],
        [0.5226643339909542, 0.6524857004237842, 0.49361126022157537],
        [0.32998671012462577, 0.4683957795363494, 1.013801099032371],
    ],
}
_DIFFERENCE_TOLERANCE = 1.0e-10


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
def test_commuting_ensembles_keep_their_recorded_states(noise, calculus, source):
    # Arrange: diagonal noise as a vector, as the matrix diag(b), and declared
    # all recorded the same bits; each is held to them, to the tolerance above.
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
    if noise == "conformal":
        digest = hashlib.sha256(final.tobytes()).hexdigest()
        assert digest == _CONFORMAL_BEFORE[(calculus, source)]
    else:
        np.testing.assert_allclose(
            final,
            _DIAGONAL_BEFORE[(calculus, source)],
            rtol=_DIFFERENCE_TOLERANCE,
            atol=0.0,
        )


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
