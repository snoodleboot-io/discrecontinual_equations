"""Generate self-contained D3 HTML plots for continuation results.

Run with ``python -m examples.continuation.webplot_examples [output_dir]``. It
continues equilibrium branches and their codim-1, codim-2, and codim-3
bifurcations, renders each with the D3 renderer using the system's own parameter
notation, and writes an atlas linking them.

The atlas is built in parallel. Each catalogue entry and each film is one unit,
built in a fresh worker process of its own, so no unit sees another's parameter
values or registry, and each writes only its own page. The index is written in
registry order, not the order the units finish in, so the atlas is the same
bytes however many workers built it::

    python examples/continuation/webplot_examples.py plots --jobs 12
    python examples/continuation/webplot_examples.py plots --only hopf cusp
"""

import argparse
import ast
import inspect
import os
import tempfile
import textwrap
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np

from discrecontinual_equations.webplot.latex import use_disk_cache
from discrecontinual_equations.webplot.manifold_film_renderer import (
    ManifoldFilmRenderer,
)
from discrecontinual_equations.webplot.renderer import D3Renderer
from discrecontinual_equations.webplot.report import AtlasEntry, PlotReport
from discrecontinual_equations.webplot.scene import Scene
from discrecontinual_equations.webplot.scene_builder import (
    with_equation,
)
from discrecontinual_equations.webplot.stochastic_stage_renderer import (
    StochasticStageRenderer,
)

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation.component_registry import ensure_registry
except ImportError:  # run as a script path: only this directory is on sys.path
    from component_registry import ensure_registry


try:  # python -m examples.continuation.webplot_examples
    from examples.continuation.pages_codim1 import (
        _hopf,
        _hysteresis,
        _pitchfork,
        _saddle_node,
        _transcritical,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from pages_codim1 import (
        _hopf,
        _hysteresis,
        _pitchfork,
        _saddle_node,
        _transcritical,
    )

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation.pages_codim2 import (
        _bogdanov_takens,
        _bogdanov_takens_curve,
        _bogdanov_takens_portrait,
        _bogdanov_takens_type_curve,
        _cusp,
        _degenerate_bautin,
        _generalized_hopf,
        _hopf_hopf,
        _swallowtail,
        _zero_hopf,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from pages_codim2 import (
        _bogdanov_takens,
        _bogdanov_takens_curve,
        _bogdanov_takens_portrait,
        _bogdanov_takens_type_curve,
        _cusp,
        _degenerate_bautin,
        _generalized_hopf,
        _hopf_hopf,
        _swallowtail,
        _zero_hopf,
    )

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation.pages_connecting import (
        _connecting_orbit_entries,
        _heteroclinic_cycle,
        _homoclinic_curve_shooting,
        _homoclinic_return,
        _manifolds,
        _saddle_focus_onset,
        _shilnikov_loop,
        _symmetry_modes,
        _three_dimensional_homoclinic,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from pages_connecting import (
        _connecting_orbit_entries,
        _heteroclinic_cycle,
        _homoclinic_curve_shooting,
        _homoclinic_return,
        _manifolds,
        _saddle_focus_onset,
        _shilnikov_loop,
        _symmetry_modes,
        _three_dimensional_homoclinic,
    )

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation.pages_network import _network_entries
except ImportError:  # run as a script path: only this directory is on sys.path
    from pages_network import _network_entries

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation.pages_cycles import (
        _adaptive_mesh,
        _collocation_convergence,
        _cycle_codim2,
        _deflated_roots,
        _fold_of_cycles_branch,
        _limit_cycle_branch,
        _snic_period_divergence,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from pages_cycles import (
        _adaptive_mesh,
        _collocation_convergence,
        _cycle_codim2,
        _deflated_roots,
        _fold_of_cycles_branch,
        _limit_cycle_branch,
        _snic_period_divergence,
    )

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation.pages_exploration import (
        _efield_detection,
        _efield_operating_map,
        _exploration_diagram,
        _exploration_skeleton,
        _exploration_tree,
        _sevencell_exploration,
        _sevencell_ring_map,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from pages_exploration import (
        _efield_detection,
        _efield_operating_map,
        _exploration_diagram,
        _exploration_skeleton,
        _exploration_tree,
        _sevencell_exploration,
        _sevencell_ring_map,
    )

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation.pages_families import (
        _bogdanov_takens_family,
        _bogdanov_takens_surface,
        _cusp_family,
        _cusp_surface,
        _fluxgate_bias_surface,
        _fluxgate_family,
        _fluxgate_surface,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from pages_families import (
        _bogdanov_takens_family,
        _bogdanov_takens_surface,
        _cusp_family,
        _cusp_surface,
        _fluxgate_bias_surface,
        _fluxgate_family,
        _fluxgate_surface,
    )

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation.pages_coloured import _coloured_thresholds
except ImportError:  # run as a script path: only this directory is on sys.path
    from pages_coloured import _coloured_thresholds

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation.pages_stochastic import (
        _density_family,
        _dynamical_bifurcation,
        _lyapunov,
        _stochastic_attractor,
        _stochastic_bifurcation,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from pages_stochastic import (
        _density_family,
        _dynamical_bifurcation,
        _lyapunov,
        _stochastic_attractor,
        _stochastic_bifurcation,
    )

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation import stage_bogdanov_takens as bt
    from examples.continuation import stage_extended_bautin as bautin
    from examples.continuation import stage_ferroelectric_ring as ring
    from examples.continuation import stage_fold_of_cycles as fold
    from examples.continuation import stage_hopf as hopf
    from examples.continuation import stage_hopf_hopf as hopfhopf
    from examples.continuation import stage_lorenz_manifold as lorenz
    from examples.continuation import stage_noisy_hopf as noisy
    from examples.continuation import stage_shilnikov as shilnikov
    from examples.continuation import stage_snic as snic
    from examples.continuation import stage_van_der_pol as vdp
    from examples.continuation import stage_zero_hopf as zerohopf
except ImportError:  # run as a script path: only this directory is on sys.path
    import stage_bogdanov_takens as bt
    import stage_extended_bautin as bautin
    import stage_ferroelectric_ring as ring
    import stage_fold_of_cycles as fold
    import stage_hopf as hopf
    import stage_hopf_hopf as hopfhopf
    import stage_lorenz_manifold as lorenz
    import stage_noisy_hopf as noisy
    import stage_shilnikov as shilnikov
    import stage_snic as snic
    import stage_van_der_pol as vdp
    import stage_zero_hopf as zerohopf

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


_VDP_STEP = 0.001
_VDP_SETTLE = 50.0


_SADDLE_JACOBIAN = np.array([[0.0, 1.0], [1.0, 0.0]])
_PHASE_WINDOW = 2.2


_FERROELECTRIC = r"\dot{x}_i = a\,x_i - x_i^3 - \lambda\,x_{i+1} + \varepsilon"
_FLUXGATE = r"\dot{x}_i = -x_i + \tanh(g\,x_i + c\,x_{i+1} + h)"
_CUSP_EQN = r"\dot{x} = \beta_1 + \beta_2 x - x^3"
_SWALLOWTAIL_EQN = r"\dot{x} = \beta_1 + \beta_2 x + \beta_3 x^2 - x^4"
_BT_FAMILY_EQN = r"\dot{x} = y,\quad \dot{y} = \beta_1 + \beta_2 y + x^2 + x y"

_EQUATIONS = {
    "saddle_node.html": r"\dot{x} = \lambda - x^2",
    "transcritical.html": r"\dot{x} = \lambda x - x^2",
    "pitchfork.html": r"\dot{x} = \lambda x - x^3",
    "hysteresis.html": r"\dot{x} = \lambda + 3x - x^3",
    "hopf.html": r"\dot{x} = \mu x - y - x(x^2+y^2),\quad "
    r"\dot{y} = x + \mu y - y(x^2+y^2)",
    "limit_cycle_branch.html": r"\dot{r} = \mu r - r^3,\quad r_{\max} = \sqrt{\mu}",
    "fold_of_cycles_branch.html": r"\dot{r} = \mu r + r^3 - r^5",
    "cycle_codim2.html": r"\dot{u} = 2(1+u) z,\quad \dot{z} = \beta_1 + u^2 "
    r"+ \beta_2 z + u z,\quad \dot\theta = 1 + 0.3 u,\quad u = r^2 - 1",
    "dynamical_bifurcation.html": r"\dot{x} = \mu x - y,\quad \dot{y} = x + \mu y",
    "collocation_convergence.html": r"\dot{x}=\mu x - y - x r^2,\ "
    r"\dot{y}=x+\mu y - y r^2",
    "deflated_roots.html": r"\dot{x}=x-x^3,\quad \dot{y}=y-y^3",
    "homoclinic_curve_shooting.html": r"\dot{x}=y,\ \dot{y}=x-x^2+\mu y+\nu x y",
    "symmetry_modes.html": r"\dot{x}_i = a x_i - x_i^3 - \lambda x_{i+1}",
    "shilnikov_loop.html": r"\dot{x}=y,\ \dot{y}=z,\ \dot{z}=-a z - y + x - x^2",
    "homoclinic_return.html": r"\dot{x} = y,\quad \dot{y} = x - x^2 + \mu y",
    "heteroclinic_cycle.html": r"\dot{x} = y,\quad \dot{y} = -x + x^3",
    "heteroclinic_network.html": r"\dot{x}_i = x_i(\lambda + a_1 x_i^2 "
    r"+ a_2 x_{i+1}^2 + a_3 x_{i+2}^2)",
    "residence_time_growth.html": r"\dot{x}_i = x_i(\lambda + a_1 x_i^2 "
    r"+ a_2 x_{i+1}^2 + a_3 x_{i+2}^2)",
    "saddle_focus_onset.html": r"\dot{x}=\sigma(y-x),\ "
    r"\dot{y}=x(\rho-z)-y,\ \dot{z}=xy-\beta z",
    "adaptive_mesh.html": r"\dot{x}=y,\quad \dot{y}=\mu(1-x^2)y - x",
    "stochastic_attractor.html": r"dx = f_{\mathrm{Lorenz}}(x)\,dt + \sigma\,dW",
    "stochastic_bifurcation.html": r"dx = \alpha x\,dt + \sigma x\,dW",
    "coloured_noise_thresholds.html": r"dx = (a x - x^3)\,dt + s x\,\eta\,dt,\ "
    r"d\eta = -\eta/\tau\,dt + dW/\tau",
    "snic_period_divergence.html": r"\dot{r} = r(1-r^2),\quad "
    r"\dot{\theta} = \mu - \sin\theta",
    "bogdanov_takens.html": r"\dot{x} = y,\quad \dot{y} = \beta_1 + \beta_2 y "
    r"+ x^2 - x y",
    "bogdanov_takens_portrait.html": r"\dot{x} = y,\quad \dot{y} = \beta_1 "
    r"+ \beta_2 y + x^2 + x y",
    "zero_hopf.html": r"\dot{x} = \mu_1 - x^2,\quad \dot{r} = \mu_2 r - r^3",
    "hopf_hopf.html": r"\dot{r}_1 = \mu_1 r_1 - r_1^3,\quad "
    r"\dot{r}_2 = \mu_2 r_2 - r_2^3",
    "cusp.html": _CUSP_EQN,
    "generalized_hopf.html": r"\dot{r} = \beta_1 r + \beta_2 r^3",
    "swallowtail.html": _SWALLOWTAIL_EQN,
    "degenerate_bautin.html": r"\dot{r} = \beta_1 r + \beta_2 r^3 + \beta_3 r^5",
    "cusp_family.html": _CUSP_EQN,
    "fluxgate_family.html": _FLUXGATE,
    "cusp_surface.html": _CUSP_EQN,
    "fluxgate_surface.html": _FLUXGATE,
    "bogdanov_takens_family.html": _BT_FAMILY_EQN,
    "bogdanov_takens_surface.html": _BT_FAMILY_EQN,
    "efield_operating_map.html": _FERROELECTRIC,
    "efield_detection.html": _FERROELECTRIC,
    "fluxgate_bias_surface.html": _FLUXGATE,
    "exploration_diagram.html": _SWALLOWTAIL_EQN,
    "exploration_skeleton.html": _SWALLOWTAIL_EQN,
    "exploration_tree.html": _SWALLOWTAIL_EQN,
    "bogdanov_takens_curve.html": r"\dot{x} = y,\quad \dot{y} = \beta_1 "
    r"+ \beta_2 x + \beta_3 y + \beta_3 x^2 + x y",
    "bogdanov_takens_type_curve.html": r"\dot{x} = y,\quad \dot{y} = \beta_1 "
    r"+ \beta_2 x + \beta_3 y + x^2 + \beta_3 x y",
    "sevencell_ring_map.html": _FERROELECTRIC,
    "sevencell_exploration.html": _FERROELECTRIC,
}


def _raw_catalogue() -> list[tuple[str, Scene]]:
    return [
        ("saddle_node.html", _saddle_node()),
        ("transcritical.html", _transcritical()),
        ("pitchfork.html", _pitchfork()),
        ("hysteresis.html", _hysteresis()),
        ("hopf.html", _hopf()),
        ("limit_cycle_branch.html", _limit_cycle_branch()),
        ("fold_of_cycles_branch.html", _fold_of_cycles_branch()),
        ("cycle_codim2.html", _cycle_codim2()),
        ("dynamical_bifurcation.html", _dynamical_bifurcation()),
        ("collocation_convergence.html", _collocation_convergence()),
        ("deflated_roots.html", _deflated_roots()),
        ("homoclinic_curve_shooting.html", _homoclinic_curve_shooting()),
        ("symmetry_modes.html", _symmetry_modes()),
        ("shilnikov_loop.html", _shilnikov_loop()),
        ("homoclinic_return.html", _homoclinic_return()),
        ("heteroclinic_cycle.html", _heteroclinic_cycle()),
        ("saddle_focus_onset.html", _saddle_focus_onset()),
        ("stochastic_bifurcation.html", _stochastic_bifurcation()),
        ("adaptive_mesh.html", _adaptive_mesh()),
        ("stochastic_attractor.html", _stochastic_attractor()),
        ("snic_period_divergence.html", _snic_period_divergence()),
        ("bogdanov_takens.html", _bogdanov_takens()),
        ("bogdanov_takens_portrait.html", _bogdanov_takens_portrait()),
        ("zero_hopf.html", _zero_hopf()),
        ("hopf_hopf.html", _hopf_hopf()),
        ("cusp.html", _cusp()),
        ("generalized_hopf.html", _generalized_hopf()),
        ("swallowtail.html", _swallowtail()),
        ("degenerate_bautin.html", _degenerate_bautin()),
        ("cusp_family.html", _cusp_family()),
        ("fluxgate_family.html", _fluxgate_family()),
        ("cusp_surface.html", _cusp_surface()),
        ("fluxgate_surface.html", _fluxgate_surface()),
        ("bogdanov_takens_family.html", _bogdanov_takens_family()),
        ("bogdanov_takens_surface.html", _bogdanov_takens_surface()),
        ("efield_operating_map.html", _efield_operating_map()),
        ("efield_detection.html", _efield_detection()),
        ("fluxgate_bias_surface.html", _fluxgate_bias_surface()),
        ("exploration_diagram.html", _exploration_diagram()),
        ("exploration_skeleton.html", _exploration_skeleton()),
        ("exploration_tree.html", _exploration_tree()),
        ("bogdanov_takens_curve.html", _bogdanov_takens_curve()),
        ("bogdanov_takens_type_curve.html", _bogdanov_takens_type_curve()),
        ("sevencell_ring_map.html", _sevencell_ring_map()),
        ("sevencell_exploration.html", _sevencell_exploration()),
        ("manifolds.html", _manifolds()),
        *_connecting_orbit_entries(),
        *_network_entries(),
        ("stochastic_density.html", _density_family()),
        ("stochastic_lyapunov.html", _lyapunov()),
        ("coloured_noise_thresholds.html", _coloured_thresholds()),
    ]


def _stages() -> tuple[tuple, ...]:
    """Every stage example: what builds it, what it is called, and its card."""
    return (
        (bt.bogdanov_takens_stage, bt.STAGE_FILENAME, bt.STAGE_ENTRY),
        (ring.ferroelectric_ring_stage, ring.STAGE_FILENAME, ring.STAGE_ENTRY),
        (vdp.van_der_pol_stage, vdp.STAGE_FILENAME, vdp.STAGE_ENTRY),
        (snic.snic_stage, snic.STAGE_FILENAME, snic.STAGE_ENTRY),
        (fold.fold_of_cycles_stage, fold.STAGE_FILENAME, fold.STAGE_ENTRY),
        (hopf.hopf_stage, hopf.STAGE_FILENAME, hopf.STAGE_ENTRY),
        (
            bautin.extended_bautin_stage,
            bautin.STAGE_FILENAME,
            bautin.STAGE_ENTRY,
        ),
        (zerohopf.zero_hopf_stage, zerohopf.STAGE_FILENAME, zerohopf.STAGE_ENTRY),
        (hopfhopf.hopf_hopf_stage, hopfhopf.STAGE_FILENAME, hopfhopf.STAGE_ENTRY),
        (
            shilnikov.shilnikov_stage,
            shilnikov.STAGE_FILENAME,
            shilnikov.STAGE_ENTRY,
        ),
        (noisy.noisy_hopf_stage, noisy.STAGE_FILENAME, noisy.STAGE_ENTRY),
    )


# How the atlas is split into units and spread over worker processes.

# The thread counts the common BLAS and OpenMP builds read when they load. Every
# worker runs one unit on one core, so a multithreaded BLAS would only fight the
# other workers for cores - and could change how a reduction is summed.
_BLAS_THREADS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# (filename, title, subtitle, kind): an atlas card, as a worker sends it back.
type _Card = tuple[str, str, str, str]


@dataclass(frozen=True, slots=True)
class _PageUnit:
    """One catalogue entry: the builder it calls and the page it names.

    ``filename`` is ``None`` for a ``*_builder()`` entry, which expands to
    several pages and names them itself.
    """

    builder: str
    filename: str | None

    @property
    def label(self) -> str:
        """What ``--only`` calls this entry."""
        return (self.filename or self.builder).removesuffix(".html").lstrip("_")


@dataclass(frozen=True, slots=True)
class _Unit:
    """One piece of work for a worker: a catalogue entry or a film."""

    kind: str
    index: int
    label: str


@dataclass(frozen=True, slots=True)
class _Built:
    """What a worker did with one unit, and what it cost."""

    cards: tuple[_Card, ...]
    seconds: float
    wall: float


def _film_kinds() -> dict[str, tuple[Callable[[], tuple[tuple, ...]], Callable]]:
    """Every kind of film: what lists them, and the renderer that draws them.

    The stochastic renderer draws a deterministic stage exactly as the stage
    renderer does, so one renderer serves every stage film. A manifold film's
    scrub bar is geodesic distance, not a parameter, and it has a renderer of its
    own.
    """
    return {
        "stage": (_stages, StochasticStageRenderer),
        "manifold": (_films, ManifoldFilmRenderer),
    }


def _page_units() -> list[_PageUnit]:
    """Every catalogue entry, read from ``_raw_catalogue`` without running it.

    Calling the catalogue builds every scene in it, one after another, so the
    driver reads which builder each entry calls and a worker calls just that
    one. The catalogue stays a plain list of ``(filename, _builder())`` and
    ``*_builder()`` entries, and an entry added to it is a unit of its own with
    no change here.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(_raw_catalogue)))
    returned = next(node for node in ast.walk(tree) if isinstance(node, ast.Return))
    if not isinstance(returned.value, ast.List):
        message = "_raw_catalogue must return a list of catalogue entries"
        raise TypeError(message)
    return [_page_unit(element) for element in returned.value.elts]


def _page_unit(element: ast.expr) -> _PageUnit:
    """The unit one catalogue entry describes, or why it cannot be split off."""
    if isinstance(element, ast.Starred):
        builder = _builder_name(element.value)
        if builder:
            return _PageUnit(builder, None)
    if isinstance(element, ast.Tuple) and len(element.elts) == 2:  # noqa: PLR2004
        name, call = element.elts
        builder = _builder_name(call)
        if isinstance(name, ast.Constant) and isinstance(name.value, str) and builder:
            return _PageUnit(builder, name.value)
    message = (
        f"catalogue entry {ast.unparse(element)} is neither "
        f'("name.html", _builder()) nor *_builder(), so it cannot be built alone'
    )
    raise ValueError(message)


def _builder_name(node: ast.expr) -> str | None:
    """The function a bare ``_builder()`` call names, or ``None``."""
    bare = isinstance(node, ast.Call) and not node.args and not node.keywords
    if bare and isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _units() -> list[_Unit]:
    """Every unit of the atlas, in the order the index lists them."""
    units = [
        _Unit("page", index, unit.label) for index, unit in enumerate(_page_units())
    ]
    for kind, (listing, _) in _film_kinds().items():
        units.extend(
            _Unit(kind, index, filename.removesuffix(".html"))
            for index, (_, filename, _) in enumerate(listing())
        )
    return units


def _schedule(units: Sequence[_Unit]) -> list[_Unit]:
    """The order units are handed to workers: films first, then pages.

    The build cannot finish before its largest unit does, and the largest are
    films - the ferroelectric ring alone costs more than half of all the pages
    together. Started first, they run alongside each other while the pages
    fill in the workers that come free.
    """
    return sorted(units, key=lambda unit: unit.kind == "page")


def _chosen(units: list[_Unit], only: Sequence[str] | None) -> list[_Unit]:
    """The units ``only`` names, in registry order; all of them without it."""
    if not only:
        return units
    known = {unit.label for unit in units}
    unknown = sorted(set(only) - known)
    if unknown:
        message = f"unknown atlas unit(s) {unknown}; choose from {sorted(known)}"
        raise SystemExit(message)
    return [unit for unit in units if unit.label in only]


@contextmanager
def _single_threaded_blas() -> Iterator[None]:
    """One BLAS thread in every process started inside this block."""
    saved = {name: os.environ.get(name) for name in _BLAS_THREADS}
    os.environ.update(dict.fromkeys(_BLAS_THREADS, "1"))
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _start_worker(latex_cache: str) -> None:
    """Fill a fresh worker's registry and share the build's typeset equations."""
    ensure_registry()
    use_disk_cache(latex_cache)


def _build_unit(kind: str, index: int, output_dir: str) -> _Built:
    """Build one unit, write its pages, and return their atlas cards."""
    started, cpu = time.perf_counter(), time.process_time()
    if kind == "page":
        cards = _build_page(index, output_dir)
    else:
        cards = _build_film(kind, index, output_dir)
    return _Built(
        tuple(cards),
        time.process_time() - cpu,
        time.perf_counter() - started,
    )


def _build_page(index: int, output_dir: str) -> list[_Card]:
    """Build one catalogue entry and write its page, or pages."""
    unit = _page_units()[index]
    built = globals()[unit.builder]()
    scenes = list(built) if unit.filename is None else [(unit.filename, built)]
    report = PlotReport(D3Renderer(), output_dir)
    cards = []
    for filename, scene in scenes:
        if filename in _EQUATIONS:
            scene = with_equation(scene, _EQUATIONS[filename])  # noqa: PLW2901
        report.write(scene, filename)
        cards.append((filename, scene.title, scene.subtitle, "plot"))
    return cards


def _build_film(kind: str, index: int, output_dir: str) -> list[_Card]:
    """Build one film and write its playable page."""
    listing, renderer = _film_kinds()[kind]
    build, filename, entry = listing()[index]
    PlotReport(renderer(), output_dir).write(build(), filename)
    return [(entry.filename, entry.title, entry.subtitle, entry.kind)]


def _build_all(
    units: Sequence[_Unit],
    output_dir: str,
    workers: int,
) -> dict[_Unit, _Built]:
    """Build every unit, each in a fresh process, ``workers`` at a time.

    A worker process is spawned for each unit and ends with it, so no unit
    inherits another's parameter values, registry or caches - only the
    equations typeset so far, which are the same bytes either way.
    """
    built: dict[_Unit, _Built] = {}
    with (
        tempfile.TemporaryDirectory(prefix="atlas-latex-") as latex_cache,
        _single_threaded_blas(),
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_start_worker,
            initargs=(latex_cache,),
            max_tasks_per_child=1,
        ) as pool,
    ):
        futures: dict[Future[_Built], _Unit] = {
            pool.submit(_build_unit, unit.kind, unit.index, output_dir): unit
            for unit in _schedule(units)
        }
        for future in as_completed(futures):
            unit = futures[future]
            try:
                built[unit] = future.result()
            except BaseException:
                pool.shutdown(cancel_futures=True)
                raise
            cost = built[unit]
            print(
                f"  {unit.label:<34} {cost.seconds:7.1f} CPU-s {cost.wall:7.1f} s",
                flush=True,
            )
    return built


def main(
    output_dir: str = "plots",
    jobs: int | None = None,
    only: Sequence[str] | None = None,
) -> None:
    """Render every scene and film to ``output_dir`` and write the atlas.

    ``jobs`` worker processes build the units, one per CPU by default; one job
    builds them one after another. ``only`` names the units to build, and the
    atlas then links just those.
    """
    units = _chosen(_units(), only)
    workers = max(1, min(jobs or os.process_cpu_count() or 1, len(units)))
    started = time.perf_counter()
    built = _build_all(units, output_dir, workers)
    entries = [AtlasEntry(*card) for unit in units for card in built[unit].cards]
    PlotReport(D3Renderer(), output_dir).write_atlas(entries)
    wall = time.perf_counter() - started
    largest = max(units, key=lambda unit: built[unit].seconds)
    print(
        f"{len(units)} units on {workers} worker(s): {wall:.1f} s wall, "
        f"{sum(cost.seconds for cost in built.values()):.1f} CPU-s in the units; "
        f"largest {largest.label} at {built[largest].seconds:.1f} CPU-s",
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the continuation atlas.")
    parser.add_argument("output_dir", nargs="?", default="plots")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="worker processes (default: one per CPU; 1 builds serially)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="build just these pages or films, e.g. hopf ferroelectric_ring",
    )
    return parser.parse_args()


_HOMOCLINIC_FILENAME = "three_dimensional_homoclinic.html"
_HOMOCLINIC_ENTRY = AtlasEntry(
    _HOMOCLINIC_FILENAME,
    "Three-dimensional homoclinic orbit",
    "A homoclinic loop to a saddle with a two-dimensional stable manifold, "
    "drawn on that manifold, grown ring by ring in geodesic distance.",
    kind="stage",
)


def _films() -> tuple[tuple, ...]:
    """Every manifold film: what builds it, what it is called, and its card."""
    return (
        (_three_dimensional_homoclinic, _HOMOCLINIC_FILENAME, _HOMOCLINIC_ENTRY),
        (lorenz.lorenz_manifold_film, lorenz.FILENAME, lorenz.ENTRY),
    )


def film_count() -> int:
    """How many films the atlas plays: every film of every kind it registers."""
    return sum(len(listing()) for listing, _ in _film_kinds().values())


if __name__ == "__main__":
    arguments = _arguments()
    main(arguments.output_dir, arguments.jobs, arguments.only)
