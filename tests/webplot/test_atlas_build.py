"""Building the atlas in parallel without changing a byte of it.

The atlas is split into units - one catalogue entry, or one film - and each unit
is built in a fresh worker process. These tests hold the driver to what makes
that safe: every catalogue entry becomes a unit without the catalogue being
rewritten, the index lists the units in registry order whatever order they
finish in, the workers' BLAS runs on one thread, and a parallel build writes
exactly the bytes a serial one does - and the bytes the old in-process loop did.
"""

# The driver's pieces are module-private, like the catalogue's builders.
# ruff: noqa: SLF001

import ast
import hashlib
import os
import re
from concurrent.futures import Future
from pathlib import Path
from typing import Self

import pytest

from discrecontinual_equations.webplot.renderer import D3Renderer
from discrecontinual_equations.webplot.report import PlotReport
from discrecontinual_equations.webplot.scene_builder import with_equation
from examples.continuation import webplot_examples as atlas

# Cheap entries, enough of them that three workers finish out of order.
_SUBSET = ("saddle_node", "transcritical", "pitchfork", "hysteresis")


def _digests(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def test_every_catalogue_entry_becomes_a_unit():
    source = Path(atlas.__file__).read_text(encoding="utf-8")
    function = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_raw_catalogue"
    )
    returned = next(n for n in ast.walk(function) if isinstance(n, ast.Return))
    assert isinstance(returned.value, ast.List)
    units = atlas._page_units()
    assert len(units) == len(returned.value.elts)
    assert all(callable(getattr(atlas, unit.builder)) for unit in units)
    labels = [unit.label for unit in units]
    assert len(set(labels)) == len(labels)
    assert labels[0] == "saddle_node"


def test_a_unit_that_expands_to_several_pages_is_one_unit():
    starred = [unit for unit in atlas._page_units() if unit.filename is None]
    assert [unit.builder for unit in starred] == [
        "_connecting_orbit_entries",
        "_network_entries",
    ]


@pytest.mark.parametrize(
    "entry",
    ['("a.html", scene)', '("a.html", _build(3))', "_build", "(name, _build())"],
)
def test_an_entry_the_driver_cannot_split_is_refused(entry):
    element = ast.parse(entry, mode="eval").body
    with pytest.raises(ValueError, match="catalogue entry"):
        atlas._page_unit(element)


def test_every_film_is_a_unit_after_the_pages():
    units = atlas._units()
    kinds = [unit.kind for unit in units]
    assert kinds == sorted(kinds, key=lambda kind: kind != "page")
    assert sum(kind == "stage" for kind in kinds) == len(atlas._stages())


def test_the_heaviest_units_are_started_first():
    """Films dominate the build, so they start before the pages queue behind."""
    order = atlas._schedule(atlas._units())
    first_page = next(i for i, unit in enumerate(order) if unit.kind == "page")
    assert all(unit.kind == "page" for unit in order[first_page:])


def test_blas_is_pinned_to_one_thread_while_workers_run(monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    with atlas._single_threaded_blas():
        assert all(os.environ[name] == "1" for name in atlas._BLAS_THREADS)
    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"
    assert "OMP_NUM_THREADS" not in os.environ


def test_unknown_names_are_refused(tmp_path):
    with pytest.raises(SystemExit, match="no_such_page"):
        atlas.main(str(tmp_path), jobs=1, only=["no_such_page"])


def test_parallel_serial_and_in_process_builds_are_byte_identical(tmp_path):
    serial, parallel, again = tmp_path / "s", tmp_path / "p", tmp_path / "q"
    atlas.main(str(serial), jobs=1, only=list(_SUBSET))
    atlas.main(str(parallel), jobs=3, only=list(_SUBSET))
    atlas.main(str(again), jobs=3, only=list(_SUBSET))
    assert _digests(serial) == _digests(parallel) == _digests(again)
    assert set(_digests(serial)) == {f"{name}.html" for name in _SUBSET} | {
        "index.html",
    }
    # The loop this replaces: every scene built and written in one process.
    atlas.ensure_registry()
    report = PlotReport(D3Renderer(), str(tmp_path / "old"))
    for name in _SUBSET:
        filename = f"{name}.html"
        scene = getattr(atlas, f"_{name}")()
        report.write(with_equation(scene, atlas._EQUATIONS[filename]), filename)
    old = _digests(tmp_path / "old")
    assert old == {k: v for k, v in _digests(parallel).items() if k != "index.html"}


def test_the_index_is_in_registry_order(tmp_path):
    chosen = ["pitchfork", "saddle_node", "hysteresis"]  # not registry order
    atlas.main(str(tmp_path), jobs=3, only=chosen)
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    links = re.findall(r'href="([^"]+)\.html"', index)
    assert links == ["saddle_node", "pitchfork", "hysteresis"]


_SEEN: dict = {}


class _Pool:
    """Stands in for the process pool: records how it was made, builds nothing."""

    def __init__(self, **options) -> None:
        _SEEN["made"] = {
            **options,
            "blas": {n: os.environ.get(n) for n in atlas._BLAS_THREADS},
        }
        _SEEN["submitted"] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        return None

    def submit(self, _function, kind: str, index: int, _output_dir: str) -> Future:
        _SEEN["submitted"].append((kind, index))
        future: Future = Future()
        card = (f"{kind}{index}.html", f"{kind} {index}", "", "plot")
        future.set_result(atlas._Built((card,), 0.0, 0.0))
        return future


def test_workers_are_fresh_spawned_single_threaded_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(atlas, "ProcessPoolExecutor", _Pool)
    # Hand results back last-submitted first, as a slow first unit would.
    monkeypatch.setattr(atlas, "as_completed", lambda futures: list(futures)[::-1])
    atlas.main(str(tmp_path), jobs=4)
    assert _SEEN["made"]["max_tasks_per_child"] == 1
    assert _SEEN["made"]["mp_context"].get_start_method() == "spawn"
    assert set(_SEEN["made"]["blas"].values()) == {"1"}
    units = atlas._units()
    assert _SEEN["submitted"] == [(u.kind, u.index) for u in atlas._schedule(units)]
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    links = re.findall(r'href="([^"]+)\.html"', index)
    assert links == [f"{unit.kind}{unit.index}" for unit in units]


def test_only_picks_the_named_units_in_registry_order():
    chosen = atlas._chosen(atlas._units(), ["hopf_stage", "cusp", "hopf"])
    assert [unit.label for unit in chosen] == ["hopf", "cusp", "hopf_stage"]
