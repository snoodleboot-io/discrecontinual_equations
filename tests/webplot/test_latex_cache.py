"""Typesetting each equation once: per process, and once per build on disk.

Every atlas page and film carries its equation as an SVG, and the first one a
process typesets pays Matplotlib's font scan. With the atlas built in many
processes that scan is paid many times over, so an equation is kept by its
expression - in memory for the process, and in a directory the build's workers
share. Either way the cache must hand back exactly the bytes a fresh typeset
would, or a cached build and an uncached one would differ.
"""

# Counting typesets means reaching the uncached typesetter itself.
# ruff: noqa: SLF001

from pathlib import Path

import pytest

from discrecontinual_equations.webplot import latex
from discrecontinual_equations.webplot.latex import latex_to_svg, use_disk_cache

_EXPRESSION = r"\dot x = \mu x - x^3"


@pytest.fixture
def fresh(tmp_path: Path):
    """No equation typeset yet in this process, and no disk cache unless asked."""
    latex_to_svg.cache_clear()
    use_disk_cache(None)
    yield tmp_path
    latex_to_svg.cache_clear()
    use_disk_cache(None)


@pytest.mark.usefixtures("fresh")
def test_an_expression_is_typeset_once_per_process(monkeypatch):
    calls = []
    typeset = latex._typeset

    def counted(expression: str, fontsize: float) -> str:
        calls.append(expression)
        return typeset(expression, fontsize)

    monkeypatch.setattr(latex, "_typeset", counted)
    first = latex_to_svg(_EXPRESSION)
    second = latex_to_svg(_EXPRESSION)
    assert second == first
    assert calls == [_EXPRESSION]


@pytest.mark.usefixtures("fresh")
def test_the_font_size_is_part_of_the_key():
    assert latex_to_svg(_EXPRESSION, 20.0) != latex_to_svg(_EXPRESSION, 12.0)


def test_a_disk_cached_equation_is_byte_identical_to_a_fresh_one(fresh):
    uncached = latex._typeset(_EXPRESSION, 20.0)
    use_disk_cache(fresh)
    written = latex_to_svg(_EXPRESSION)
    assert len(list(fresh.glob("*.svg"))) == 1
    latex_to_svg.cache_clear()  # as a second worker process would start
    read = latex_to_svg(_EXPRESSION)
    assert written == uncached
    assert read == uncached


def test_a_disk_cached_equation_is_not_typeset_again(fresh, monkeypatch):
    use_disk_cache(fresh)
    latex_to_svg(_EXPRESSION)
    latex_to_svg.cache_clear()

    def refuse(expression: str, fontsize: float) -> str:
        message = f"typeset {expression!r} at {fontsize} despite the disk cache"
        raise AssertionError(message)

    monkeypatch.setattr(latex, "_typeset", refuse)
    assert latex_to_svg(_EXPRESSION).lstrip().startswith("<svg")


def test_different_expressions_have_different_files(fresh):
    use_disk_cache(fresh)
    latex_to_svg(_EXPRESSION)
    latex_to_svg(r"\dot y = y")
    assert len(list(fresh.glob("*.svg"))) == 2
    assert not list(fresh.glob("*.tmp"))  # every write was completed by a rename
