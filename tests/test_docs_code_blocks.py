"""Every Python example in the docs runs as written (DEQ-131, DEQ-173).

The user guide, the README and the design notes inside the package
(``discrecontinual_equations/**/*.md``) are all covered.

A page's ``python`` blocks are run in order in one namespace, the way a reader
pastes them into one session, from a scratch working directory so plots land
there. A fence left open swallows the rest of a page without an error, so the
fences are checked to pair up first.
"""

import re
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_DOCS = _ROOT / "docs"
_GUIDE = sorted([_DOCS / "index.md", *(_DOCS / "user-guide").glob("*.md")])
_PACKAGE = sorted(
    [_ROOT / "README.md", *(_ROOT / "discrecontinual_equations").rglob("*.md")]
)
_PAGES = _GUIDE + _PACKAGE
_FENCE = re.compile(r"^```", re.MULTILINE)
_PYTHON = re.compile(r"^```python[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
# Pages whose examples take more than about 3 s on an idle machine (continuation
# of connecting and periodic orbits; the quasipotential page's parabolic march on
# an 81 x 81 grid and its error check, about 10 s) run in the nightly slow set,
# not the gate.
_SLOW_PAGES = {
    "connecting-orbits.md",
    "periodic-orbits.md",
    "quasipotential-first-passage.md",
}
_RUNNABLE = [
    pytest.param(page, marks=pytest.mark.slow) if page.name in _SLOW_PAGES else page
    for page in _PAGES
]


def _blocks(page: Path) -> list[tuple[int, str]]:
    text = page.read_text(encoding="utf-8")
    return [
        (text[: match.start()].count("\n") + 1, match.group(1))
        for match in _PYTHON.finditer(text)
    ]


def _page_id(page: Path) -> str:
    # User-guide pages keep their bare names, as earlier tickets quote them.
    return page.name if page in _GUIDE else str(page.relative_to(_ROOT))


@pytest.mark.parametrize("page", _PAGES, ids=_page_id)
def test_fences_pair_up(page: Path) -> None:
    fences = len(_FENCE.findall(page.read_text(encoding="utf-8")))
    assert fences % 2 == 0, f"{page.name} has an unclosed code fence"


@pytest.mark.parametrize("page", _RUNNABLE, ids=_page_id)
def test_python_blocks_run(page: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace: dict[str, object] = {"__name__": "__docs__"}
    with TemporaryDirectory() as scratch:
        monkeypatch.chdir(scratch)
        for line, source in _blocks(page):
            code = compile(source, f"{page.relative_to(_ROOT)}:{line}", "exec")
            exec(code, namespace)  # noqa: S102  # nosec B102 - our own docs
        assert Path.cwd() == Path(scratch).resolve()


def test_the_getting_started_pages_have_examples() -> None:
    # A page whose examples vanished (an unclosed fence, a renamed language tag)
    # would pass the test above vacuously.
    for name in ("getting-started.md", "basic-usage.md"):
        assert _blocks(_DOCS / "user-guide" / name), f"{name} has no python block"
    assert _blocks(_DOCS / "index.md")


def test_the_readme_and_package_pages_have_examples() -> None:
    # The same guard for the pages DEQ-173 added.
    for page in (
        _ROOT / "README.md",
        _ROOT / "discrecontinual_equations" / "plot" / "PLOT.md",
    ):
        assert _blocks(page), f"{page.name} has no python block"
