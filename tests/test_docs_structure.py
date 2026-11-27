"""The documentation is stitched together without duplicates or dead links.

Several lanes merging independently left the same reference key in two area
files, and the same module rendered twice in the API pages (DEQ-182); the
package's own ``.md`` files linked to pages that never existed (DEQ-157).
"""

import re
from collections import Counter
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_DOCS = _ROOT / "docs"
_PACKAGE = _ROOT / "discrecontinual_equations"
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_KEY = re.compile(r"^- \*\*\[(\w+)\]\*\*", re.MULTILINE)
_SNIPPET = re.compile(r'^--8<-- "([^"]+)"', re.MULTILINE)
_RENDER = re.compile(r"^::: ([\w.]+)", re.MULTILINE)
_LINK = re.compile(r"\]\(([^)\s]+)\)")


def _stitched_references() -> str:
    # The references page as mkdocs sees it: each snippet line replaced by the
    # area file it names.
    page = (_DOCS / "references.md").read_text(encoding="utf-8")
    return _SNIPPET.sub(
        lambda match: (_ROOT / match.group(1)).read_text(encoding="utf-8"), page
    )


def _duplicates(items: list[str]) -> list[str]:
    return sorted(item for item, count in Counter(items).items() if count > 1)


def test_the_references_page_stitches_every_area_file() -> None:
    page = (_DOCS / "references.md").read_text(encoding="utf-8")
    stitched = {Path(name).name for name in _SNIPPET.findall(page)}
    assert stitched == {path.name for path in (_DOCS / "references").glob("*.md")}


def test_each_reference_key_has_one_entry() -> None:
    keys = _KEY.findall(_COMMENT.sub("", _stitched_references()))
    assert len(keys) > 100, "the entry pattern no longer matches the area files"
    assert _duplicates(keys) == []


def test_each_module_is_rendered_once() -> None:
    renders = [
        module
        for page in sorted(_DOCS.rglob("*.md"))
        for module in _RENDER.findall(page.read_text(encoding="utf-8"))
    ]
    assert len(renders) > 50, "the ::: pattern no longer matches the API pages"
    assert _duplicates(renders) == []


_PACKAGE_PAGES = sorted(_PACKAGE.rglob("*.md"))


@pytest.mark.parametrize(
    "page", _PACKAGE_PAGES, ids=lambda page: str(page.relative_to(_PACKAGE))
)
def test_package_page_links_resolve(page: Path) -> None:
    text = re.sub(r"```.*?```", "", page.read_text(encoding="utf-8"), flags=re.DOTALL)
    broken = []
    for target in _LINK.findall(text):
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            continue  # external URLs and in-page anchors
        if not (page.parent / target.split("#", 1)[0]).exists():
            broken.append(target)
    assert broken == []
