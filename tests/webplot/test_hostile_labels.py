"""Every renderer, fed labels that are trying to be markup.

Each string a page shows - titles, subtitles, notes, axis and series labels,
frame labels, marker names, the atlas's cards - is set to :data:`HOSTILE`,
which closes the payload's ``<script>``, opens one of its own and plants an
element. The page must come out with only its own scripts, the payload must
read back as the very same strings, and, where a browser is at hand, the page
it builds must hold no planted element and show the string as text.
"""

import html
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from discrecontinual_equations.webplot.escaping import escape_text, script_json
from discrecontinual_equations.webplot.manifold_film import (
    FilmCurve,
    FilmMarker,
    ManifoldFilm,
)
from discrecontinual_equations.webplot.manifold_film_renderer import (
    ManifoldFilmRenderer,
)
from discrecontinual_equations.webplot.renderer import D3Renderer
from discrecontinual_equations.webplot.report import AtlasEntry, _atlas_html
from discrecontinual_equations.webplot.scene import Axes, Scene, Series
from discrecontinual_equations.webplot.stage import (
    CycleBranch,
    Equilibrium,
    Frame,
    Lattice,
    StageScene,
    StageSystem,
    Timeline,
)
from discrecontinual_equations.webplot.stage_renderer import StageRenderer
from discrecontinual_equations.webplot.stochastic_stage_renderer import (
    StochasticStageRenderer,
)
from tests.webplot.test_manifold_film import _film as _plane_film

HOSTILE = (
    "</script><script>document.title='pwned'</script>"
    '<img id="pwned" src="x"><!-- & \' " -->'
)
_PAYLOAD = re.compile(r"window\.__[A-Z]+__ = (.*?);</script>", re.DOTALL)


def _scene() -> Scene:
    return Scene(
        HOSTILE,
        HOSTILE,
        Axes(HOSTILE, HOSTILE),
        [
            Series("line", "stable", HOSTILE, [(0.0, 0.0), (1.0, 1.0)]),
            Series("marker", "fold", HOSTILE, [(0.5, 0.5)]),
        ],
    )


def _stage() -> StageScene:
    frame = Frame(
        0.0,
        [0.0] * 8,
        [Equilibrium(0.0, 0.0, "stable", [(-1.0, 2.0), (-1.0, -2.0)])],
        [],
        [0],
    )
    frame.label = HOSTILE
    return StageScene(
        StageSystem(HOSTILE, HOSTILE, HOSTILE, note=HOSTILE),
        Lattice(((-1.0, 1.0), (-1.0, 1.0)), 2, HOSTILE, HOSTILE),
        Timeline([], []),
        CycleBranch([], HOSTILE),
        [frame],
    )


def _film() -> ManifoldFilm:
    plain = _plane_film()
    plain.mesh.stopped = HOSTILE
    plain.mesh.warnings = [HOSTILE]
    film = ManifoldFilm(
        HOSTILE,
        HOSTILE,
        plain.mesh,
        axes=(HOSTILE, HOSTILE, HOSTILE),
        note=HOSTILE,
    )
    film.curves.append(FilmCurve(HOSTILE, HOSTILE, [[0, 0, 0], [0, 0, 0.5]]))
    film.markers.append(FilmMarker(HOSTILE, "saddle", [0.0, 0.0, 0.0]))
    return film


def _pages() -> dict[str, str]:
    entry = AtlasEntry(HOSTILE, HOSTILE, HOSTILE, kind="stage")
    return {
        "d3": D3Renderer(library="").render(_scene()),
        "stage": StageRenderer(library="").render(_stage()),
        "stochastic stage": StochasticStageRenderer(library="").render(_stage()),
        "manifold film": ManifoldFilmRenderer().render(_film()),
        "atlas": _atlas_html([entry]),
    }


@pytest.fixture(scope="module")
def pages() -> dict[str, str]:
    return _pages()


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for item in value.values() for s in _strings(item)]
    if isinstance(value, list):
        return [s for item in value for s in _strings(item)]
    return []


def test_the_escapes_are_html_and_script_safe():
    assert escape_text('<a href="x">&\'') == "&lt;a href=&quot;x&quot;&gt;&amp;&#x27;"
    text = script_json({"label": HOSTILE})
    for character in "<>&":
        assert character not in text
    assert json.loads(text) == {"label": HOSTILE}


@pytest.mark.parametrize(
    "filename",
    [
        "javascript:alert(1)",
        " JaVaScRiPt:alert(1)",
        "java\tscript:alert(1)",
        "data:text/html,x",
    ],
)
def test_an_atlas_card_never_links_to_a_script_url(filename):
    # Escaping keeps a filename inside its attribute, but a filename that
    # is a ``javascript:`` URL is still one, and runs when the card is
    # clicked (measured in headless Chrome). A card links to a file beside
    # the atlas, so its href must be a relative path, whatever the name.
    page = _atlas_html([AtlasEntry(filename, "title", "subtitle")])
    href = re.search(r'<a class="card" href="([^"]*)"', page).group(1)
    assert ":" not in html.unescape(href)


def test_an_ordinary_filename_links_as_it_is():
    page = _atlas_html([AtlasEntry("lorenz_manifold.html", "title", "subtitle")])
    assert 'href="lorenz_manifold.html"' in page


RENDERERS = ["d3", "stage", "stochastic stage", "manifold film", "atlas"]


@pytest.mark.parametrize("name", RENDERERS)
def test_only_the_pages_own_scripts_open_and_close(pages, name):
    page = pages[name]
    assert page.count("</script") == page.count("<script")
    assert '<img id="pwned"' not in page
    assert "<script>document.title" not in page


@pytest.mark.parametrize("name", [n for n in RENDERERS if n != "atlas"])
def test_the_payload_reads_back_the_strings_unchanged(pages, name):
    match = _PAYLOAD.search(pages[name])
    assert match is not None
    payload = json.loads(match.group(1))
    found = _strings(payload)
    assert HOSTILE in found
    assert all(s == HOSTILE or "pwned" not in s for s in found)


def test_scripts_escape_labels_they_put_into_markup(pages):
    # The tooltip and the film's SVG text are the places a label joins a
    # string of markup; both go through ``esc``.
    assert "esc(s.label)" in pages["d3"]
    assert "esc(m.label)" in pages["manifold film"]
    assert "esc(name)" in pages["manifold film"]


_CHROME = next(
    (
        path
        for path in (shutil.which("google-chrome"), shutil.which("chromium"))
        if path
    ),
    None,
)


@pytest.mark.skipif(_CHROME is None, reason="no headless Chrome to load the page")
@pytest.mark.parametrize("name", RENDERERS)
def test_the_browser_builds_no_planted_element(pages, name):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "page.html"
        path.write_text(pages[name], encoding="utf-8")
        result = subprocess.run(  # noqa: S603 (a fixed local binary on our own file)
            [
                _CHROME,
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--virtual-time-budget=4000",
                "--dump-dom",
                path.as_uri(),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    dom = result.stdout
    assert "<html" in dom
    assert '<img id="pwned"' not in dom
    assert "<title>pwned</title>" not in dom
    assert '&lt;img id="pwned"' in dom or "&lt;img id=&quot;pwned&quot;" in dom
