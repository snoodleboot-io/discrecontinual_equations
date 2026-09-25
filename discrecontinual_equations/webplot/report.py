"""Write rendered scenes to disk and an atlas linking them.

:class:`PlotReport` pairs an :class:`~.renderer.HtmlRenderer` with an output
directory: it writes one HTML document per scene and an ``index.html`` atlas that
links them, so a set of analyses becomes a small browsable site.
"""

from pathlib import Path

from discrecontinual_equations.webplot.renderer import HtmlRenderer


class AtlasEntry:
    """One card on the atlas index page.

    ``kind`` is ``"plot"`` for a still and ``"stage"`` for a playable page, which
    the atlas marks so a reader knows which cards move.
    """

    __slots__ = ["filename", "kind", "subtitle", "title"]

    def __init__(
        self,
        filename: str,
        title: str,
        subtitle: str,
        kind: str = "plot",
    ) -> None:
        self.filename = filename
        self.title = title
        self.subtitle = subtitle
        self.kind = kind


class PlotReport:
    """Render scenes to a directory and build an atlas index.

    The renderer decides what a scene is: a :class:`~.renderer.D3Renderer` writes
    :class:`~.scene.Scene` stills, a :class:`~.stage_renderer.StageRenderer`
    writes :class:`~.stage.StageScene` pages. Two reports on one directory share
    an atlas by passing both sets of entries to :meth:`write_atlas`.
    """

    __slots__ = ["_directory", "_renderer"]

    def __init__(self, renderer: HtmlRenderer, output_dir: str) -> None:
        self._renderer = renderer
        self._directory = Path(output_dir)

    def write(self, scene: object, filename: str) -> Path:
        """Render ``scene`` to ``filename`` in the output directory."""
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / filename
        path.write_text(self._renderer.render(scene))
        return path

    def write_atlas(self, entries: list[AtlasEntry]) -> Path:
        """Write ``index.html`` linking the listed documents."""
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / "index.html"
        path.write_text(_atlas_html(entries))
        return path


def _atlas_html(entries: list[AtlasEntry]) -> str:
    cards = "\n".join(_card(entry) for entry in entries)
    return _ATLAS_TEMPLATE.replace("__CARDS__", cards)


def _card(entry: AtlasEntry) -> str:
    stage = entry.kind == "stage"
    classes = "card stage" if stage else "card"
    tag = '<span class="tag">&#9654; play</span>' if stage else ""
    return (
        f'<a class="{classes}" href="{entry.filename}">'
        f"{tag}<h2>{_escape(entry.title)}</h2>"
        f"<p>{_escape(entry.subtitle)}</p></a>"
    )


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_ATLAS_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Continuation atlas</title>
<style>
  *{box-sizing:border-box}
  body{margin:0;color:#eef1f7;
    background:radial-gradient(1200px 700px at 50% -8%, #151a28 0%, #0a0c12 60%),
      #0a0c12;
    font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,
      Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1040px;margin:0 auto;padding:44px 22px 72px}
  h1{font-size:30px;margin:0 0 8px;letter-spacing:-.02em;font-weight:680}
  .lead{color:#9aa3b8;margin:0 0 30px;max-width:76ch}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(292px,1fr));
    gap:18px}
  a.card{display:block;position:relative;overflow:hidden;
    background:linear-gradient(180deg,#141824,#10131d);
    border:1px solid #222838;border-radius:16px;padding:20px;
    text-decoration:none;color:inherit;transition:transform .15s,border-color .15s,
      box-shadow .15s;
    box-shadow:0 18px 44px -28px rgba(0,0,0,.9)}
  a.card:hover{transform:translateY(-3px);border-color:#38bdf8;
    box-shadow:0 26px 60px -28px rgba(56,189,248,.35)}
  a.card::before{content:"";position:absolute;left:0;top:0;height:3px;width:100%;
    background:linear-gradient(90deg,#38bdf8,#a78bfa,#e879f9);opacity:.0;
    transition:opacity .15s}
  a.card:hover::before{opacity:.9}
  a.card.stage{border-color:#2c3a5a;
    background:linear-gradient(180deg,#151c30,#10131d)}
  a.card .tag{display:inline-block;font-size:11px;letter-spacing:.12em;
    text-transform:uppercase;color:#8ab4ff;margin:0 0 8px}
  a.card h2{font-size:16.5px;margin:0 0 7px;font-weight:620}
  a.card p{color:#9aa3b8;font-size:13px;margin:0}
</style>
</head>
<body>
<div class="wrap">
<h1>Continuation &amp; bifurcation atlas</h1>
<p class="lead">Self-contained D3 plots of numerical continuation results:
equilibrium branches and their codim-1, codim-2, and codim-3 bifurcations. Curves
are centripetal Catmull-Rom splines through the computed points.</p>
<div class="grid">
__CARDS__
</div>
</div>
</body>
</html>
"""
