"""Render LaTeX math to a self-contained, theme-adaptive inline SVG.

Equations are typeset once at build time with Matplotlib's mathtext engine and
emitted as vector SVG, so they stay crisp at any zoom and add only a few kilobytes
per figure. The glyph fill is rewritten to ``currentColor`` so the surrounding
theme (light, dark, or print) controls the ink without re-rendering.

The date Matplotlib would otherwise record in the SVG's metadata is suppressed,
so the same equation typesets to the same bytes on every build.

That makes an equation safe to keep by its expression. Each process keeps what
it has typeset, and :func:`use_disk_cache` names a directory that processes
share: the first equation a process typesets pays Matplotlib's font scan, about
a second, and a build spread over many worker processes would otherwise pay it
in every one of them.
"""

import functools
import hashlib
import io
import os
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

_HEX_STYLE = re.compile(r"fill:\s*#[0-9a-fA-F]{6}")
_HEX_ATTR = re.compile(r'fill="#[0-9a-fA-F]{6}"')
_SVG_OPEN = re.compile(r"<svg\b")
_PT_SIZE = re.compile(r'(width|height)="[0-9.]+pt"')

_disk_cache: Path | None = None


def use_disk_cache(directory: str | Path | None) -> None:
    """Keep typeset equations in ``directory`` as well; ``None`` stops.

    The directory belongs to one build: a cached file is keyed by the
    expression, the size, Matplotlib's version and its math font set, not by
    the post-processing below, so a directory must not outlive a change to it.
    """
    global _disk_cache  # noqa: PLW0603 (module-level setting)
    _disk_cache = None if directory is None else Path(directory)


@functools.cache
def latex_to_svg(expression: str, fontsize: float = 20.0) -> str:
    """Typeset ``expression`` (a LaTeX math string) as a currentColor SVG."""
    if _disk_cache is None:
        return _typeset(expression, fontsize)
    path = _disk_cache / f"{_key(expression, fontsize)}.svg"
    if path.is_file():
        return path.read_bytes().decode("utf-8")
    svg = _typeset(expression, fontsize)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Two workers may typeset the same equation at once. Each writes its own
    # file and renames it into place, so a reader sees a whole file or none.
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    temporary.write_bytes(svg.encode("utf-8"))
    temporary.replace(path)
    return svg


def _key(expression: str, fontsize: float) -> str:
    """What an equation's SVG depends on, as a file name."""
    parts = (mpl.__version__, mpl.rcParams["mathtext.fontset"], fontsize, expression)
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()


def _typeset(expression: str, fontsize: float) -> str:
    """Typeset ``expression`` with Matplotlib, every time it is asked."""
    figure = plt.figure(figsize=(0.1, 0.1))
    figure.patch.set_alpha(0.0)
    figure.text(0, 0, f"${expression}$", fontsize=fontsize)
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
        # Matplotlib stamps the current time into the SVG's Dublin Core metadata
        # as <dc:date>, which made two builds of the same commit differ - the
        # same page, the same length, eight digits apart. An equation is the
        # same equation whenever it was typeset, so the date is dropped.
        metadata={"Date": None},
    )
    plt.close(figure)
    svg = buffer.getvalue().decode("utf-8")
    svg = svg[svg.index("<svg") :]
    svg = _HEX_STYLE.sub("fill:currentColor", svg)
    svg = _HEX_ATTR.sub('fill="currentColor"', svg)
    svg = _PT_SIZE.sub("", svg)
    return _SVG_OPEN.sub('<svg fill="currentColor"', svg, count=1)
