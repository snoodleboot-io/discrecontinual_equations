"""Render LaTeX math to a self-contained, theme-adaptive inline SVG.

Equations are typeset once at build time with Matplotlib's mathtext engine and
emitted as vector SVG, so they stay crisp at any zoom and add only a few kilobytes
per figure. The glyph fill is rewritten to ``currentColor`` so the surrounding
theme (light, dark, or print) controls the ink without re-rendering.
"""

import io
import re

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

_HEX_STYLE = re.compile(r"fill:\s*#[0-9a-fA-F]{6}")
_HEX_ATTR = re.compile(r'fill="#[0-9a-fA-F]{6}"')
_SVG_OPEN = re.compile(r"<svg\b")
_PT_SIZE = re.compile(r'(width|height)="[0-9.]+pt"')


def latex_to_svg(expression: str, fontsize: float = 20.0) -> str:
    """Typeset ``expression`` (a LaTeX math string) as a currentColor SVG."""
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
    )
    plt.close(figure)
    svg = buffer.getvalue().decode("utf-8")
    svg = svg[svg.index("<svg") :]
    svg = _HEX_STYLE.sub("fill:currentColor", svg)
    svg = _HEX_ATTR.sub('fill="currentColor"', svg)
    svg = _PT_SIZE.sub("", svg)
    return _SVG_OPEN.sub('<svg fill="currentColor"', svg, count=1)
