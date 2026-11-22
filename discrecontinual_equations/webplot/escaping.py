"""Putting our strings into a page without letting them become markup.

Every renderer writes text into HTML (titles, notes, labels) and a JSON
payload into a ``<script>`` element. Today every string is ours, but a label
holding ``<`` or ``</script>`` would still break the page, and a page is the
kind of file that ends up showing strings someone else wrote. So:

* text goes through :func:`html.escape`, quotes included, so it is safe in an
  element and in an attribute alike;
* JSON is written with ``<``, ``>`` and ``&`` as ``\\u003c``, ``\\u003e`` and
  ``\\u0026`` - the same JSON, which the script reads back unchanged, but with
  no ``</script>`` and no ``<!--`` for the HTML parser to act on.
  ``json.dumps`` already writes every non-ASCII character escaped, the line
  separators older JavaScript took for line ends inside a string among them;
* a script puts a label into markup only through ``esc``
  (:data:`ESCAPE_JS`), the same escape on the browser's side.
"""

import html
import json

_SCRIPT_UNSAFE = {"<": "\\u003c", ">": "\\u003e", "&": "\\u0026"}

# The browser-side escape: what a script wraps a label in before it joins a
# string of markup (an SVG <text>, a tooltip).
ESCAPE_JS = (
    'const esc = s => String(s).replace(/[&<>"\']/g, c => ({"&": "&amp;", '
    '"<": "&lt;", ">": "&gt;", \'"\': "&quot;", "\'": "&#39;"})[c]);'
)


def escape_text(text: str) -> str:
    """``text`` as HTML: safe in an element's content and in an attribute."""
    return html.escape(text, quote=True)


def script_json(payload: object) -> str:
    """``payload`` as JSON that is safe inside a ``<script>`` element."""
    text = json.dumps(payload)
    for character, escaped in _SCRIPT_UNSAFE.items():
        text = text.replace(character, escaped)
    return text
