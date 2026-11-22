"""Guard the examples against a format spec printed in place of its value.

An f-string whose replacement field was lost leaves its format spec behind as a
plain literal - ``print(".2f")`` - which runs and prints ".2f". No test runs the
examples, so this reads their source instead: a string literal that is nothing
but a format spec, outside a replacement field, is that failure.
"""

import ast
import re
from pathlib import Path
from unittest import TestCase

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
_BARE_SPEC = re.compile(r"^[+\- ]?\d*,?\.\d+[eEfFgG%]$")


def _bare_specs(tree: ast.AST) -> list[ast.Constant]:
    """String literals that are only a format spec, outside any ``{...:spec}``."""
    in_specs = {
        id(part)
        for node in ast.walk(tree)
        if isinstance(node, ast.FormattedValue) and node.format_spec is not None
        for part in ast.walk(node.format_spec)
    }
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in in_specs
        and _BARE_SPEC.match(node.value)
    ]


class TestExamplePrints(TestCase):
    def test_the_detector_sees_a_lost_field_and_not_a_kept_one(self):
        assert len(_bare_specs(ast.parse('print(".2f")'))) == 1
        assert _bare_specs(ast.parse('x = 1\nprint(f"x = {x:.2f}")')) == []

    def test_no_example_prints_a_bare_format_spec(self):
        found = [
            f"{path.relative_to(_EXAMPLES)}:{node.lineno}: {node.value!r}"
            for path in sorted(_EXAMPLES.rglob("*.py"))
            for node in _bare_specs(ast.parse(path.read_text(encoding="utf-8")))
        ]
        assert found == [], "format specs printed as text:\n" + "\n".join(found)
