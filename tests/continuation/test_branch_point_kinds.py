"""The kinds a branch point is reported under, and the examples that match them.

The branch-point detector is asked for as ``"branch_point"`` but reports the kind
it classified - ``"transcritical"``, ``"pitchfork"``, or ``"branch_point"`` when
neither normal-form coefficient settles it. A caller that matches the requested
name finds nothing: the atlas pages did until DEQ-14, the ferroelectric ring film
did until DEQ-123. These tests keep the shared set equal to what the detector
returns, and keep the examples from matching any one of its members by name.
"""

import ast
import inspect
import textwrap
from pathlib import Path
from unittest import TestCase

from discrecontinual_equations.continuation.branch_point_kinds import (
    BRANCH_POINT_KINDS,
    is_branch_point,
)
from discrecontinual_equations.continuation.detection import BranchPoint

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _returned_literals(function: object) -> set[str]:
    """Every string literal ``function`` returns, read from its source."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def _literal_branch_matches(tree: ast.AST) -> list[ast.Compare]:
    """Comparisons of a ``.kind`` against a branch-point kind spelled out."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        sides = [node.left, *node.comparators]
        if not any(isinstance(s, ast.Attribute) and s.attr == "kind" for s in sides):
            continue
        literals = {
            leaf.value
            for side in sides
            for leaf in ast.walk(side)
            if isinstance(leaf, ast.Constant) and isinstance(leaf.value, str)
        }
        if literals & set(BRANCH_POINT_KINDS):
            found.append(node)
    return found


class TestBranchPointKinds(TestCase):
    def test_the_set_is_what_the_detector_can_report(self):
        reported = _returned_literals(BranchPoint.refine_kind)
        assert reported | {BranchPoint().kind} == set(BRANCH_POINT_KINDS)

    def test_each_classified_kind_is_a_branch_point(self):
        for kind in ("branch_point", "transcritical", "pitchfork"):
            assert is_branch_point(kind)

    def test_other_kinds_are_not(self):
        for kind in ("fold", "hopf", "cusp", "Pitchfork", ""):
            assert not is_branch_point(kind)

    def test_the_finder_sees_a_literal_match_and_not_the_helper(self):
        assert len(_literal_branch_matches(ast.parse('p.kind == "branch_point"'))) == 1
        assert len(_literal_branch_matches(ast.parse('p.kind in ("pitchfork",)'))) == 1
        assert _literal_branch_matches(ast.parse("is_branch_point(p.kind)")) == []
        assert _literal_branch_matches(ast.parse('p.kind == "hopf"')) == []

    def test_no_example_matches_a_branch_point_kind_by_name(self):
        found = [
            f"{path.relative_to(_EXAMPLES)}:{node.lineno}"
            for path in sorted(_EXAMPLES.rglob("*.py"))
            for node in _literal_branch_matches(
                ast.parse(path.read_text(encoding="utf-8")),
            )
        ]
        assert found == [], "match through is_branch_point:\n" + "\n".join(found)
