"""The ferroelectric ring film, built, carries the labels its stage sets.

The stage labels its pitchfork "symmetry breaking (lambda = a)". It used to look
for that point under the kind the detector was asked for, ``"branch_point"``,
while the detector reports the kind it classified, ``"pitchfork"``; nothing
matched, and the built film showed a bare "pitchfork" (DEQ-123).

The gate builds the film as ``main`` does but with two frames and no rotating
wave. The labels come from the equilibrium branches, so dropping the wave's cycle
continuation and most frames leaves them as the full film has them. It then reads
the labels back out of the page. The full film, about 100 s, is built only in the
``slow`` set (DEQ-144).

Like the noisy Hopf film's test, it imports an example: what it checks is the
page that example writes, and a copy of the stage here would not be that page.
"""

import json
import re
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import pytest

from discrecontinual_equations.continuation.branch_point_kinds import is_branch_point
from examples.continuation import stage_ferroelectric_ring as film

# The detector lands on the pitchfork at lambda = a to its own accuracy.
_LOCATED = 1.0e-3
# Every label the stage sets, and the parameter it belongs at.
_EXPECTED = {"symmetry breaking (λ = a)": film.GAIN}
# Two frames are the fewest a film has: its first and last coupling.
_REDUCED_FRAMES = 2


def _payload(html: str) -> dict:
    match = re.search(r"window.__STAGE__ = (.*?);</script>", html, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def _built_specials() -> list[dict]:
    with tempfile.TemporaryDirectory() as directory:
        film.main(directory)
        page = Path(directory) / film.STAGE_FILENAME
        return _payload(page.read_text(encoding="utf-8"))["branch"]["special"]


class _LabelChecks:
    special: list[dict]

    def test_every_expected_label_is_on_the_page(self):
        labels = {point["label"] for point in self.special}
        missing = sorted(set(_EXPECTED) - labels)
        assert missing == [], f"labels missing from the built film: {missing}"

    def test_each_label_marks_the_branch_point_it_names(self):
        for label, parameter in _EXPECTED.items():
            marked = [point for point in self.special if point["label"] == label]
            assert len(marked) == 1, label
            assert is_branch_point(marked[0]["kind"])
            assert abs(marked[0]["p"] - parameter) < _LOCATED


class TestFerroelectricRingFilmReduced(_LabelChecks, TestCase):
    """The page from a two-frame film without the rotating wave."""

    @classmethod
    def setUpClass(cls):
        with (
            patch.object(film, "FRAMES", _REDUCED_FRAMES),
            patch.object(film, "wave_branch", list),
        ):
            cls.special = _built_specials()


@pytest.mark.slow
class TestFerroelectricRingFilm(_LabelChecks, TestCase):
    """The page ``main`` writes, every frame and the rotating wave included."""

    @classmethod
    def setUpClass(cls):
        cls.special = _built_specials()
