"""The ferroelectric ring film, built, carries the labels its stage sets.

The stage labels its pitchfork "symmetry breaking (lambda = a)". It used to look
for that point under the kind the detector was asked for, ``"branch_point"``,
while the detector reports the kind it classified, ``"pitchfork"``; nothing
matched, and the built film showed a bare "pitchfork" (DEQ-123). This builds
the film the way ``main`` does and reads the labels back out of the page.

Like the noisy Hopf film's test, it imports an example: what it checks is the
page that example writes, and a copy of the stage here would not be that page.
"""

import json
import re
import tempfile
from pathlib import Path
from unittest import TestCase

from discrecontinual_equations.continuation.branch_point_kinds import is_branch_point
from examples.continuation import stage_ferroelectric_ring as film

# The detector lands on the pitchfork at lambda = a to its own accuracy.
_LOCATED = 1.0e-3
# Every label the stage sets, and the parameter it belongs at.
_EXPECTED = {"symmetry breaking (λ = a)": film.GAIN}


def _payload(html: str) -> dict:
    match = re.search(r"window.__STAGE__ = (.*?);</script>", html, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


class TestFerroelectricRingFilm(TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            film.main(directory)
            page = Path(directory) / film.STAGE_FILENAME
            cls.special = _payload(page.read_text(encoding="utf-8"))["branch"][
                "special"
            ]

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
