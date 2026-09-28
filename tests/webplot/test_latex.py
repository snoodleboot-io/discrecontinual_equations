"""Typesetting an equation, and the reproducibility that depends on it.

Matplotlib stamps the current time into the Dublin Core metadata of every SVG it
writes. Because every atlas page carries its equation as one of those SVGs, two
builds of the same commit produced different HTML - the same page, the same
length, eight digits apart - and comparing two builds could not tell a real
change from the clock. These tests are what keeps that from coming back.
"""

from unittest import TestCase

from discrecontinual_equations.webplot.latex import latex_to_svg

_EXPRESSION = r"\dot x = \mu - x^2"


class TestLatexToSvg(TestCase):
    def test_typesets_to_an_svg(self):
        svg = latex_to_svg(_EXPRESSION)
        assert svg.lstrip().startswith("<svg")
        assert "currentColor" in svg  # the theme controls the ink
        assert len(svg) > 1000

    def test_records_no_date(self):
        """The date is what made two builds of one commit differ."""
        assert "dc:date" not in latex_to_svg(_EXPRESSION)

    def test_the_same_equation_typesets_to_the_same_bytes(self):
        """An equation is the same equation whenever it was typeset."""
        assert latex_to_svg(_EXPRESSION) == latex_to_svg(_EXPRESSION)

    def test_different_equations_differ(self):
        """A guard on the two above: they would pass on a constant, too."""
        assert latex_to_svg(_EXPRESSION) != latex_to_svg(r"\dot y = y")
