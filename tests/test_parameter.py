from unittest import TestCase

from discrecontinual_equations.parameter import Parameter


class TestParameter(TestCase):

    def test_parameter(self):
        expected_value = 1.0
        parameter = Parameter(value=expected_value, name="x", abbreviation="x")
        self.assertEqual(parameter.value, expected_value)
