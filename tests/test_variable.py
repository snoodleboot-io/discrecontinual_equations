from unittest import TestCase

from discrecontinual_equations.variable import Variable


class TestVariable(TestCase):
    def test_variable(self):
        variable = Variable(name="Displacement", abbreviation="x", range=[0, 1])

        variable.discretization = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        self.assertIsInstance(variable, Variable)
        self.assertEqual(variable.abbreviation, "x")
        self.assertEqual(variable.name, "Displacement")
        self.assertEqual(
            variable.discretization,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        )
