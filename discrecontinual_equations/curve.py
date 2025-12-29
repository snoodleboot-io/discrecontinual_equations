from discrecontinual_equations.variable import Variable


class Curve:
    """
    Representation of a curve in time or space for differential equation solutions.

    A Curve object stores the discretized solution of a differential equation,
    containing time points and corresponding variable values. This class provides
    a structured way to store and access numerical solution data.

    The curve can represent either time-dependent solutions (with a time variable)
    or spatial solutions (without time). Each point in the discretization contains:
    - Time value (if time-dependent)
    - Variable values at that point
    - Result/derivative values at that point

    Attributes:
        time: Time variable (None for spatial curves)
        variables: List of system variables
        results: List of computed results/derivatives
    """

    __slots__ = ["_results", "_time", "_variables"]

    def __init__(
        self,
        variables: list[Variable],
        results: list[Variable],
        time: Variable | None,
    ) -> None:
        """
        Initialize a curve with variables, results, and optional time.

        Args:
            variables: List of Variable objects representing system variables
            results: List of Variable objects for storing computed results
            time: Time variable for time-dependent problems (None for spatial)
        """
        self._time: Variable | None = time
        self._variables: list[Variable] = variables
        self._results: list[Variable] = results

    @property
    def time(self):
        return self._time

    @property
    def variables(self):
        return self._variables

    @property
    def results(self):
        return self._results

    def __getitem__(self, index: int) -> list[float | list[float]]:
        """
        Get a point from the curve at the specified index.

        Returns [time, variables, results] for time-dependent curves,
        or [variables, results] for spatial curves.

        Args:
            index: Index of the point to retrieve

        Returns:
            List containing time (if applicable), variable values, and results
        """
        points = (
            [
                self._time.discretization[index],
                [variable.discretization[index] for variable in self._variables],
                [result.discretization[index] for result in self._results],
            ]
            if self._time
            else [
                [variable.discretization[index] for variable in self._variables],
                [result.discretization[index] for result in self._results],
            ]
        )
        return points

    def __setitem__(self, index: int, values: list[float | list[float]]):
        """
        Set a point in the curve at the specified index.

        Args:
            index: Index of the point to set
            values: Values to set in format [time, variables, results] or [variables, results]
        """
        if self._time:
            self._time.discretization[index] = values[0]
            for variable, value in zip(self._variables, values[1], strict=False):
                variable.discretization[index] = value
            for result, value in zip(self._results, values[2], strict=False):
                result.discretization[index] = value
        else:
            for variable, value in zip(self._variables, values[0], strict=False):
                variable.discretization[index] = value
            for result, value in zip(self._results, values[1], strict=False):
                result.discretization[index] = value

    def append(self, values: list[float | list[float]]):
        """
        Append a new point to the curve.

        Args:
            values: Values to append in format [time, variables, results] or [variables, results]
        """
        if self._time:
            self._time.discretization.append(values[0])
            [
                variable.discretization.append(value)
                for variable, value in zip(self._variables, values[1], strict=False)
            ]
            [
                variable.discretization.append(value)
                for variable, value in zip(self._results, values[2], strict=False)
            ]
        else:
            [
                variable.discretization.append(value)
                for variable, value in zip(self._variables, values[0], strict=False)
            ]

            [
                variable.discretization.append(value)
                for variable, value in zip(self._results, values[1], strict=False)
            ]

    def __len__(self):
        """Return the length of the curve."""
        if self._time:
            return len(self._time.discretization)
        return len(self._variables[0].discretization) if self._variables else 0
