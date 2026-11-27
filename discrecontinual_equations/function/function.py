"""The base class for a right-hand side: variables, parameters and ``eval``.

A ``Function`` holds the state variables, the parameters, the result variables
and an optional time variable, and evaluates the field at a point with
``eval(point, time)``: the derivative ``f(t, y)`` of an ODE or the drift of an
SDE. ``eval_batch`` evaluates many points at once, by default one ``eval`` per
row.
"""

import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class Function:
    """Base class for differential equation functions."""

    @property
    def variables(self) -> list[Variable]:
        """The state variables."""
        return self._variables

    @property
    def parameters(self) -> list[Parameter]:
        """The parameters."""
        return self._parameters

    @property
    def time(self) -> Variable | None:
        """The time variable, or ``None``."""
        return self._time

    __slots__ = ["_curve", "_parameters", "_time", "_variables"]

    def __init__(
        self,
        variables: list[Variable],
        parameters: list[Parameter],
        results: list[Variable],
        time: Variable | None = None,
    ):
        """Hold the variables, parameters, results and time.

        Args:
            variables: The state variables.
            parameters: The parameters.
            results: The result variables.
            time: The time variable, if any.
        """
        self._variables: list[Variable] = variables
        self._parameters: list[Parameter] = parameters
        self._time: Variable | None = time
        self._curve: Curve = Curve(
            variables=self._variables,
            time=self._time,
            results=results,
        )

    @property
    def curve(self) -> Curve:
        """A curve over this function's own variables; the solvers return their own."""
        return self._curve

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        """Evaluate the function at a given point and time.

        For ODEs: returns the derivative f(x, t)
        For SDEs: returns the drift term μ(x, t)

        Args:
            point: Current state vector
            time: Current time

        Returns:
            Function evaluation result
        """
        raise NotImplementedError

    def eval_batch(self, points: np.ndarray, time: float | None = None) -> np.ndarray:
        """Evaluate at every row of ``points`` (``(count, n)``), as ``(count, n)``.

        The default calls :meth:`eval` once per row, so every function supports it
        unchanged; a function that overrides it with array arithmetic makes the
        callers that evaluate it on a whole grid at once (the stationary
        Fokker-Planck assembly) cost a few array operations instead of a Python call
        per point.

        Each row is handed over as Python floats, as
        :func:`~discrecontinual_equations.continuation.noise.field_at` hands a single
        state, so the two agree to the bit. NumPy scalars would round the same in
        arithmetic but not in the builtin ``sum``, which since Python 3.12 adds floats
        with compensation and NumPy scalars without it.

        Args:
            points: The points, one per row.
            time: The time, shared by all rows.

        Returns:
            The values, one row per point.
        """
        rows = np.asarray(points, dtype=float).tolist()
        return np.array(
            [self.eval(point=row, time=time) for row in rows],
            dtype=float,
        )
