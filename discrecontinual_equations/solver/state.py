"""A placeholder for a solver's state vector.

``State`` declares a ``values`` list and nothing else; no solver uses it yet.
Solvers carry their state as plain lists or NumPy arrays.
"""


class State:
    """A state vector, ``values``; declared but not used by any solver."""

    values: list[float]
