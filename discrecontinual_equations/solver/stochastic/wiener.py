"""The Wiener increments a stochastic Runge-Kutta step consumes.

A stochastic Runge-Kutta scheme is not an ODE Runge-Kutta scheme with a random
number added at the end. Its derivation (an Ito-Taylor expansion truncated at
some order, with derivatives replaced by supporting values) tells it exactly
which functionals of the driving Brownian motion it needs over one step, and the
stage structure only has the order it was derived for when those functionals
are supplied with the joint distribution the derivation assumed. Drawing one
fresh normal per stage, or using ``sqrt(dt)`` where an increment belongs, gives
a different scheme whose order nobody has worked out. This module is where the
schemes get their increments, so that the correlation is in one place and can
be checked once.

Two functionals cover every scheme in this package:

* ``delta_w = W(t + h) - W(t)``, the increment, which is ``N(0, h)``.
* ``delta_z = integral_t^{t+h} (W(s) - W(t)) ds``, the time integral of the
  increment (Kloeden and Platen's ``I_(1,0)``), which the order 1.5 strong
  scheme needs. It is Gaussian with ``Var delta_z = h^3 / 3`` and
  ``Cov(delta_w, delta_z) = h^2 / 2``, so with two independent standard normals
  ``u1`` and ``u2`` the pair is exactly
  ``delta_w = sqrt(h) u1``, ``delta_z = (h^{3/2} / 2) (u1 + u2 / sqrt(3))``.

Both are per noise channel. The diffusion interface of this library returns one
coefficient per state component, so the noise is diagonal: component ``i`` is
driven by its own Wiener process ``W_i`` and the channels are independent. What
no source here provides are the Levy areas ``I_(j,k)`` between channels, which
have no closed-form sampler; that is why every order claim in this package is
stated for scalar noise and lowered for systems.

Why the increments come from a source object rather than from a generator inside
each solver: a strong-order measurement is the expected pathwise distance
between the scheme and the exact solution, driven by the same Brownian path. It
is only meaningful if every step size sees the same path, which means the path
has to exist before the solver runs. :class:`BrownianPath` is that pre-generated
path, reading each step's ``delta_w`` and ``delta_z`` off a fine grid;
:class:`GaussianWienerSource` is the ordinary case, drawing fresh increments from
a generator the solver owns - never ``np.random.seed``, which seeds the
process-wide stream and lets one solver alter another's results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WienerIncrements:
    """What one step of Brownian motion gives a scheme, one entry per channel."""

    delta_w: np.ndarray
    """``W(t + h) - W(t)``."""

    delta_z: np.ndarray
    """``integral_t^{t+h} (W(s) - W(t)) ds``, Kloeden and Platen's ``I_(1,0)``."""


class WienerSource(ABC):
    """Supplies the Wiener functionals for one step of a scheme."""

    @abstractmethod
    def increments(self, time: float, step: float, dimension: int) -> WienerIncrements:
        """The increments over ``[time, time + step]`` for ``dimension`` channels."""
        raise NotImplementedError


class GaussianWienerSource(WienerSource):
    """Fresh increments from a generator this source owns.

    ``seed=None`` means fresh entropy, as ``np.random.default_rng`` does.
    """

    __slots__ = ["_generator"]

    def __init__(self, seed: int | None = None) -> None:
        self._generator = np.random.default_rng(seed)

    def increments(self, time: float, step: float, dimension: int) -> WienerIncrements:  # noqa: ARG002
        u1 = self._generator.standard_normal(dimension)
        u2 = self._generator.standard_normal(dimension)
        root = np.sqrt(step)
        delta_w = root * u1
        delta_z = 0.5 * step * root * (u1 + u2 / np.sqrt(3.0))
        return WienerIncrements(delta_w=delta_w, delta_z=delta_z)


class BrownianPath(WienerSource):
    """A Brownian path fixed in advance, read at whatever step a scheme uses.

    The path is held on a uniform grid of spacing ``resolution``. A step must be
    a whole number of grid cells, so that every step size that divides the
    horizon sees exactly the same path: ``delta_w`` is then the exact difference
    of grid values, and ``delta_z`` is the trapezoidal integral of ``W - W(t)``
    over the cells inside the step, whose error is of order ``resolution^2`` per
    step and so well below any scheme error at the step sizes a convergence
    study uses.
    """

    __slots__ = ["_resolution", "_start", "_values"]

    def __init__(self, start: float, resolution: float, values: np.ndarray) -> None:
        if values.ndim != 2:  # noqa: PLR2004 (a matrix: grid nodes by channels)
            message = "values must have shape (nodes, channels)"
            raise ValueError(message)
        self._start = float(start)
        self._resolution = float(resolution)
        self._values = np.asarray(values, dtype=float)

    @classmethod
    def sample(
        cls,
        start: float,
        end: float,
        resolution: float,
        dimension: int,
        seed: int | None = None,
    ) -> "BrownianPath":
        """Draw a path on ``[start, end]`` with ``W(start) = 0`` on every channel."""
        generator = np.random.default_rng(seed)
        cells = round((end - start) / resolution)
        if cells < 1 or not np.isclose(start + cells * resolution, end):
            message = "the horizon must be a whole number of resolution cells"
            raise ValueError(message)
        increments = generator.standard_normal((cells, dimension)) * np.sqrt(resolution)
        values = np.vstack([np.zeros((1, dimension)), np.cumsum(increments, axis=0)])
        return cls(start=start, resolution=resolution, values=values)

    @property
    def dimension(self) -> int:
        return self._values.shape[1]

    @property
    def end(self) -> float:
        return self._start + (self._values.shape[0] - 1) * self._resolution

    def _node(self, time: float) -> int:
        offset = (time - self._start) / self._resolution
        node = round(offset)
        if (
            not np.isclose(offset, node, atol=1.0e-6)
            or node < 0
            or node >= len(self._values)
        ):
            message = f"time {time} is not a grid node of this path"
            raise ValueError(message)
        return node

    def value(self, time: float) -> np.ndarray:
        """``W(time)`` on every channel; ``time`` must be a grid node."""
        return self._values[self._node(time)]

    def increments(self, time: float, step: float, dimension: int) -> WienerIncrements:
        if dimension != self.dimension:
            message = (
                f"path has {self.dimension} channels, scheme asked for {dimension}"
            )
            raise ValueError(message)
        first = self._node(time)
        last = self._node(time + step)
        segment = self._values[first : last + 1] - self._values[first]
        delta_w = segment[-1].copy()
        delta_z = np.trapezoid(segment, dx=self._resolution, axis=0)
        return WienerIncrements(delta_w=delta_w, delta_z=delta_z)
