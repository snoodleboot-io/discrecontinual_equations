"""The Fokker-Planck coefficients gathered in one batched pass (DEQ-53 item 3).

The face and cell coefficients used to be gathered by a Python loop, one field call
and a handful of small NumPy operations per face. They are now gathered with one
``eval_batch`` per field per face family. The per-point loop is kept here, written
from the public :mod:`~...continuation.noise` helpers, as the reference: every
coefficient must equal it bit for bit, and a field that overrides ``eval_batch``
with array arithmetic must never be called point by point.
"""

from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    StationaryFokkerPlanck,
)
from discrecontinual_equations.continuation.noise import (
    NoiseMatrix,
    diffusion_matrix,
    field_at,
    fields_at,
    jacobian_at,
    jacobians_along,
    state_drift_shift,
    stratonovich_factor,
)
from discrecontinual_equations.continuation.stochastic import Ito, Stratonovich
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import (
    Alpha,
    ConstantNoiseColumn,
    HopfField,
    IsotropicNoiseColumn,
    Kappa,
    Sigma,
    State,
)


class Hopf3(DeterministicFunction):
    """The planar Hopf field with a cubic contraction in a third direction."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        mu = self.parameters[0].value
        x, y, z = point[0], point[1], point[2]
        radius = x * x + y * y
        return [mu * x - y - x * radius, x + mu * y - y * radius, -z - z * z * z]


class Calls:
    scalar = 0
    batched = 0


class ArrayHopf(HopfField):
    """The Hopf field with array arithmetic for whole batches."""

    def eval(self, point, time=None):
        Calls.scalar += 1
        return super().eval(point, time)

    def eval_batch(self, points, time=None):  # noqa: ARG002 (base signature)
        Calls.batched += 1
        mu = self.parameters[0].value
        x, y = points[:, 0], points[:, 1]
        radius = x * x + y * y
        return np.stack([mu * x - y - x * radius, x + mu * y - y * radius], axis=1)


class ArrayIsotropic(IsotropicNoiseColumn):
    def eval(self, point, time=None):
        Calls.scalar += 1
        return super().eval(point, time)

    def eval_batch(self, points, time=None):  # noqa: ARG002 (base signature)
        Calls.batched += 1
        sigma = self.parameters[1].value
        kappa = self.parameters[2].value
        amplitude = sigma * np.sqrt(1.0 + kappa * np.sum(points * points, axis=1))
        out = np.zeros_like(points)
        out[:, self._index] = amplitude
        return out


def _states(dimension):
    return [State() for _ in range(dimension)]


def _problem(dimension, convention, *, correlated, drift_type=None, column_type=None):
    drift_type = drift_type or (HopfField if dimension == 2 else Hopf3)
    column_type = column_type or IsotropicNoiseColumn
    drift = drift_type(
        variables=_states(dimension),
        parameters=[Alpha(value=0.3)],
        results=_states(dimension),
        time=None,
    )
    columns = [
        column_type(
            index,
            _states(dimension),
            [Alpha(value=0.0), Sigma(value=0.5), Kappa(value=0.4)],
            _states(dimension),
            None,
        )
        for index in range(dimension)
    ]
    if correlated:
        columns.append(
            ConstantNoiseColumn(
                variables=_states(dimension),
                parameters=[Alpha(value=0.2)] * dimension,
                results=_states(dimension),
                time=None,
            ),
        )
    side = 9 if dimension == 2 else 6
    grid = DensityGrid.box([-2.0] * dimension, [2.0] * dimension, [side] * dimension)
    return StationaryFokkerPlanck(drift, NoiseMatrix(columns), convention, grid), grid


def _looped_face_coefficients(problem, faces, axis):
    """The per-face loop the batched pass replaced, from the public helpers."""
    factor = stratonovich_factor(problem._convention)  # noqa: SLF001
    noise = problem._noise  # noqa: SLF001
    drift = np.empty(faces.shape[0])
    diffusion = np.empty(faces.shape[0])
    for row, state in enumerate(faces):
        columns = noise.columns_at(state)
        value = field_at(problem._drift, state)[axis]  # noqa: SLF001
        if factor != 0.0:
            jacobians = noise.jacobians_at(state)
            value += state_drift_shift(jacobians, columns, factor)[axis]
        drift[row] = value
        diffusion[row] = diffusion_matrix(columns)[axis, axis]
    return drift, diffusion


def _faces(grid, axis):
    points = grid.points()
    indices = np.arange(grid.size).reshape(grid.shape)
    count = grid.shape[axis]
    low = np.take(indices, np.arange(count - 1), axis=axis).ravel()
    high = np.take(indices, np.arange(1, count), axis=axis).ravel()
    return 0.5 * (points[low] + points[high])


class TestBatchedHelpers(TestCase):
    def test_fields_and_jacobians_along_rows_equal_the_pointwise_ones(self):
        problem, grid = _problem(3, Ito(), correlated=True)
        states = grid.points() * 7.5
        field = problem._drift  # noqa: SLF001
        base = fields_at(field, states)
        jacobians = jacobians_along(field, states, base)
        for row, state in enumerate(states):
            np.testing.assert_array_equal(base[row], field_at(field, state))
            np.testing.assert_array_equal(jacobians[row], jacobian_at(field, state))

    def test_noise_columns_and_jacobians_along_rows_equal_the_pointwise_ones(self):
        problem, grid = _problem(2, Ito(), correlated=True)
        noise = problem._noise  # noqa: SLF001
        states = grid.points()
        columns = noise.columns_along(states)
        jacobians = noise.jacobians_along(states, columns)
        for row, state in enumerate(states):
            np.testing.assert_array_equal(columns[row], noise.columns_at(state))
            np.testing.assert_array_equal(jacobians[row], noise.jacobians_at(state))


class TestBatchedCoefficients(TestCase):
    def test_face_coefficients_equal_the_per_face_loop_bit_for_bit(self):
        for dimension in (2, 3):
            for convention in (Ito(), Stratonovich()):
                for correlated in (False, True):
                    problem, grid = _problem(
                        dimension,
                        convention,
                        correlated=correlated,
                    )
                    for axis in range(dimension):
                        faces = _faces(grid, axis)
                        coefficients = problem._face_coefficients  # noqa: SLF001
                        batched = coefficients(faces, axis)
                        looped = _looped_face_coefficients(problem, faces, axis)
                        np.testing.assert_array_equal(batched[0], looped[0])
                        np.testing.assert_array_equal(batched[1], looped[1])

    def test_cell_diffusion_equals_the_per_cell_product_bit_for_bit(self):
        problem, grid = _problem(3, Stratonovich(), correlated=True)
        points = grid.points()
        cells = problem._diffusion(points)  # noqa: SLF001
        noise = problem._noise  # noqa: SLF001
        for row, state in enumerate(points):
            np.testing.assert_array_equal(
                cells[row],
                diffusion_matrix(noise.columns_at(state)),
            )

    def test_an_array_field_is_never_called_point_by_point(self):
        problem, grid = _problem(
            2,
            Stratonovich(),
            correlated=False,
            drift_type=ArrayHopf,
            column_type=ArrayIsotropic,
        )
        Calls.scalar = 0
        Calls.batched = 0
        density = problem.solve()

        assert Calls.scalar == 0
        # cells (2 columns) + per axis: drift + 2 columns + 2 columns x 2 shifts
        assert Calls.batched == 2 + 2 * (1 + 2 + 2 * 2)
        reference, _grid = _problem(2, Stratonovich(), correlated=False)
        np.testing.assert_allclose(
            density.values,
            reference.solve().values,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        assert abs(density.mass() - 1.0) < 1.0e-12
        assert grid.size == density.values.size
