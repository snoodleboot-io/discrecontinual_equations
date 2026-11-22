"""A cycle's manifold on the stage: the tube of fibres from its chart (DEQ-38).

The fold-of-cycles field ``r' = (mu + r^2 - r^4) r`` has a repelling cycle at the
smaller root of ``r^4 - r^2 = mu``: its unstable manifold is the band between
the attracting origin and the attracting outer cycle, so every fibre must start
on the inner cycle and end near one of the two attractors.
"""

import json
import math
from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.periodic_orbit import (
    PeriodicOrbitSolution,
)
from discrecontinual_equations.webplot.cycle_tube import (
    TubeSettings,
    add_cycle_tubes,
    cycle_tube,
)
from discrecontinual_equations.webplot.stage import (
    Cycle,
    CycleBranch,
    Frame,
    Lattice,
    StageScene,
    StageSystem,
    Timeline,
    stage_payload,
)
from tests.continuation.cycle_fields import circle, fold_equation

MU = -0.15
BOX = ((-1.45, 1.45), (-1.45, 1.45))


def _radius(root: float) -> float:
    return math.sqrt((1.0 + root * math.sqrt(1.0 + 4.0 * MU)) / 2.0)


INNER, OUTER = _radius(-1.0), _radius(1.0)


def _cycle(radius: float) -> PeriodicOrbitSolution:
    return PeriodicOrbitSolution(circle(radius, 60), 2.0 * math.pi)


class TestCycleTube(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.field = fold_equation(MU).derivative
        cls.settings = TubeSettings(fibres=6, time=40.0)
        cls.fibres = cycle_tube(
            cls.field,
            _cycle(INNER),
            BOX,
            unstable=True,
            settings=cls.settings,
        )

    def test_one_fibre_each_side_of_each_phase(self):
        assert len(self.fibres) == 2 * self.settings.fibres
        assert {fibre.kind for fibre in self.fibres} == {"unstable-surface"}

    def test_every_fibre_starts_on_the_cycle(self):
        for fibre in self.fibres:
            assert abs(math.hypot(*fibre.points[0]) - INNER) < 1.0e-9
            assert fibre.curve[0][:2] == list(fibre.points[0])

    def test_fibres_end_at_the_two_attractors(self):
        ends = sorted(math.hypot(*fibre.points[-1]) for fibre in self.fibres)
        half = len(ends) // 2
        assert all(end < 0.05 for end in ends[:half])
        assert all(abs(end - OUTER) < 1.0e-3 for end in ends[half:])

    def test_a_stable_cycle_has_no_unstable_tube(self):
        with pytest.raises(ValueError, match="selection picks 0"):
            cycle_tube(self.field, _cycle(OUTER), BOX, unstable=True)


class TestAddCycleTubes(TestCase):
    def test_only_the_unstable_cycle_gets_a_tube(self):
        equation = fold_equation(MU)
        parameter = equation.derivative.parameters[0]
        inner_multiplier = math.exp(2.0 * math.pi * (2 * INNER**2 - 4 * INNER**4))
        outer_multiplier = math.exp(2.0 * math.pi * (2 * OUTER**2 - 4 * OUTER**4))
        cycles = [
            Cycle(
                MU,
                2 * math.pi,
                INNER,
                [(1.0, 0.0), (inner_multiplier, 0.0)],
                [(float(x), float(y)) for x, y in circle(INNER, 60)],
            ),
            Cycle(
                MU,
                2 * math.pi,
                OUTER,
                [(1.0, 0.0), (outer_multiplier, 0.0)],
                [(float(x), float(y)) for x, y in circle(OUTER, 60)],
            ),
        ]
        frame = Frame(MU, [], [], [], [0, 1])
        scene = StageScene(
            StageSystem("Fold of cycles", "", "mu"),
            Lattice(BOX, 3),
            Timeline([], []),
            CycleBranch(cycles),
            [frame],
        )
        drawn = add_cycle_tubes(
            scene, equation.derivative, parameter, TubeSettings(fibres=4)
        )
        assert drawn == 1
        assert len(frame.manifolds) == 8
        starts = {round(math.hypot(*m.points[0]), 6) for m in frame.manifolds}
        assert starts == {round(INNER, 6)}
        json.dumps(stage_payload(scene))
        assert np.isfinite(
            [x for m in frame.manifolds for p in m.points for x in p]
        ).all()
