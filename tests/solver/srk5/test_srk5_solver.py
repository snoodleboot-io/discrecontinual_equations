"""SRK5 refuses to run, and says why.

A solver of unknown order is the defect of DEQ-25 in a harder-to-notice form, so
the refusal is the behaviour under test: constructing the solver is allowed (so
that configurations can still be built and inspected), solving is not, and the
message names the solvers that do have a measured order.
"""

import pytest

from discrecontinual_equations.solver.stochastic.srk5.srk5_config import SRK5Config
from discrecontinual_equations.solver.stochastic.srk5.srk5_solver import SRK5Solver
from tests.solver.geometric_brownian_motion import GeometricBrownianMotionCase


class TestSRK5Refusal:
    def test_solve_raises_and_points_at_measured_solvers(self):
        solver = SRK5Solver(SRK5Config(end_time=1.0, step_size=0.1, random_seed=1))
        with pytest.raises(NotImplementedError, match="SRK3Solver") as excinfo:
            solver.solve(GeometricBrownianMotionCase().equation(), [1.0])
        assert "SRK4Solver" in str(excinfo.value)
        assert solver.solution is None
