"""Every Runge-Kutta tableau in the package against its order conditions, exactly.

A Runge-Kutta method has order ``p`` when, for every rooted tree ``t`` with at
most ``p`` vertices, ``sum_i b_i Phi_i(t) = 1 / gamma(t)`` (Butcher 1963, eqs.
(27)-(37)): ``Phi_i`` is the elementary weight built from ``a`` over the tree's
branches and ``gamma`` the tree's density. That is 1, 2, 4, 8, 17, 37, 85, 200
and 486 conditions for orders 1 to 9. A non-autonomous field adds one more
requirement, ``c_i = sum_j a_ij``: Butcher's reduction treats ``t`` as a state
component, and the stage time is then ``t + h sum_j a_ij``.

The checks are in :class:`fractions.Fraction` arithmetic, so a wrong digit
anywhere fails by its full size rather than hiding under a float tolerance. The
one tableau whose published coefficients are rational *approximations*, Prince
and Dormand's 8(7), is held to 1e-16 (its true residuals are below 1.1e-17);
its one-digit typo (DEQ-139) missed by 3.1e-7.

Each row's order is checked from both sides: every condition up to the stated
order holds, and some condition one order higher fails. The second half is
what catches swapped rows (DEQ-138), where the stated order 5 row was order 4
and the stated order 4 row was order 5.
"""

import itertools
import math
from fractions import Fraction
from functools import cache

import numpy as np
import pytest

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.solver.deterministic.butcher_tableau import (
    ButcherTableau,
    row,
)
from discrecontinual_equations.solver.deterministic.cash_karp.cash_karp_config import (
    CashKarpConfig,
)
from discrecontinual_equations.solver.deterministic.cash_karp.cash_karp_solver import (
    CASH_KARP_54,
    CashKarpSolver,
)
from discrecontinual_equations.solver.deterministic.dormand_prince.dormand_prince_config import (  # noqa: E501
    DormandPrinceConfig,
)
from discrecontinual_equations.solver.deterministic.dormand_prince.dormand_prince_solver import (  # noqa: E501
    DORMAND_PRINCE_54,
    DormandPrinceSolver,
)
from discrecontinual_equations.solver.deterministic.fehlberg.fehlberg_config import (
    FehlbergConfig,
)
from discrecontinual_equations.solver.deterministic.fehlberg.fehlberg_solver import (
    FEHLBERG_45,
    FehlbergSolver,
)
from discrecontinual_equations.solver.deterministic.prince_dormand87.prince_dormand87_config import (  # noqa: E501
    PrinceDormand87Config,
)
from discrecontinual_equations.solver.deterministic.prince_dormand87.prince_dormand87_solver import (  # noqa: E501
    PRINCE_DORMAND_87,
    PrinceDormand87Solver,
)
from discrecontinual_equations.solver.stochastic.srk2.srk2_config import SRK2Config
from discrecontinual_equations.solver.stochastic.srk2.srk2_solver import SRK2Solver
from discrecontinual_equations.solver.stochastic.srk3.srk3_config import SRK3Config
from discrecontinual_equations.solver.stochastic.srk3.srk3_solver import SRK3Solver
from discrecontinual_equations.solver.stochastic.srk4.srk4_config import SRK4Config
from discrecontinual_equations.solver.stochastic.srk4.srk4_solver import SRK4Solver
from discrecontinual_equations.variable import Variable
from discrecontinual_equations.webplot import manifold_flow

Tree = tuple

# How far a condition one order above the stated one must miss for the order
# to count as exactly the stated one, not more.
MISS = Fraction(1, 10**12)


@cache
def trees(order: int) -> tuple[Tree, ...]:
    """Every rooted tree with ``order`` vertices, as a sorted tuple of subtrees."""
    if order == 1:
        return ((),)
    found = set()
    for sizes in _partitions(order - 1, order - 1):
        for children in itertools.product(*(trees(size) for size in sizes)):
            found.add(tuple(sorted(children)))
    return tuple(sorted(found))


def _partitions(total: int, largest: int):
    if total == 0:
        yield []
        return
    for part in range(min(total, largest), 0, -1):
        for rest in _partitions(total - part, part):
            yield [part, *rest]


def size(tree: Tree) -> int:
    return 1 + sum(size(child) for child in tree)


def density(tree: Tree) -> int:
    """``gamma(t)``: the tree's size times its children's densities."""
    return size(tree) * math.prod(density(child) for child in tree)


def residuals(a, weights, top: int) -> dict[int, Fraction]:
    """Largest ``|sum_i b_i Phi_i(t) - 1/gamma(t)|`` over the trees of each order."""
    stages = len(weights)
    full = [list(r) + [Fraction(0)] * (stages - len(r)) for r in a]
    weight_of: dict[Tree, list[Fraction]] = {}

    def elementary(tree: Tree) -> list[Fraction]:
        if tree not in weight_of:
            value = [Fraction(1)] * stages
            for child in tree:
                inner = elementary(child)
                value = [
                    value[i] * sum(full[i][j] * inner[j] for j in range(i))
                    for i in range(stages)
                ]
            weight_of[tree] = value
        return weight_of[tree]

    return {
        order: max(
            abs(
                sum(b * w for b, w in zip(weights, elementary(tree), strict=True))
                - Fraction(1, density(tree))
            )
            for tree in trees(order)
        )
        for order in range(1, top + 1)
    }


def order_of(a, weights, tolerance: Fraction, top: int) -> int:
    """The largest ``p <= top`` whose conditions all hold within ``tolerance``."""
    found = 0
    for order, residual in residuals(a, weights, top).items():
        if residual > tolerance:
            break
        found = order
    return found


class TestRootedTrees:
    """The tree machinery itself, against counts and methods known independently."""

    def test_tree_counts_are_the_rooted_tree_numbers(self):
        # OEIS A000081: 1, 1, 2, 4, 9, 20, 48, 115, 286.
        assert [len(trees(n)) for n in range(1, 10)] == [
            1,
            1,
            2,
            4,
            9,
            20,
            48,
            115,
            286,
        ]

    def test_densities_of_the_order_three_trees(self):
        # The chain [[.]] has gamma 6 and the bush [., .] has gamma 3.
        assert sorted(density(tree) for tree in trees(3)) == [3, 6]

    def test_classical_runge_kutta_is_exactly_fourth_order(self):
        a = (row(""), row("1/2"), row("0 1/2"), row("0 0 1"))
        weights = row("1/6 1/3 1/3 1/6")
        assert order_of(a, weights, Fraction(0), 6) == 4

    def test_heun_is_exactly_second_order(self):
        assert order_of((row(""), row("1")), row("1/2 1/2"), Fraction(0), 4) == 2


# The 8(7) coefficients are rational approximations, as published; the others
# are exact.
TABLEAUX = {
    "dormand_prince_54": (DORMAND_PRINCE_54, Fraction(0)),
    "cash_karp_54": (CASH_KARP_54, Fraction(0)),
    "fehlberg_45": (FEHLBERG_45, Fraction(0)),
    "prince_dormand_87": (PRINCE_DORMAND_87, Fraction(1, 10**16)),
}


def _cases(failing: dict[str, str]):
    return [
        pytest.param(
            *TABLEAUX[name],
            id=name,
            marks=(
                [pytest.mark.xfail(strict=True, reason=failing[name])]
                if name in failing
                else []
            ),
        )
        for name in TABLEAUX
    ]


class TestTableaux:
    @pytest.mark.parametrize(
        ("tableau", "tolerance"),
        _cases({}),
    )
    def test_nodes_are_the_row_sums(self, tableau: ButcherTableau, tolerance):
        for node, stage in zip(tableau.c, tableau.a, strict=True):
            assert abs(node - sum(stage, Fraction(0))) <= tolerance

    @pytest.mark.parametrize(
        ("tableau", "tolerance"),
        _cases(
            {},
        ),
    )
    def test_advancing_row_has_exactly_its_order(
        self,
        tableau: ButcherTableau,
        tolerance,
    ):
        found = residuals(tableau.a, tableau.b, tableau.order + 1)
        assert all(found[p] <= tolerance for p in range(1, tableau.order + 1))
        assert found[tableau.order + 1] > MISS

    @pytest.mark.parametrize(
        ("tableau", "tolerance"),
        _cases(
            {},
        ),
    )
    def test_embedded_row_has_exactly_its_order(
        self,
        tableau: ButcherTableau,
        tolerance,
    ):
        found = residuals(tableau.a, tableau.b_hat, tableau.embedded_order + 1)
        assert all(found[p] <= tolerance for p in range(1, tableau.embedded_order + 1))
        assert found[tableau.embedded_order + 1] > MISS

    @pytest.mark.parametrize(
        ("solver", "config", "tableau", "advance", "embedded"),
        [
            (DormandPrinceSolver, DormandPrinceConfig, DORMAND_PRINCE_54, "b5", "b4"),
            (CashKarpSolver, CashKarpConfig, CASH_KARP_54, "b5", "b4"),
            (FehlbergSolver, FehlbergConfig, FEHLBERG_45, "b5", "b4"),
            (
                PrinceDormand87Solver,
                PrinceDormand87Config,
                PRINCE_DORMAND_87,
                "b8",
                "b7",
            ),
        ],
    )
    def test_solver_steps_with_its_tableau(
        self,
        solver,
        config,
        tableau,
        advance,
        embedded,
    ):
        built = solver(config())
        a, c, b, b_hat = tableau.arrays()
        np.testing.assert_array_equal(built.a, a)
        np.testing.assert_array_equal(built.c, c)
        np.testing.assert_array_equal(getattr(built, advance), b)
        np.testing.assert_array_equal(getattr(built, embedded), b_hat)


def test_manifold_flow_steps_with_the_dormand_prince_tableau():
    """The film's vectorised Dormand-Prince pair is the solver's, entry for entry.

    It drops the seventh stage (first same as last) and stores the error row as
    ``b_hat - b``; each entry is the exact value rounded once, so the
    comparison allows one rounding of the difference.
    """
    tableau = DORMAND_PRINCE_54
    a, c, b, _ = tableau.arrays()
    np.testing.assert_array_equal(manifold_flow.C, c[:6])
    np.testing.assert_array_equal(manifold_flow.A, a[:6, :5])
    np.testing.assert_array_equal(manifold_flow.B, b[:6])
    error = [float(h - w) for h, w in zip(tableau.b_hat, tableau.b, strict=True)]
    np.testing.assert_allclose(manifold_flow.E, error, rtol=1e-15, atol=0)


class _DriftOnly(StochasticFunction):
    """``dy = y cos t dt + 0 dW``: an SDE that is the ODE ``y' = y cos t``."""

    def eval(self, point, time=None):
        return [point[0] * math.cos(time)]

    def diffusion(self, point, time=None):  # noqa: ARG002
        return [0.0]


@pytest.mark.parametrize(
    ("solver", "config", "order"),
    [
        (SRK2Solver, SRK2Config, 1),
        (SRK3Solver, SRK3Config, 2),
        (SRK4Solver, SRK4Config, 2),
    ],
)
def test_stochastic_runge_kutta_drift_has_its_deterministic_order(
    solver, config, order
):
    """With no noise each stochastic Runge-Kutta step is a deterministic tableau.

    The ``srk*`` steps are written as formulas, not stored tableaux, so their
    drift part is checked by its measured order on a field that depends on
    ``t``: SRK2 reduces to Euler, SRK3 and SRK4 to Heun's method with the
    predictor at ``t + h`` (``c = (0, 1)``, the row sum). A node at the wrong
    time would cost an order, as it did for Dormand-Prince (DEQ-137).
    """
    t, y, r = Variable(name="t"), Variable(name="y"), Variable(name="r")
    errors = []
    steps = [0.1, 0.05, 0.025, 0.0125]
    for h in steps:
        equation = DifferentialEquation(
            variables=[y], time=t, parameters=[], derivative=_DriftOnly([y], [], [r], t)
        )
        curve = solver(
            config(start_time=0.0, end_time=2.0, step_size=h, random_seed=1),
        ).solve(equation, [1.0])
        assert abs(curve.time.discretization[-1] - 2.0) < 1e-12
        errors.append(
            abs(curve.results[0].discretization[-1] - math.exp(math.sin(2.0)))
        )
    observed = [math.log2(errors[i] / errors[i + 1]) for i in range(len(steps) - 1)]
    assert all(abs(p - order) < 0.1 for p in observed), observed
