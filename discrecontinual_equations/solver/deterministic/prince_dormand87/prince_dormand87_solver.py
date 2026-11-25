"""The Prince-Dormand 8(7) embedded Runge-Kutta pair, with adaptive steps.

Prince and Dormand (1981) give a thirteen-stage 8(7) pair: an eighth-order
formula to advance with and an embedded seventh-order formula whose difference
from it estimates the local error. The step advances with the eighth-order row
``b8`` (local extrapolation, as RKSUITE does), so ``|y8 - y7|`` estimates the
error of the seventh-order solution and is pessimistic for the one returned.
The published coefficients are rational approximations, accurate to about
``1e-17``, to the irrational values of the pair; the order conditions hold to
that accuracy, not exactly.

Until DEQ-139 this solver was ``DormandPrince89Solver``, an "8(9)" pair
citing Dormand and Prince's "Runge-Kutta triples" (1987). There is no
ninth-order row: the embedded one, then called ``b9``, is the seventh-order
row, as the order conditions and RKSUITE ("the one of order 7") agree. The old
names remain as deprecated aliases. One eighth-order weight was also
mistyped: ``b8[7]`` read ``561292735/797845732`` where RKSUITE has
0.703510669403443, i.e. ``561292985/797845732``. The weights summed to
``1 - 3.1e-7``, so every step lost about ``3.1e-7 h f`` and the global error
stalled near ``1e-7`` whatever the step (``y' = -y^2`` to ``t = 2``: 7.1e-8 at
``h = 1/32``); it now falls as ``h^8`` to 4.4e-16 at ``h = 1/16``.

Steps are taken, measured in the RMS norm and sized by the PI controller of
:mod:`~discrecontinual_equations.solver.deterministic.embedded_runge_kutta`; a
step rejected at ``min_step_size`` raises ``StepSizeTooSmallError`` (DEQ-140).

References:
    Prince, P. J. and Dormand, J. R. (1981). High order embedded Runge-Kutta
    formulae. Journal of Computational and Applied Mathematics 7(1):67-75.
    DOI: 10.1016/0771-050X(81)90010-3. The thirteen-stage tableau (not read
    directly: every coefficient checked against RKSUITE to 1e-16 and the order
    conditions solved in exact arithmetic). The code's earlier citation,
    Dormand and Prince's "Runge-Kutta triples" (1987), was not checked;
    RKSUITE attributes this tableau to the 1981 paper.

    Brankin, R. W., Gladwell, I. and Shampine, L. F. (1991). RKSUITE, Release
    1.0, Netlib ``ode/rksuite``. Subroutine ``CONST``, ``METHD = 3``: the same
    pair to 30 digits, with ``BHAT`` "the formula used to advance the
    integration, here the one of order 8" and ``B`` "the one of order 7".

    See ``docs/references.md`` (PrinceDormand1981,
    BrankinGladwellShampine1991, HairerWanner1996).
"""

import numpy as np

from discrecontinual_equations.solver.deterministic.butcher_tableau import (
    ButcherTableau,
    row,
)
from discrecontinual_equations.solver.deterministic.embedded_runge_kutta import (
    EmbeddedRungeKuttaSolver,
)
from discrecontinual_equations.solver.deterministic.prince_dormand87.prince_dormand87_config import (
    PrinceDormand87Config,
)

# Prince and Dormand's thirteen-stage 8(7) pair, as RKSUITE carries it: b is the
# eighth-order row the step advances with, b_hat the seventh-order one.
PRINCE_DORMAND_87 = ButcherTableau(
    name="Prince-Dormand 8(7)",
    a=(
        row(""),
        row("1/18"),
        row("1/48 1/16"),
        row("1/32 0 3/32"),
        row("5/16 0 -75/64 75/64"),
        row("3/80 0 0 3/16 3/20"),
        row(
            "29443841/614563906 0 0 77736538/692538347 -28693883/1125000000"
            " 23124283/1800000000",
        ),
        row(
            "16016141/946692911 0 0 61564180/158732637 22789713/633445777"
            " 545815736/2771057229 -180193667/1043307555",
        ),
        row(
            "39632708/573591083 0 0 -433636366/683701615 -421739975/2616292301"
            " 100302831/723423059 790204164/839813087 800635310/3783071287",
        ),
        row(
            "246121993/1340847787 0 0 -37695042795/15268766246"
            " -309121744/1061227803 -12992083/490766935 6005943493/2108947869"
            " 393006217/1396673457 123872331/1001029789",
        ),
        row(
            "-1028468189/846180014 0 0 8478235783/508512852"
            " 1311729495/1432422823 -10304129995/1701304382"
            " -48777925059/3047939560 15336726248/1032824649"
            " -45442868181/3398467696 3065993473/597172653",
        ),
        row(
            "185892177/718116043 0 0 -3185094517/667107341"
            " -477755414/1098053517 -703635378/230739211 5731566787/1027545527"
            " 5232866602/850066563 -4093664535/808688257 3962137247/1805957418"
            " 65686358/487910083",
        ),
        row(
            "403863854/491063109 0 0 -5068492393/434740067"
            " -411421997/543043805 652783627/914296604 11173962825/925320556"
            " -13158990841/6184727034 3936647629/1978049680"
            " -160528059/685178525 248638103/1413531060 0",
        ),
    ),
    c=row(
        "0 1/18 1/12 1/8 5/16 3/8 59/400 93/200 5490023248/9719169821 13/20"
        " 1201146811/1299019798 1 1",
    ),
    b=row(
        "14005451/335480064 0 0 0 0 -59238493/1068277825 181606767/758867731"
        " 561292985/797845732 -1041891430/1371343529 760417239/1151165299"
        " 118820643/751138087 -528747749/2220607170 1/4",
    ),
    b_hat=row(
        "13451932/455176623 0 0 0 0 -808719846/976000145 1757004468/5645159321"
        " 656045339/265891186 -3867574721/1518517206 465885868/322736535"
        " 53011238/667516719 2/45 0",
    ),
    order=8,
    embedded_order=7,
)


class PrinceDormand87Solver(EmbeddedRungeKuttaSolver):
    """Prince-Dormand 8(7): thirteen stages, eighth order to advance.

    ``a`` and ``c`` hold the tableau, ``b8`` the eighth-order weights that
    advance the solution and ``b7`` the seventh-order weights of the error
    estimate. Thirteen field evaluations a step buy a high order: on smooth
    problems at tight tolerances it takes far fewer steps than the 5(4) pairs.

    References:
        Prince and Dormand (1981), the 8(7) pair; Brankin, Gladwell and Shampine
        (1991), RKSUITE. See ``docs/references.md`` (PrinceDormand1981,
        BrankinGladwellShampine1991).
    """

    tableau = PRINCE_DORMAND_87

    def __init__(self, solver_config: PrinceDormand87Config) -> None:
        """Store the configuration and build the tableau.

        Args:
            solver_config: The interval, first step, tolerances, step bounds,
                ``safety_factor`` and ``beta``.
        """
        super().__init__(solver_config=solver_config)

    @property
    def b8(self) -> np.ndarray:
        """The eighth-order weights the step advances with: ``b``."""
        return self.b

    @property
    def b7(self) -> np.ndarray:
        """The embedded seventh-order weights: ``b_hat``."""
        return self.b_hat
