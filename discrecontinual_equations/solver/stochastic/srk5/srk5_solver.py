from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.solver.stochastic.srk5.srk5_config import SRK5Config


class SRK5Solver(Solver):
    """Refused: no five-stage scheme of strong order 2.5 exists for this interface.

    This solver refuses to run. The previous version of this file claimed "strong
    order 2.5 and weak order 5.0" from five Runge-Kutta stages whose noise terms
    were weighted by ``sqrt(dt)`` in place of a Wiener increment, so it drew no
    random number and integrated the ordinary differential equation
    ``dX = (a + b / sqrt(h)) dt`` - a drift with a deterministic bias that grows,
    relative to the drift, as the step shrinks (DEQ-25). The other three schemes in
    this package were rebuilt from derivations in Kloeden and Platen and their
    orders measured; for this one there was nothing defensible to build.

    **Why not.** For a scalar SDE with multiplicative noise the strong order of a
    scheme is set by which iterated Ito integrals it carries with their exact joint
    law. Order 1.0 needs ``dW`` and ``I_(1,1) = (dW^2 - h) / 2`` (:class:`SRK2Solver`);
    order 1.5 adds ``I_(1,0)`` and ``I_(1,1,1)`` (:class:`SRK3Solver`); order 2.0 adds
    ``I_(1,1,0)``, ``I_(1,0,1)``, ``I_(0,1,1)`` and ``I_(1,1,1,1)``, of which the
    mixed ones have no closed-form sampler and are usually approximated by series
    (Kloeden-Platen Section 10.5 and Chapter 5), and order 2.5 adds a further layer.
    Strong order 2.0 and above are therefore written down only for additive noise
    (``b`` constant), where the mixed integrals drop out, and this library's
    interface cannot tell additive noise from multiplicative noise before the fact.
    Weak order 5.0 is not a scheme at all for a general SDE; the only route to it in
    Kloeden and Platen is Richardson extrapolation of expectations across step
    sizes (Section 15.3), which produces a number, not a path, and so does not fit
    a solver that returns a trajectory. Finally, a "stage count" of five says
    nothing about stochastic order - the deterministic order of a Runge-Kutta
    tableau does not survive the addition of noise, since the noise terms need
    their own order conditions (Burrage and Burrage, 1996; Roessler, 2010).

    **What would be needed.** Either a strong order 2.0 scheme for additive noise
    with ``(dW, I_(1,0), I_(1,0,0))`` drawn jointly and a check that ``b`` is
    constant, measured on an additive-noise SDE with an exact solution (the
    Ornstein-Uhlenbeck process); or an honest weak order 3.0 scheme for additive
    noise. Both are real derivations with their own convergence tests. Until one
    is done, ``solve`` raises, because a solver named ``srk5`` that silently ran at
    an unknown order would be the present defect in a harder-to-notice form.

    Use :class:`SRK3Solver` for the highest measured strong order (1.5, scalar
    noise) or :class:`SRK4Solver` for the highest measured weak order (2.0, scalar
    noise).

    References:
    - Kloeden, P. E. and Platen, E. "Numerical Solution of Stochastic Differential
      Equations", Springer, 1992, Chapters 10, 11, 14 and 15.
    - Burrage, K. and Burrage, P. M. "High strong order explicit Runge-Kutta methods
      for stochastic ordinary differential equations", Applied Numerical Mathematics
      22, 1996.
    - Roessler, A. "Runge-Kutta methods for the strong approximation of solutions of
      stochastic differential equations", SIAM Journal on Numerical Analysis 48, 2010.
    """

    def __init__(self, solver_config: SRK5Config):
        super().__init__(solver_config=solver_config)

    def solve(self, equation: DifferentialEquation, initial_values: list[float]):
        message = (
            "SRK5Solver has no implementation: a stochastic Runge-Kutta scheme of "
            "strong order 2.5 or weak order 5.0 for general multiplicative noise does "
            "not exist, and the previous version of this solver was a deterministic "
            "ODE step mislabelled as one (DEQ-25). Use SRK3Solver (strong order 1.5, "
            "scalar noise) or SRK4Solver (weak order 2.0, scalar noise)."
        )
        raise NotImplementedError(message)
