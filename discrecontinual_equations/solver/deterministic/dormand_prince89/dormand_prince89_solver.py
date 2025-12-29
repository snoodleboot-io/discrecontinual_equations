import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.deterministic.dormand_prince89.dormand_prince89_config import (
    DormandPrince89Config,
)
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.variable import Variable


class DormandPrince89Solver(Solver):
    """
    Dormand-Prince Runge-Kutta method (8,9) for solving ODEs.

    An 8th-order Runge-Kutta method with embedded 9th-order solution for error estimation.
    Uses 13 stages and provides adaptive step size control through local error estimation.

    This is a very high-order method suitable for problems requiring high accuracy.
    The method provides both 8th-order (y8) and 9th-order (y9) solutions, where the
    difference |y9 - y8| serves as an estimate of the local truncation error.

    References:
    - Dormand, John R. and Prince, Peter J. "Runge-Kutta triples"
      Computational Mathematics with Applications, 1987
    """

    def __init__(self, solver_config: DormandPrince89Config):
        super().__init__(solver_config=solver_config)

        # Dormand-Prince (8,9) coefficients - 13 stages
        # This is a very large coefficient matrix
        self.a = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1 / 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1 / 48, 1 / 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1 / 32, 0, 3 / 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5 / 16, 0, -75 / 64, 75 / 64, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [3 / 80, 0, 0, 3 / 16, 3 / 20, 0, 0, 0, 0, 0, 0, 0, 0],
                [
                    29443841 / 614563906,
                    0,
                    0,
                    77736538 / 692538347,
                    -28693883 / 1125000000,
                    23124283 / 1800000000,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    16016141 / 946692911,
                    0,
                    0,
                    61564180 / 158732637,
                    22789713 / 633445777,
                    545815736 / 2771057229,
                    -180193667 / 1043307555,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    39632708 / 573591083,
                    0,
                    0,
                    -433636366 / 683701615,
                    -421739975 / 2616292301,
                    100302831 / 723423059,
                    790204164 / 839813087,
                    800635310 / 3783071287,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    246121993 / 1340847787,
                    0,
                    0,
                    -37695042795 / 15268766246,
                    -309121744 / 1061227803,
                    -12992083 / 490766935,
                    6005943493 / 2108947869,
                    393006217 / 1396673457,
                    123872331 / 1001029789,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    -1028468189 / 846180014,
                    0,
                    0,
                    8478235783 / 508512852,
                    1311729495 / 1432422823,
                    -10304129995 / 1701304382,
                    -48777925059 / 3047939560,
                    15336726248 / 1032824649,
                    -45442868181 / 3398467696,
                    3065993473 / 597172653,
                    0,
                    0,
                    0,
                ],
                [
                    185892177 / 718116043,
                    0,
                    0,
                    -3185094517 / 667107341,
                    -477755414 / 1098053517,
                    -703635378 / 230739211,
                    5731566787 / 1027545527,
                    5232866602 / 850066563,
                    -4093664535 / 808688257,
                    3962137247 / 1805957418,
                    65686358 / 487910083,
                    0,
                    0,
                ],
                [
                    403863854 / 491063109,
                    0,
                    0,
                    -5068492393 / 434740067,
                    -411421997 / 543043805,
                    652783627 / 914296604,
                    11173962825 / 925320556,
                    -13158990841 / 6184727034,
                    3936647629 / 1978049680,
                    -160528059 / 685178525,
                    248638103 / 1413531060,
                    0,
                    0,
                ],
            ],
        )

        # 8th order solution coefficients
        self.b8 = np.array(
            [
                14005451 / 335480064,
                0,
                0,
                0,
                0,
                -59238493 / 1068277825,
                181606767 / 758867731,
                561292735 / 797845732,
                -1041891430 / 1371343529,
                760417239 / 1151165299,
                118820643 / 751138087,
                -528747749 / 2220607170,
                1 / 4,
            ],
        )

        # 9th order solution coefficients (for error estimation)
        self.b9 = np.array(
            [
                13451932 / 455176623,
                0,
                0,
                0,
                0,
                -808719846 / 976000145,
                1757004468 / 5645159321,
                656045339 / 265891186,
                -3867574721 / 1518517206,
                465885868 / 322736535,
                53011238 / 667516719,
                2 / 45,
                0,
            ],
        )

        self.c = np.array(
            [
                0,
                1 / 18,
                1 / 12,
                1 / 8,
                5 / 16,
                3 / 8,
                59 / 400,
                93 / 200,
                5490023248 / 9719169821,
                13 / 20,
                1201146811 / 1299019798,
                1,
                1,
            ],
        )

    def solve(self, equation: DifferentialEquation, initial_values: list[float]):
        results = [
            Variable(name=f"Integral of {variable.name}")
            for variable in equation.derivative.variables
        ]
        self.solution = Curve(
            time=equation.derivative.time,
            variables=equation.derivative.variables,
            results=results,
        )

        # Initialize
        t = self.solver_config.start_time
        y = np.array(initial_values)
        h = self.solver_config.initial_step_size

        # Append initial point
        self.solution.append([t, [0] * len(initial_values), initial_values])

        # Adaptive integration loop
        while t < self.solver_config.end_time:
            if t + h > self.solver_config.end_time:
                h = self.solver_config.end_time - t

            # Perform one step
            y_new, error, h_new = self._step(equation, t, y, h)

            # Check if step is accepted
            if self._accept_step(error, y, y_new, h):
                # Accept step
                t += h
                y = y_new
                self.solution.append([t, [0] * len(initial_values), y.tolist()])
                h = h_new
            else:
                # Reject step, try smaller step
                h = max(h_new, self.solver_config.min_step_size)
                if h < self.solver_config.min_step_size:
                    raise RuntimeError("Step size became too small")

    def _step(self, equation: DifferentialEquation, t: float, y: np.ndarray, h: float):
        """Perform one Dormand-Prince (8,9) step."""
        k = np.zeros((13, len(y)))

        # Compute stages
        for i in range(13):
            t_stage = t + self.c[i] * h
            y_stage = y + h * np.dot(self.a[i, : i + 1], k[: i + 1])
            k[i] = np.array(
                equation.derivative.eval(point=y_stage.tolist(), time=t_stage),
            )

        # Compute solutions
        y8 = y + h * np.dot(self.b8, k)  # 8th order solution
        y9 = y + h * np.dot(self.b9, k)  # 9th order solution

        # Error estimate
        error = np.abs(y9 - y8)

        # Step size control
        h_new = self._compute_new_step_size(error, y, h)

        return y8, error, h_new

    def _accept_step(
        self,
        error: np.ndarray,
        y: np.ndarray,
        y_new: np.ndarray,
        h: float,
    ) -> bool:
        """Check if the step should be accepted based on error."""
        # Compute scaled error
        tol = (
            self.solver_config.absolute_tolerance
            + self.solver_config.relative_tolerance
            * np.maximum(np.abs(y), np.abs(y_new))
        )
        scaled_error = np.max(error / tol)

        return scaled_error <= 1.0

    def _compute_new_step_size(
        self,
        error: np.ndarray,
        y: np.ndarray,
        h: float,
    ) -> float:
        """Compute new step size using PI control."""
        tol = (
            self.solver_config.absolute_tolerance
            + self.solver_config.relative_tolerance * np.abs(y)
        )
        scaled_error = np.max(error / tol)

        if scaled_error == 0:
            return self.solver_config.max_step_size

        # PI controller
        h_new = (
            h
            * self.solver_config.safety_factor
            * (1 / scaled_error) ** (1 / 8)
            * (1 / scaled_error) ** (self.solver_config.beta / 8)
        )

        # Limit step size
        h_new = np.clip(
            h_new,
            self.solver_config.min_step_size,
            self.solver_config.max_step_size,
        )

        return h_new
