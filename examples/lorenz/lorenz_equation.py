#!/usr/bin/env python3
"""
Example: Solving the Lorenz Equations

This example demonstrates solving the Lorenz system of differential equations,
a classic example of chaotic behavior in dynamical systems.

The Lorenz equations are:
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz

With classical parameters: σ = 10, ρ = 28, β = 8/3
"""

import os

import plotly.graph_objects as go

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.plot.line_plot import LinePlot
from discrecontinual_equations.plot.phase_plot import PhasePlot
from discrecontinual_equations.solver.deterministic.adams_bashforth2.adams_bashforth2_config import (
    AdamsBashforth2Config,
)
from discrecontinual_equations.solver.deterministic.adams_bashforth2.adams_bashforth2_solver import (
    AdamsBashforth2Solver,
)
from discrecontinual_equations.solver.deterministic.adams_bashforth3.adams_bashforth3_config import (
    AdamsBashforth3Config,
)
from discrecontinual_equations.solver.deterministic.adams_bashforth3.adams_bashforth3_solver import (
    AdamsBashforth3Solver,
)
from discrecontinual_equations.solver.deterministic.cash_karp.cash_karp_config import (
    CashKarpConfig,
)
from discrecontinual_equations.solver.deterministic.cash_karp.cash_karp_solver import (
    CashKarpSolver,
)
from discrecontinual_equations.solver.deterministic.dormand_prince.dormand_prince_config import (
    DormandPrinceConfig,
)
from discrecontinual_equations.solver.deterministic.dormand_prince.dormand_prince_solver import (
    DormandPrinceSolver,
)
from discrecontinual_equations.solver.deterministic.dormand_prince89.dormand_prince89_config import (
    DormandPrince89Config,
)
from discrecontinual_equations.solver.deterministic.dormand_prince89.dormand_prince89_solver import (
    DormandPrince89Solver,
)
from discrecontinual_equations.solver.deterministic.euler.euler_config import (
    EulerConfig,
)
from discrecontinual_equations.solver.deterministic.euler.euler_solver import (
    EulerSolver,
)
from discrecontinual_equations.solver.deterministic.fehlberg.fehlberg_config import (
    FehlbergConfig,
)
from discrecontinual_equations.solver.deterministic.fehlberg.fehlberg_solver import (
    FehlbergSolver,
)
from discrecontinual_equations.variable import Variable


class X(Variable, name="X", abbreviation="x"):
    pass


class Y(Variable, name="Y", abbreviation="y"):
    pass


class Z(Variable, name="Z", abbreviation="z"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


class PrandtlNumber(Parameter, name="Prandtl Number", abbreviation="Pr"):
    """Corresponds to σ in the standard Lorenz equations."""


class RayleighNumber(Parameter, name="Rayleigh Number", abbreviation="Ra"):
    """Corresponds to ρ in the standard Lorenz equations."""


class AspectRatio(Parameter, name="Aspect Ratio", abbreviation="b"):
    """Corresponds to β in the standard Lorenz equations."""


class LorenzFunction(DeterministicFunction):
    """The Lorenz system function."""

    __slots__ = ["beta", "rho", "sigma"]

    def __init__(
        self,
        variables: list[Variable],
        parameters: list[Parameter],
        time: Variable | None = None,
    ):
        # For Lorenz, results are dx/dt, dy/dt, dz/dt
        results = [
            Variable(name="dX/dt", discretization=[]),
            Variable(name="dY/dt", discretization=[]),
            Variable(name="dZ/dt", discretization=[]),
        ]
        super().__init__(variables, parameters, results, time)
        self.sigma = parameters[0].value
        self.rho = parameters[1].value
        self.beta = parameters[2].value

    def eval(
        self, point: list[float], time: float | None = None,
    ) -> list[float]:
        x, y, z = point
        # Lorenz equations
        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z
        return [dx_dt, dy_dt, dz_dt]


def main():
    """Run the Lorenz equation example with multiple solvers."""
    # Parameters for Lorenz (classical values)
    prandtl_number = 10.0  # σ = 10
    rayleigh_number = 28.0  # ρ = 28
    aspect_ratio = 8.0 / 3.0  # β = 8/3

    print("Solving the Lorenz Equations with Multiple Solvers")
    print(f"Parameters: Prandtl={prandtl_number}, Rayleigh={rayleigh_number}, Aspect={aspect_ratio:.3f}")
    print("Initial conditions: x=1, y=1, z=1")
    print(f"Total simulation time: {150.0} units")  # end_time is 150.0
    print()

    parameters = [
        PrandtlNumber(value=prandtl_number),
        RayleighNumber(value=rayleigh_number),
        AspectRatio(value=aspect_ratio),
    ]

    # Initial conditions
    initial_values = [1.0, 1.0, 1.0]  # x=1, y=1, z=1

    # Define solvers to test
    solvers = [
        ("Euler", EulerConfig, EulerSolver),
        ("Adams-Bashforth 2", AdamsBashforth2Config, AdamsBashforth2Solver),
        ("Adams-Bashforth 3", AdamsBashforth3Config, AdamsBashforth3Solver),
        ("Cash-Karp", CashKarpConfig, CashKarpSolver),
        ("Fehlberg", FehlbergConfig, FehlbergSolver),
        ("Dormand-Prince 5(4)", DormandPrinceConfig, DormandPrinceSolver),
        ("Dormand-Prince 8(9)", DormandPrince89Config, DormandPrince89Solver),
    ]

    # Store results from all solvers
    results = {}

    for solver_name, config_class, solver_class in solvers:
        print(f"Solving with {solver_name} method...")

        # Create fresh variables for each solver to avoid sharing
        x_var = X()
        y_var = Y()
        z_var = Z()
        time_var = Time()
        variables = [x_var, y_var, z_var]

        # Create the Lorenz function
        lorenz = LorenzFunction(variables=variables, parameters=parameters, time=time_var)

        # Create the differential equation
        equation = DifferentialEquation(
            variables=variables,
            time=time_var,
            parameters=parameters,
            derivative=lorenz,
        )

        if solver_name in ["Cash-Karp", "Fehlberg", "Dormand-Prince 5(4)", "Dormand-Prince 8(9)"]:
            # Adaptive solvers
            config = config_class(
                start_time=0.0,
                end_time=150.0,
                initial_step_size=0.001,
                absolute_tolerance=1e-6,
                relative_tolerance=1e-6,
            )
            print(f"    Adaptive solver with tolerances atol={config.absolute_tolerance}, rtol={config.relative_tolerance}")
        else:
            # Fixed-step solvers
            config = config_class(
                start_time=0.0,
                end_time=150.0,
                step_size=0.001,
            )
            print(f"    Config times length: {len(config.times)}")

        solver = solver_class(config)

        solver.solve(equation, initial_values)

        print(f"  Solution computed with {len(solver.solution.time.discretization)} time steps")

        # Extract the results
        x_values = solver.solution.results[0].discretization
        y_values = solver.solution.results[1].discretization
        z_values = solver.solution.results[2].discretization

        results[solver_name] = {
            "x": x_values,
            "y": y_values,
            "z": z_values,
        }

        print(f"  Final X: {x_values[-1]:.3f}")
        print(f"  Final Y: {y_values[-1]:.3f}")
        print(f"  Final Z: {z_values[-1]:.3f}")
        print()

    # Create a single 3D plot with all trajectories
    plot = LinePlot(output_dir=os.path.join(os.path.dirname(__file__), "images"))

    colors = ["blue", "red", "green", "purple", "orange", "brown", "pink"]
    for i, (solver_name, data) in enumerate(results.items()):
        plot.figure.add_trace(
            go.Scatter3d(
                x=data["x"],
                y=data["y"],
                z=data["z"],
                mode="lines",
                line=dict(width=2, color=colors[i]),
                name=solver_name,
            ),
        )

    plot.figure.update_layout(
        title="Lorenz Attractor Comparison (sigma=10, rho=28, beta=8/3)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
    )

    plot.figure.show()
    # Save the combined plot
    safe_title = "Lorenz_Attractor_Comparison_sigma10_rho28_beta83"
    script_dir = os.path.dirname(__file__)
    png_filename = os.path.join(script_dir, "images", f"{safe_title}.png")
    plot.figure.write_image(png_filename)
    print(f"Combined plot saved as {os.path.basename(png_filename)}")

    # Create 2D phase space plot (X vs Z projection)
    phase_plot = PhasePlot(output_dir=os.path.join(script_dir, "images"))
    # Use the first solver's results for the phase plot
    first_solver_data = list(results.values())[0]
    x_var = Variable(name="X", discretization=first_solver_data["x"])
    z_var = Variable(name="Z", discretization=first_solver_data["z"])

    phase_plot.plot(x_var, z_var, title="Lorenz Phase Space (X vs Z)")
    phase_plot.show()

    print("All solvers completed. The Lorenz system shows chaotic behavior with sensitive dependence on initial conditions.")
    print("Note: Different numerical methods will produce different trajectories due to the chaotic nature of the system.")


if __name__ == "__main__":
    main()
