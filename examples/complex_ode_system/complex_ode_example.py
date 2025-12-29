"""
Complex ODE System Example

This example demonstrates solving a complex system of ordinary differential equations
using the Euler method. The system involves coupled equations with absolute values
and fractional powers.

Equation:
dx_i/dt = -(x_i * x_{i+1}) / |x_i - x_{i+1}|^(1/n) + |c(x_i + λ x_{i+1} + (-1)^i ε)|^(1/m)

For i = 1 to k, where k is odd and ≥ 3.

Parameters used:
- k = 5 (system size)
- n = 2, m = 3 (power parameters)
- λ = -1.0, c = 3.0, ε = -0.03
"""

import os

import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.function import Function
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.plot.delay_plot import DelayPlot
from discrecontinual_equations.plot.line_plot import LinePlot
from discrecontinual_equations.solver.deterministic.euler.euler_config import (
    EulerConfig,
)
from discrecontinual_equations.solver.deterministic.euler.euler_solver import (
    EulerSolver,
)
from discrecontinual_equations.variable import Variable


class ComplexODESystem(Function):
    """
    Complex system of ODEs as specified by the user.

    dx_i/dt = -(x_i * x_{i+1}) / |x_i - x_{i+1}|^(1/n) + |c(x_i + λ x_{i+1} + (-1)^i ε)|^(1/m)

    For i = 1 to k, where k is odd and ≥ 3.
    """

    def __init__(self, variables, parameters, time=None, k=5, n=2, m=3):
        # Create result variables for the ODE system
        results = [
            Variable(
                name=f"d{variable.name}/dt",
                abbreviation=f"d{variable.abbreviation}/dt",
            )
            for variable in variables
        ]
        super().__init__(
            variables=variables,
            parameters=parameters,
            results=results,
            time=time,
        )

        self.k = k  # System size (must be odd, >= 3)
        self.n = n  # Power parameter for denominator
        self.m = m  # Power parameter for absolute value term

        # Extract parameters
        self.c = parameters[0].value  # c parameter
        self.lam = parameters[1].value  # λ parameter
        self.eps = parameters[2].value  # ε parameter

        print("Complex ODE System initialized:")
        print(f"  System size k = {k}")
        print(f"  Parameters: c = {self.c}, λ = {self.lam}, ε = {self.eps}")
        print(f"  Powers: n = {n}, m = {m}")

    def eval(self, point, time=None):
        """
        Evaluate the right-hand side of the ODE system.

        Args:
            point: Current state vector [x1, x2, ..., xk]
            time: Current time (not used in this autonomous system)

        Returns:
            List of derivatives [dx1/dt, dx2/dt, ..., dxk/dt]
        """
        x = np.array(point)  # Convert to numpy array for easier manipulation

        derivatives = []

        # print(x)
        for i in range(self.k):
            # 0-based indexing in code, but corresponds to i+1 in mathematical notation
            x_i = x[i]  # x_{i+1} in 1-based notation
            x_ip1 = x[(i + 1) % self.k]  # x_{i+2} in 1-based notation, wraps around

            # print(x_i, x_ip1)
            # First term: -(x_i * x_{i+1}) / |x_i - x_{i+1}|^(1/n)
            # In 1-based notation: -(x_{i+1} * x_{i+2}) / |x_{i+1} - x_{i+2}|^(1/n)
            numerator1 = -(x_i * x_ip1)
            denominator1 = abs(x_i - x_ip1) ** (1.0 / self.n)
            term1 = numerator1 / denominator1 if denominator1 != 0 else 0.0

            # Second term: |c(x_i + λ x_{i+1} + (-1)^i ε)|^(1/m)
            # In 1-based notation: |c(x_{i+1} + λ x_{i+2} + (-1)^{i+1} ε)|^(1/m)
            sign_factor = (-1) ** i
            inside_abs = self.c * (x_i + self.lam * x_ip1 + sign_factor * self.eps)
            term2 = abs(inside_abs) ** (1.0 / self.m)

            # Total derivative for variable i (0-based) = dx_{i+1}/dt (1-based)
            dx_dt = term1 + term2
            derivatives.append(dx_dt)

        return derivatives


def run_complex_ode_example():
    """
    Solve the complex ODE system and visualize the results.
    """
    print("Complex ODE System Simulation")
    print("=" * 50)

    # System parameters
    k = 5  # System size (odd, >= 3)
    n = 3  # Power for denominator
    m = 5  # Power for absolute value term

    # Create variables for the system
    variables = []
    for i in range(1, k + 1):
        var = Variable(name=f"x{i}", abbreviation=f"x{i}")
        variables.append(var)

    time = Variable(name="Time", abbreviation="t")

    # Parameters: c, λ, ε
    c_param = Parameter(name="c", value=3.0)
    lambda_param = Parameter(name="lambda", value=-1.0)
    epsilon_param = Parameter(name="epsilon", value=-0.03)

    # Create the ODE system
    ode_system = ComplexODESystem(
        variables=variables,
        parameters=[c_param, lambda_param, epsilon_param],
        time=time,
        k=k,
        n=n,
        m=m,
    )

    # Create differential equation
    equation = DifferentialEquation(
        variables=variables,
        time=time,
        parameters=[c_param, lambda_param, epsilon_param],
        derivative=ode_system,
    )

    # Simulation parameters
    end_time = 2000.0  # Simulation time
    n_steps = 10000  # Number of steps
    step_size = end_time / n_steps

    print("Simulation parameters:")
    print(f"  End time: {end_time}")
    print(f"  Steps: {n_steps}")
    print(f"  Step size: {step_size:.4f}")
    print()

    # Euler solver configuration
    config = EulerConfig(
        start_time=0.0,
        end_time=end_time,
        step_size=step_size,
    )

    # Create solver
    solver = EulerSolver(config)

    # Initial conditions - small random perturbations around 1.0
    # np.random.seed(42)
    initial_conditions = 0.0 + 0.1 * np.random.randn(k)
    print(f"Initial conditions: {initial_conditions}")

    # Solve the system
    print("Solving ODE system...")
    solver.solve(equation, initial_conditions.tolist())

    # Extract solution data
    time_points = [point[0] for point in solver.solution]
    solutions = []

    for var_idx in range(k):
        var_solution = [point[2][var_idx] for point in solver.solution]
        solutions.append(var_solution)

    # Slice data to show only steps 4000 to 6000
    start_step = 0
    end_step = n_steps
    if len(time_points) > end_step:
        time_points_plot = time_points[start_step:end_step]
        solutions_plot = [sol[start_step:end_step] for sol in solutions]
        print(f"Plotting data from steps {start_step} to {end_step}")
        print(".1f")
    else:
        # If we don't have enough data, use all available
        time_points_plot = time_points
        solutions_plot = solutions
        print("Warning: Not enough steps available, using all data")
        print(f"Available steps: {len(time_points)}")

    print("Solution computed successfully!")
    print(f"Final time: {time_points[-1]:.2f}")
    final_values = [sol[-1] for sol in solutions]
    print(f"Final values: {final_values}")

    # Create plots
    plot = LinePlot(
        output_dir=os.path.join(os.path.dirname(__file__), "images"),
        output_format="png",
    )

    # Plot all variables
    plot.figure.update_layout(
        title=f"Complex ODE System (k={k}, n={n}, m={m})",
        xaxis_title="Time (t)",
        yaxis_title="Variable Values",
        hovermode="x unified",
        showlegend=True,
    )

    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
    ]
    for i in range(k):
        plot.figure.add_trace(
            dict(
                x=time_points_plot,
                y=solutions_plot[i],
                mode="lines",
                name=f"x{i + 1}",
                line=dict(color=colors[i % len(colors)], width=2),
                showlegend=True,
            ),
        )

    # Add parameter information
    param_info = f"""
    Complex ODE System Parameters:<br>
    • System size: k = {k}<br>
    • Powers: n = {n}, m = {m}<br>
    • Coefficients: c = {c_param.value}, λ = {lambda_param.value}, ε = {epsilon_param.value}<br>
    • Solver: Euler method<br>
    • Initial conditions: perturbed around 1.0
    """

    plot.figure.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=param_info,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        align="left",
    )

    # Save the plot
    script_dir = os.path.dirname(__file__)
    png_path = os.path.join(script_dir, "images", "complex_ode_system.png")
    html_path = os.path.join(script_dir, "images", "complex_ode_system.html")
    plot.figure.write_image(png_path)
    plot.figure.write_html(html_path)

    print("Plots saved as:")
    print("- complex_ode_system.png")
    print("- complex_ode_system.html")

    # Create phase plot for first two variables
    if k >= 2:
        phase_plot = LinePlot(
            output_dir=os.path.join(os.path.dirname(__file__), "images"),
            output_format="png",
        )

        phase_plot.figure.update_layout(
            title="Phase Plot: x₁ vs x₂",
            xaxis_title="x₁",
            yaxis_title="x₂",
            hovermode="closest",
        )

        phase_plot.figure.add_trace(
            dict(
                x=solutions_plot[0],
                y=solutions_plot[1],
                mode="markers",
                name="Phase trajectory",
                marker=dict(color="blue", size=2, opacity=0.6),
                showlegend=False,
            ),
        )

        # Mark start and end points of the plotted segment
        phase_plot.figure.add_trace(
            dict(
                x=[solutions_plot[0][0]],
                y=[solutions_plot[1][0]],
                mode="markers",
                marker=dict(color="green", size=8, symbol="circle"),
                name="Start of segment",
                showlegend=True,
            ),
        )

        phase_plot.figure.add_trace(
            dict(
                x=[solutions_plot[0][-1]],
                y=[solutions_plot[1][-1]],
                mode="markers",
                marker=dict(color="red", size=8, symbol="x"),
                name="End of segment",
                showlegend=True,
            ),
        )

        phase_png_path = os.path.join(script_dir, "images", "complex_ode_phase.png")
        phase_html_path = os.path.join(script_dir, "images", "complex_ode_phase.html")
        phase_plot.figure.write_image(phase_png_path)
        phase_plot.figure.write_html(phase_html_path)

        print("Phase plot saved as:")
        print("- complex_ode_phase.png")
        print("- complex_ode_phase.html")

    # Create delay plots for all variables with delay=1 using sliced data
    delay_plot_dir = os.path.join(os.path.dirname(__file__), "images")

    # Create delay plots for all variables
    for i in range(k):
        var_name = f"x{i + 1}"
        var_data = solutions_plot[i]

        # Create variable object
        var_obj = Variable(name=var_name, abbreviation=var_name)
        var_obj.discretization = var_data

        # Create delay plot
        delay_plot = DelayPlot(output_dir=delay_plot_dir, output_format="png")
        delay_plot.plot(
            var_obj,
            delay=1,
            title=f"Delay Plot: {var_name}(t) vs {var_name}(t+1) [Steps {start_step}-{end_step}]",
        )
        delay_plot.show()

    # Show the plots
    # # plot.figure.show()
    # if k >= 2:
    #     phase_plot.figure.show()

    return solutions, time_points


def analyze_system_properties(solutions, time_points, k):
    """
    Analyze some properties of the system solution.
    """
    print("\nSystem Analysis")
    print("=" * 20)

    # Check for periodicity or chaos
    final_values = [sol[-1] for sol in solutions]
    initial_values = [sol[0] for sol in solutions]

    print(f"Initial values: {initial_values}")
    print(f"Final values: {final_values}")

    # Check if system reached steady state
    tolerance = 1e-6
    derivatives_at_end = []

    # Approximate derivatives at the end
    dt = time_points[1] - time_points[0]
    for i in range(k):
        # Simple finite difference for last few points
        if len(solutions[i]) >= 3:
            deriv_approx = (solutions[i][-1] - solutions[i][-2]) / dt
            derivatives_at_end.append(deriv_approx)

    print("Approximate derivatives at end:")
    for i, deriv in enumerate(derivatives_at_end):
        print(f"dx{i + 1}/dt = {deriv:.6f}")

    # Check for steady state
    steady_state_threshold = 1e-4
    near_steady = all(abs(d) < steady_state_threshold for d in derivatives_at_end)
    print(f"\nSystem near steady state: {near_steady}")

    return derivatives_at_end


def main():
    """
    Run the complex ODE system example.
    """
    print("Complex ODE System Example")
    print("=" * 60)
    print("This example demonstrates solving a complex system of coupled ODEs")
    print("using the Euler method. The system includes absolute values and")
    print("fractional powers, making it challenging for numerical solvers.")
    print()

    # Run the simulation
    solutions, time_points = run_complex_ode_example()

    # Analyze results
    k = len(solutions)
    analyze_system_properties(solutions, time_points, k)

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("Check the generated PNG and HTML files for visualizations.")
    print("\nSystem characteristics:")
    print("• Coupled nonlinear ODEs with circular boundary conditions")
    print("• Involves absolute values and fractional powers")
    print("• Parameters create complex interaction dynamics")
    print("• Euler method provides stable numerical integration")


if __name__ == "__main__":
    main()
