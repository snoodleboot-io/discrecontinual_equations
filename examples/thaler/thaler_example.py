"""
Thaler Method for alpha-Stable Stochastic Differential Equations

This example demonstrates how to use the Thaler solver to simulate
stochastic differential equations driven by alpha-stable Levy processes.

The Thaler method uses deterministic homogenisation to approximate Marcus SDEs
with non-Lipschitz coefficients and alpha-stable noise for alpha in (1,2).

Two examples are provided:
1. Pure alpha-stable diffusion: dZ = sigma * dW^alpha,eta,beta
2. SDE with natural boundary: dZ = -Z^3 dt + Z^2 * dW^alpha,eta,beta
"""

import os

import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.plot.line_plot import LinePlot
from discrecontinual_equations.solver.stochastic.gottwald_melbourne.gottwald_melbourne_config import (
    GottwaldMelbourneConfig,
)
from discrecontinual_equations.solver.stochastic.gottwald_melbourne.gottwald_melbourne_solver import (
    GottwaldMelbourneSolver,
)
from discrecontinual_equations.variable import Variable


class AlphaStableDiffusion(StochasticFunction):
    """
    Pure alpha-stable diffusion SDE: dZ = sigma * dW^alpha,eta,beta

    This represents pure alpha-stable noise with no drift term.
    The solution exhibits heavy tails and discontinuous sample paths.
    """

    def __init__(self, variables: list[Variable], parameters: list[Parameter], time: Variable | None = None):
        # Create result variables for the SDE
        results = [
            Variable(name=f"d{variable.name}/dt", abbreviation=f"d{variable.abbreviation}/dt")
            for variable in variables
        ]
        super().__init__(variables=variables, parameters=parameters, results=results, time=time)

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        """
        Drift term: μ(z,t) = 0 (pure diffusion)
        """
        return [0.0]  # No drift

    def diffusion(self, point: list[float], time: float | None = None) -> list[float]:
        """
        Diffusion term: σ(z,t) = σ (constant volatility)
        """
        sigma = self.parameters[0].value  # Diffusion coefficient
        return [sigma]


class BoundarySDE(StochasticFunction):
    """
    SDE with natural boundary: dZ = -Z^3 dt + Z^2  dW^α,η,β

    This SDE has a natural boundary at Z = 0. Solutions starting with Z > 0
    remain positive for all time. The Thaler method preserves this boundary
    better than traditional Euler-Maruyama discretizations.
    """

    def __init__(self, variables: list[Variable], parameters: list[Parameter], time: Variable | None = None):
        # Create result variables for the SDE
        results = [
            Variable(name=f"d{variable.name}/dt", abbreviation=f"d{variable.abbreviation}/dt")
            for variable in variables
        ]
        super().__init__(variables=variables, parameters=parameters, results=results, time=time)

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        """
        Drift term: μ(z,t) = -z^3 (attractive force toward origin)
        """
        z = point[0]
        return [-z**3]

    def diffusion(self, point: list[float], time: float | None = None) -> list[float]:
        """
        Diffusion term: σ(z,t) = z^2 (non-Lipschitz, vanishes at boundary)
        """
        z = point[0]
        return [z**2]


def run_alpha_stable_diffusion_example():
    """
    Simulate α-stable diffusion and demonstrate heavy-tailed behavior.
    """
    print("α-Stable Diffusion Simulation")
    print("=" * 40)

    # Define variables and parameters
    z = Variable(name="Position", abbreviation="Z")
    t = Variable(name="Time", abbreviation="t")
    diffusion_coeff = Parameter(name="Diffusion Coefficient", value=1.0)

    # Create the stochastic function
    alpha_stable_process = AlphaStableDiffusion(variables=[z], parameters=[diffusion_coeff], time=t)

    # Create differential equation
    equation = DifferentialEquation(
        variables=[z],
        time=t,
        parameters=[diffusion_coeff],
        derivative=alpha_stable_process,
    )

    # Simulation parameters
    n_simulations = 3  # Fewer simulations due to computational cost
    end_time = 1.0     # Shorter time due to heavy-tailed behavior
    n_steps = 100      # Fewer steps for faster computation
    step_size = end_time / n_steps

    # α-stable parameters
    alpha = 1.5  # Stability index (1 < α < 2 for finite variance but infinite mean)
    eta = 1.0    # Scale parameter
    beta = 0.0   # Skewness (symmetric)

    # Solver configuration
    config = GottwaldMelbourneConfig(
        alpha=alpha,
        eta=eta,
        beta=beta,
        epsilon=0.01,  # Homogenisation parameter
        start_time=0.0,
        end_time=end_time,
        step_size=step_size,
        random_seed=42,
    )

    print(f"Simulating α-stable diffusion with α={alpha}, η={eta}, β={beta}")
    print(f"Number of paths: {n_simulations}")
    print(".2f")
    print(f"Time steps: {n_steps}")
    print(f"Step size: {step_size:.4f}")
    print(f"Homogenisation ε: {config.epsilon}")
    print()

    # Create plotter
    plot = LinePlot(output_dir=os.path.join(os.path.dirname(__file__), "images"), output_format="png")

    # Simulate multiple paths
    all_paths = []

    for i in range(n_simulations):
        # Create solver with fresh random seed for each path
        config.random_seed = 42 + i
        solver = GottwaldMelbourneSolver(config)

        # Solve with initial condition Z_0 = 0
        solver.solve(equation, [0.0])

        # Extract the solution path
        path = [point[2][0] for point in solver.solution]  # Extract position values
        times = [point[0] for point in solver.solution]    # Extract time values

        all_paths.append((times, path))
        print(f"Path {i+1} final position: {path[-1]:.4f}")

    print("\nPlotting results...")

    # Create plot
    plot.figure.update_layout(
        title=f"α-Stable Diffusion (α={alpha}) - Multiple Paths",
        xaxis_title="Time (t)",
        yaxis_title="Position Z(t)",
        hovermode="x unified",
        showlegend=True,
    )

    # Plot each path
    colors = ["blue", "red", "green"]
    for i, (times, path) in enumerate(all_paths):
        plot.figure.add_trace(
            dict(
                x=times,
                y=path,
                mode="lines",
                name=f"Path {i+1}",
                line=dict(color=colors[i], width=2),
                showlegend=True,
            ),
        )

    # Add theoretical information
    theoretical_info = ".2f"".2f"f"""
α-Stable Process Properties:
• Stability index: α = {alpha}
• Scale parameter: η = {eta}
• Skewness: β = {beta}
• Heavy tails: P(|Z| > x) ~ x^(-α) as x → ∞
• Variance: {'finite' if alpha > 2 else 'infinite'}
• Mean: {'finite' if alpha > 1 else 'infinite'}"""

    plot.figure.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=theoretical_info,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        align="left",
    )

    # Save the plot
    script_dir = os.path.dirname(__file__)
    png_path = os.path.join(script_dir, "images", "alpha_stable_diffusion.png")
    html_path = os.path.join(script_dir, "images", "alpha_stable_diffusion.html")
    plot.figure.write_image(png_path)
    plot.figure.write_html(html_path)

    print("Plots saved as:")
    print("- alpha_stable_diffusion.png")
    print("- alpha_stable_diffusion.html")

    # Show the plot
    plot.figure.show()

    return all_paths


def run_boundary_preservation_example():
    """
    Demonstrate boundary preservation for SDEs with natural boundaries.
    """
    print("\nSDE with Natural Boundary")
    print("=" * 40)

    # Define variables and parameters
    z = Variable(name="Position", abbreviation="Z")
    t = Variable(name="Time", abbreviation="t")

    # Create the stochastic function
    boundary_process = BoundarySDE(variables=[z], parameters=[], time=t)

    # Create differential equation
    equation = DifferentialEquation(
        variables=[z],
        time=t,
        parameters=[],
        derivative=boundary_process,
    )

    # Simulation parameters
    end_time = 2.0
    n_steps = 200
    step_size = end_time / n_steps

    # α-stable parameters
    alpha = 1.5
    eta = 0.5
    beta = 0.0

    # Solver configuration
    config = GottwaldMelbourneConfig(
        alpha=alpha,
        eta=eta,
        beta=beta,
        epsilon=0.005,  # Smaller ε for better boundary preservation
        start_time=0.0,
        end_time=end_time,
        step_size=step_size,
        random_seed=123,
    )

    print(f"Solving SDE: dZ = -Z³ dt + Z²  dW^α,η,β")
    print(f"Natural boundary at Z = 0 (solutions remain positive)")
    print(".2f")
    print(f"Time steps: {n_steps}")
    print(f"Homogenisation ε: {config.epsilon}")
    print()

    # Create solver
    solver = GottwaldMelbourneSolver(config)

    # Solve with initial condition Z_0 = 0.1 (positive)
    solver.solve(equation, [0.1])

    # Extract the solution path
    path = [point[2][0] for point in solver.solution]
    times = [point[0] for point in solver.solution]

    print("Boundary preservation check:")
    negative_values = [val for val in path if val < 0]
    print(f"  Values below boundary: {len(negative_values)}")
    print(f"  Minimum value: {min(path):.6f}")
    print(f"  Final value: {path[-1]:.6f}")

    # Create plot
    plot = LinePlot(output_dir=os.path.join(os.path.dirname(__file__), "images"), output_format="png")

    plot.figure.update_layout(
        title="SDE with Natural Boundary - Boundary Preservation",
        xaxis_title="Time (t)",
        yaxis_title="Position Z(t)",
        hovermode="x unified",
    )

    plot.figure.add_trace(
        dict(
            x=times,
            y=path,
            mode="lines",
            name="Solution",
            line=dict(color="blue", width=2),
        ),
    )

    # Add boundary line
    plot.figure.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Natural Boundary Z=0",
        annotation_position="bottom right",
    )

    # Add information about boundary preservation
    boundary_info = f"""
Boundary Preservation:
• Natural boundary at Z = 0
• Initial condition: Z₀ = 0.1
• Minimum value: {min(path):.6f}
• Values < 0: {len(negative_values)}
• Gottwald-Melbourne method preserves boundaries"""

    plot.figure.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=boundary_info,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        align="left",
    )

    # Save the plot
    script_dir = os.path.dirname(__file__)
    png_path = os.path.join(script_dir, "images", "boundary_preservation.png")
    html_path = os.path.join(script_dir, "images", "boundary_preservation.html")
    plot.figure.write_image(png_path)
    plot.figure.write_html(html_path)

    print("Plots saved as:")
    print("- boundary_preservation.png")
    print("- boundary_preservation.html")

    # Show the plot
    plot.figure.show()

    return path


def main():
    """
    Run both Thaler method examples.
    """
    print("Thaler Method Examples for alpha-Stable SDEs")
    print("=" * 50)
    print("This example demonstrates the Thaler method for solving")
    print("stochastic differential equations with alpha-stable Levy noise.")
    print()

    # Run α-stable diffusion example
    diffusion_paths = run_alpha_stable_diffusion_example()

    # Run boundary preservation example
    boundary_path = run_boundary_preservation_example()

    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("Check the generated PNG and HTML files for visualizations.")
    print("\nKey advantages of the Thaler method:")
    print("• Handles non-Lipschitz coefficients")
    print("• Preserves natural boundaries")
    print("• Deterministic noise generation")
    print("• Marcus interpretation of SDEs")


if __name__ == "__main__":
    main()
