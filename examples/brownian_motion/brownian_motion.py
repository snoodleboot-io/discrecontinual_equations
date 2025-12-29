"""
Brownian Motion Example

This example demonstrates how to use the Euler-Maruyama solver to simulate
Brownian motion, a fundamental stochastic process in physics and finance.

Brownian motion is described by the stochastic differential equation:
dX_t = μ dt + σ dW_t

For standard Brownian motion:
- Drift term μ = 0 (no deterministic component)
- Diffusion term σ = 1 (unit volatility)
- Initial condition X_0 = 0

The Euler-Maruyama discretization gives:
X_{n+1} = X_n + μ Δt + σ ΔW_n
where ΔW_n ~ N(0, Δt)
"""

import os

import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.plot.line_plot import LinePlot
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config import (
    EulerMaruyamaConfig,
)
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_solver import (
    EulerMaruyamaSolver,
)
from discrecontinual_equations.variable import Variable


class BrownianMotion(StochasticFunction):
    """
    Standard Brownian motion SDE: dX_t = σ dW_t

    This represents pure random motion with no drift term.
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
        Drift term: μ(x,t) = 0 for standard Brownian motion
        """
        return [0.0]  # No drift

    def diffusion(self, point: list[float], time: float | None = None) -> list[float]:
        """
        Diffusion term: σ(x,t) = σ (constant volatility)
        """
        sigma = self.parameters[0].value  # Volatility parameter
        return [sigma]


def run_brownian_motion_example():
    """
    Simulate multiple paths of Brownian motion and visualize the results.
    """
    print("Brownian Motion Simulation")
    print("=" * 40)

    # Define variables and parameters
    x = Variable(name="Position", abbreviation="X")
    t = Variable(name="Time", abbreviation="t")
    diffusion_coefficient = Parameter(name="Diffusion Coefficient", value=1.0)  # Corresponds to σ in standard Brownian motion

    # Create the stochastic function
    brownian_process = BrownianMotion(variables=[x], parameters=[diffusion_coefficient], time=t)

    # Create differential equation
    equation = DifferentialEquation(
        variables=[x],
        time=t,
        parameters=[diffusion_coefficient],
        derivative=brownian_process,
    )

    # Simulation parameters
    n_simulations = 5  # Number of Brownian motion paths to simulate
    end_time = 2.0     # Total simulation time
    n_steps = 1000     # Number of time steps
    step_size = end_time / n_steps

    # Solver configuration
    config = EulerMaruyamaConfig(
        start_time=0.0,
        end_time=end_time,
        step_size=step_size,
        random_seed=42,  # For reproducible results
    )

    print(f"Simulating {n_simulations} Brownian motion paths...")
    print(".2f")
    print(f"Time steps: {n_steps}")
    print(f"Step size: {step_size:.4f}")
    print()

    # Create plotter
    plot = LinePlot(output_dir=os.path.join(os.path.dirname(__file__), "images"), output_format="png")

    # Simulate multiple paths
    all_paths = []

    for i in range(n_simulations):
        # Create solver with fresh random seed for each path
        config.random_seed = 42 + i  # Different seed for each path
        solver = EulerMaruyamaSolver(config)

        # Solve with initial condition X_0 = 0
        solver.solve(equation, [0.0])

        # Extract the solution path
        path = [point[2][0] for point in solver.solution]  # Extract position values
        times = [point[0] for point in solver.solution]    # Extract time values

        all_paths.append((times, path))
        print(f"Final position: {path[-1]:.4f}")

        # For the first path, also show some intermediate values
        if i == 0:
            print("\nFirst path intermediate values:")
            for j in [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps]:
                print(f"  t={times[j]:.3f}: X={path[j]:.4f}")

    print("\nPlotting results...")

    # Create comprehensive plot
    plot.figure.update_layout(
        title="Brownian Motion Simulation - Multiple Paths",
        xaxis_title="Time (t)",
        yaxis_title="Position X(t)",
        hovermode="x unified",
        showlegend=True,
    )

    # Plot each path
    colors = ["blue", "red", "green", "orange", "purple"]
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

    # Add analytical properties
    plot.figure.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text="Expected value: E[X_t] = 0<br>Variance: Var(X_t) = t",
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
    )

    # Save the plot
    script_dir = os.path.dirname(__file__)
    png_path = os.path.join(script_dir, "images", "brownian_motion_simulation.png")
    html_path = os.path.join(script_dir, "images", "brownian_motion_simulation.html")
    plot.figure.write_image(png_path)
    plot.figure.write_html(html_path)

    print("Plots saved as:")
    print("- brownian_motion_simulation.png")
    print("- brownian_motion_simulation.html")

    # Statistical analysis
    print("\nStatistical Analysis:")
    print("-" * 30)

    final_positions = [path[-1] for _, path in all_paths]
    mean_final = np.mean(final_positions)
    std_final = np.std(final_positions)
    var_final = np.var(final_positions)

    print(f"Mean final position: {mean_final:.4f}")
    print(f"Std final position: {std_final:.4f}")
    print(f"Var final position: {var_final:.4f}")

    # Theoretical expectation: E[X_T] = 0
    # Theoretical variance: Var(X_T) = T = 2.0
    theoretical_variance = end_time
    print(f"Theoretical variance: {theoretical_variance:.4f}")
    print(f"Ratio: {var_final/theoretical_variance:.1f}")

    return all_paths


if __name__ == "__main__":
    # Run Brownian motion simulation
    paths = run_brownian_motion_example()

    print("\nSimulation completed successfully!")
    print("Check the generated PNG and HTML files for visualizations.")
