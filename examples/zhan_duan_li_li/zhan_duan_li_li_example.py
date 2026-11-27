"""
Zhan-Duan-Li-Li Method for Stochastic Differential Equations

This example runs the solver named Zhan-Duan-Li-Li: an explicit predictor-corrector
step whose corrector weights the predicted point by theta (method_parameter). At
theta = 1/2 it is the stochastic Heun step. The equations here are Ito equations
(the config's default calculus), and the solver adjusts the drift so that the step
converges to the Ito solution for every theta (DEQ-133). See the solver's docstring
for the method and its sources; the arXiv paper the name comes from
(arXiv:2010.07691) describes a different, symplectic Marcus scheme.
"""

import os

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
from discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_config import (
    ZhanDuanLiLiConfig,
)
from discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_solver import (
    ZhanDuanLiLiSolver,
)
from discrecontinual_equations.variable import Variable


class SimpleStochasticProcess(StochasticFunction):
    """
    Simple stochastic process: dX = μX dt + σX dW

    This represents geometric Brownian motion, commonly used in
    financial modeling and population dynamics.
    """

    def __init__(self, variables, parameters, time=None):
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

    def eval(self, point, time=None):
        """
        Drift term: μ(x,t) = μ * x (geometric drift)
        """
        x = point[0]
        mu = self.parameters[0].value
        return [mu * x]

    def diffusion(self, point, time=None):
        """
        Diffusion term: σ(x,t) = σ * x (geometric diffusion)
        """
        x = point[0]
        sigma = self.parameters[1].value
        return [sigma * x]


def run_zhan_duan_li_li_example():
    """
    Demonstrate the Zhan-Duan-Li-Li stochastic theta method on geometric Brownian motion.
    """
    print("Zhan-Duan-Li-Li Method Example")
    print("=" * 40)

    # Define variables and parameters
    x = Variable(name="Price", abbreviation="X")
    t = Variable(name="Time", abbreviation="t")
    drift_param = Parameter(name="Drift Rate", value=0.1)  # μ = 10% growth
    diffusion_param = Parameter(name="Volatility", value=0.2)  # σ = 20% volatility

    # Create the stochastic function
    gbm_process = SimpleStochasticProcess(
        variables=[x],
        parameters=[drift_param, diffusion_param],
        time=t,
    )

    # Create differential equation
    equation = DifferentialEquation(
        variables=[x],
        time=t,
        parameters=[drift_param, diffusion_param],
        derivative=gbm_process,
    )

    # Simulation parameters
    n_simulations = 3  # Number of sample paths
    end_time = 1.0  # One year simulation
    n_steps = 100  # Time steps
    step_size = end_time / n_steps

    # Zhan-Duan-Li-Li solver configuration
    # Theta parameter: 0.5 = stochastic Heun, strong order 1.0 on this equation
    config = ZhanDuanLiLiConfig(
        method_parameter=0.5,  # Stochastic Heun
        start_time=0.0,
        end_time=end_time,
        step_size=step_size,
        random_seed=42,
    )

    print("Solving Geometric Brownian Motion:")
    print("dX = μX dt + σX dW")
    print(f"Total time: {end_time:.2f}")
    print(f"Time steps: {n_steps:.0f}")
    print(f"Step size: {step_size:.3f}")
    print(f"Zhan-Duan-Li-Li theta parameter: {config.method_parameter}")
    print()

    # Create plotter
    plot = LinePlot(
        output_dir=os.path.join(os.path.dirname(__file__), "images"),
        output_format="png",
    )

    # Simulate multiple paths
    all_paths = []

    for i in range(n_simulations):
        # Create solver with fresh random seed for each path
        config.random_seed = 42 + i
        solver = ZhanDuanLiLiSolver(config)

        # Solve with initial condition X_0 = 100
        solver.solve(equation, [100.0])

        # Extract the solution path
        path = [point[2][0] for point in solver.solution]  # Extract price values
        times = [point[0] for point in solver.solution]  # Extract time values

        all_paths.append((times, path))
        final_price = path[-1]
        print(f"Path {i + 1}: final price {final_price:.2f}")

    print("\nPlotting results...")

    # Create plot
    plot.figure.update_layout(
        title="Zhan-Duan-Li-Li Method - Geometric Brownian Motion",
        xaxis_title="Time (years)",
        yaxis_title="Price X(t)",
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
                name=f"Path {i + 1}",
                line=dict(color=colors[i], width=2),
                showlegend=True,
            ),
        )

    # Add theoretical information
    plot.figure.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text="""
        Geometric Brownian Motion:<br>
        • Drift: μ = 10% per year<br>
        • Volatility: σ = 20% per year<br>
        • Initial price: X₀ = $100<br>
        • Method: Zhan-Duan-Li-Li (θ = 0.5)
        """,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        align="left",
    )

    # Save the plot
    script_dir = os.path.dirname(__file__)
    png_path = os.path.join(script_dir, "images", "zhan_duan_li_li_gbm.png")
    html_path = os.path.join(script_dir, "images", "zhan_duan_li_li_gbm.html")
    plot.figure.write_image(png_path)
    plot.figure.write_html(html_path)

    print("Plots saved as:")
    print("- zhan_duan_li_li_gbm.png")
    print("- zhan_duan_li_li_gbm.html")

    # Show the plot
    plot.figure.show()

    return all_paths


def run_method_comparison():
    """
    Compare Zhan-Duan-Li-Li with Euler-Maruyama on the same Brownian path.

    Both solvers draw one normal per step from a generator seeded with the same
    seed, so they see the same increments, and both converge to the Ito solution.
    """
    print("\nMethod Comparison")
    print("=" * 20)

    # Set up the same equation
    x = Variable(name="Value", abbreviation="X")
    t = Variable(name="Time", abbreviation="t")
    drift_param = Parameter(name="Drift", value=0.05)
    diffusion_param = Parameter(name="Diffusion", value=0.1)

    process = SimpleStochasticProcess(
        variables=[x],
        parameters=[drift_param, diffusion_param],
        time=t,
    )

    equation = DifferentialEquation(
        variables=[x],
        time=t,
        parameters=[drift_param, diffusion_param],
        derivative=process,
    )

    # Compare methods
    methods = ["Zhan-Duan-Li-Li", "Euler-Maruyama"]
    final_values = []

    for method_name in methods:
        print(f"\nTesting {method_name}:")

        if method_name == "Zhan-Duan-Li-Li":
            config = ZhanDuanLiLiConfig(
                method_parameter=1.0,
                start_time=0.0,
                end_time=1.0,
                step_size=0.01,
                random_seed=123,
            )
            solver = ZhanDuanLiLiSolver(config)
        else:
            solver = EulerMaruyamaSolver(
                EulerMaruyamaConfig(
                    start_time=0.0,
                    end_time=1.0,
                    step_size=0.01,
                    random_seed=123,
                ),
            )

        solver.solve(equation, [1.0])
        final_value = solver.solution.results[0].discretization[-1]
        final_values.append(final_value)
        print(f"Final value: {final_value:.4f}")

    print("Both are Ito schemes on the same increments; they differ by their")
    print("discretisation error, which shrinks with the step size.")

    return final_values


def main():
    """
    Run Zhan-Duan-Li-Li method examples.
    """
    print("Zhan-Duan-Li-Li Method Examples")
    print("=" * 50)
    print(
        "This example demonstrates the theta-weighted stochastic Heun step (Zhan-Duan-Li-Li).",
    )
    print(
        "The corrector weights the predicted point by theta; the step stays explicit.",
    )
    print()

    # Run main example
    run_zhan_duan_li_li_example()

    # Run comparison
    run_method_comparison()

    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("Check the generated PNG and HTML files for visualizations.")
    print("\nImplementation Notes:")
    print("• Theta weights the predicted point in the corrector; the step is explicit")
    print("• θ = 0.5 (stochastic Heun) has strong order 1.0 for scalar noise")
    print("• Other θ converge to the same solution with strong order 0.5")


if __name__ == "__main__":
    main()
