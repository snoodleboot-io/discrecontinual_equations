#!/usr/bin/env python3
"""
Example: Couple-Cell Electric Field Sensors (CCEFS)

This example demonstrates solving the stochastic differential equations for
Couple-Cell Electric Field Sensors as described in the ACSESS 2008 poster.

The system consists of coupled cells with the following SDEs:
dX_i = (a X_i - b X_i^3 + ε_0 + λ (X_i - X_{i+1})) dt + sqrt(2D) dB_t

For this example, we implement a 3-cell system (N=3, odd number of oscillators)
with periodic coupling to form a ring.
"""

import os

import numpy as np
import plotly.graph_objects as go  # Add this import

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.plot.line_plot import LinePlot
from discrecontinual_equations.solver.stochastic.milstein.milstein_config import (
    MilsteinConfig,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_solver import (
    MilsteinSolver,
)
from discrecontinual_equations.variable import Variable


class CCEFSFunction(StochasticFunction):
    """
    Stochastic function for Couple-Cell Electric Field Sensors system.

    For a 3-cell system with periodic coupling.
    """

    def __init__(
        self,
        variables: list[Variable],
        parameters: list[Parameter],
        time: Variable | None = None,
    ):
        # Results are dX1/dt, dX2/dt, dX3/dt
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

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        """
        Drift terms for the CCEFS system with periodic coupling.
        """
        x1, x2, x3 = point
        a = self.parameters[0].value  # a parameter
        b = self.parameters[1].value  # b parameter
        epsilon0 = self.parameters[2].value  # ε_0
        lambda_param = self.parameters[3].value  # λ

        # Drift for X1: a*X1 - b*X1^3 + ε0 + λ*(X1 - X2)
        drift1 = a * x1 - b * x1**3 + epsilon0 + lambda_param * (x1 - x2)
        # Drift for X2: a*X2 - b*X2^3 + ε0 + λ*(X2 - X3)
        drift2 = a * x2 - b * x2**3 + epsilon0 + lambda_param * (x2 - x3)
        # Drift for X3: a*X3 - b*X3^3 + ε0 + λ*(X3 - X1)  [periodic]
        drift3 = a * x3 - b * x3**3 + epsilon0 + lambda_param * (x3 - x1)

        return [drift1, drift2, drift3]

    def diffusion(self, point: list[float], time: float | None = None) -> list[float]:
        """
        Diffusion terms: sqrt(2D) for each cell
        """
        D = self.parameters[4].value  # Diffusion coefficient D
        return [np.sqrt(2 * D), np.sqrt(2 * D), np.sqrt(2 * D)]


def main():
    """Run the CCEFS example."""
    print("Couple-Cell Electric Field Sensors (CCEFS) Simulation")
    print("=" * 55)

    # Define variables
    x1 = Variable(name="X1", abbreviation="X₁")
    x2 = Variable(name="X2", abbreviation="X₂")
    x3 = Variable(name="X3", abbreviation="X₃")
    t = Variable(name="Time", abbreviation="t")

    # Parameters
    nonlinearity_coeff_a = Parameter(
        name="Nonlinearity Coefficient A",
        value=1.0,
    )  # Corresponds to 'a' in the CCEFS equation
    nonlinearity_coeff_b = Parameter(
        name="Nonlinearity Coefficient B",
        value=0.1,
    )  # Corresponds to 'b' in the CCEFS equation
    bias_voltage = Parameter(
        name="Bias Voltage",
        value=0.0,
    )  # Corresponds to ε₀ in the CCEFS equation
    coupling_strength = Parameter(
        name="Coupling Strength",
        value=0.5,
    )  # Corresponds to λ in the CCEFS equation
    noise_intensity = Parameter(
        name="Noise Intensity",
        value=0.01,
    )  # Corresponds to D in the CCEFS equation

    parameters = [
        nonlinearity_coeff_a,
        nonlinearity_coeff_b,
        bias_voltage,
        coupling_strength,
        noise_intensity,
    ]

    # Create the stochastic function
    ccefs = CCEFSFunction(variables=[x1, x2, x3], parameters=parameters, time=t)

    # Create differential equation
    equation = DifferentialEquation(
        variables=[x1, x2, x3],
        time=t,
        parameters=parameters,
        derivative=ccefs,
    )

    # Simulation parameters
    end_time = 500.0  # Total simulation time
    n_steps = 5000  # Number of time steps
    step_size = end_time / n_steps

    # Solver configuration - Using Milstein method for higher accuracy
    config = MilsteinConfig(
        start_time=0.0,
        end_time=end_time,
        step_size=step_size,
        random_seed=42,  # For reproducible results
    )

    print("Simulating 3-cell CCEFS system with Milstein method...")
    print(f"Total time: {end_time:.2f}")
    print(f"Time steps: {n_steps}")
    print(f"Step size: {step_size:.4f}")
    print(
        f"Parameters: A={nonlinearity_coeff_a.value}, B={nonlinearity_coeff_b.value}, Bias={bias_voltage.value}, Coupling={coupling_strength.value}, Noise={noise_intensity.value}",
    )
    print()

    # Create solver
    solver = MilsteinSolver(config)

    # Solve with initial conditions
    initial_conditions = [0.1, 0.2, 0.3]  # X1=0.1, X2=0.2, X3=0.3
    solver.solve(equation, initial_conditions)

    print("Simulation completed.")

    # Extract results
    times = [point[0] for point in solver.solution]
    x1_values = [point[2][0] for point in solver.solution]
    x2_values = [point[2][1] for point in solver.solution]
    x3_values = [point[2][2] for point in solver.solution]

    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)

    # Create plot
    plot = LinePlot(output_dir=script_dir)

    # Plot X1, X2, X3 time series - Fixed to use go.Scatter
    plot.figure.add_trace(
        go.Scatter(
            x=times,
            y=x1_values,
            mode="lines",
            name="X₁ (Cell 1)",
            line=dict(color="blue", width=1),
        ),
    )

    plot.figure.add_trace(
        go.Scatter(
            x=times,
            y=x2_values,
            mode="lines",
            name="X₂ (Cell 2)",
            line=dict(color="red", width=1),
        ),
    )

    plot.figure.add_trace(
        go.Scatter(
            x=times,
            y=x3_values,
            mode="lines",
            name="X₃ (Cell 3)",
            line=dict(color="green", width=1),
        ),
    )

    plot.figure.update_layout(
        title="CCEFS 3-Cell System Time Series",
        xaxis_title="Time (t)",
        yaxis_title="Cell State X_i",
        hovermode="x unified",
        showlegend=True,
    )

    plot.figure.show()

    # Save time series plot
    ts_png = os.path.join(script_dir, "ccefs_timeseries.png")
    ts_html = os.path.join(script_dir, "ccefs_timeseries.html")
    plot.figure.write_image(ts_png)
    plot.figure.write_html(ts_html)

    print("Time series plot saved as:")
    print(f"- {os.path.basename(ts_png)}")
    print(f"- {os.path.basename(ts_html)}")

    # Create phase space plot (X1 vs X2)
    plot.figure.data = []  # Clear previous traces

    plot.figure.add_trace(
        go.Scatter(
            x=x1_values,
            y=x2_values,
            mode="lines",
            name="X₁ vs X₂",
            line=dict(color="purple", width=1),
        ),
    )

    plot.figure.update_layout(
        title="CCEFS Phase Space (X₁ vs X₂)",
        xaxis_title="X₁",
        yaxis_title="X₂",
        hovermode="closest",
    )

    plot.figure.show()

    # Save phase space plot
    ps_png = os.path.join(script_dir, "ccefs_phasespace.png")
    ps_html = os.path.join(script_dir, "ccefs_phasespace.html")
    plot.figure.write_image(ps_png)
    plot.figure.write_html(ps_html)

    print("Phase space plot saved as:")
    print(f"- {os.path.basename(ps_png)}")
    print(f"- {os.path.basename(ps_html)}")

    # Statistical analysis
    print("\nStatistical Analysis:")
    print("-" * 25)

    # Final values statistics
    final_x1 = x1_values[-1]
    final_x2 = x2_values[-1]
    final_x3 = x3_values[-1]

    print(f"Final X₁: {final_x1:.4f}")
    print(f"Final X₂: {final_x2:.4f}")
    print(f"Final X₃: {final_x3:.4f}")

    # Time-averaged statistics
    mean_x1 = np.mean(x1_values)
    mean_x2 = np.mean(x2_values)
    mean_x3 = np.mean(x3_values)

    print(f"Mean X₁: {mean_x1:.4f}")
    print(f"Mean X₂: {mean_x2:.4f}")
    print(f"Mean X₃: {mean_x3:.4f}")

    # Correlations
    corr12 = np.corrcoef(x1_values, x2_values)[0, 1]
    corr13 = np.corrcoef(x1_values, x3_values)[0, 1]
    corr23 = np.corrcoef(x2_values, x3_values)[0, 1]
    print(f"X₁-X₂ correlation: {corr12:.4f}")
    print(f"X₁-X₃ correlation: {corr13:.4f}")
    print(f"X₂-X₃ correlation: {corr23:.4f}")


if __name__ == "__main__":
    main()
