#!/usr/bin/env python3
"""
Geometric Brownian Motion Example

This example demonstrates geometric Brownian motion, a stochastic process
commonly used in financial modeling (Black-Scholes model).

The geometric Brownian motion is described by the SDE:
dX_t = μ X_t dt + σ X_t dW_t

This represents multiplicative noise and is suitable for modeling
asset prices, population growth, etc.
"""

import os

import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.plot.line_plot import LinePlot
from discrecontinual_equations.solver.stochastic.srk2.srk2_config import SRK2Config
from discrecontinual_equations.solver.stochastic.srk2.srk2_solver import SRK2Solver
from discrecontinual_equations.variable import Variable


class GeometricBrownianMotion(StochasticFunction):
    """
    Geometric Brownian motion: dX_t = μ X_t dt + σ X_t dW_t
    """

    def __init__(
        self,
        variables: list[Variable],
        parameters: list[Parameter],
        time: Variable | None = None,
    ):
        # Create result variables for the SDE
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
        """Drift term: μ X_t"""
        x = point[0]
        mu = self.parameters[0].value  # Drift rate
        return [mu * x]

    def diffusion(self, point: list[float], time: float | None = None) -> list[float]:
        """Diffusion term: σ X_t"""
        x = point[0]
        sigma = self.parameters[1].value  # Volatility
        return [sigma * x]


def main():
    """Run the geometric Brownian motion example."""
    print("Geometric Brownian Motion Example")
    print("=" * 40)

    # Parameters
    x = Variable(name="Asset Price", abbreviation="S")
    t = Variable(name="Time", abbreviation="t")
    drift_rate = Parameter(
        name="Drift Rate",
        value=0.1,
    )  # Corresponds to μ in geometric Brownian motion
    volatility = Parameter(
        name="Volatility",
        value=0.2,
    )  # Corresponds to σ in geometric Brownian motion

    # Create equation
    gbm_process = GeometricBrownianMotion(
        variables=[x],
        parameters=[drift_rate, volatility],
        time=t,
    )
    equation = DifferentialEquation(
        variables=[x],
        time=t,
        parameters=[drift_rate, volatility],
        derivative=gbm_process,
    )

    # Simulate using SRK2 method
    config = SRK2Config(
        start_time=0.0,
        end_time=1.0,  # 1 year
        step_size=0.001,
        random_seed=123,
    )

    solver = SRK2Solver(config)
    solver.solve(equation, [100.0])  # Start with $100

    print("Simulation completed.")

    # Extract results
    times = [point[0] for point in solver.solution]
    prices = [point[2][0] for point in solver.solution]

    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)

    # Plot
    plot = LinePlot(output_dir=os.path.join(script_dir, "images"))
    plot.figure.add_trace(
        dict(
            x=times,
            y=prices,
            mode="lines",
            name="Asset Price",
            line=dict(color="blue", width=2),
        ),
    )

    plot.figure.update_layout(
        title="Geometric Brownian Motion - Asset Price Simulation",
        xaxis_title="Time (years)",
        yaxis_title="Asset Price ($)",
    )

    plot.figure.show()

    # Save the plot
    png_path = os.path.join(script_dir, "images", "geometric_brownian_motion.png")
    html_path = os.path.join(script_dir, "images", "geometric_brownian_motion.html")
    plot.figure.write_image(png_path)
    plot.figure.write_html(html_path)

    print("Geometric Brownian motion plot saved as:")
    print(f"- {os.path.basename(png_path)}")
    print(f"- {os.path.basename(html_path)}")

    # Statistical analysis
    print("\nStatistical Analysis:")
    print("-" * 25)

    final_price = prices[-1]
    print(f"Final price: ${final_price:.2f}")

    # Theoretical expectation for GBM: S_0 * exp(μ T)
    theoretical_expectation = 100.0 * np.exp(drift_rate.value * 1.0)
    print(f"Theoretical expectation: ${theoretical_expectation:.2f}")
    print(f"Actual vs Theoretical: {final_price / theoretical_expectation:.2f}")

    # Theoretical variance: S_0^2 * exp(2μ T) * (exp(σ^2 T) - 1)
    theoretical_variance = (
        100.0**2
        * np.exp(2 * drift_rate.value * 1.0)
        * (np.exp(volatility.value**2 * 1.0) - 1)
    )
    actual_variance = np.var(prices)
    print(f"Theoretical variance: {theoretical_variance:.2f}")
    print(f"Actual variance: {actual_variance:.2f}")


if __name__ == "__main__":
    main()
