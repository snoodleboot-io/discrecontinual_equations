# Examples

This directory contains example scripts demonstrating the usage of the discrecontinual_equations library.

## Available Examples

### Quartic Equation (`quartic_equation.py`)
- Demonstrates evaluation of a quartic potential function: E(x) = -a*x^4 + b*x^2
- Shows a double-well potential structure
- Plots the energy landscape

### Lorenz Equation (`lorenz_equation.py`)
- Solves the famous Lorenz system of chaotic differential equations
- Shows the iconic Lorenz attractor in 3D space
- Demonstrates chaotic behavior with sensitive dependence on initial conditions

### Thaler Method (`thaler/thaler_example.py`)
- Demonstrates the Thaler method for α-stable stochastic differential equations
- Shows α-stable diffusion with heavy-tailed behavior
- Illustrates boundary preservation for SDEs with natural boundaries
- Compares deterministic homogenisation approach with traditional methods

### Zhan-Duan-Li-Li Method (`zhan_duan_li_li/zhan_duan_li_li_example.py`)
- Demonstrates the Zhan-Duan-Li-Li method for stochastic differential equations
- Shows geometric Brownian motion simulation
- Placeholder implementation - needs to be updated with actual method from paper
- Framework for comparing different SDE solution methods

### Complex ODE System (`complex_ode_system/complex_ode_example.py`)
- Demonstrates solving a complex system of coupled ODEs using Euler method
- Shows time evolution and phase space plots for the system
- Implements nonlinear equations with absolute values and fractional powers
- Analyzes system behavior and stability properties

## Running Examples

Each example can be run directly:

```bash
uv run python examples/quartic_equation.py
uv run python examples/lorenz_equation.py
uv run python examples/thaler/thaler_example.py
uv run python examples/zhan_duan_li_li/zhan_duan_li_li_example.py
uv run python examples/complex_ode_system/complex_ode_example.py
```

The examples will compute the solutions and display interactive plots using Plotly.
