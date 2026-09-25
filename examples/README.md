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

## Building a site of every example

`build_sites.py` runs every example in its own process and gathers what each
one writes into a single directory with an index page linking them all:

```bash
python examples/build_sites.py                   # everything, into examples_site/
python examples/build_sites.py out --only lorenz thaler
python examples/build_sites.py --list            # names and rough timings
```

Three kinds of example are handled. Those that write self-contained HTML (the
continuation atlas and its two playable stages) are handed the output
directory; those that write PNG and HTML beside their own source have the
figures they just wrote harvested into a gallery page; and those that only
print have their console output made into a page.

Each example runs with a timeout and its own process, so one failure does not
stop the rest - the report at the end names what failed and the exit status is
non-zero. `--jobs` runs several at once, which is worth it because the atlas
takes far longer than everything else put together.

Still images come from Plotly, which drives a real browser to export them. The
script finds an installed Chrome or Chromium (including the copy Playwright
keeps in its cache) and points Plotly at it; if there is none, run
`plotly_get_chrome` once and the figures will export.

Everything is written locally and nothing is uploaded. The output directory is
build output and is not tracked.

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
