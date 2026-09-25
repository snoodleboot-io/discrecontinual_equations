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
continuation atlas and its ten stage films) are handed the output
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

## Stage films

A stage film is one self-contained page per bifurcation: the flow drawn as
advected particles, the equilibria and cycles standing on top of it, a spectral
clock, and a bifurcation diagram you scrub to move the parameter. Each one
builds on its own, writing its page and a one-card atlas into the directory you
name:

```bash
uv run python -m examples.continuation.stage_hopf plots
```

Swap the module for any of the ten. Roughly how long each takes, and what it is
for:

| Module | Minutes | What it shows |
| --- | --- | --- |
| `stage_hopf` | ~1 | The canonical Hopf: a pair crosses, a cycle of radius √μ opens |
| `stage_fold_of_cycles` | ~4 | Two cycles meeting and annihilating at a fold |
| `stage_van_der_pol` | ~3 | A relaxation oscillator stiffening as μ grows |
| `stage_snic` | ~6 | A cycle whose period runs to infinity as two equilibria land on it |
| `stage_extended_bautin` | ~2 | Three nested cycles at once, between two folds of cycles |
| `stage_zero_hopf` | ~2 | A fold that produces not two points but two circles |
| `stage_hopf_hopf` | ~2 | Four eigenvalues on the clock, one pair crossing |
| `stage_shilnikov` | ~1 | A saddle-focus whose saddle index crosses one |
| `stage_bogdanov_takens` | ~5 | Fold, Hopf and homoclinic meeting at one point |
| `stage_ferroelectric_ring` | ~5 | A unidirectionally coupled ring of three cells |

The cycle branches dominate those timings: `CycleContinuation` builds a dense
finite-difference Jacobian at every step, so a film with cycles costs about a
second per continuation step at 80 mesh intervals. The two films without one
(`stage_shilnikov`, and the equilibrium half of the rest) are the quick ones.

`build_sites.py` builds all ten as part of the `continuation` example, along
with the rest of the atlas.

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
