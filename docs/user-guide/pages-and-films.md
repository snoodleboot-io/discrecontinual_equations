# Atlas pages and films

The library draws its results as web pages. Each page is one HTML file with D3 and
its data inlined, so it opens from disk with no server and no network. The pages
come in three kinds:

- a **still**: one picture, such as a bifurcation diagram, a two-parameter curve, a
  family surface or an exploration tree;
- a **stage**, also called a **film**: a one-parameter continuation played as
  motion. A scrub bar moves the parameter, particles flow through the vector field,
  and the equilibria, their eigenvalues, the saddle manifolds and the cycles move
  with it;
- a **manifold film**: an invariant manifold grown ring by ring, such as the Lorenz
  manifold.

The **atlas** is a directory of these pages with an `index.html` of cards that
link to them. A card is marked when its page moves.

This area is infrastructure, not mathematics. The few places where an algorithm is
involved are covered [below](#the-numerics-inside-a-page), with their sources on
the [References](../references.md#pages-films-and-numerics-infrastructure) page.

## Builders and renderers

Every page is made in two steps that never mix:

1. A **builder** turns the library's results into a render-agnostic value object.
   [`scene_builder`](../api/plotting.md#discrecontinual_equations.webplot.scene_builder)
   makes a [`Scene`](../api/plotting.md#discrecontinual_equations.webplot.scene.Scene),
   and [`stage_scene`](../api/plotting.md#discrecontinual_equations.webplot.stage_builder.stage_scene)
   makes a [`StageScene`](../api/plotting.md#discrecontinual_equations.webplot.stage.StageScene).
   These objects hold numbers and labels, never drawing code, so they are built and
   unit-tested without a browser.
2. A **renderer** turns the object into HTML.
   [`D3Renderer`](../api/plotting.md#discrecontinual_equations.webplot.renderer.D3Renderer)
   draws stills.
   [`StageRenderer`](../api/plotting.md#discrecontinual_equations.webplot.stage_renderer.StageRenderer)
   draws stages, and the stochastic stage renderer draws stochastic ones.
   [`ManifoldFilmRenderer`](../api/plotting.md#discrecontinual_equations.webplot.manifold_film_renderer.ManifoldFilmRenderer)
   draws manifold films.

[`PlotReport`](../api/plotting.md#discrecontinual_equations.webplot.report.PlotReport)
pairs a renderer with an output directory. Its `write(scene, filename)` method
writes one page, and its `write_atlas(entries)` method writes the `index.html`
of [`AtlasEntry`](../api/plotting.md#discrecontinual_equations.webplot.report.AtlasEntry)
cards. Two reports writing to one directory share an atlas when every entry is
passed to a single `write_atlas` call.

## What a stage is

A stage is built from two inputs and one optional setting:

- [`Continued`](../api/plotting.md#discrecontinual_equations.webplot.stage_builder.Continued)
  holds **what was computed**: the equation, the index of the continuation
  parameter, one or more equilibrium branches, and optionally the cycle branch
  from `CycleContinuation`.
- [`Film`](../api/plotting.md#discrecontinual_equations.webplot.stage_builder.Film)
  holds **how it is shot**: the parameter values of the frames, the
  [`Lattice`](../api/plotting.md#discrecontinual_equations.webplot.stage.Lattice)
  (the box, the grid the field is sampled on and the axis names), the
  [`StageSystem`](../api/plotting.md#discrecontinual_equations.webplot.stage.StageSystem)
  narration (the title, the subtitle, the parameter's symbol, an equation in LaTeX
  and a note), and an optional `describe(frame)` that names each frame's regime.
- [`StageSettings`](../api/plotting.md#discrecontinual_equations.webplot.stage_builder.StageSettings)
  sets how far, and to what Taylor order, the saddle manifolds are followed.

For each frame value $p_k$, `stage_scene` does the following:

- **Equilibria.** It walks each branch pair by pair and takes every crossing of
  $p_k$. A branch that rounds a fold therefore gives one equilibrium on each arc.
  The state is interpolated linearly between the two points that straddle $p_k$.
  The stability and eigenvalues are taken from the nearer of the two, because they
  are measurements, not quantities to average.
- **Saddle manifolds.** Each one-dimensional stable or unstable manifold starts on
  its Taylor chart, a distance $10^{-3}$ from the equilibrium, and is carried by the
  flow until it leaves the box. A two-dimensional manifold is drawn as a fan of
  trajectories through a ring of chart points. All the branches of a frame are
  integrated together; see [Manifold branches](#manifold-branches).
- **Cycles.** Each sheet of the cycle branch contributes its point nearest
  $p_k$. A folded cycle branch therefore shows both its stable and its unstable
  cycle.
- **Field.** For a planar system, the field is sampled on the lattice's
  $n \times n$ nodes, with $n$ the lattice's `grid`.

A system of three or more dimensions is filmed through a
[`View`](../api/plotting.md#discrecontinual_equations.webplot.stage.View). The view
holds one or more $2\times N$ projections, the bounds the particles live in, and the
field written as polynomial
[`Term`](../api/plotting.md#discrecontinual_equations.webplot.stage.Term) lists.
The page integrates the particles in full dimension from those terms. Before
anything is built, the terms are checked against the system at eight random states.
A mistyped coefficient therefore stops the build instead of sending particles
through a different flow.

The detected bifurcations of every branch go on the **timeline**, the bifurcation
diagram under the stage.

## Building one film

The example below continues the saddle-node $\dot x = \mu - x^2,\ \dot y = -y$ and
films it. It writes the stage page, a still of the same branch, and an atlas with a
card for each. It takes a few seconds.

```python
from pathlib import Path

import numpy as np
from sweet_tea.registry import Registry

import discrecontinual_equations
from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable
from discrecontinual_equations.webplot import (
    AtlasEntry,
    Continued,
    D3Renderer,
    Film,
    Lattice,
    PlotReport,
    StageRenderer,
    StageSystem,
    bifurcation_scene,
    stage_scene,
)

# The continuation builders look their components up by name.
Registry.fill_registry(
    path=str(Path(discrecontinual_equations.__file__).parent),
    module="discrecontinual_equations",
    exclude=["*.tests", "*.examples", "*.plot"],
)


class Mu(Parameter, name="Mu", abbreviation="mu"):
    """The continuation parameter."""


class State(Variable, name="State", abbreviation="v"):
    """One state coordinate."""


class Time(Variable, name="Time", abbreviation="t"):
    """Time."""


class Fold(DeterministicFunction):
    """x' = mu - x^2, y' = -y: two equilibria meet at mu = 0 and vanish."""

    def eval(self, point, time=None):
        mu = self.parameters[0].value
        return [mu - point[0] ** 2, -point[1]]


mu = Mu(value=1.0)
equation = DifferentialEquation(
    variables=[State(), State()],
    time=Time(),
    parameters=[mu],
    derivative=Fold(
        variables=[State(), State()],
        parameters=[mu],
        results=[State(), State()],
        time=None,
    ),
)
config = ContinuationConfig(
    continuation_parameter_index=0,
    detectors=["fold"],
    initial_parameter=1.0,
    direction=-1,
    measure="component",
    parameter_lower_bound=-0.5,
    parameter_upper_bound=1.2,
    maximum_points=400,
    maximum_step=0.05,
)
branch = ContinuerBuilder.build(config, equation).solve(equation, [1.0, 0.0])

stage = stage_scene(
    Continued(equation, 0, branch),
    Film(
        frames=np.linspace(-0.5, 1.0, 31),
        lattice=Lattice(((-1.5, 1.5), (-1.0, 1.0)), 24, "x", "y"),
        system=StageSystem("Saddle-node", "Two equilibria meet and vanish.", "μ"),
    ),
)
still = bifurcation_scene(branch, "Saddle-node", "The fold branch")

out = "site"
PlotReport(StageRenderer(), out).write(stage, "fold_stage.html")
report = PlotReport(D3Renderer(), out)
report.write(still, "fold.html")
report.write_atlas(
    [
        AtlasEntry("fold.html", "Saddle-node", "The branch", kind="plot"),
        AtlasEntry("fold_stage.html", "Saddle-node film", "Playable", kind="stage"),
    ],
)
print([len(frame.equilibria) for frame in stage.frames][::10])  # [0, 0, 2, 2]
```

Open `site/index.html`. Below $\mu = 0$ there is no equilibrium, and the particles
drift through the bottleneck where the saddle and node used to be. Above $\mu = 0$
the saddle at $x = -\sqrt\mu$ and the node at $x = \sqrt\mu$ appear, and the
saddle's manifolds are drawn.

The library's own films live in `examples/continuation/stage_*.py`. Each one takes
an output directory, for example:

```bash
uv run --locked python examples/continuation/stage_hopf.py outputs/films/hopf
uv run --locked python examples/continuation/stage_lorenz_manifold.py outputs/films/lorenz
```

A film run into a directory that already has an `index.html` leaves that index
alone, so filming into a built atlas does not throw away its other cards.

## Building the atlas

`examples/build_sites.py` runs every example in its own process. It collects what
each one writes under one directory and adds an index page:

```bash
OPENBLAS_NUM_THREADS=1 uv run --locked python examples/build_sites.py outputs/atlas --jobs 4
uv run --locked python examples/build_sites.py --list                 # names and rough timings
uv run --locked python examples/build_sites.py out --only lorenz thaler     # a subset
```

The `continuation` example, `examples/continuation/webplot_examples.py`, is the
atlas proper. It writes every still with `D3Renderer` and every stage film with the
stochastic renderer, which draws a deterministic stage exactly as `StageRenderer`
does. It also writes the manifold films, then one `write_atlas` call for all of
them. It is by far the longest example to build, which is why `--jobs` runs
examples side by side. `OPENBLAS_NUM_THREADS=1` stops each process from starting
one BLAS thread per core. Plotly's PNG export needs a Chrome or Chromium, and
`build_sites.py` looks for one.

## Stochastic stages and the quasipotential panel

[`stochastic_stage`](../api/plotting.md#discrecontinual_equations.webplot.stochastic_stage_builder.stochastic_stage)
turns a stage built through a view into a stochastic one, with
$dx = f\,dt + G\,dW$:

- The particles become sample paths, which the page integrates by Euler-Maruyama
  from the drift and noise terms.
- Each frame gets the stationary Fokker-Planck density at its parameter.
- The clock shows the top Lyapunov exponent across the parameter.

Both thresholds are root-found and put on the timeline: the **P** (phenomenological)
bifurcation, where the density changes shape, and the **D** (dynamical) bifurcation,
where the top exponent changes sign. The noise terms are checked against the
`NoiseMatrix` before anything is built. The view's drift is checked against the Itô
drift, which is the drift an Euler-Maruyama step needs. A Stratonovich equation
whose correction is missing from the view is therefore refused.

[`with_quasipotential`](../api/plotting.md#discrecontinual_equations.webplot.quasipotential_stage.with_quasipotential)
adds a panel with the quasipotential $U$ and the minimum action paths, at the cost
of one ordered-line-integral march per frame.

The mathematics belongs to other pages:
[Lyapunov exponents and noise](lyapunov-noise.md) and
[Quasipotential, Fokker-Planck and first passage](quasipotential-first-passage.md).

## The numerics inside a page

### Manifold branches

A frame's saddles hand the builder up to a few dozen starting points.
[`flow_branches`](../api/plotting.md#discrecontinual_equations.webplot.manifold_flow.flow_branches)
carries them all at once, as rows of one array. It steps them with the
Dormand-Prince 5(4) pair [[Dormand1980](../references.md#pages-films-and-numerics-infrastructure)],
and each row keeps its own step size, its own acceptance and its own stopping time.
A step from $y_n$ with stages $k_1,\dots,k_7$ is accepted when

$$
\operatorname{rms}\!\left(\frac{h\sum_j e_j k_j}{\mathrm{atol} + \mathrm{rtol}\,\max(|y_n|,|y_{n+1}|)}\right) < 1,
$$

Here $e_j$ are the embedded error weights. After an accepted step the next step
is $h \cdot \min\!\big(10,\ 0.9\,\mathrm{err}^{-1/5}\big)$, and no growth is allowed
straight after a rejection. After a rejected step the retry is
$h \cdot \max\!\big(0.2,\ 0.9\,\mathrm{err}^{-1/5}\big)$. The first step follows the starting rule of Hairer, Nørsett &
Wanner, §II.4 [[Hairer1993](../references.md#pages-films-and-numerics-infrastructure)].

Between steps, each row has Shampine's free quartic interpolant
[[Shampine1986](../references.md#pages-films-and-numerics-infrastructure)]:

$$
y(t_n + \theta h) = y_n + h \sum_{j=1}^{7} k_j \sum_{m=1}^{4} P_{jm}\,\theta^m .
$$

A row stops where it crosses the edge of the box, widened by a margin. The crossing
is found by Brent's method on the interpolant
[[Brent1973](../references.md#pages-films-and-numerics-infrastructure)]. The branch
is then resampled from the interpolant at evenly spaced times. The tableau and the
interpolant matrix are SciPy's `RK45` ones, entry for entry, so a branch is the one
`solve_ivp(method="RK45")` would draw, to rounding.

### Curves on the page

The page scripts draw branches and cycle curves as centripetal Catmull-Rom splines
(`d3.curveCatmullRom.alpha(0.5)`) through the computed points
[[Catmull1974](../references.md#pages-films-and-numerics-infrastructure)]. The knots
are spaced by the chord length raised to a power, $t_{i+1} - t_i = |P_{i+1}-P_i|^{\alpha}$,
which is how the bundled D3 computes them. $\alpha = 0$ is the uniform spline,
$\alpha = 1$ the chordal one, and $\alpha = \tfrac12$ the centripetal one. It is the only member of the
family that forms no cusp or self-intersection within a segment
[[Yuksel2009](../references.md#pages-films-and-numerics-infrastructure)]. A spline
still interpolates, so between widely spaced points it shows a smooth curve that
was never computed.

### Interpolation between frames

The page plays a continuous parameter from a finite set of frames, and everything
between two frames is linear:

- the sampled field is bilinear in space and linear between frames;
- equilibria and manifold points are blended between neighbouring frames;
- cycles are matched by amplitude, and a cycle with no partner fades in or out.

A view's polynomial field is evaluated exactly at the current parameter. The
particles are visual only. They take explicit midpoint steps of fixed size and are
respawned when they stall, leave the box or jump.

Two more interpolations happen at build time, in `scene_builder`:

- A family's slices are coloured by mixing the two end colours linearly in RGB.
- A family surface resamples each slice to the same number of nodes, by linear
  interpolation. It uses evenly spaced measures when the measure is monotone along
  the slice, and evenly spaced arclengths otherwise.

## Static figures: the `plot` package

The `plot` package writes Plotly figures (PNG through kaleido, or HTML):

- [`BifurcationPlot`](../api/plotting.md#discrecontinual_equations.plot.bifurcation_plot.BifurcationPlot)
  draws a branch with its stability and bifurcations.
- [`LinePlot`](../api/plotting.md#discrecontinual_equations.plot.line_plot.LinePlot)
  draws time series and trajectories.
- [`PhasePlot`](../api/plotting.md#discrecontinual_equations.plot.phase_plot.PhasePlot)
  draws phase portraits.
- [`DelayPlot`](../api/plotting.md#discrecontinual_equations.plot.delay_plot.DelayPlot)
  draws delay embeddings $(x_i, x_{i+k})$ and $(x_i, x_{i+k_1}, x_{i+k_2})$. By
  Takens' theorem, $2d+1$ delays of a generic observable embed a $d$-dimensional
  attractor [[Takens1981](../references.md#pages-films-and-numerics-infrastructure),
  Theorems 1 and 2]. Two or three delays are a picture, not a guaranteed
  embedding, and the delay is counted in samples.
- [`SpectrumPlot`](../api/plotting.md#discrecontinual_equations.plot.spectrum_plot.SpectrumPlot)
  draws $|\mathrm{FFT}|^2$ and the periodogram. Both are single raw periodograms
  with a rectangular window and no averaging, so each bin is as noisy as its own
  mean. Use `scipy.signal.welch` first if you need a smooth estimate.

## Limits and failure modes

- **Frames are samples.** Anything between two frames is drawn by linear
  interpolation, as described above. Frames must be close enough that the flow does
  not jump. A bifurcation between two frames is shown on the timeline at its
  detected value, but the stage only changes at the next frame.
- **What a stage shows is what was continued.** Equilibria come only from the
  branches handed in, and cycles only from the cycle branch. A branch that was
  never continued is simply missing.
- **Saddle manifolds** are drawn only for one- and two-dimensional manifolds, only
  inside the box plus a margin, and for at most `manifold_time` time units. A branch
  spiralling into an attractor is cut off where time runs out.
- **Views must be polynomial.** The page integrates the `Term` lists, not your
  Python field. A non-polynomial system can be filmed in the plane through the
  sampled lattice, but not through a view.
- **Page size.** Each page inlines D3 and every frame's data. Stage pages run from
  about a megabyte up, so many frames on a fine lattice make a heavy page.
- **Stochastic stages refuse rather than guess.** A threshold whose bracket holds
  no sign change, noise terms that disagree with the matrix, or a drift that is not
  Itô all stop the build with a `ValueError`.

## API

See the [Plotting API](../api/plotting.md). The main entries are:

- `webplot.stage_builder`: `stage_scene`, `Continued`, `Film`, `StageSettings`.
- `webplot.stage`: `StageScene`, `Frame`, `Lattice`, `View`, `Term`.
- `webplot.scene_builder` and `webplot.scene`: stills.
- `webplot.stochastic_stage_builder` and `webplot.quasipotential_stage`: stochastic
  stages.
- `webplot.manifold_flow`: `flow_branches`.
- `webplot.report`: `PlotReport`, `AtlasEntry`.
