# Plotting API

Two packages draw results. `webplot` writes self-contained D3 pages: stills,
playable stages (films) and the atlas that links them. `plot` writes Plotly
figures of time series, phase portraits, delay embeddings and spectra. The
[Atlas pages and films](../user-guide/pages-and-films.md) guide explains how
they fit together.

## Pages and films (`webplot`)

::: discrecontinual_equations.webplot
    options:
      members: false

### Stills

::: discrecontinual_equations.webplot.scene

::: discrecontinual_equations.webplot.scene_builder

### Stages

::: discrecontinual_equations.webplot.stage

::: discrecontinual_equations.webplot.stage_builder

::: discrecontinual_equations.webplot.manifold_flow

### Stochastic stages

::: discrecontinual_equations.webplot.stochastic_stage_builder

::: discrecontinual_equations.webplot.quasipotential_stage

### Manifold films

::: discrecontinual_equations.webplot.manifold_film

### Renderers and the atlas

::: discrecontinual_equations.webplot.renderer

::: discrecontinual_equations.webplot.stage_renderer

::: discrecontinual_equations.webplot.stochastic_stage_renderer

::: discrecontinual_equations.webplot.manifold_film_renderer

::: discrecontinual_equations.webplot.report

::: discrecontinual_equations.webplot.latex

## Figures (`plot`)

::: discrecontinual_equations.plot.bifurcation_plot

::: discrecontinual_equations.plot.line_plot

::: discrecontinual_equations.plot.phase_plot

::: discrecontinual_equations.plot.delay_plot

::: discrecontinual_equations.plot.spectrum_plot
