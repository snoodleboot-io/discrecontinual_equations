"""Generate self-contained D3 HTML plots for continuation results.

Run with ``python -m examples.continuation.webplot_examples [output_dir]``. It
continues equilibrium branches and their codim-1, codim-2, and codim-3
bifurcations, renders each with the D3 renderer using the system's own parameter
notation, and writes an atlas linking them.
"""

import sys

import numpy as np

from discrecontinual_equations.webplot.renderer import D3Renderer
from discrecontinual_equations.webplot.report import AtlasEntry, PlotReport
from discrecontinual_equations.webplot.scene import Scene
from discrecontinual_equations.webplot.scene_builder import (
    with_equation,
)
from discrecontinual_equations.webplot.stage_renderer import StageRenderer

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation.component_registry import ensure_registry
except ImportError:  # run as a script path: only this directory is on sys.path
    from component_registry import ensure_registry


try:  # python -m examples.continuation.webplot_examples
    from examples.continuation.pages_codim1 import (
        _hopf,
        _hysteresis,
        _pitchfork,
        _saddle_node,
        _transcritical,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from pages_codim1 import (
        _hopf,
        _hysteresis,
        _pitchfork,
        _saddle_node,
        _transcritical,
    )

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation.pages_codim2 import (
        _bogdanov_takens,
        _bogdanov_takens_curve,
        _bogdanov_takens_portrait,
        _bogdanov_takens_type_curve,
        _cusp,
        _degenerate_bautin,
        _generalized_hopf,
        _hopf_hopf,
        _swallowtail,
        _zero_hopf,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from pages_codim2 import (
        _bogdanov_takens,
        _bogdanov_takens_curve,
        _bogdanov_takens_portrait,
        _bogdanov_takens_type_curve,
        _cusp,
        _degenerate_bautin,
        _generalized_hopf,
        _hopf_hopf,
        _swallowtail,
        _zero_hopf,
    )

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation.pages_connecting import (
        _connecting_orbit_entries,
        _heteroclinic_cycle,
        _homoclinic_curve_shooting,
        _homoclinic_return,
        _manifolds,
        _saddle_focus_onset,
        _shilnikov_loop,
        _symmetry_modes,
        _three_dimensional_homoclinic,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from pages_connecting import (
        _connecting_orbit_entries,
        _heteroclinic_cycle,
        _homoclinic_curve_shooting,
        _homoclinic_return,
        _manifolds,
        _saddle_focus_onset,
        _shilnikov_loop,
        _symmetry_modes,
        _three_dimensional_homoclinic,
    )

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation.pages_cycles import (
        _adaptive_mesh,
        _collocation_convergence,
        _deflated_roots,
        _fold_of_cycles_branch,
        _limit_cycle_branch,
        _snic_period_divergence,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from pages_cycles import (
        _adaptive_mesh,
        _collocation_convergence,
        _deflated_roots,
        _fold_of_cycles_branch,
        _limit_cycle_branch,
        _snic_period_divergence,
    )

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation.pages_exploration import (
        _efield_detection,
        _efield_operating_map,
        _exploration_diagram,
        _exploration_skeleton,
        _exploration_tree,
        _sevencell_exploration,
        _sevencell_ring_map,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from pages_exploration import (
        _efield_detection,
        _efield_operating_map,
        _exploration_diagram,
        _exploration_skeleton,
        _exploration_tree,
        _sevencell_exploration,
        _sevencell_ring_map,
    )

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation.pages_families import (
        _bogdanov_takens_family,
        _bogdanov_takens_surface,
        _cusp_family,
        _cusp_surface,
        _fluxgate_bias_surface,
        _fluxgate_family,
        _fluxgate_surface,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from pages_families import (
        _bogdanov_takens_family,
        _bogdanov_takens_surface,
        _cusp_family,
        _cusp_surface,
        _fluxgate_bias_surface,
        _fluxgate_family,
        _fluxgate_surface,
    )

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation.pages_stochastic import (
        _density_family,
        _dynamical_bifurcation,
        _lyapunov,
        _stochastic_attractor,
        _stochastic_bifurcation,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from pages_stochastic import (
        _density_family,
        _dynamical_bifurcation,
        _lyapunov,
        _stochastic_attractor,
        _stochastic_bifurcation,
    )

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation import stage_bogdanov_takens as bt
    from examples.continuation import stage_extended_bautin as bautin
    from examples.continuation import stage_ferroelectric_ring as ring
    from examples.continuation import stage_fold_of_cycles as fold
    from examples.continuation import stage_hopf as hopf
    from examples.continuation import stage_hopf_hopf as hopfhopf
    from examples.continuation import stage_shilnikov as shilnikov
    from examples.continuation import stage_snic as snic
    from examples.continuation import stage_van_der_pol as vdp
    from examples.continuation import stage_zero_hopf as zerohopf
except ImportError:  # run as a script path: only this directory is on sys.path
    import stage_bogdanov_takens as bt
    import stage_extended_bautin as bautin
    import stage_ferroelectric_ring as ring
    import stage_fold_of_cycles as fold
    import stage_hopf as hopf
    import stage_hopf_hopf as hopfhopf
    import stage_shilnikov as shilnikov
    import stage_snic as snic
    import stage_van_der_pol as vdp
    import stage_zero_hopf as zerohopf

MU = "\u03bc"
ALPHA = "\u03b1"
LAMBDA = "\u03bb"
_FOLD_WINDOW = 0.02
_LOOP_ESCAPE = 50.0
_LOOP_DEPARTURE = 0.6
_LOOP_RETURN = 0.25
_IMAG_TOLERANCE = 1.0e-9
_MODE_IMAG = 1.0e-6
_PERIOD_TOLERANCE = 1.0e-2
BETA1, BETA2, BETA3 = "\u03b2\u2081", "\u03b2\u2082", "\u03b2\u2083"
NORM = "\u2016x\u2016"
_IMAGINARY_TOLERANCE = 1.0e-9


_VDP_STEP = 0.001
_VDP_SETTLE = 50.0


_SADDLE_JACOBIAN = np.array([[0.0, 1.0], [1.0, 0.0]])
_PHASE_WINDOW = 2.2


_FERROELECTRIC = r"\dot{x}_i = a\,x_i - x_i^3 - \lambda\,x_{i+1} + \varepsilon"
_FLUXGATE = r"\dot{x}_i = -x_i + \tanh(g\,x_i + c\,x_{i+1} + h)"
_CUSP_EQN = r"\dot{x} = \beta_1 + \beta_2 x - x^3"
_SWALLOWTAIL_EQN = r"\dot{x} = \beta_1 + \beta_2 x + \beta_3 x^2 - x^4"
_BT_FAMILY_EQN = r"\dot{x} = y,\quad \dot{y} = \beta_1 + \beta_2 y + x^2 + x y"

_EQUATIONS = {
    "saddle_node.html": r"\dot{x} = \lambda - x^2",
    "transcritical.html": r"\dot{x} = \lambda x - x^2",
    "pitchfork.html": r"\dot{x} = \lambda x - x^3",
    "hysteresis.html": r"\dot{x} = \lambda + 3x - x^3",
    "hopf.html": r"\dot{x} = \mu x - y - x(x^2+y^2),\quad "
    r"\dot{y} = x + \mu y - y(x^2+y^2)",
    "limit_cycle_branch.html": r"\dot{r} = \mu r - r^3,\quad r_{\max} = \sqrt{\mu}",
    "fold_of_cycles_branch.html": r"\dot{r} = \mu r + r^3 - r^5",
    "dynamical_bifurcation.html": r"\dot{x} = \mu x - y,\quad \dot{y} = x + \mu y",
    "collocation_convergence.html": r"\dot{x}=\mu x - y - x r^2,\ "
    r"\dot{y}=x+\mu y - y r^2",
    "deflated_roots.html": r"\dot{x}=x-x^3,\quad \dot{y}=y-y^3",
    "homoclinic_curve_shooting.html": r"\dot{x}=y,\ \dot{y}=x-x^2+\mu y+\nu x y",
    "symmetry_modes.html": r"\dot{x}_i = a x_i - x_i^3 - \lambda x_{i+1}",
    "shilnikov_loop.html": r"\dot{x}=y,\ \dot{y}=z,\ \dot{z}=-a z - y + x - x^2",
    "homoclinic_return.html": r"\dot{x} = y,\quad \dot{y} = x - x^2 + \mu y",
    "three_dimensional_homoclinic.html": r"\dot{u}=v,\ \dot{v}=u-u^2,\ \dot{w}=-2w",
    "heteroclinic_cycle.html": r"\dot{x} = y,\quad \dot{y} = -x + x^3",
    "saddle_focus_onset.html": r"\dot{x}=\sigma(y-x),\ "
    r"\dot{y}=x(\rho-z)-y,\ \dot{z}=xy-\beta z",
    "adaptive_mesh.html": r"\dot{x}=y,\quad \dot{y}=\mu(1-x^2)y - x",
    "stochastic_attractor.html": r"dx = f_{\mathrm{Lorenz}}(x)\,dt + \sigma\,dW",
    "stochastic_bifurcation.html": r"dx = \alpha x\,dt + \sigma x\,dW",
    "snic_period_divergence.html": r"\dot{r} = r(1-r^2),\quad "
    r"\dot{\theta} = \mu - \sin\theta",
    "bogdanov_takens.html": r"\dot{x} = y,\quad \dot{y} = \beta_1 + \beta_2 y "
    r"+ x^2 - x y",
    "bogdanov_takens_portrait.html": r"\dot{x} = y,\quad \dot{y} = \beta_1 "
    r"+ \beta_2 y + x^2 + x y",
    "zero_hopf.html": r"\dot{x} = \mu_1 - x^2,\quad \dot{r} = \mu_2 r - r^3",
    "hopf_hopf.html": r"\dot{r}_1 = \mu_1 r_1 - r_1^3,\quad "
    r"\dot{r}_2 = \mu_2 r_2 - r_2^3",
    "cusp.html": _CUSP_EQN,
    "generalized_hopf.html": r"\dot{r} = \beta_1 r + \beta_2 r^3",
    "swallowtail.html": _SWALLOWTAIL_EQN,
    "degenerate_bautin.html": r"\dot{r} = \beta_1 r + \beta_2 r^3 + \beta_3 r^5",
    "cusp_family.html": _CUSP_EQN,
    "fluxgate_family.html": _FLUXGATE,
    "cusp_surface.html": _CUSP_EQN,
    "fluxgate_surface.html": _FLUXGATE,
    "bogdanov_takens_family.html": _BT_FAMILY_EQN,
    "bogdanov_takens_surface.html": _BT_FAMILY_EQN,
    "efield_operating_map.html": _FERROELECTRIC,
    "efield_detection.html": _FERROELECTRIC,
    "fluxgate_bias_surface.html": _FLUXGATE,
    "exploration_diagram.html": _SWALLOWTAIL_EQN,
    "exploration_skeleton.html": _SWALLOWTAIL_EQN,
    "exploration_tree.html": _SWALLOWTAIL_EQN,
    "bogdanov_takens_curve.html": r"\dot{x} = y,\quad \dot{y} = \beta_1 "
    r"+ \beta_2 x + \beta_3 y + \beta_3 x^2 + x y",
    "bogdanov_takens_type_curve.html": r"\dot{x} = y,\quad \dot{y} = \beta_1 "
    r"+ \beta_2 x + \beta_3 y + x^2 + \beta_3 x y",
    "sevencell_ring_map.html": _FERROELECTRIC,
    "sevencell_exploration.html": _FERROELECTRIC,
}


def _catalogue() -> list[tuple[str, Scene]]:
    entries = _raw_catalogue()
    return [
        (name, with_equation(scene, _EQUATIONS[name]) if name in _EQUATIONS else scene)
        for name, scene in entries
    ]


def _raw_catalogue() -> list[tuple[str, Scene]]:
    return [
        ("saddle_node.html", _saddle_node()),
        ("transcritical.html", _transcritical()),
        ("pitchfork.html", _pitchfork()),
        ("hysteresis.html", _hysteresis()),
        ("hopf.html", _hopf()),
        ("limit_cycle_branch.html", _limit_cycle_branch()),
        ("fold_of_cycles_branch.html", _fold_of_cycles_branch()),
        ("dynamical_bifurcation.html", _dynamical_bifurcation()),
        ("collocation_convergence.html", _collocation_convergence()),
        ("deflated_roots.html", _deflated_roots()),
        ("homoclinic_curve_shooting.html", _homoclinic_curve_shooting()),
        ("symmetry_modes.html", _symmetry_modes()),
        ("shilnikov_loop.html", _shilnikov_loop()),
        ("homoclinic_return.html", _homoclinic_return()),
        ("three_dimensional_homoclinic.html", _three_dimensional_homoclinic()),
        ("heteroclinic_cycle.html", _heteroclinic_cycle()),
        ("saddle_focus_onset.html", _saddle_focus_onset()),
        ("stochastic_bifurcation.html", _stochastic_bifurcation()),
        ("adaptive_mesh.html", _adaptive_mesh()),
        ("stochastic_attractor.html", _stochastic_attractor()),
        ("snic_period_divergence.html", _snic_period_divergence()),
        ("bogdanov_takens.html", _bogdanov_takens()),
        ("bogdanov_takens_portrait.html", _bogdanov_takens_portrait()),
        ("zero_hopf.html", _zero_hopf()),
        ("hopf_hopf.html", _hopf_hopf()),
        ("cusp.html", _cusp()),
        ("generalized_hopf.html", _generalized_hopf()),
        ("swallowtail.html", _swallowtail()),
        ("degenerate_bautin.html", _degenerate_bautin()),
        ("cusp_family.html", _cusp_family()),
        ("fluxgate_family.html", _fluxgate_family()),
        ("cusp_surface.html", _cusp_surface()),
        ("fluxgate_surface.html", _fluxgate_surface()),
        ("bogdanov_takens_family.html", _bogdanov_takens_family()),
        ("bogdanov_takens_surface.html", _bogdanov_takens_surface()),
        ("efield_operating_map.html", _efield_operating_map()),
        ("efield_detection.html", _efield_detection()),
        ("fluxgate_bias_surface.html", _fluxgate_bias_surface()),
        ("exploration_diagram.html", _exploration_diagram()),
        ("exploration_skeleton.html", _exploration_skeleton()),
        ("exploration_tree.html", _exploration_tree()),
        ("bogdanov_takens_curve.html", _bogdanov_takens_curve()),
        ("bogdanov_takens_type_curve.html", _bogdanov_takens_type_curve()),
        ("sevencell_ring_map.html", _sevencell_ring_map()),
        ("sevencell_exploration.html", _sevencell_exploration()),
        ("manifolds.html", _manifolds()),
        *_connecting_orbit_entries(),
        ("stochastic_density.html", _density_family()),
        ("stochastic_lyapunov.html", _lyapunov()),
    ]


def main(output_dir: str = "plots") -> None:
    """Render every scene to ``output_dir`` and write the atlas."""
    ensure_registry()
    report = PlotReport(D3Renderer(), output_dir)
    entries: list[AtlasEntry] = []
    for filename, scene in _catalogue():
        report.write(scene, filename)
        entries.append(AtlasEntry(filename, scene.title, scene.subtitle))
    entries.extend(_stage_entries(output_dir))
    report.write_atlas(entries)


def _stages() -> tuple[tuple, ...]:
    """Every stage example: what builds it, what it is called, and its card."""
    return (
        (bt.bogdanov_takens_stage, bt.STAGE_FILENAME, bt.STAGE_ENTRY),
        (ring.ferroelectric_ring_stage, ring.STAGE_FILENAME, ring.STAGE_ENTRY),
        (vdp.van_der_pol_stage, vdp.STAGE_FILENAME, vdp.STAGE_ENTRY),
        (snic.snic_stage, snic.STAGE_FILENAME, snic.STAGE_ENTRY),
        (fold.fold_of_cycles_stage, fold.STAGE_FILENAME, fold.STAGE_ENTRY),
        (hopf.hopf_stage, hopf.STAGE_FILENAME, hopf.STAGE_ENTRY),
        (
            bautin.extended_bautin_stage,
            bautin.STAGE_FILENAME,
            bautin.STAGE_ENTRY,
        ),
        (zerohopf.zero_hopf_stage, zerohopf.STAGE_FILENAME, zerohopf.STAGE_ENTRY),
        (hopfhopf.hopf_hopf_stage, hopfhopf.STAGE_FILENAME, hopfhopf.STAGE_ENTRY),
        (
            shilnikov.shilnikov_stage,
            shilnikov.STAGE_FILENAME,
            shilnikov.STAGE_ENTRY,
        ),
    )


def _stage_entries(output_dir: str) -> list[AtlasEntry]:
    """Write the playable pages with their own renderer; they share the atlas."""
    report = PlotReport(StageRenderer(), output_dir)
    entries = []
    for build, filename, entry in _stages():
        report.write(build(), filename)
        entries.append(entry)
    return entries


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plots")
