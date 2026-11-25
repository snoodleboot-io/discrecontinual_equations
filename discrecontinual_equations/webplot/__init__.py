# Copyright 2025 snoodleboot, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained D3.js pages for continuation and bifurcation results.

Every page is one HTML file with D3 inlined, so it opens from disk with no
server and no network. A page is built in two steps that never mix: a *builder*
turns the library's results into a render-agnostic value object, and a
*renderer* turns that object into HTML. The value objects hold no drawing code,
so they are built and unit-tested without a browser.

There are three kinds of page:

* a **still**, a :class:`Scene` built by the functions of
  :mod:`~discrecontinual_equations.webplot.scene_builder` (a bifurcation
  diagram, a two- or three-parameter curve, a family surface, an exploration
  tree) and drawn by :class:`D3Renderer`;
* a **stage** (a *film*), a :class:`StageScene` built by :func:`stage_scene`
  from a continued branch and its cycles: one :class:`Frame` per parameter value,
  with the equilibria, their eigenvalues, saddle manifolds and cycles, which the
  page plays as motion with particles flowing through the field. It is drawn by
  :class:`StageRenderer`, and a stochastic stage (paths, densities, the top
  Lyapunov exponent) by the stochastic renderer;
* a **manifold film**, a :class:`ManifoldFilm` of growing invariant-manifold
  rings, drawn by :class:`ManifoldFilmRenderer`.

:class:`PlotReport` writes pages into a directory and an ``index.html`` atlas
of :class:`AtlasEntry` cards linking them. ``examples/build_sites.py`` builds
the full atlas from every example; see the "Atlas pages and films" guide.
"""

from discrecontinual_equations.webplot.manifold_film import (
    FilmCurve,
    FilmMarker,
    ManifoldFilm,
    film_payload,
)
from discrecontinual_equations.webplot.manifold_film_renderer import (
    ManifoldFilmRenderer,
)
from discrecontinual_equations.webplot.renderer import D3Renderer, HtmlRenderer
from discrecontinual_equations.webplot.report import AtlasEntry, PlotReport
from discrecontinual_equations.webplot.scene import (
    Axes,
    Gradient,
    Locus,
    Region,
    Scene,
    SceneExtras,
    Series,
    Surface,
    Tree,
    TreeNode,
)
from discrecontinual_equations.webplot.scene_builder import (
    bifurcation_scene,
    codim2_scene,
    codim3_scene,
    curve_family_scene,
    curve_scene,
    exploration_diagram,
    exploration_skeleton,
    exploration_tree,
    family_scene,
    surface_scene,
    with_equation,
    with_regions,
)
from discrecontinual_equations.webplot.stage import (
    BranchPoint,
    Cycle,
    CycleBranch,
    Equilibrium,
    Frame,
    Lattice,
    Manifold,
    SpecialPoint,
    StageScene,
    StageSystem,
    Term,
    Timeline,
    View,
    evaluate_terms,
)
from discrecontinual_equations.webplot.stage_builder import (
    Continued,
    Film,
    StageSettings,
    stage_scene,
)
from discrecontinual_equations.webplot.stage_renderer import StageRenderer

__all__ = [
    "AtlasEntry",
    "Axes",
    "BranchPoint",
    "Continued",
    "Cycle",
    "CycleBranch",
    "D3Renderer",
    "Equilibrium",
    "Film",
    "FilmCurve",
    "FilmMarker",
    "Frame",
    "Gradient",
    "HtmlRenderer",
    "Lattice",
    "Locus",
    "Manifold",
    "ManifoldFilm",
    "ManifoldFilmRenderer",
    "PlotReport",
    "Region",
    "Scene",
    "SceneExtras",
    "Series",
    "SpecialPoint",
    "StageRenderer",
    "StageScene",
    "StageSettings",
    "StageSystem",
    "Surface",
    "Term",
    "Timeline",
    "Tree",
    "TreeNode",
    "View",
    "bifurcation_scene",
    "codim2_scene",
    "codim3_scene",
    "curve_family_scene",
    "curve_scene",
    "evaluate_terms",
    "exploration_diagram",
    "exploration_skeleton",
    "exploration_tree",
    "family_scene",
    "film_payload",
    "stage_scene",
    "surface_scene",
    "with_equation",
    "with_regions",
]
