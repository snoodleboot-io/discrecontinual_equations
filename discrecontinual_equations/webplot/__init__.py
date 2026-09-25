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

"""Self-contained D3.js plotting for continuation and bifurcation results."""

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
    Timeline,
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
    "Frame",
    "Gradient",
    "HtmlRenderer",
    "Lattice",
    "Locus",
    "Manifold",
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
    "Timeline",
    "Tree",
    "TreeNode",
    "bifurcation_scene",
    "codim2_scene",
    "codim3_scene",
    "curve_family_scene",
    "curve_scene",
    "exploration_diagram",
    "exploration_skeleton",
    "exploration_tree",
    "family_scene",
    "stage_scene",
    "surface_scene",
    "with_equation",
    "with_regions",
]
