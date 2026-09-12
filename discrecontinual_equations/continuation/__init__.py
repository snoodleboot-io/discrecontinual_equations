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

"""Numerical continuation and codim-1 bifurcation detection for equilibria."""

from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.center_manifold import (
    CenterManifold,
    EigenvalueSplit,
    SpectralSplit,
    TaylorCenterManifold,
)
from discrecontinual_equations.continuation.codim2 import Codim2Point
from discrecontinual_equations.continuation.codim2_builder import (
    Codim2Driver,
    Codim2Result,
    CurveSeed,
)
from discrecontinual_equations.continuation.codim2_config import Codim2Config
from discrecontinual_equations.continuation.codim3 import Codim3Point
from discrecontinual_equations.continuation.codim3_builder import (
    Codim3Driver,
    Codim3Result,
    Codim3Seed,
)
from discrecontinual_equations.continuation.codim3_config import Codim3Config
from discrecontinual_equations.continuation.components import ContinuationComponents
from discrecontinual_equations.continuation.connecting_orbit import (
    ConnectingOrbit,
    HeteroclinicOrbit,
    HomoclinicOrbit,
    MeshSpec,
    OrbitSolution,
    Terminus,
)
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.continuation.continuation_point import ContinuationPoint
from discrecontinual_equations.continuation.continuer import Continuer
from discrecontinual_equations.continuation.cycle_continuation import (
    CycleBifurcation,
    CycleContinuation,
    CyclePoint,
    CycleSeed,
    classify_transition,
)
from discrecontinual_equations.continuation.deflation import (
    DeflatedSolver,
    DeflationSettings,
)
from discrecontinual_equations.continuation.diagram import (
    BifurcationDiagram,
    StabilitySegment,
)
from discrecontinual_equations.continuation.exploration import (
    BifurcationExplorer,
    DetectedBifurcation,
    ExplorationConfig,
    ExplorationNode,
)
from discrecontinual_equations.continuation.family_builder import (
    FamilyDriver,
    FamilySlice,
    ParameterFamily,
)
from discrecontinual_equations.continuation.family_config import FamilyConfig
from discrecontinual_equations.continuation.homoclinic_curve import (
    HomoclinicCurve,
    HomoclinicCurvePoint,
)
from discrecontinual_equations.continuation.homoclinic_shooting import (
    Departure,
    HomoclinicShooting,
    ReturnSettings,
)
from discrecontinual_equations.continuation.lyapunov import (
    LyapunovSettings,
    LyapunovSpectrum,
    StochasticLyapunovSettings,
    StochasticSystem,
    lyapunov_spectrum,
    stochastic_lyapunov,
)
from discrecontinual_equations.continuation.manifold import (
    ManifoldChart,
    ManifoldSelection,
    StableManifold,
    TaylorManifold,
    UnstableManifold,
)
from discrecontinual_equations.continuation.periodic_orbit import (
    AdaptivePeriodicOrbit,
    AnalyticPeriodicOrbit,
    HermiteSimpsonOrbit,
    PeriodicOrbit,
    PeriodicOrbitSolution,
    ResolutionLevel,
    ResolutionSettings,
    ResolutionStudy,
    ResolvedCycle,
    RobustPeriodicOrbit,
)
from discrecontinual_equations.continuation.shilnikov import (
    SaddleFocus,
    classify_saddle_focus,
)
from discrecontinual_equations.continuation.snic import (
    SnicCharacterization,
    characterize_snic,
)
from discrecontinual_equations.continuation.stochastic import (
    DensityModes,
    Ito,
    LyapunovExponent,
    NoiseConvention,
    StationaryDensity,
    Stratonovich,
)
from discrecontinual_equations.continuation.symmetry import (
    FourierReduction,
    cyclic_action,
    equivariance_defect,
    fourier_reduce,
)

__all__ = [
    "AdaptivePeriodicOrbit",
    "AnalyticPeriodicOrbit",
    "BifurcationDiagram",
    "BifurcationExplorer",
    "Branch",
    "CenterManifold",
    "Codim2Config",
    "Codim2Driver",
    "Codim2Point",
    "Codim2Result",
    "Codim3Config",
    "Codim3Driver",
    "Codim3Point",
    "Codim3Result",
    "Codim3Seed",
    "ConnectingOrbit",
    "ContinuationComponents",
    "ContinuationConfig",
    "ContinuationPoint",
    "Continuer",
    "ContinuerBuilder",
    "CurveSeed",
    "CycleBifurcation",
    "CycleContinuation",
    "CyclePoint",
    "CycleSeed",
    "DeflatedSolver",
    "DeflationSettings",
    "DensityModes",
    "Departure",
    "DetectedBifurcation",
    "EigenvalueSplit",
    "ExplorationConfig",
    "ExplorationNode",
    "FamilyConfig",
    "FamilyDriver",
    "FamilySlice",
    "FourierReduction",
    "HermiteSimpsonOrbit",
    "HeteroclinicOrbit",
    "HomoclinicCurve",
    "HomoclinicCurvePoint",
    "HomoclinicOrbit",
    "HomoclinicShooting",
    "Ito",
    "LyapunovExponent",
    "LyapunovSettings",
    "LyapunovSpectrum",
    "ManifoldChart",
    "ManifoldSelection",
    "MeshSpec",
    "NoiseConvention",
    "OrbitSolution",
    "ParameterFamily",
    "PeriodicOrbit",
    "PeriodicOrbitSolution",
    "ResolutionLevel",
    "ResolutionSettings",
    "ResolutionStudy",
    "ResolvedCycle",
    "ReturnSettings",
    "RobustPeriodicOrbit",
    "SaddleFocus",
    "SnicCharacterization",
    "SpectralSplit",
    "StabilitySegment",
    "StableManifold",
    "StationaryDensity",
    "StochasticLyapunovSettings",
    "StochasticSystem",
    "Stratonovich",
    "TaylorCenterManifold",
    "TaylorManifold",
    "Terminus",
    "UnstableManifold",
    "characterize_snic",
    "classify_saddle_focus",
    "classify_transition",
    "cyclic_action",
    "equivariance_defect",
    "fourier_reduce",
    "lyapunov_spectrum",
    "stochastic_lyapunov",
]
