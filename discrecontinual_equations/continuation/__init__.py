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
from discrecontinual_equations.continuation.codim2_normal_form import (
    BogdanovTakensNormalForm,
    BogdanovTakensUnfolding,
    CuspNormalForm,
    HopfHopfNormalForm,
    HopfNormalForm,
    ZeroHopfNormalForm,
    bogdanov_takens_normal_form,
    cusp_normal_form,
    hopf_hopf_normal_form,
    hopf_normal_form,
    zero_hopf_normal_form,
)
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
from discrecontinual_equations.continuation.covariant_lyapunov import (
    AngleDistribution,
    CovariantLyapunov,
    CovariantLyapunovSettings,
    FiniteTimeExponents,
    FlowTangent,
    MapTangent,
    TangentDynamics,
    covariant_lyapunov_vectors,
    stochastic_finite_time_exponents,
)
from discrecontinual_equations.continuation.cycle_codim2 import (
    CycleCodim2Point,
    CycleCurve,
    CycleCurveContinuation,
    CycleCurvePoint,
)
from discrecontinual_equations.continuation.cycle_continuation import (
    CycleBifurcation,
    CycleContinuation,
    CyclePoint,
    CycleSeed,
    ResolvedBranch,
    classify_transition,
    resolved_branch,
)
from discrecontinual_equations.continuation.cycle_normal_form import CycleNormalForm
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
from discrecontinual_equations.continuation.first_passage import (
    Ball,
    EnsembleFirstPassage,
    EnsembleResidence,
    GridFirstPassage,
    HalfSpace,
    MeanFirstPassage,
    ResidenceSample,
    ResidenceTracker,
    TimeSample,
    eyring_kramers_time,
)
from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    GridDensity,
    StationaryFokkerPlanck,
)
from discrecontinual_equations.continuation.geodesic_level_sets import (
    CrochetRound,
    GeodesicLevelSets,
    GeodesicMesh,
    GeodesicSettings,
)
from discrecontinual_equations.continuation.heteroclinic_network import (
    ConnectionOrbit,
    CycleSpectrum,
    HeteroclinicNetwork,
    NetworkConnection,
    NetworkSaddle,
    NotATrajectoryError,
    coordinate_plane,
)
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
    MatrixNoiseSystem,
    StochasticLyapunovSettings,
    StochasticSystem,
    lyapunov_spectrum,
    matrix_noise_lyapunov,
    stochastic_lyapunov,
)
from discrecontinual_equations.continuation.manifold import (
    ManifoldChart,
    ManifoldSelection,
    StableManifold,
    TaylorManifold,
    UnstableManifold,
)
from discrecontinual_equations.continuation.network_continuation import (
    ConnectionBreak,
    NetworkContinuation,
    NetworkPoint,
)
from discrecontinual_equations.continuation.network_stability import (
    CycleStability,
    NetworkCompetition,
    f_index,
    stability_indices,
    transition_matrices,
)
from discrecontinual_equations.continuation.noise import (
    NoiseMatrix,
    NoiseMatrixFunction,
    diffusion_matrix,
    field_at,
    jacobian_at,
    noise_matrix_from,
    state_drift_shift,
    stratonovich_factor,
    variational_drift_shift,
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
from discrecontinual_equations.continuation.quasipotential import (
    CycleAttractor,
    EquilibriumAttractor,
    GeometricMinimumAction,
    GridQuasipotential,
    MinimumActionPath,
    OrderedLineIntegral,
)
from discrecontinual_equations.continuation.residence import Itinerary, Visit
from discrecontinual_equations.continuation.shilnikov import (
    SaddleFocus,
    classify_saddle_focus,
)
from discrecontinual_equations.continuation.slow_manifold import (
    CanardSearch,
    FastSlowSystem,
    FenichelExpansion,
    MaximalCanard,
    SlowManifold,
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
from discrecontinual_equations.continuation.stochastic_threshold import (
    DensityProbe,
    DynamicalThreshold,
    MeanTopExponent,
    ParameterThreshold,
    PhenomenologicalThreshold,
    RadialCrater,
    StochasticSample,
    StochasticScan,
    ThresholdCurve,
    ThresholdPoint,
)
from discrecontinual_equations.continuation.symmetry import (
    FourierReduction,
    cyclic_action,
    equivariance_defect,
    fourier_reduce,
)
from discrecontinual_equations.continuation.topological_modes import (
    KernelDensity,
    ModeCountThreshold,
    PersistencePair,
    PersistenceProbe,
    TopologicalModes,
    superlevel_persistence,
)

__all__ = [
    "AdaptivePeriodicOrbit",
    "AnalyticPeriodicOrbit",
    "AngleDistribution",
    "Ball",
    "BifurcationDiagram",
    "BifurcationExplorer",
    "BogdanovTakensNormalForm",
    "BogdanovTakensUnfolding",
    "Branch",
    "CanardSearch",
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
    "ConnectionBreak",
    "ConnectionOrbit",
    "ContinuationComponents",
    "ContinuationConfig",
    "ContinuationPoint",
    "Continuer",
    "ContinuerBuilder",
    "CovariantLyapunov",
    "CovariantLyapunovSettings",
    "CrochetRound",
    "CurveSeed",
    "CuspNormalForm",
    "CycleAttractor",
    "CycleBifurcation",
    "CycleCodim2Point",
    "CycleContinuation",
    "CycleCurve",
    "CycleCurveContinuation",
    "CycleCurvePoint",
    "CycleNormalForm",
    "CyclePoint",
    "CycleSeed",
    "CycleSpectrum",
    "CycleStability",
    "DeflatedSolver",
    "DeflationSettings",
    "DensityGrid",
    "DensityModes",
    "DensityProbe",
    "Departure",
    "DetectedBifurcation",
    "DynamicalThreshold",
    "EigenvalueSplit",
    "EnsembleFirstPassage",
    "EnsembleResidence",
    "EquilibriumAttractor",
    "ExplorationConfig",
    "ExplorationNode",
    "FamilyConfig",
    "FamilyDriver",
    "FamilySlice",
    "FastSlowSystem",
    "FenichelExpansion",
    "FiniteTimeExponents",
    "FlowTangent",
    "FourierReduction",
    "GeodesicLevelSets",
    "GeodesicMesh",
    "GeodesicSettings",
    "GeometricMinimumAction",
    "GridDensity",
    "GridFirstPassage",
    "GridQuasipotential",
    "HalfSpace",
    "HermiteSimpsonOrbit",
    "HeteroclinicNetwork",
    "HeteroclinicOrbit",
    "HomoclinicCurve",
    "HomoclinicCurvePoint",
    "HomoclinicOrbit",
    "HomoclinicShooting",
    "HopfHopfNormalForm",
    "HopfNormalForm",
    "Itinerary",
    "Ito",
    "KernelDensity",
    "LyapunovExponent",
    "LyapunovSettings",
    "LyapunovSpectrum",
    "ManifoldChart",
    "ManifoldSelection",
    "MapTangent",
    "MatrixNoiseSystem",
    "MaximalCanard",
    "MeanFirstPassage",
    "MeanTopExponent",
    "MeshSpec",
    "MinimumActionPath",
    "ModeCountThreshold",
    "NetworkCompetition",
    "NetworkConnection",
    "NetworkContinuation",
    "NetworkPoint",
    "NetworkSaddle",
    "NoiseConvention",
    "NoiseMatrix",
    "NoiseMatrixFunction",
    "NotATrajectoryError",
    "OrbitSolution",
    "OrderedLineIntegral",
    "ParameterFamily",
    "ParameterThreshold",
    "PeriodicOrbit",
    "PeriodicOrbitSolution",
    "PersistencePair",
    "PersistenceProbe",
    "PhenomenologicalThreshold",
    "RadialCrater",
    "ResidenceSample",
    "ResidenceTracker",
    "ResolutionLevel",
    "ResolutionSettings",
    "ResolutionStudy",
    "ResolvedBranch",
    "ResolvedCycle",
    "ReturnSettings",
    "RobustPeriodicOrbit",
    "SaddleFocus",
    "SlowManifold",
    "SnicCharacterization",
    "SpectralSplit",
    "StabilitySegment",
    "StableManifold",
    "StationaryDensity",
    "StationaryFokkerPlanck",
    "StochasticLyapunovSettings",
    "StochasticSample",
    "StochasticScan",
    "StochasticSystem",
    "Stratonovich",
    "TangentDynamics",
    "TaylorCenterManifold",
    "TaylorManifold",
    "Terminus",
    "ThresholdCurve",
    "ThresholdPoint",
    "TimeSample",
    "TopologicalModes",
    "UnstableManifold",
    "Visit",
    "ZeroHopfNormalForm",
    "bogdanov_takens_normal_form",
    "characterize_snic",
    "classify_saddle_focus",
    "classify_transition",
    "coordinate_plane",
    "covariant_lyapunov_vectors",
    "cusp_normal_form",
    "cyclic_action",
    "diffusion_matrix",
    "equivariance_defect",
    "eyring_kramers_time",
    "f_index",
    "field_at",
    "fourier_reduce",
    "hopf_hopf_normal_form",
    "hopf_normal_form",
    "jacobian_at",
    "lyapunov_spectrum",
    "matrix_noise_lyapunov",
    "noise_matrix_from",
    "resolved_branch",
    "stability_indices",
    "state_drift_shift",
    "stochastic_finite_time_exponents",
    "stochastic_lyapunov",
    "stratonovich_factor",
    "superlevel_persistence",
    "transition_matrices",
    "variational_drift_shift",
    "zero_hopf_normal_form",
]
