"""The bundle of collaborators injected into a :class:`~.continuer.Continuer`.

Grouping the wired strategies into one object keeps the driver's constructor to a
single dependency while preserving injection: the composition root builds each
collaborator and assembles them here. It carries no behaviour of its own.
"""

from pydantic import BaseModel, ConfigDict

from discrecontinual_equations.continuation.corrector import Corrector
from discrecontinual_equations.continuation.detection import BifurcationDetector
from discrecontinual_equations.continuation.jacobian import JacobianProvider
from discrecontinual_equations.continuation.localizer import Localizer
from discrecontinual_equations.continuation.point_builder import (
    ContinuationPointBuilder,
)
from discrecontinual_equations.continuation.predictor import Predictor
from discrecontinual_equations.continuation.seed_refiner import SeedRefiner
from discrecontinual_equations.continuation.step_controller import StepController
from discrecontinual_equations.continuation.switching import BranchSwitcher
from discrecontinual_equations.continuation.tangent import TangentComputer


class ContinuationComponents(BaseModel):
    """Passive holder of the abstractions the driver depends on."""

    jacobian: JacobianProvider
    seed_refiner: SeedRefiner
    predictor: Predictor
    corrector: Corrector
    tangent_computer: TangentComputer
    step_controller: StepController
    detectors: list[BifurcationDetector]
    localizer: Localizer
    point_builder: ContinuationPointBuilder
    branch_switcher: BranchSwitcher

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
