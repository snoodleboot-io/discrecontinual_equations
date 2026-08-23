"""Detect a saddle-node on an invariant circle (SNIC) from the period divergence.

At a SNIC bifurcation a saddle-node pair of equilibria appears directly on a limit
cycle. Approaching it, the cycle keeps its shape but its period diverges, because
the flow crawls through the region where the equilibria are about to form. The
divergence follows a universal inverse-square-root law,

    T(mu) ~ (mu - mu_c) ** (-1/2),

so ``1 / T**2`` is linear in the parameter and reaches zero at the critical value.
This is the signature that distinguishes a SNIC from a saddle homoclinic orbit,
whose period instead diverges logarithmically (``T ~ -log|mu - mu_c|``) and whose
``1 / T**2`` is therefore markedly non-linear.

:func:`characterize_snic` fits that law to a run of (parameter, period) samples,
returning the extrapolated critical parameter and how well the inverse-square-root
law holds. The equilibria that emerge can be confirmed to lie on the former cycle
with :func:`discrecontinual_equations.continuation.region_analysis.find_equilibria`.
"""

import numpy as np

_LINEARITY_THRESHOLD = 0.999
_EXPONENT_TOLERANCE = 0.08
_SNIC_EXPONENT = -0.5
_SEARCH_SAMPLES = 600


class SnicCharacterization:
    """The outcome of fitting the SNIC period-divergence law to a branch."""

    __slots__ = ["_critical_parameter", "_exponent", "_linearity"]

    def __init__(
        self,
        critical_parameter: float,
        exponent: float,
        linearity: float,
    ) -> None:
        self._critical_parameter = critical_parameter
        self._exponent = exponent
        self._linearity = linearity

    @property
    def critical_parameter(self) -> float:
        """Parameter value where the period diverges."""
        return self._critical_parameter

    @property
    def exponent(self) -> float:
        """Fitted power-law exponent of ``T ~ (mu - mu_c) ** exponent``."""
        return self._exponent

    @property
    def linearity(self) -> float:
        """Coefficient of determination of the power-law (log-log) fit."""
        return self._linearity

    @property
    def is_inverse_square_root(self) -> bool:
        """Whether the divergence follows the inverse-square-root SNIC law.

        True when the period is a clean power law of the distance to threshold and
        its exponent is ``-1/2``; a logarithmic homoclinic divergence fails this.
        """
        return (
            self._linearity > _LINEARITY_THRESHOLD
            and abs(self._exponent - _SNIC_EXPONENT) < _EXPONENT_TOLERANCE
        )


def _power_law_fit(distance: np.ndarray, log_period: np.ndarray) -> tuple[float, float]:
    design = np.vstack([np.log(distance), np.ones_like(distance)]).T
    coefficients, *_ = np.linalg.lstsq(design, log_period, rcond=None)
    predicted = design @ coefficients
    residual = float(np.sum((log_period - predicted) ** 2))
    return float(coefficients[0]), residual


def characterize_snic(
    parameters: list[float],
    periods: list[float],
) -> SnicCharacterization:
    """Fit the SNIC period-divergence law to (parameter, period) samples.

    The period is modelled as ``T = C (mu - mu_c) ** p``. The critical parameter
    ``mu_c`` is found by searching for the value that best linearises ``log T``
    against ``log(mu - mu_c)``; the fitted exponent ``p`` (``-1/2`` for a SNIC) and
    the fit quality separate a SNIC from a logarithmic homoclinic divergence.
    """
    parameter = np.asarray(parameters, dtype=float)
    log_period = np.log(np.asarray(periods, dtype=float))
    lowest = float(np.min(parameter))
    span = float(np.max(parameter) - lowest) or 1.0
    lower, upper = lowest - 4.0 * span, lowest - 1.0e-4 * span
    best_exponent, best_critical, best_residual = _SNIC_EXPONENT, lowest, np.inf
    for _ in range(2):  # coarse sweep, then refine around the best candidate
        for critical in np.linspace(lower, upper, _SEARCH_SAMPLES):
            exponent, residual = _power_law_fit(parameter - critical, log_period)
            if residual < best_residual:
                best_exponent = exponent
                best_critical = critical
                best_residual = residual
        window = (upper - lower) / _SEARCH_SAMPLES
        lower = best_critical - window
        upper = min(best_critical + window, lowest - 1.0e-9)
    total = float(np.sum((log_period - log_period.mean()) ** 2))
    linearity = 1.0 - best_residual / total if total > 0.0 else 0.0
    return SnicCharacterization(best_critical, best_exponent, linearity)
