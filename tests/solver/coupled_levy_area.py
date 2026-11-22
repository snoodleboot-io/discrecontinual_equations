"""A Levy-area approximation and a much finer reference, coupled pathwise.

The sampler's pathwise error can only be measured against a reference built from
the same randomness. This is the construction of Kastner and Rossler (2022,
Section 5): draw the Fourier coefficients ``X_k``, ``Y_k`` up to a large ``K``
once; the reference is the ``K``-term Mrongowius-Rossler area; the ``n``-term
approximation uses the first ``n`` coefficients and, for its two remainder
terms, the standard normals recovered from the coefficients beyond ``n``
(Mrongowius and Rossler (21) and (24)):

* ``Psi_1 = (sum_{n<k<=K} X_k / k + sqrt(tail(K)) Psi_1^K) / sqrt(tail(n))``,
  which is exactly ``N(0, I)``;
* for two drivers, ``Psi_2 = R_2 / sqrt(V)`` with ``R_2`` the reference's second
  remainder beyond ``n`` and ``V = h^2 / (4 pi^2) sum_{n<k<=K} (X_1k^2 + X_2k^2)
  / k^2 + h^2 / (2 pi^2) tail(K)`` its conditional variance given the ``X``:
  ``N(0, 1)`` and independent of the ``X``.

The two then differ only in the second remainder, which is the error the
algorithm makes. ``tail(n) = sum_{k>n} 1/k^2`` is computed here by the trigamma
function, independently of the code under test.
"""

import numpy as np
from scipy import special

from discrecontinual_equations.solver.stochastic.wiener import LevyAreaSampler


def tail(terms: int) -> float:
    """``sum_{k > terms} 1 / k^2``."""
    return float(special.polygamma(1, terms + 1))


def coupled_areas(
    generator: np.random.Generator,
    delta_w: np.ndarray,
    step: float,
    terms: int,
    reference_terms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``(approximation, reference)`` areas for increments ``delta_w``, ``(s, 2)``."""
    samples = delta_w.shape[0]
    x = generator.standard_normal((samples, reference_terms, 2))
    y = generator.standard_normal((samples, reference_terms, 2))
    psi1 = generator.standard_normal((samples, 2))
    psi2 = generator.standard_normal((samples, 2, 2))
    reference = LevyAreaSampler.assemble(delta_w, step, x, y, psi1, psi2)

    k = np.arange(1, reference_terms + 1)
    beyond = slice(terms, reference_terms)
    weights = 1.0 / k[beyond]
    first = (
        np.einsum("k,skm->sm", weights, x[:, beyond])
        + np.sqrt(tail(reference_terms)) * psi1
    ) / np.sqrt(tail(terms))
    remainder = (
        step
        / (2 * np.pi)
        * np.einsum(
            "k,sk->s",
            weights,
            x[:, beyond, 0] * y[:, beyond, 1] - x[:, beyond, 1] * y[:, beyond, 0],
        )
        + step / (np.sqrt(2) * np.pi) * np.sqrt(tail(reference_terms)) * psi2[:, 0, 1]
    )
    variance = step**2 / (4 * np.pi**2) * np.einsum(
        "k,sk->s",
        weights**2,
        x[:, beyond, 0] ** 2 + x[:, beyond, 1] ** 2,
    ) + step**2 / (2 * np.pi**2) * tail(reference_terms)
    second = np.zeros((samples, 2, 2))
    second[:, 0, 1] = remainder / np.sqrt(variance)
    approximation = LevyAreaSampler.assemble(
        delta_w,
        step,
        x[:, :terms],
        y[:, :terms],
        first,
        second,
    )
    return approximation, reference
