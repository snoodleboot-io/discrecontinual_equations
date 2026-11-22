"""Eigenvalues of a matrix product without forming it: the periodic QR algorithm.

The Floquet multipliers of a discretised cycle are the eigenvalues of a product
``A_p ... A_2 A_1`` of per-interval transfer matrices. Forming the product and
calling an eigensolver gives every eigenvalue to an absolute accuracy of about
``eps * ||A_p ... A_1||``, so a multiplier far below the largest one keeps no
relative accuracy at all: on van der Pol at ``mu = 5`` an exact multiplier of
7.7e-38 came out as 1e-15 noise. The periodic QR algorithm (Bojanczyk, Golub and
Van Dooren, Proc. SPIE 1770, 1992) works on the factors instead. Orthogonal
``Q_1 ... Q_p`` bring every factor to upper triangular form, ``Q_{k+1}^H A_k Q_k``
(with ``Q_{p+1} = Q_1``), and each eigenvalue of the product is then the product of
the factors' diagonal entries in one position, each of which carries its own
relative accuracy.

The steps follow Kressner, "The periodic QR algorithm is a disguised QR
algorithm" (Linear Algebra Appl., 2006): Algorithm 2 for the reduction to
periodic Hessenberg form (``A_1 ... A_{p-1}`` upper triangular, ``A_p`` upper
Hessenberg), Algorithm 4 with one shift for the QR step, and criterion (5) on the
Hessenberg factor's subdiagonal for deflation. Kressner shows this performs
exactly the operations the standard QR algorithm performs on the perfectly
shuffled block-cyclic matrix, so it inherits that algorithm's convergence and is
backward stable factor by factor. The arithmetic is complex with a single
Wilkinson shift, so a conjugate pair needs no double-shift bulge; Givens
rotations replace Householder reflections, which for a single shift is the same
transformation.
"""

import numpy as np

# How many QR steps one eigenvalue may take before the iteration gives up. The
# standard algorithm needs two or three; this only guards against a cycle.
_STEPS_PER_EIGENVALUE = 60
# Steps without a deflation after which one exceptional shift is tried.
_EXCEPTIONAL_EVERY = 12
_EXCEPTIONAL_FACTOR = 0.75


class PeriodicSchurError(ArithmeticError):
    """The periodic QR iteration did not converge."""


def _householder(vector: np.ndarray) -> np.ndarray | None:
    """The unit ``v`` for which ``I - 2 v v^H`` takes ``vector`` onto its first axis."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return None
    first = vector[0]
    phase = first / abs(first) if first != 0 else 1.0
    reflector = vector.astype(complex)
    reflector[0] += phase * norm
    return reflector / np.linalg.norm(reflector)


def _reflect_rows(matrix: np.ndarray, start: int, reflector: np.ndarray) -> None:
    """``matrix[start:] <- (I - 2 v v^H) matrix[start:]``, in place."""
    rows = matrix[start:]
    rows -= 2.0 * np.outer(reflector, reflector.conj() @ rows)


def _reflect_columns(matrix: np.ndarray, start: int, reflector: np.ndarray) -> None:
    """``matrix[:, start:] <- matrix[:, start:] (I - 2 v v^H)``, in place."""
    columns = matrix[:, start:]
    columns -= 2.0 * np.outer(columns @ reflector, reflector.conj())


def _hessenberg_triangular(factors: np.ndarray) -> None:
    """Kressner's Algorithm 2, in place: triangular factors, ``A_p`` Hessenberg.

    Column ``l`` of each triangular factor is cleared below the diagonal by a
    reflection from the left, which the next factor absorbs from the right; the
    Hessenberg factor is cleared below its subdiagonal and passes its reflection
    round to ``A_1``. A reflection acting on rows ``l`` onward mixes only columns
    ``l`` onward of the factor it is passed to, so the columns already cleared
    stay cleared.
    """
    count, size = factors.shape[0], factors.shape[1]
    for column in range(size - 1):
        for index in range(count - 1):
            reflector = _householder(factors[index, column:, column])
            if reflector is not None:
                _reflect_rows(factors[index], column, reflector)
                _reflect_columns(factors[index + 1], column, reflector)
                factors[index, column + 1 :, column] = 0.0
        if column + 2 <= size - 1:
            reflector = _householder(factors[-1, column + 1 :, column])
            if reflector is not None:
                _reflect_rows(factors[-1], column + 1, reflector)
                _reflect_columns(factors[0], column + 1, reflector)
                factors[-1, column + 2 :, column] = 0.0


def _givens(first: complex, second: complex) -> np.ndarray:
    """The unitary ``G`` with ``G [first, second]^T = [r, 0]^T``."""
    norm = float(np.hypot(abs(first), abs(second)))
    if norm == 0.0:
        return np.eye(2, dtype=complex)
    if first == 0:
        cosine, sine = 0.0, np.conj(second) / abs(second)
    else:
        phase = first / abs(first)
        cosine, sine = abs(first) / norm, phase * np.conj(second) / norm
    return np.array([[cosine, sine], [-np.conj(sine), cosine]], dtype=complex)


def _rotate(window: np.ndarray, index: int, row: int, rotation: np.ndarray) -> None:
    """``A_index <- G A_index`` on rows ``row, row+1``; ``A_{index+1} <- A G^H``.

    Rotating rows of one factor and the same columns of the next leaves the
    product unchanged up to the similarity ``G P G^H``; the next factor after the
    last is the first.
    """
    pair = slice(row, row + 2)
    window[index, pair, :] = rotation @ window[index, pair, :]
    following = (index + 1) % window.shape[0]
    window[following, :, pair] = window[following, :, pair] @ rotation.conj().T


def _trailing_block(window: np.ndarray) -> np.ndarray:
    """The product's trailing ``2 x 2``, from the trailing columns of the factors."""
    size = window.shape[1]
    columns = np.eye(size, dtype=complex)[:, size - 2 :]
    for factor in window:
        columns = factor @ columns
    return columns[size - 2 :]


def _shift(window: np.ndarray, *, exceptional: bool) -> complex:
    """Wilkinson's shift: the trailing block's eigenvalue nearer its last entry.

    The block is scaled by a power of two to unit size first. Its determinant
    is the product of two eigenvalues, which leaves the floating-point range
    for two below 1e-154 or above 1e154 - a cycle with two strongly
    contracting directions has those - and a zero or infinite determinant
    gives a shift that never converges. A power of two scales exactly, so
    every shift that was representable before is the same to the last bit.
    """
    block = _trailing_block(window)
    if exceptional:
        return complex(block[1, 1] + _EXCEPTIONAL_FACTOR * abs(block[1, 0]))
    largest = float(np.max(np.abs(block)))
    scale = 1.0
    if np.isfinite(largest) and largest > 0.0:
        scale = float(np.ldexp(1.0, np.frexp(largest)[1]))
        block = block / scale
    half_trace = 0.5 * (block[0, 0] + block[1, 1])
    determinant = block[0, 0] * block[1, 1] - block[0, 1] * block[1, 0]
    root = np.sqrt(half_trace * half_trace - determinant)
    candidates = (half_trace + root, half_trace - root)
    shift = scale * min(candidates, key=lambda value: abs(value - block[1, 1]))
    return complex(shift) if np.isfinite(shift) else 0j


def _qr_step(window: np.ndarray, *, exceptional: bool) -> None:
    """One implicitly shifted periodic QR step on an unreduced window (Algorithm 4).

    The first rotation is the one the shifted product's first column asks for,
    ``(P - s I) e_1``, whose two nonzero entries need only the first diagonal of
    the triangular factors. It disturbs ``A_1`` below its diagonal; each factor in
    turn is restored by a rotation from the left that the next one absorbs, and
    the disturbance reaches the Hessenberg factor one row lower, until it falls
    off the bottom.
    """
    count, size = window.shape[0], window.shape[1]
    hessenberg = count - 1
    leading = complex(np.prod(window[:hessenberg, 0, 0]))
    shift = _shift(window, exceptional=exceptional)
    first = window[hessenberg, 0, 0] * leading - shift
    second = window[hessenberg, 1, 0] * leading
    for row in range(size - 1):
        if row == 0:
            rotation = _givens(first, second)
        else:
            rotation = _givens(
                window[hessenberg, row, row - 1],
                window[hessenberg, row + 1, row - 1],
            )
        _rotate(window, hessenberg, row, rotation)
        if row > 0:
            window[hessenberg, row + 1, row - 1] = 0.0
        for index in range(hessenberg):
            rotation = _givens(window[index, row, row], window[index, row + 1, row])
            _rotate(window, index, row, rotation)
            window[index, row + 1, row] = 0.0


def _diagonal_products(factors: np.ndarray) -> np.ndarray:
    """Each position's product of diagonal entries, summed in logarithms.

    A product of a thousand factors can leave the floating-point range on the
    way even when its value does not; the logarithm cannot.
    """
    diagonals = np.diagonal(factors, axis1=1, axis2=2)
    magnitudes = np.abs(diagonals)
    zero = np.any(magnitudes == 0.0, axis=0)
    logarithm = np.sum(np.log(np.where(magnitudes == 0.0, 1.0, magnitudes)), axis=0)
    angle = np.sum(np.angle(diagonals), axis=0)
    values = np.exp(logarithm + 1j * angle)
    values[zero] = 0.0
    return values


def periodic_eigenvalues(factors: np.ndarray) -> np.ndarray:
    """Eigenvalues of ``factors[-1] @ ... @ factors[0]``, without forming the product.

    ``factors`` has shape ``(p, n, n)``; ``factors[0]`` acts first. The result is
    complex, in no particular order, and each eigenvalue is accurate relative to
    its own size wherever the factors determine it so - which a strongly
    contracting direction of a cycle's flow does and the explicit product does not.
    Raises :class:`PeriodicSchurError` if the iteration does not converge.
    """
    work = np.array(factors, dtype=complex)
    if work.ndim != 3 or work.shape[1] != work.shape[2]:  # noqa: PLR2004 (a stack)
        message = f"factors must have shape (p, n, n), got {work.shape}"
        raise ValueError(message)
    size = work.shape[1]
    _hessenberg_triangular(work)
    hessenberg = work[-1]
    epsilon = np.finfo(float).eps
    high = size - 1
    steps = 0
    since = 0
    while high > 0:
        low = 0
        for row in range(high, 0, -1):
            beside = abs(hessenberg[row - 1, row - 1]) + abs(hessenberg[row, row])
            if abs(hessenberg[row, row - 1]) <= epsilon * beside:
                hessenberg[row, row - 1] = 0.0
                low = row
                break
        if low == high:
            high -= 1
            since = 0
            continue
        steps += 1
        since += 1
        if steps > _STEPS_PER_EIGENVALUE * size:
            message = f"periodic QR did not converge in {steps - 1} steps"
            raise PeriodicSchurError(message)
        window = work[:, low : high + 1, low : high + 1]
        _qr_step(window, exceptional=since % _EXCEPTIONAL_EVERY == 0)
    return _diagonal_products(work)
