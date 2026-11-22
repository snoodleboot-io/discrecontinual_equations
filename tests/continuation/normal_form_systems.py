"""Systems whose codim-2 normal-form coefficients are known in advance.

Each embedded system is a normal form written in coordinates ``z`` - the critical
block first, then extra stable directions - and handed to the library in
coordinates ``X = T z`` for a random invertible ``T``. Nothing about ``T`` is
visible to the reduction, so the coefficients it returns can only come out right
if the whole chain (eigenvectors, bordered solves, multilinear forms) is right.

The stable directions are driven by the critical ones, so the center manifold is
curved, but they do not feed back into the critical equations: the restricted
dynamics is the written normal form exactly, and its coefficients are the oracle.
The systems with feedback are checked against an independent reference instead
(:mod:`tests.continuation.normal_form_reference`).
"""

import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class First(Parameter, name="First", abbreviation="m1"):
    pass


class Second(Parameter, name="Second", abbreviation="m2"):
    pass


class State(Variable, name="State", abbreviation="s"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


def random_transform(dimension: int, seed: int) -> np.ndarray:
    """A well-conditioned random similarity (condition number below ~30)."""
    generator = np.random.default_rng(seed)
    while True:
        transform = generator.normal(size=(dimension, dimension))
        if np.linalg.cond(transform) < 30.0:
            return transform


class Embedded(DeterministicFunction):
    """A field written in ``z`` and evaluated in ``X = T z``.

    Subclasses set ``transform`` and implement :meth:`core`. The arithmetic is
    written with explicit sums so the field evaluates on jets and series as well
    as on floats.
    """

    transform = np.eye(1)

    def core(self, z: list, mu1: object, mu2: object) -> list:
        raise NotImplementedError

    def eval(
        self,
        point: list,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list:
        transform = type(self).transform
        inverse = np.linalg.inv(transform)
        size = transform.shape[0]
        z = [
            sum(float(inverse[i, k]) * point[k] for k in range(size))
            for i in range(size)
        ]
        mu1 = self.parameters[0].value
        mu2 = self.parameters[1].value
        field = self.core(z, mu1, mu2)
        return [
            sum(float(transform[i, k]) * field[k] for k in range(size))
            for i in range(size)
        ]


class EmbeddedBogdanovTakens(Embedded):
    """x' = y, y' = m1 + m2 y + a x^2 + b x y + 0.3 x s1, with s driven by x.

    The ``x s1`` feedback is cubic on the center manifold, so it cannot move the
    quadratic coefficients ``a`` and ``b``.
    """

    a = 1.0
    b = -1.0

    def core(self, z: list, mu1: object, mu2: object) -> list:
        x, y, s1, s2 = z
        cls = type(self)
        return [
            y,
            mu1 + mu2 * y + cls.a * x * x + cls.b * x * y + 0.3 * x * s1,
            -s1 + x * x + 0.5 * x * y,
            -2.5 * s2 + 0.4 * s1 * x,
        ]


class EmbeddedZeroHopf(Embedded):
    """The fold-Hopf normal form (Kuznetsov 1999, eq. 3.4) with two stable modes.

    ``u' = G200/2 u^2 + G011 |w|^2 + G300/6 u^3 + G111 u |w|^2 + m1``,
    ``w' = (i omega + m2) w + G110 u w + G210/2 u^2 w + G021/2 w |w|^2``,
    with ``w = x + i y``.
    """

    omega = 1.7
    g200 = -1.3
    g011 = 0.9
    g300 = 0.4
    g111 = -0.6
    g110 = complex(0.7, -0.25)
    g210 = complex(-0.35, 0.8)
    g021 = complex(-1.1, 0.45)

    def core(self, z: list, mu1: object, mu2: object) -> list:
        u, x, y, s1, s2 = z
        cls = type(self)
        radius = x * x + y * y
        u_dot = (
            0.5 * cls.g200 * u * u
            + cls.g011 * radius
            + cls.g300 / 6.0 * u * u * u
            + cls.g111 * u * radius
            + mu1
        )
        factor_real = (
            cls.g110.real * u
            + 0.5 * cls.g210.real * u * u
            + 0.5 * cls.g021.real * radius
            + mu2
        )
        factor_imag = (
            cls.omega
            + cls.g110.imag * u
            + 0.5 * cls.g210.imag * u * u
            + 0.5 * cls.g021.imag * radius
        )
        return [
            u_dot,
            factor_real * x - factor_imag * y,
            factor_real * y + factor_imag * x,
            -s1 + u * u + x * x,
            -1.7 * s2 + u * s1,
        ]


class EmbeddedHopfHopf(Embedded):
    """The double-Hopf normal form (Kuznetsov 1999, eq. 3.6) with a stable mode.

    ``w1' = (i omega1 + m1) w1 + G2100/2 w1 |w1|^2 + G1011 w1 |w2|^2``,
    ``w2' = (i omega2 + m2) w2 + G1110 w2 |w1|^2 + G0021/2 w2 |w2|^2``.
    """

    omega1 = 2.0
    omega2 = 1.3
    g2100 = complex(-1.2, 0.3)
    g1011 = complex(0.5, -0.7)
    g1110 = complex(-0.8, 0.2)
    g0021 = complex(0.9, 0.6)

    def core(self, z: list, mu1: object, mu2: object) -> list:
        x1, y1, x2, y2, s = z
        cls = type(self)
        first = x1 * x1 + y1 * y1
        second = x2 * x2 + y2 * y2
        real1 = mu1 + 0.5 * cls.g2100.real * first + cls.g1011.real * second
        imag1 = cls.omega1 + 0.5 * cls.g2100.imag * first + cls.g1011.imag * second
        real2 = mu2 + cls.g1110.real * first + 0.5 * cls.g0021.real * second
        imag2 = cls.omega2 + cls.g1110.imag * first + 0.5 * cls.g0021.imag * second
        return [
            real1 * x1 - imag1 * y1,
            real1 * y1 + imag1 * x1,
            real2 * x2 - imag2 * y2,
            real2 * y2 + imag2 * x2,
            -s + x1 * x1 + x2 * y2,
        ]


class PolynomialField(Embedded):
    """``z' = L z + Q(z) + C(z)`` with dense random quadratic and cubic terms.

    ``L`` carries the critical spectrum, so every term couples every direction:
    the stable modes feed back into the critical ones. Used where the oracle is the
    independent reference rather than the written normal form.
    """

    linear = np.zeros((1, 1))
    quadratic = np.zeros((1, 1, 1))
    cubic = np.zeros((1, 1, 1, 1))

    def core(self, z: list, mu1: object, mu2: object) -> list:  # noqa: ARG002
        cls = type(self)
        size = len(z)
        rows = []
        for i in range(size):
            total = 0.0 * z[0]
            for j in range(size):
                if cls.linear[i, j] != 0.0:
                    total = total + float(cls.linear[i, j]) * z[j]
                for k in range(j, size):
                    if cls.quadratic[i, j, k] != 0.0:
                        total = total + float(cls.quadratic[i, j, k]) * z[j] * z[k]
                    for m in range(k, size):
                        if cls.cubic[i, j, k, m] != 0.0:
                            total = (
                                total
                                + float(cls.cubic[i, j, k, m]) * z[j] * z[k] * z[m]
                            )
            rows.append(total)
        return rows


def polynomial_field(
    linear: np.ndarray,
    seed: int,
    scale: float = 0.6,
) -> type[PolynomialField]:
    """A :class:`PolynomialField` subclass with random terms and a random ``T``."""
    generator = np.random.default_rng(seed)
    size = linear.shape[0]
    quadratic = np.zeros((size, size, size))
    cubic = np.zeros((size, size, size, size))
    for i in range(size):
        for j in range(size):
            for k in range(j, size):
                quadratic[i, j, k] = scale * generator.normal()
                for m in range(k, size):
                    cubic[i, j, k, m] = scale * generator.normal()
    return type(
        f"Polynomial{seed}",
        (PolynomialField,),
        {
            "linear": linear,
            "quadratic": quadratic,
            "cubic": cubic,
            "transform": random_transform(size, seed + 1000),
        },
    )


def build(
    function_type: type[DeterministicFunction],
    size: int,
) -> DifferentialEquation:
    """Wrap a field as an equation with two parameters at zero."""
    parameters = [First(value=0.0), Second(value=0.0)]
    derivative = function_type(
        variables=[State() for _ in range(size)],
        parameters=parameters,
        results=[State() for _ in range(size)],
        time=None,
    )
    return DifferentialEquation(
        variables=[State() for _ in range(size)],
        time=Time(),
        parameters=parameters,
        derivative=derivative,
    )


class CurvedBogdanovTakens(Embedded):
    """A BT whose unfolding is nonlinear in the parameters and moves the manifold.

    ``mu1 = m1 + 0.7 m2^2 + 0.4 m1 m2 + 0.3 m1^2`` and
    ``mu2 = m2 + 0.3 m1 - 0.5 m2^2`` enter the BT block, and ``m2`` also pushes
    the stable mode ``s1``, which feeds back through ``0.3 x s1``: the parameter
    map has a ``beta2^2`` term and the center manifold moves with the parameters,
    so ``K02``, ``delta1`` and ``gamma3`` all matter.
    """

    a = -0.8
    b = 1.6

    def core(self, z: list, mu1: object, mu2: object) -> list:
        x, y, s1, s2 = z
        cls = type(self)
        first = mu1 + 0.7 * mu2 * mu2 + 0.4 * mu1 * mu2 + 0.3 * mu1 * mu1
        second = mu2 + 0.3 * mu1 - 0.5 * mu2 * mu2
        return [
            y,
            first + second * y + cls.a * x * x + cls.b * x * y + 0.3 * x * s1,
            -s1 + x * x + 0.5 * x * y + 0.6 * mu2,
            -2.5 * s2 + 0.4 * s1 * x,
        ]


class StackedBautin(Embedded):
    """``z' = i z + 0.5 z |z|^4`` (so ``eta4 = 0``) with any number of stable modes.

    Each stable mode is driven by the center (``x^2``, ``x y``, ``y^3``) and none
    feeds back, so the restricted dynamics is the written planar one whatever the
    dimension: its second focal value must not depend on how many modes there are.
    """

    def core(self, z: list, mu1: object, mu2: object) -> list:  # noqa: ARG002
        x, y = z[0], z[1]
        radius = x * x + y * y
        drive = 0.5 * radius * radius
        rows = [drive * x - y, x + drive * y]
        for index, mode in enumerate(z[2:]):
            rate = -(1.0 + 0.7 * index)
            rows.append(rate * mode + x * x + 0.5 * (index + 1) * x * y + 0.3 * y**3)
        return rows


class StackedHopfHopf(EmbeddedHopfHopf):
    """``EmbeddedHopfHopf``'s critical block with any number of driven stable modes."""

    def core(self, z: list, mu1: object, mu2: object) -> list:
        rows = super().core([*z[:4], 0.0 * z[0]], mu1, mu2)[:4]
        x1, y1, x2, y2 = z[:4]
        for index, mode in enumerate(z[4:]):
            rate = -(1.2 + 0.5 * index)
            rows.append(rate * mode + x1 * x1 + (index + 1) * x2 * y2 + y1 * x2)
        return rows
