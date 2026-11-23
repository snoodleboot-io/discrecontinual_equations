"""Systems whose cycles, and their bifurcations in two parameters, are known exactly.

Each but the last two is a planar system turned about an axis: with
``rho = x^2 + y^2`` the angle advances at ``Omega(rho)`` and the radius obeys a
reduced system, so the cycles are the reduced system's equilibria, their
periods ``2 pi / Omega`` and their nontrivial multipliers ``exp(T lambda)`` for
the reduced eigenvalues ``lambda``. ``Omega`` depends on the radius so that a
fold of cycles carries the Jordan block the periodic normal form assumes (with
``Omega = 1`` the double multiplier at the fold is semisimple and ``b`` is not
defined).

The last two are published models: the Rossler variant of the MATCONT
tutorials and the seasonally forced two-patch predator-prey model of De Witte,
Della Rossa, Govaerts and Kuznetsov (SIADS 12, 2013, eq. 4.2).
"""

from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class First(Parameter, name="first", abbreviation="p1"):
    pass


class Second(Parameter, name="second", abbreviation="p2"):
    pass


class Third(Parameter, name="third", abbreviation="p3"):
    pass


class Coordinate(Variable, name="coordinate", abbreviation="x"):
    pass


class ExtendedBautin(DeterministicFunction):
    """``r' = r (m1 + m2 r^2 + c r^4 - r^6)``, ``theta' = 1 + e r^2``.

    Cycles at the positive roots ``rho`` of ``m1 + m2 rho + c rho^2 - rho^3``; a
    double root is a fold of cycles, so the fold curve is
    ``(m1, m2) = (-2 rho^3 + c rho^2, 3 rho^2 - 2 c rho)`` and its cusp, the
    triple root, sits at ``rho = c / 3``: ``(c^3 / 27, -c^2 / 3)``. In polar
    coordinates the fold's normal form coefficient is ``-R'' Omega / (2
    Omega')``, here ``-rho (2 c - 6 rho)(1 + e rho) / e``.
    """

    C = 1.0
    E = 0.5

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        m1 = self.parameters[0].value
        m2 = self.parameters[1].value
        x, y = point[0], point[1]
        rho = x * x + y * y
        growth = m1 + m2 * rho + self.C * rho * rho - rho * rho * rho
        omega = 1.0 + self.E * rho
        return [growth * x - omega * y, growth * y + omega * x]


class RotatedTakens(DeterministicFunction):
    """A Bogdanov-Takens unfolding in ``(u, z)``, ``u = rho - 1``, turned about z.

    ``u' = 2 (1 + u) z``, ``z' = b1 + u^2 + b2 z + u z``, ``Omega = 1 + E u``.
    Cycles at ``z = 0``, ``u = -+ sqrt(-b1)``: the fold curve is ``b1 = 0`` and
    the torus (Neimark-Sacker) curve ``b1 = -b2^2`` (reduced Hopf at
    ``u = -b2``), with frequency ``omega^2 = -4 (1 + u) u`` and multipliers
    ``exp(+- i omega T)``; they meet at the 1:1 resonance at the origin.
    """

    E = 0.3

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        x, y, z = point[0], point[1], point[2]
        u = x * x + y * y - 1.0
        omega = 1.0 + self.E * u
        return [z * x - omega * y, z * y + omega * x, b1 + u * u + b2 * z + u * z]


class RotatedBautin(DeterministicFunction):
    """The Bautin normal form in ``(U, Z)``, ``U = rho - 1``, turned about z.

    ``U' + i Z' = (b1 + i W0) w + b2 w |w|^2 - w |w|^4`` with ``w = U + i Z``:
    the cycle sits at ``U = Z = 0``, the torus curve is ``b1 = 0`` and its
    Chenciner point is ``b2 = 0``, where the torus's ``Re d`` - linear in
    ``b2`` - changes sign.
    """

    E = 0.3
    W0 = 0.37

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        x, y, z = point[0], point[1], point[2]
        rho = x * x + y * y
        u = rho - 1.0
        size = u * u + z * z
        radial = b1 * u - self.W0 * z + b2 * u * size - u * size * size
        vertical = self.W0 * u + b1 * z + b2 * z * size - z * size * size
        omega = 1.0 + self.E * u
        h = radial / (2.0 * rho)
        return [h * x - omega * y, h * y + omega * x, vertical]


class FoldTorus(DeterministicFunction):
    """A fold of cycles at ``b1 = 0`` and a transverse torus at ``b2 = 0``.

    ``rho' = 2 rho (b1 + u^2)`` with ``u = rho - 1``, and a Hopf block
    ``(p, q)`` at rate ``b2`` and frequency ``W``: the two curves cross at the
    fold-Neimark-Sacker point, the origin.
    """

    E = 0.3
    W = 0.37

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        x, y, p, q = point[0], point[1], point[2], point[3]
        u = x * x + y * y - 1.0
        omega = 1.0 + self.E * u
        h = b1 + u * u
        size = p * p + q * q
        return [
            h * x - omega * y,
            h * y + omega * x,
            b2 * p - self.W * q - p * size,
            self.W * p + b2 * q - q * size,
        ]


class Rossler(DeterministicFunction):
    """MATCONT's tutorial Rossler, ``x' = -y - z, y' = x + A y, z' = B x - C z + x z``.

    (Kuznetsov and Neirynck, tutorials I and III, MATCONT 7.3, 2021.)
    """

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        a = self.parameters[0].value
        b = self.parameters[1].value
        c = self.parameters[2].value
        x, y, z = point[0], point[1], point[2]
        return [-y - z, x + a * y, b * x - c * z + x * z]


class Adaptive(DeterministicFunction):
    """MATCONT's ``adaptx``: ``x' = y, y' = z, z' = -a z - b y - x + x^2``."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        a = self.parameters[0].value
        b = self.parameters[1].value
        x, y, z = point[0], point[1], point[2]
        return [y, z, -a * z - b * y - x + x * x]


class PredatorPrey(DeterministicFunction):
    """De Witte et al. (2013), eq. (4.2); parameters ``eps`` then ``b2``."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        eps = self.parameters[0].value
        b2 = self.parameters[1].value
        b1, gamma, c = 0.4, 0.1, 2.0
        x1, x2, y1, y2, v1, v2 = (point[i] for i in range(6))
        half = x1 + b1 * (1.0 + eps * v1)
        size = v1 * v1 + v2 * v2
        return [
            x1 * (1.0 - x1) - c * x1 * x2 / half,
            -x2 + c * x1 * x2 / half + gamma * (y2 - x2),
            y1 * (1.0 - y1) - c * y1 * y2 / (y1 + b2),
            -y2 + c * y1 * y2 / (y1 + b2) + gamma * (x2 - y2),
            -v2 + v1 * (1.0 - size),
            v1 + v2 * (1.0 - size),
        ]


class ShearedBautin(DeterministicFunction):
    """:class:`ExtendedBautin` in the coordinates ``(X, Y) = (x + S y^2, y)``.

    A nonlinear change of coordinates leaves every cycle, multiplier and normal
    form coefficient as it was, and breaks the rotation symmetry under which
    terms of ``b`` such as ``int <phi*, A v>`` vanish identically.
    """

    S = 0.5

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        m1 = self.parameters[0].value
        m2 = self.parameters[1].value
        x, y = point[0] - self.S * point[1] * point[1], point[1]
        rho = x * x + y * y
        growth = m1 + m2 * rho + ExtendedBautin.C * rho * rho - rho * rho * rho
        omega = 1.0 + ExtendedBautin.E * rho
        dx, dy = growth * x - omega * y, growth * y + omega * x
        return [dx + 2.0 * self.S * y * dy, dy]


class ShearedTakens(DeterministicFunction):
    """:class:`RotatedTakens` in the coordinates ``(x + S y^2, y, z)``.

    The same cycles and normal form coefficients without the rotation
    symmetry, under which terms of ``d`` multiplied by ``a`` vanish.
    """

    S = 0.5

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        x, y, z = point[0] - self.S * point[1] * point[1], point[1], point[2]
        u = x * x + y * y - 1.0
        omega = 1.0 + RotatedTakens.E * u
        dx, dy = z * x - omega * y, z * y + omega * x
        return [dx + 2.0 * self.S * y * dy, dy, b1 + u * u + b2 * z + u * z]


def _theta_forced(x, y, u_rate, z_rate, speed):
    """The field with ``rho' = W U``, ``theta' = W``, ``z' = W Z``, ``u = rho - 1``."""
    rho = x * x + y * y
    g = speed * u_rate / (2.0 * rho)
    return [x * g - y * speed, y * g + x * speed, speed * z_rate]


class ThetaFlip(DeterministicFunction):
    """A flip of cycles with no symmetry and amplitude-dependent speed.

    In the angle ``theta`` the pair ``(u, z)``, ``u = rho - 1``, obeys a
    ``2 pi``-periodic system whose linear part turns half a revolution per lap
    (multipliers ``-exp(2 pi (m +- K))`` on the unforced circle, a flip near
    ``m = -K``), with generic quadratic and cubic terms and a small forcing
    that moves the cycle off the circle. The speed ``W`` depends on ``u`` and
    ``z``, so the period varies with the amplitude and the normal form's
    ``a`` does not vanish, as it does in any forced system. The return map to
    ``theta = 0`` is the ``theta``-flow over ``2 pi``: no section to hit.
    """

    K = 0.3

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        m = self.parameters[0].value
        cubic = self.parameters[1].value
        x, y, z = point[0], point[1], point[2]
        u = x * x + y * y - 1.0
        u_rate = (
            -0.5 * z
            + m * u
            + self.K * (x * u + y * z)
            + 0.05 * y
            + 0.3 * u * u
            - 0.4 * u * z
            + 0.2 * x * z * z
            + cubic * u * u * u
            + 0.1 * y * u * u
        )
        z_rate = (
            0.5 * u
            + m * z
            + self.K * (y * u - x * z)
            + 0.25 * z * z
            + 0.3 * x * u * z
            - 0.2 * z * z * z
            + 0.15 * u * u
            + 0.1 * u * z * z
        )
        speed = 1.0 + 0.4 * u + 0.3 * z + 0.2 * x + 0.1 * y
        return _theta_forced(x, y, u_rate, z_rate, speed)


class ThetaTorus(DeterministicFunction):
    """A Neimark-Sacker of cycles built as :class:`ThetaFlip`, rotating at ``W0``."""

    W0 = 0.37

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        m = self.parameters[0].value
        cubic = self.parameters[1].value
        x, y, z = point[0], point[1], point[2]
        u = x * x + y * y - 1.0
        u_rate = (
            m * u
            - self.W0 * z
            + 0.1 * (x * u - y * z)
            + 0.05 * y
            + 0.3 * u * u
            - 0.2 * u * z
            + 0.4 * z * z
            + cubic * u * u * u
            + 0.2 * u * z * z
            + 0.3 * x * u * u * z
        )
        z_rate = (
            self.W0 * u
            + m * z
            + 0.1 * (y * u + x * z)
            - 0.25 * u * u
            + 0.35 * u * z
            - 0.1 * z * z
            - 0.4 * z * z * z
            + 0.15 * u * u * z
            + 0.2 * y * u * u
        )
        speed = 1.0 + 0.4 * u + 0.3 * z + 0.2 * x + 0.1 * y
        return _theta_forced(x, y, u_rate, z_rate, speed)
