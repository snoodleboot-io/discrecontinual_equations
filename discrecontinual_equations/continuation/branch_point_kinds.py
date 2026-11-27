"""The kinds a detected branch point is reported under.

A branch point is asked for as ``"branch_point"``, but the detector reports the
kind it classified from the branch's normal form (see
:class:`~.detection.BranchPoint`): ``"transcritical"`` when the quadratic
coefficient is non-zero, ``"pitchfork"`` when only the cubic one is, and
``"branch_point"`` when neither settles it. Code that wants every branch point
matches all three through :func:`is_branch_point`; matching the requested name
alone misses each one the detector managed to classify.
"""

BRANCH_POINT_KINDS = ("branch_point", "transcritical", "pitchfork")


def is_branch_point(kind: str) -> bool:
    """Whether ``kind`` is one a detected branch point can be reported under.

    A branch point is where two equilibrium branches cross: the state Jacobian
    has a simple zero eigenvalue with right and left null vectors ``q`` and
    ``p``, and the bordered extended Jacobian is singular as well. On the center
    manifold the field reads ``w' = (a / 2) w^2 + (c / 6) w^3 + ...``. With
    ``a = <p, B(q, q)>`` non-zero the two branches exchange stability through
    each other, a transcritical bifurcation (normal form ``w' = r w - w^2``);
    when ``a`` vanishes, as a reflection symmetry forces, and ``c`` does not, one
    branch splits into three, a pitchfork (``w' = r w -+ w^3``). The detector is
    configured under the one name ``"branch_point"`` and reports whichever of the
    three it could decide, so a match on any single name is incomplete.

    The set is the detector's own: a test reads the literals
    :meth:`~.detection.BranchPoint.refine_kind` returns and fails if they and
    :data:`BRANCH_POINT_KINDS` part ways. It covers equilibria only; cycle
    bifurcations (``"fold_of_cycles"``, ``"period_doubling"``, ...) are not
    branch points in this sense.

    Args:
        kind: A special point's ``kind``, as continuation reported it.

    Returns:
        ``True`` for ``"branch_point"``, ``"transcritical"`` and ``"pitchfork"``;
        ``False`` for every other kind. The match is exact and case-sensitive.

    References:
        - Strogatz, S. H. *Nonlinear Dynamics and Chaos: With Applications to
          Physics, Biology, Chemistry, and Engineering*. Westview Press, 2001,
          ISBN 0-7382-0453-6: the normal forms ``x' = r x - x^2`` (transcritical)
          and ``x' = r x -+ x^3`` (pitchfork), and the pitchfork's conditions
          ``f_x = f_xx = 0``, ``f_xxx != 0`` with odd symmetry. Key
          ``Strogatz2001`` on the references page.
        - The coefficients ``a`` and ``c`` and the classification by them are
          :class:`~.detection.BranchPoint`'s; its docstring carries their
          citation. This function only names the labels it can return.
    """
    return kind in BRANCH_POINT_KINDS
