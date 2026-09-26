"""Filling the component registry, once, for every example that needs it.

The continuation builders resolve their pieces through sweet_tea's registry, so
it has to be filled before any of them runs. Four separate example modules had
their own byte-identical copy of the call - which is the kind of duplication
that stays correct right up until the arguments need to change.

Not named ``registry`` so there is no chance of confusion with
:mod:`sweet_tea.registry`, which this wraps.
"""

from pathlib import Path

from sweet_tea.registry import Registry

import discrecontinual_equations

_registered = False


def ensure_registry() -> None:
    """Fill the component registry once; the continuation builders need it."""
    global _registered  # noqa: PLW0603 (module-level guard)
    if _registered:
        return
    Registry.fill_registry(
        path=str(Path(discrecontinual_equations.__file__).parent),
        module="discrecontinual_equations",
        exclude=["*.tests", "*.examples", "*.plot"],
    )
    _registered = True
