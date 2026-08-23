"""Pytest bootstrap for the class registry.

In the installed library the registry is populated by the package root __init__
via sweet_tea's ``fill_registry`` (that file is not modified here). This conftest
reproduces the same automatic package scan for the test environment - it is the
auto-discovery mechanism, not manual per-class registration.
"""

from pathlib import Path

from sweet_tea.registry import Registry

import discrecontinual_equations

Registry.fill_registry(
    path=str(Path(discrecontinual_equations.__file__).parent),
    module="discrecontinual_equations",
    exclude=["*.tests", "*.examples", "*.plot"],
)
