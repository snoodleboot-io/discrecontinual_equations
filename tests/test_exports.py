"""Guard against a package advertising names in ``__all__`` it does not import.

A name listed in ``__all__`` but never imported into the package raises ImportError
only when a caller asks for it by name - the rest of the suite, which imports from
submodules directly, never notices. This test makes that failure loud.
"""

import importlib
from unittest import TestCase

_PACKAGES = [
    "discrecontinual_equations.continuation",
]


class TestPackageExports(TestCase):
    def test_all_names_are_importable(self):
        for name in _PACKAGES:
            module = importlib.import_module(name)
            exported = getattr(module, "__all__", [])
            missing = [entry for entry in exported if not hasattr(module, entry)]
            assert missing == [], f"{name} lists un-importable names: {missing}"
