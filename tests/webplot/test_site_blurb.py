"""The site index's line about the atlas counts its films from the catalogue."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_the_atlas_blurb_counts_the_catalogues_films():
    # It said "ten" while the atlas played thirteen.
    sys.path.insert(0, str(ROOT / "examples"))
    try:
        import build_sites  # noqa: PLC0415 (a script, not a package)

        from examples.continuation.webplot_examples import (  # noqa: PLC0415
            _films,
            _stages,
        )
    finally:
        sys.path.remove(str(ROOT / "examples"))
    atlas = next(e for e in build_sites.EXAMPLES if e.name == "continuation")
    count = re.search(r"plus (\d+) scrubbable films", atlas.description)
    assert count is not None
    assert int(count.group(1)) == len(_stages()) + len(_films())
    assert "ten" not in atlas.description
