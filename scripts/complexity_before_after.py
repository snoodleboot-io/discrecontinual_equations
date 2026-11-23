"""Refit the before/after scaling runs that back docs/development/complexity.md.

Each ticket's run lives outside the repository, in the main checkout's git-ignored
``outputs/complexity/before-after/<TICKET>/``. Most of those folders hold a CSV with
the columns ``algorithm, size_param, size, variant, seconds, peak_bytes`` and,
where CPU time was recorded, ``cpu_seconds``; several rounds per size are allowed.

For every ``(algorithm, variant)`` this takes the best (smallest) wall and CPU time
at each size and the largest traced peak, fits ``log(cost) = slope log(size) + c``
by least squares, and prints one Markdown row per pair, then the after/before
ratios at the largest size both variants reached. It replaces the per-folder
``summarize.py`` scripts, which fitted every round rather than the best of each
size (DEQ-50's ran its two passes as ten points), and adds the CPU column and the
ratios.

Usage::

    python scripts/complexity_before_after.py outputs/.../DEQ-49/deq49.csv [...]
"""

import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_MINIMUM_POINTS = 2


@dataclass
class _Series:
    """Every recorded value of one ``(algorithm, variant)`` pair, by size."""

    wall: dict[float, list[float]] = field(default_factory=lambda: defaultdict(list))
    cpu: dict[float, list[float]] = field(default_factory=lambda: defaultdict(list))
    peak: dict[float, list[float]] = field(default_factory=lambda: defaultdict(list))


def _positive(text: str | None) -> float | None:
    """A finite positive number, or None for blanks, ``nan`` and refusals."""
    try:
        value = float(text) if text is not None else math.nan
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def slope(sizes: list[float], costs: list[float]) -> float:
    """Least-squares slope of ``log(cost)`` against ``log(size)``."""
    xs = [math.log(size) for size in sizes]
    ys = [math.log(cost) for cost in costs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    spread = sum((x - mean_x) ** 2 for x in xs)
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / (
        spread
    )


def best(values: dict[float, list[float]], *, largest: bool = False) -> dict:
    """One value per size: the smallest (times) or the largest (peaks)."""
    pick = max if largest else min
    return {size: pick(found) for size, found in values.items() if found}


def read(path: Path) -> dict[tuple[str, str, str], _Series]:
    """Group one CSV's rows by ``(algorithm, variant, size_param)``."""
    series: dict[tuple[str, str, str], _Series] = defaultdict(_Series)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            size = _positive(row.get("size"))
            if size is None:
                continue
            key = (row["algorithm"], row["variant"], row.get("size_param", ""))
            for column, store in (
                ("seconds", series[key].wall),
                ("cpu_seconds", series[key].cpu),
                ("peak_bytes", series[key].peak),
            ):
                value = _positive(row.get(column))
                if value is not None:
                    store[size].append(value)
    return series


def _fit(values: dict[float, float]) -> str:
    if len(values) < _MINIMUM_POINTS:
        return "-"
    sizes = sorted(values)
    return f"{slope(sizes, [values[size] for size in sizes]):.2f}"


def _last(values: dict[float, float], scale: float, unit: str) -> str:
    if not values:
        return "-"
    return f"{values[max(values)] / scale:.4g} {unit}"


def report(path: Path) -> list[str]:
    """Markdown rows for one CSV: fits per pair, then after/before ratios."""
    series = read(path)
    lines = [
        f"### {path}",
        "",
        (
            "| algorithm | variant | param | sizes | wall slope | CPU slope "
            "| memory slope | wall at max | CPU at max | peak at max |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (algorithm, variant, parameter), found in series.items():
        wall = best(found.wall)
        cpu = best(found.cpu)
        peak = best(found.peak, largest=True)
        sizes = sorted(set(wall) | set(peak))
        listed = ",".join(f"{size:g}" for size in sizes)
        lines.append(
            f"| {algorithm} | {variant} | {parameter} | {listed} | {_fit(wall)} "
            f"| {_fit(cpu)} | {_fit(peak)} | {_last(wall, 1.0, 's')} "
            f"| {_last(cpu, 1.0, 's')} | {_last(peak, 1e6, 'MB')} |",
        )
    lines += ["", "| algorithm | size | after/before wall | after/before CPU |"]
    lines.append("|---|---|---|---|")
    for algorithm, _variant, parameter in dict.fromkeys(
        (key[0], "", key[2]) for key in series
    ):
        before = series.get((algorithm, "before", parameter))
        after = series.get((algorithm, "after", parameter))
        if before is None or after is None:
            continue
        common = sorted(set(best(before.wall)) & set(best(after.wall)))
        if not common:
            continue
        size = common[-1]
        wall = best(after.wall)[size] / best(before.wall)[size]
        cpu_before, cpu_after = best(before.cpu), best(after.cpu)
        cpu = (
            f"{cpu_after[size] / cpu_before[size]:.3g}"
            if size in cpu_before and size in cpu_after
            else "-"
        )
        lines.append(f"| {algorithm} | {size:g} | {wall:.3g} | {cpu} |")
    return [*lines, ""]


def main(paths: list[str]) -> None:
    """Print the report for every CSV named on the command line."""
    for name in paths:
        sys.stdout.write("\n".join(report(Path(name))) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
