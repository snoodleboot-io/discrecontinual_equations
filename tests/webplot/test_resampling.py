"""Surface-slice resampling in one merge walk (DEQ-53 item 4).

The scans these replace looked for each sample's segment from the start of the
branch, ``O(samples x points)``. They are kept here verbatim as the reference: the
merge walk must return the same list, float for float, on monotone and folded
branches, with repeated points and zero-length segments.
"""

from itertools import pairwise
from unittest import TestCase

import numpy as np

from discrecontinual_equations.webplot.scene_builder import (
    _resample_by_arclength,
    _resample_by_measure,
)


def _scan_by_measure(raw, samples):
    ordered = sorted(raw, key=lambda pair: pair[1])
    low, high = ordered[0][1], ordered[-1][1]
    span = (high - low) or 1.0
    result = []
    for i in range(samples):
        measure = low + span * i / (samples - 1)
        parameter = ordered[-1][0]
        for (p0, m0), (p1, m1) in pairwise(ordered):
            if m1 >= measure:
                width = (m1 - m0) or 1.0
                parameter = p0 + (measure - m0) / width * (p1 - p0)
                break
        result.append((parameter, measure))
    return result


def _scan_by_arclength(raw, samples):
    lengths = [0.0]
    for (x0, z0), (x1, z1) in pairwise(raw):
        lengths.append(lengths[-1] + ((x1 - x0) ** 2 + (z1 - z0) ** 2) ** 0.5)
    total = lengths[-1] or 1.0
    result = []
    for i in range(samples):
        target = total * i / (samples - 1)
        point = raw[-1]
        for j in range(1, len(lengths)):
            if lengths[j] >= target:
                span = lengths[j] - lengths[j - 1] or 1.0
                fraction = (target - lengths[j - 1]) / span
                point = (
                    raw[j - 1][0] + fraction * (raw[j][0] - raw[j - 1][0]),
                    raw[j - 1][1] + fraction * (raw[j][1] - raw[j - 1][1]),
                )
                break
        result.append(point)
    return result


def _branches():
    generator = np.random.default_rng(53)
    branches = []
    for points in (2, 3, 7, 40, 301):
        steps = generator.uniform(0.0, 1.0, points)
        monotone = [
            (float(p), float(m))
            for p, m in zip(
                generator.normal(size=points),
                np.cumsum(steps),
                strict=True,
            )
        ]
        angle = np.linspace(0.0, 3.0 * np.pi, points)
        folded = [(float(np.cos(a) * (1.0 + a)), float(np.sin(a))) for a in angle]
        branches.extend([monotone, folded])
    branches.append([(0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 2.0), (1.0, 2.0)])
    branches.append([(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])
    return branches


class TestMergeWalkResampling(TestCase):
    def test_by_measure_matches_the_scan_float_for_float(self):
        for raw in _branches():
            for samples in (2, 3, 40, 257):
                assert _resample_by_measure(raw, samples) == _scan_by_measure(
                    raw,
                    samples,
                ), (len(raw), samples)

    def test_by_arclength_matches_the_scan_float_for_float(self):
        for raw in _branches():
            for samples in (2, 3, 40, 257):
                assert _resample_by_arclength(raw, samples) == _scan_by_arclength(
                    raw,
                    samples,
                ), (len(raw), samples)

    def test_a_target_past_the_last_length_takes_the_last_point(self):
        # A total that rounds below the last target falls through to the end.
        raw = [(0.0, 0.0), (0.1, 0.2), (0.30000000000000004, 0.7)]
        assert _resample_by_arclength(raw, 7) == _scan_by_arclength(raw, 7)
        assert _resample_by_measure(raw, 7) == _scan_by_measure(raw, 7)

    def test_by_measure_compares_each_point_and_sample_about_once(self):
        # Measures that count their comparisons: the scan compares up to
        # samples x points times, the walk about samples + points (plus the
        # points - 1 that sorting an already ordered branch costs both).
        points, samples = 400, 300
        raw = [(float(i), _Counted(i * 0.5)) for i in range(points)]
        _Counted.comparisons = 0
        walked = _resample_by_measure(raw, samples)
        comparisons = _Counted.comparisons
        _Counted.comparisons = 0
        scanned = _scan_by_measure(raw, samples)

        assert walked == scanned
        assert comparisons <= 2 * points + samples
        assert _Counted.comparisons > 50 * (points + samples)


class _Counted(float):
    """A float that counts the order comparisons made against it."""

    comparisons = 0

    def __lt__(self, other):
        _Counted.comparisons += 1
        return float(self) < other

    def __ge__(self, other):
        _Counted.comparisons += 1
        return float(self) >= other
