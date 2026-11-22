"""Which saddles a trajectory visits, in what order, and for how long.

Near an attracting heteroclinic cycle a trajectory lingers near each saddle and
moves quickly between them, and the lingering grows: the time near the next
saddle is asymptotically ``c_j / e_{j+1}`` times the time near this one
(Gaunersdorfer 1992, Lemma 1). Each passage time also carries an additive
constant from the entry and exit geometry, ``T_{k+1} = r T_k + K``, so the plain
ratio ``T_{k+1} / T_k`` converges only like ``1 / T_k``. Differences remove the
constant exactly - ``T_{k+2} - T_{k+1} = r (T_{k+1} - T_k)`` - and are what the
ratios here are built from.

A visit is a maximal interval during which the trajectory is within ``radius`` of
one saddle. Entry and exit times are interpolated linearly between samples, and a
visit cut off by the start or end of the record is discarded, since its length is
not known.
"""

import numpy as np


class Visit:
    """One passage near one saddle: which saddle, and when it entered and left."""

    __slots__ = ["entry", "exit", "saddle"]

    def __init__(self, saddle: int, entry: float, exit: float) -> None:  # noqa: A002
        self.saddle = saddle
        self.entry = entry
        self.exit = exit

    @property
    def duration(self) -> float:
        """Time spent within the radius on this passage."""
        return self.exit - self.entry


class Itinerary:
    """The visits a trajectory makes to a set of saddles, in time order."""

    __slots__ = ["_visits"]

    def __init__(self, visits: list[Visit]) -> None:
        self._visits = sorted(visits, key=lambda visit: visit.entry)

    @classmethod
    def measure(
        cls,
        times: np.ndarray,
        states: np.ndarray,
        saddles: list[np.ndarray],
        radius: float,
    ) -> "Itinerary":
        """Find every complete visit of ``states`` to a ball around each saddle."""
        visits: list[Visit] = []
        for index, saddle in enumerate(saddles):
            distance = np.linalg.norm(states - np.asarray(saddle), axis=1) - radius
            inside = distance < 0.0
            changes = np.flatnonzero(np.diff(inside.astype(int)))
            entries = [int(k) for k in changes if not inside[k]]
            exits = [int(k) for k in changes if inside[k]]
            for entry in entries:
                later = [k for k in exits if k > entry]
                if not later:
                    break
                visits.append(
                    Visit(
                        index,
                        _crossing(times, distance, entry),
                        _crossing(times, distance, later[0]),
                    ),
                )
        return cls(visits)

    @property
    def visits(self) -> list[Visit]:
        """Every complete visit, ordered by entry time."""
        return list(self._visits)

    @property
    def order(self) -> list[int]:
        """The saddle index of each visit: the switching sequence."""
        return [visit.saddle for visit in self._visits]

    @property
    def durations(self) -> np.ndarray:
        """The length of each visit, in time order."""
        return np.array([visit.duration for visit in self._visits])

    def growth_ratios(self) -> np.ndarray:
        """``(T_{k+2} - T_{k+1}) / (T_{k+1} - T_k)`` along the whole itinerary.

        For a cycle whose steps all grow by the same ratio - a symmetric cycle -
        this converges to that ratio, free of the additive constant.
        """
        steps = np.diff(self.durations)
        return steps[1:] / steps[:-1]

    def step_ratios(self, period: int) -> np.ndarray:
        """The growth of each step of a cycle of ``period`` saddles, measured.

        With ``D_k = T_{k+period} - T_k`` - one revolution's growth at the same
        saddle, so the constants cancel - ``D_{k+1} / D_k`` converges to the growth
        of the step from visit ``k`` to visit ``k + 1``. Entry ``s`` of the result is
        the latest such ratio for the step that *leaves* saddle ``s``, so for saddles
        passed to :meth:`measure` in cycle order it lines up with
        ``CycleSpectrum.residence_ratios``.
        """
        durations = self.durations
        revolutions = durations[period:] - durations[:-period]
        ratios = revolutions[1:] / revolutions[:-1]
        order = self.order
        result = np.full(period, np.nan)
        for k in range(len(ratios) - 1, -1, -1):
            if np.isnan(result[order[k]]):
                result[order[k]] = ratios[k]
            if not np.any(np.isnan(result)):
                break
        return result


def _crossing(times: np.ndarray, distance: np.ndarray, index: int) -> float:
    """Time at which ``distance`` crosses zero between samples ``index`` and next."""
    before, after = distance[index], distance[index + 1]
    fraction = before / (before - after)
    return float(times[index] + fraction * (times[index + 1] - times[index]))
