"""Measure the BSP parameters of the machine (adaptation of ``bsmlprobe``).

The BSP model describes a parallel machine with a few numbers: *p*, the
number of processors, *r*, their speed, *g*, the cost of communication, and
*l*, the cost of a synchronisation barrier.  The cost of a superstep where
each processor computes for at most *w* and sends or receives at most *h* is
then ``w + h*g + l``.

Run the benchmark on the machine and with the number of processes you plan
to use::

    mpiexec -n 8 python -m bsp4py.probe

or call :func:`probe` from a program.  Like the original, the benchmark is
itself a BSML program: *r* is measured with local computations, *g* and *l*
by timing supersteps that exchange more and more data and fitting a line
through the timings.

Adaptation to Python.  ``bsmlprobe`` expresses *g* and *l* in the unit of
the local computations (flops), which is convenient when the cost of a
program is counted in floating-point operations.  The local computations of
a Python program are anything but uniform, so here the parameters are in
units that can be used directly:

* *g* is in **seconds per byte** (bytes of serialised messages, the unit of
  :func:`bsp4py.stats`),
* *l* is in **seconds**; it is measured directly, as the time of a superstep
  that exchanges no data, rather than as the intercept of the fitted line,
  which is unreliable over a range of several megabytes,
* *r* is the speed of the interpreter on the kernel of ``bsmlprobe`` (two
  DAXPY-like loops), in **flops per second**; ``g * r`` and ``l * r`` are the
  values that ``bsmlprobe`` reports.

With these units ``sum(h) * g + supersteps * l`` is the time a program spends
communicating (:meth:`BspParameters.communication_time`).

Multi-level machines.  When the machine has several levels (several
computers with several cores each, for example: see ``bsp4py.init``), one
``g`` and one ``l`` do not describe it well: the communications inside a
computer are much cheaper than between two computers.  The benchmark then
also measures a ``g`` and an ``l`` for each level, and the predicted time of
a superstep of level ``k`` that exchanges ``h_d`` bytes at each distance
``d`` is ``sum(h_d * g_d) + l_k``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Callable, List, Optional, Sequence, Tuple

from ._core import (
    Par,
    Stats,
    backend_name,
    bsp_distance,
    bsp_levels,
    bsp_p,
    bsp_shape,
    get_cost,
    mkpar,
    put,
    start_timing,
    stop_timing,
)
from .base import parfun, print_once, to_list

__all__ = ["BspParameters", "LevelParameters", "probe", "main"]


@dataclass(frozen=True)
class LevelParameters:
    """Parameters of one level of a multi-level machine."""

    level: int
    #: cost of communication between two processors at this distance (in
    #: the same group of this level but not of the level below), seconds per byte
    g: float
    #: cost of a superstep of this level without data, in seconds
    l: float  # noqa: E741


@dataclass(frozen=True)
class BspParameters:
    """BSP parameters measured by :func:`probe`.

    ``g`` and ``l`` describe the machine as a flat BSP machine, all the
    processors being treated alike.  ``levels`` describes it level by level,
    from level 1 to the number of levels of the machine (on a flat machine
    there is one level, with the same values).
    """

    p: int  #: number of processors
    r: float  #: speed of the local computations, in flops per second
    g: float  #: cost of communication, in seconds per byte
    l: float  #: cost of a superstep without data, in seconds  # noqa: E741
    levels: Tuple[LevelParameters, ...] = ()

    def flat_communication_time(self, stats: Stats) -> float:
        """Predicted time (seconds) of the communications counted in
        ``stats`` with the flat model: ``sum(h_i) * g + supersteps * l``."""
        return sum(h or 0 for h in stats.h_relations) * self.g + stats.supersteps * self.l

    def communication_time(self, stats: Stats) -> float:
        """Predicted time (seconds) of the communications counted in
        ``stats``.

        With the multi-level model when the parameters of the levels and
        the levels of the supersteps are known: a superstep of level ``k``
        that exchanges ``h_d`` bytes at each distance ``d`` costs ``sum(h_d
        * g_d) + l_k``.  Otherwise with the flat model
        (:meth:`flat_communication_time`).  Use the counters of
        ``bsp4py.global_stats()`` with the parallel backends.
        """
        steps = stats.steps
        if not self.levels or len(steps) != stats.supersteps:
            return self.flat_communication_time(stats)
        if any(len(step.h) != len(self.levels) for step in steps):
            return self.flat_communication_time(stats)
        return sum(
            self.levels[step.level - 1].l
            + sum((h or 0) * level.g for h, level in zip(step.h, self.levels))
            for step in steps
        )

    def environment(self) -> str:
        """Shell line that makes these values the ones returned by
        ``bsp_g()``, ``bsp_l()`` and ``bsp_r()``."""
        return f"export BSP4PY_G={self.g!r} BSP4PY_L={self.l!r} BSP4PY_R={self.r!r}"


# --------------------------------------------------------------------------
# r: speed of the local computations
# --------------------------------------------------------------------------


def _kernel(niters: int, n: int) -> float:
    """Time (seconds) of ``niters`` runs of two DAXPY-like loops on vectors
    of size ``n``: ``4 * niters * n`` floating-point operations."""
    a, b = 1.0 / 3.0, 4.0 / 9.0
    x = [float(i) for i in range(n)]
    y = [float(i) for i in range(n)]
    z = [float(i) for i in range(n)]
    start = time.perf_counter()
    for _ in range(niters):
        for i in range(n):
            y[i] = a * x[i] + y[i]
        for i in range(n):
            z[i] = z[i] - b * x[i]
    return time.perf_counter() - start


def _flops_per_second(niters: int, n: int) -> float:
    """Average over the processors of the speed measured on vectors of size n."""
    seconds = to_list(mkpar(lambda pid: _kernel(niters, n)))
    speeds = [4 * niters * n / t for t in seconds if t > 0.0]
    return mean(speeds) if speeds else 0.0


def _determine_r(niters: int, maxn: int) -> float:
    sizes = []
    n = 16
    while n <= maxn:
        sizes.append(n)
        n *= 2
    return mean(_flops_per_second(niters, n) for n in sizes)


# --------------------------------------------------------------------------
# g and l: cost of the supersteps
# --------------------------------------------------------------------------


def _messages(p: int, h: int) -> Callable[[int], Callable[[int], Optional[bytes]]]:
    """Messages of an exact h-relation: every processor sends ``h`` bytes in
    all, shared between the ``p - 1`` other processors, and receives ``h``."""
    size, rest = divmod(h, p - 1)
    small = bytes(size) if size > 0 else None
    big = bytes(size + 1)

    def messages(pid: int) -> Callable[[int], Optional[bytes]]:
        def message(dst: int) -> Optional[bytes]:
            distance = (dst - pid) % p  # 1 .. p-1 for the other processors
            if distance == 0:
                return None
            return big if distance <= rest else small

        return message

    return messages


def _messages_at(distance: int, h: int) -> Callable[[int], Callable[[int], Optional[bytes]]]:
    """Messages between the processors that are at the given distance from
    each other: every processor that has such partners sends them ``h``
    bytes in all (and receives ``h`` when the machine is regular)."""
    p = bsp_p()

    def messages(pid: int) -> Callable[[int], Optional[bytes]]:
        partners = [j for j in range(p) if bsp_distance(pid, j) == distance]
        if not partners:
            return lambda dst: None
        size, rest = divmod(h, len(partners))
        payload = {
            j: bytes(size + (1 if index < rest else 0)) or None for index, j in enumerate(partners)
        }
        return lambda dst: payload.get(dst)

    return messages


def _superstep_time(
    niters: int, messages: Callable[[int], Any], level: Optional[int] = None
) -> float:
    """Time (seconds) of one superstep of the given level (global by
    default) that exchanges the given messages: the median of ``niters``
    supersteps on each processor, averaged over the processors."""
    vector: Par[Any] = mkpar(messages)
    times: Par[List[float]] = mkpar(lambda pid: [])
    for _ in range(niters):
        start_timing()
        put(vector, level=level)
        stop_timing()
        times = parfun(lambda ts, t: [*ts, t], times, get_cost())
    return mean(to_list(parfun(median, times)))


def _least_squares(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Slope and intercept of the line that fits the points best."""
    mean_x, mean_y = mean(xs), mean(ys)
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance == 0.0:
        return 0.0, mean_y
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance
    return slope, mean_y - slope * mean_x


def _determine_g_and_l(niters: int, maxh: int, points: int) -> Tuple[float, float]:
    """*l* is the time of a superstep without data and *g* the slope of the
    line that fits the times of supersteps exchanging 0 to ``maxh`` bytes."""
    p = bsp_p()
    nothing = lambda pid: lambda dst: None
    _superstep_time(1, nothing)  # warm up
    l = _superstep_time(5 * niters, nothing)  # noqa: E741
    if p == 1:  # one processor: no communication, only the barrier costs
        return 0.0, l
    sizes = sorted({(maxh * k) // (points - 1) for k in range(points)})
    times = [_superstep_time(niters, _messages(p, h)) for h in sizes]
    g, _ = _least_squares(sizes, times)
    return max(g, 0.0), l


def _determine_level(level: int, niters: int, maxh: int, points: int) -> LevelParameters:
    """Parameters of one level: *l* is the time of a superstep of this level
    without data and *g* the slope of the line that fits the times of
    supersteps of this level in which the processors exchange 0 to ``maxh``
    bytes with the processors at this distance."""
    p = bsp_p()
    nothing = lambda pid: lambda dst: None
    _superstep_time(1, nothing, level)  # warm up (and creation of the group communicators)
    l = _superstep_time(5 * niters, nothing, level)  # noqa: E741
    if not any(bsp_distance(i, j) == level for i in range(p) for j in range(i)):
        return LevelParameters(level, 0.0, l)  # nobody is at this distance
    sizes = sorted({(maxh * k) // (points - 1) for k in range(points)})
    times = [_superstep_time(niters, _messages_at(level, h), level) for h in sizes]
    g, _ = _least_squares(sizes, times)
    return LevelParameters(level, max(g, 0.0), l)


# --------------------------------------------------------------------------
# The benchmark
# --------------------------------------------------------------------------


def probe(
    niters: int = 20, maxn: int = 512, maxh: int = 1 << 20, points: int = 17
) -> BspParameters:
    """Measure the BSP parameters of the current machine.

    :param niters: number of repetitions of each measure.
    :param maxn: the speed *r* is measured on vectors of 16, 32, ... ``maxn``
        numbers.
    :param maxh: the supersteps that are timed exchange from 0 to ``maxh``
        bytes per processor.
    :param points: number of sizes between 0 and ``maxh`` that are timed.

    On a multi-level machine the parameters of every level are measured as
    well (``levels`` of the result): the cost of the communications between
    processors at each distance, and the cost of a superstep of each level.

    The result depends on the backend, on the number of processors and on
    the load of the machine: measure in the conditions of the real runs, and
    several times.  With the sequential backend the values describe the
    simulator, not a parallel machine.
    """
    if niters < 1 or maxn < 16 or maxh < 1 or points < 2:
        raise ValueError("probe: niters >= 1, maxn >= 16, maxh >= 1 and points >= 2 are needed")
    r = _determine_r(niters, maxn)
    g, l = _determine_g_and_l(niters, maxh, points)  # noqa: E741
    if bsp_levels() == 1:
        levels: Tuple[LevelParameters, ...] = (LevelParameters(1, g, l),)
    else:
        levels = tuple(
            _determine_level(level, niters, maxh, points) for level in range(1, bsp_levels() + 1)
        )
    return BspParameters(p=bsp_p(), r=r, g=g, l=l, levels=levels)


def main(argv: Optional[List[str]] = None) -> None:
    """Command line of the benchmark: ``python -m bsp4py.probe --help``."""
    parser = argparse.ArgumentParser(
        prog="python -m bsp4py.probe",
        description="Measure the BSP parameters r, g and l of the machine.",
    )
    parser.add_argument("--niters", type=int, default=20, help="repetitions of each measure")
    parser.add_argument("--maxn", type=int, default=512, help="biggest vector used to measure r")
    parser.add_argument(
        "--maxh", type=int, default=1 << 20, help="biggest exchange timed, in bytes per processor"
    )
    parser.add_argument("--points", type=int, default=17, help="number of exchange sizes timed")
    options = parser.parse_args(argv)

    print_once(f"Benchmark starts ({backend_name()} backend)")
    parameters = probe(options.niters, options.maxn, options.maxh, options.points)
    r, g, l = parameters.r, parameters.g, parameters.l  # noqa: E741
    print_once(f"p = {parameters.p}")
    print_once(f"r = {r / 1e6:.3f} Mflops/s")
    print_once(f"g = {g:.3e} s/byte   ({g * r:.3f} flops/byte)")
    print_once(f"l = {l:.3e} s        ({l * r:.1f} flops)")
    if len(parameters.levels) > 1:
        print_once(f"shape = {bsp_shape()}; g and l above describe the machine as a flat one")
        for level in parameters.levels:
            print_once(f"level {level.level}: g = {level.g:.3e} s/byte, l = {level.l:.3e} s")
    print_once(parameters.environment())
    if backend_name() == "seq":
        print_once(
            "Note: sequential backend, these values describe the simulator; "
            "start the benchmark with mpiexec to measure a parallel machine."
        )


if __name__ == "__main__":
    main()
