"""Multi-level structure of the machine.

A BSP machine is flat: ``p`` processors that are all at the same distance
from each other.  Real machines are not: a few computers, each with several
cores, form a machine with two levels, and the communications between two
cores of one computer are much cheaper than between two computers.  Deeper
hierarchies exist (cores, sockets, computers, racks...).

A :class:`Topology` describes such a machine as a tree whose leaves are the
processors.  Each processor has a *path*, the sequence of the indices of the
groups it belongs to, from the outermost one to the processor itself; all the
paths have the same length ``levels``.  With two computers of two cores::

    processor   path     level-1 group (computer)   level-2 group (machine)
    0           (0, 0)   {0, 1}                      {0, 1, 2, 3}
    1           (0, 1)   {0, 1}                      {0, 1, 2, 3}
    2           (1, 0)   {2, 3}                      {0, 1, 2, 3}
    3           (1, 1)   {2, 3}                      {0, 1, 2, 3}

Levels are numbered from the processors up: the level-0 groups are the
processors themselves, the level-``k`` groups gather the processors that
share the first ``levels - k`` elements of their paths, and the only group of
level ``levels`` is the whole machine.  The *distance* between two processors
is the level of the smallest group that contains both.  A flat machine has
one level: paths ``(0,)``, ``(1,)``...

The tree of groups and the cost parameters per level come from the work on
hierarchical BSP machines: Valiant's Multi-BSP model, the SGL model of Chong
Li and Gaétan Hains, and the Multi-ML language of Victor Allombert (see the
credits in the README).  This is a much simpler design than either: programs
stay flat BSML programs whose supersteps can be restricted to a level.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

Path = Tuple[int, ...]
Shape = Union[str, Sequence[int]]


def parse_shape(shape: Shape) -> Optional[List[int]]:
    """Numbers of sub-groups per level, from the outermost level to the
    processors: ``"2x4"`` or ``[2, 4]`` is 2 groups of 4 processors.
    ``"flat"`` (or an empty text) gives ``None``: a machine with one level."""
    if isinstance(shape, str):
        text = shape.strip().lower()
        if text in ("", "flat"):
            return None
        try:
            sizes = [int(part) for part in text.replace("*", "x").split("x")]
        except ValueError:
            raise ValueError(
                f"invalid shape {shape!r}: expected numbers separated by 'x', like '2x4'"
            ) from None
    else:
        sizes = [int(n) for n in shape]
    if not sizes or any(n < 1 for n in sizes):
        raise ValueError(f"invalid shape {shape!r}: the sizes must be at least 1")
    return sizes


class Topology:
    """The tree of the groups of processors of a machine."""

    def __init__(self, paths: Sequence[Sequence[int]]) -> None:
        self.paths: List[Path] = [tuple(int(x) for x in path) for path in paths]
        if not self.paths:
            raise ValueError("a machine needs at least one processor")
        self.p = len(self.paths)
        self.levels = len(self.paths[0])
        if self.levels < 1 or any(len(path) != self.levels for path in self.paths):
            raise ValueError("the paths of the processors must have the same length, at least 1")
        if len(set(self.paths)) != self.p:
            raise ValueError("two processors cannot have the same path")
        self._groups: Dict[int, List[List[int]]] = {}
        self._members: Dict[int, List[List[int]]] = {}
        self._distances: Dict[int, List[int]] = {}

    @classmethod
    def flat(cls, p: int) -> "Topology":
        """The flat machine of ``p`` processors."""
        return cls([(i,) for i in range(p)])

    @classmethod
    def from_shape(cls, sizes: Sequence[int]) -> "Topology":
        """The regular machine with ``sizes[0]`` groups of ``sizes[1]`` groups
        of ... of ``sizes[-1]`` processors, numbered in the order of the tree."""
        paths: List[Path] = [()]
        for n in sizes:
            paths = [path + (i,) for path in paths for i in range(n)]
        return cls(paths)

    @classmethod
    def from_hosts(cls, names: Sequence[str]) -> Optional["Topology"]:
        """The machine with two levels made of the computers whose names are
        given, one name per processor, and of the processors of each
        computer.  ``None`` when there is only one computer: the machine is
        flat."""
        hosts: Dict[str, int] = {}
        counts: List[int] = []
        paths = []
        for name in names:
            if name not in hosts:
                hosts[name] = len(hosts)
                counts.append(0)
            index = hosts[name]
            paths.append((index, counts[index]))
            counts[index] += 1
        return cls(paths) if len(hosts) > 1 else None

    def _check(self, level: int, lowest: int = 0) -> None:
        if isinstance(level, bool) or not isinstance(level, int):
            raise TypeError(f"a level must be an integer, got {level!r}")
        if not lowest <= level <= self.levels:
            raise ValueError(
                f"invalid level {level}: this machine has the levels {lowest} to {self.levels}"
            )

    def groups(self, level: int) -> List[List[int]]:
        """The groups of the given level, each one the sorted list of its
        processors, ordered by their smallest processor."""
        self._check(level)
        if level not in self._groups:
            by_key: Dict[Path, List[int]] = {}
            for pid, path in enumerate(self.paths):
                by_key.setdefault(path[: self.levels - level], []).append(pid)
            groups = sorted(by_key.values(), key=lambda members: members[0])
            members: List[List[int]] = [[] for _ in range(self.p)]
            for group in groups:
                for pid in group:
                    members[pid] = group
            self._groups[level] = groups
            self._members[level] = members
        return self._groups[level]

    def group(self, pid: int, level: int) -> List[int]:
        """The processors of the group of the given level that contains ``pid``."""
        self.groups(level)
        return self._members[level][pid]

    def leader(self, pid: int, level: int) -> int:
        """The smallest processor of the group of ``pid`` at the given level."""
        return self.group(pid, level)[0]

    def distances(self, pid: int) -> List[int]:
        """Distance from ``pid`` to every processor (0 for itself)."""
        if pid not in self._distances:
            mine = self.paths[pid]
            row = []
            for path in self.paths:
                common = 0
                while common < self.levels and path[common] == mine[common]:
                    common += 1
                row.append(self.levels - common)
            self._distances[pid] = row
        return self._distances[pid]

    def distance(self, i: int, j: int) -> int:
        """Level of the smallest group that contains both ``i`` and ``j``."""
        return self.distances(i)[j]

    def describe(self) -> str:
        """Short text like ``2x4`` (regular) or ``3+5`` (two unequal groups)."""
        sizes = []
        regular = True
        for level in range(self.levels, 0, -1):
            counts = {
                len({self.leader(pid, level - 1) for pid in group}) for group in self.groups(level)
            }
            regular = regular and len(counts) == 1
            sizes.append(min(counts))
        if regular:
            return "x".join(str(n) for n in sizes)
        return f"{self.levels} levels, groups of level {self.levels - 1}: " + "+".join(
            str(len(group)) for group in self.groups(self.levels - 1)
        )
