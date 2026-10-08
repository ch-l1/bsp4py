"""Skeletons on distributed lists (port of the module ``Skeleton`` of BSML).

A *distributed list* is a parallel vector holding one list per processor;
it represents the concatenation of these lists.  OCaml BSML provides the
same skeletons for arrays and for lists (``MakeArray``, ``MakeList``); in
Python both are covered by this module, which works on Python lists.

Adaptations to Python: the sizes of the blocks built by :func:`make` and
:func:`par` are computed locally instead of with a parallel scan, and
:func:`length` and :func:`extract` use a single projection, which saves
supersteps without changing the results.

The second part of the module is not in BSML: :func:`reduce`, :func:`scan`,
:func:`sort` and :func:`dh` are the data-parallel skeletons that Chong Li
programmed with the SGL model (Section 5.2 of his PhD thesis, "Parallel
Skeletons Implementation"; see also https://ieeexplore.ieee.org/document/6341490).
SGL programs are recursive: a master makes its children compute, gathers
their results, computes, and scatters data back.  Here the processors are
the workers and the leader of each group of processors plays the master of
the group, so the skeletons follow the levels of the machine (see
``bsp4py.init``): on a machine with several levels the data goes up and
down the tree of the groups with supersteps restricted to each level, and
on a flat machine they are the algorithms of the thesis for one master and
``p`` workers.
"""

from __future__ import annotations

import builtins
import functools
from bisect import bisect_right
from collections.abc import Sequence
from heapq import merge as _merge
from itertools import accumulate
from typing import Any, Callable, List, Optional, Tuple, TypeVar

from ._core import (
    Par,
    _replicated,
    apply,
    bsp_distance,
    bsp_groups,
    bsp_levels,
    bsp_p,
    mkpar,
    put,
)
from .base import parfun, this
from .base import to_list as _vector_to_list
from .comm import bcast_multi, fold_direct, fold_multi, scan_direct, sgl_gps
from .comm import shift_left as _shift_left
from .comm import shift_right as _shift_right

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")

__all__ = [
    "dh",
    "extract",
    "length",
    "make",
    "map",
    "map_index",
    "par",
    "reduce",
    "scan",
    "shift_left",
    "shift_right",
    "sort",
    "to_array",
    "to_list",
    "zip",
    "zip_index",
]


def _block(size: int, p: int, i: int) -> tuple[int, int]:
    """(offset, length) of the block of processor ``i`` when ``size``
    elements are distributed over ``p`` processors, bigger blocks first."""
    quotient, remainder = divmod(size, p)
    return i * quotient + min(i, remainder), quotient + (1 if i < remainder else 0)


def make(f: Callable[[int], T], size: int) -> Par[List[T]]:
    """Distributed list ``[f(0), f(1), ..., f(size-1)]``, evenly distributed:
    ``< [f(0), ..., f(k-1)], [f(k), ...], ... >``.  No communication."""
    p = bsp_p()

    def local(i: int) -> List[T]:
        offset, n = _block(size, p, i)
        return [f(offset + z) for z in range(n)]

    return mkpar(local)


def length(v: Par[Sequence[Any]]) -> int:
    """Number of elements of the distributed list.  One superstep."""
    return sum(_vector_to_list(parfun(len, v)))


def extract(v: Par[Sequence[T]]) -> List[T]:
    """Gather the distributed list into a (global) list.  One superstep."""
    return [x for block in _vector_to_list(v) for x in block]


def par(a: Sequence[T]) -> Par[List[T]]:
    """Dual of :func:`extract`: distribute evenly a (global) sequence over
    the processors.  No communication."""
    p = bsp_p()
    size = len(a)

    def local(i: int) -> List[T]:
        offset, n = _block(size, p, i)
        return list(a[offset : offset + n])

    return mkpar(local)


def map(f: Callable[[T], U], v: Par[Sequence[T]]) -> Par[List[U]]:
    """Apply ``f`` to each element of the distributed list."""
    return parfun(lambda l: [f(x) for x in l], v)


def zip(f: Callable[[T, U], V], v: Par[Sequence[T]], w: Par[Sequence[U]]) -> Par[List[V]]:
    """Apply ``f`` to each pair of elements of two distributed lists that
    have the same distribution."""

    def local(a: Sequence[T], b: Sequence[U]) -> List[V]:
        if len(a) != len(b):
            raise ValueError("skeleton.zip: the lists have different distributions")
        return [f(x, y) for x, y in builtins.zip(a, b)]

    return parfun(local, v, w)


def _offsets(v: Par[Sequence[Any]]) -> Par[int]:
    """Global index of the first element of each processor.  One superstep."""
    sizes = parfun(len, v)
    return parfun(lambda end, n: end - n, scan_direct(lambda a, b: a + b, sizes), sizes)


def map_index(f: Callable[[int, T], U], v: Par[Sequence[T]]) -> Par[List[U]]:
    """Apply ``f(index, element)`` to each element of the distributed list,
    ``index`` being the position of the element in the whole list."""
    return parfun(lambda offset, l: [f(offset + i, x) for i, x in enumerate(l)], _offsets(v), v)


def zip_index(
    f: Callable[[int, T, U], V], v: Par[Sequence[T]], w: Par[Sequence[U]]
) -> Par[List[V]]:
    """Apply ``f(index, x, y)`` to each pair of elements of two distributed
    lists that have the same distribution."""

    def local(offset: int, a: Sequence[T], b: Sequence[U]) -> List[V]:
        if len(a) != len(b):
            raise ValueError("skeleton.zip_index: the lists have different distributions")
        return [f(offset + i, x, y) for i, (x, y) in enumerate(builtins.zip(a, b))]

    return parfun(local, _offsets(v), v, w)


def to_list(v: Par[Sequence[T]]) -> Par[List[T]]:
    """Convert the local blocks to Python lists (the vector stays
    distributed; use :func:`extract` to get a global list)."""
    return parfun(list, v)


to_array = to_list


def _non_empty(l: Sequence[Any]) -> Sequence[Any]:
    if len(l) == 0:
        raise ValueError("skeleton shifts need at least one element on every processor")
    return l


def shift_left(k: T, v: Par[Sequence[T]]) -> Par[List[T]]:
    """Shift the distributed list to the left and insert ``k`` at its end:
    ``[x_0, x_1, ..., x_{n-1}]`` becomes ``[x_1, ..., x_{n-1}, k]``.  Every
    processor must hold at least one element.  One superstep."""
    firsts = parfun(lambda l: _non_empty(l)[0], v)
    firsts = parfun(lambda pid, x: k if pid == 0 else x, this(), firsts)
    incoming = _shift_left(firsts)
    return parfun(lambda x, l: list(l[1:]) + [x], incoming, v)


def shift_right(k: T, v: Par[Sequence[T]]) -> Par[List[T]]:
    """Shift the distributed list to the right and insert ``k`` at its
    beginning: ``[x_0, ..., x_{n-2}, x_{n-1}]`` becomes ``[k, x_0, ...,
    x_{n-2}]``.  Every processor must hold at least one element.  One
    superstep."""
    lasts = parfun(lambda l: _non_empty(l)[-1], v)
    incoming = _shift_right(lasts)
    incoming = parfun(lambda pid, x: k if pid == 0 else x, this(), incoming)
    return parfun(lambda x, l: [x] + list(l[:-1]), incoming, v)


# --------------------------------------------------------------------------
# The data-parallel skeletons of SGL: reduce, scan, sort and dh
# --------------------------------------------------------------------------

Op = Callable[[T, T], T]
#: An optional value: ``()`` for none, ``(x,)`` for the value ``x``.
Maybe = Tuple[Any, ...]

#: One tier of the tree of the groups: for each processor the leader of its
#: group of the level below and of its group of this level, and the level of
#: the supersteps between them.
Tier = Tuple[List[int], List[int], int]


def _tiers(ordered: bool) -> List[Tier]:
    """The tiers of the machine, from the processors up.

    The leader of a group (its smallest processor) plays the master of SGL
    and the leaders of the groups of the level below play its children.
    With ``ordered``, the skeleton needs the groups to follow the order of
    the processors (it combines the data in the order of the tree): when
    they do not (groups that are not made of consecutive processors), the
    machine is used as a flat one, with processor 0 as only master.
    """
    p, levels = bsp_p(), bsp_levels()
    groups = [bsp_groups(level) for level in range(levels + 1)]
    consecutive = all(
        group == list(range(group[0], group[-1] + 1)) for level in groups for group in level
    )
    if ordered and not consecutive:
        return [(list(range(p)), [0] * p, levels)]
    leaders = []
    for level in groups:
        row = [0] * p
        for group in level:
            for pid in group:
                row[pid] = group[0]
        leaders.append(row)
    return [(leaders[level - 1], leaders[level], level) for level in range(1, levels + 1)]


def _children(tier: Tier, pid: int) -> List[int]:
    """The children of the master ``pid`` in a tier, in order."""
    below, here, _ = tier
    return sorted({below[j] for j in range(len(here)) if here[j] == pid})


def _gather(tier: Tier, values: Par[Any]) -> Par[Any]:
    """SGL ``gather``: each master gets the list of the pairs (child, value
    of the child); the other processors get ``None``."""
    below, here, level = tier

    def send(pid: int) -> Callable[[Any], Callable[[int], Any]]:
        if below[pid] != pid:
            return lambda x: lambda dst: None
        return lambda x: lambda dst: (x,) if dst == here[pid] else None

    def receive(pid: int) -> Callable[[Callable[[int], Any]], Any]:
        if here[pid] != pid:
            return lambda f: None
        children = _children(tier, pid)
        return lambda f: [(child, f(child)[0]) for child in children]

    received = put(apply(mkpar(send), values), level=level)
    return apply(mkpar(receive), received)


def _scatter(tier: Tier, tables: Par[Any], default: Any = None) -> Par[Any]:
    """SGL ``scatter``: each master holds a dictionary from its children to
    values, and each child gets its value; the processors that are not
    children in this tier get ``default``."""
    below, here, level = tier

    def send(pid: int) -> Callable[[Any], Callable[[int], Any]]:
        if here[pid] != pid:
            return lambda table: lambda dst: None
        return lambda table: lambda dst: (table[dst],) if dst in table else None

    def receive(pid: int) -> Callable[[Callable[[int], Any]], Any]:
        if below[pid] != pid:
            return lambda f: default
        return lambda f: f(here[pid])[0]

    received = put(apply(mkpar(send), tables), level=level)
    return apply(mkpar(receive), received)


def _lift(op: Op) -> Callable[[Maybe, Maybe], Maybe]:
    """``op`` on optional values: a missing value is neutral."""

    def lifted(a: Maybe, b: Maybe) -> Maybe:
        if not a:
            return b
        if not b:
            return a
        return (op(a[0], b[0]),)

    return lifted


def reduce(op: Op, v: Par[Sequence[T]]) -> T:
    """Parallel reduction of a distributed list: ``x_0 op x_1 op ... op
    x_{n-1}``, as an ordinary (global) value.

    Algorithm 1 of the thesis: each worker reduces its own elements, each
    master gathers the results of its children and reduces them.  The result
    of the root is then sent back down, so that it is known everywhere.
    ``2L - 1`` supersteps on a machine with ``L`` levels (one on a flat
    machine).  ``op`` must be associative; the list must not be empty.
    """
    lifted = _lift(op)
    partial = parfun(lambda l: (functools.reduce(op, l),) if len(l) else (), v)
    # fold_multi combines in the order of the tree: fold_direct when it is
    # not the order of the processors
    fold = fold_multi if len(_tiers(ordered=True)) == bsp_levels() else fold_direct
    result = _replicated(fold(lifted, partial))
    if not result:
        raise ValueError("reduce of an empty distributed list")
    return result[0]


def scan(op: Op, v: Par[Sequence[T]]) -> Par[List[T]]:
    """Parallel prefix of a distributed list: the distributed list
    ``[x_0, x_0 op x_1, ..., x_0 op ... op x_{n-1}]``, with the same
    distribution.

    Algorithm 2 of the thesis, in two steps.  Step 1: each worker scans its
    own elements; each master gathers the last element of each child (the
    total of the child) and scans them, which gives the offset of each
    child.  Step 2: each master scatters the offsets to its children, which
    add them to what they hold.  ``2L`` supersteps on a machine with ``L``
    levels.  ``op`` must be associative; local lists may be empty.
    """
    lifted = _lift(op)
    tiers = _tiers(ordered=True)
    local = parfun(lambda l: list(accumulate(l, op)), v)

    # step 1, up the tree: totals[pid] is the total of the group led by pid
    totals: Par[Any] = parfun(lambda l: (l[-1],) if l else (), local)
    gathered = []
    for tier in tiers:
        parts = _gather(tier, totals)
        gathered.append(parts)
        totals = parfun(
            lambda ps: functools.reduce(lifted, [t for _, t in ps], ()) if ps is not None else (),
            parts,
        )

    # step 2, down the tree: offsets[pid] is what comes before the group led by pid
    def offsets_of_children(offset: Maybe, parts: Any) -> Any:
        if parts is None:
            return {}
        table = {}
        for child, total in parts:
            table[child] = offset
            offset = lifted(offset, total)
        return table

    offsets: Par[Any] = mkpar(lambda pid: ())
    for tier, parts in builtins.zip(reversed(tiers), reversed(gathered)):
        offsets = _scatter(tier, parfun(offsets_of_children, offsets, parts), default=())
    return parfun(lambda off, l: [op(off[0], x) for x in l] if off else l, offsets, local)


def sort(v: Par[Sequence[T]], *, key: Optional[Callable[[T], Any]] = None) -> Par[List[T]]:
    """Parallel sorting by regular sampling of a distributed list.

    The result is a distributed list: each local list is sorted and the
    elements of a processor are before the elements of the next one.  The
    sort is stable.  ``key`` is used like in ``sorted``.

    Algorithms 3 and 4 of the thesis.  (1) Each worker sorts its elements
    and selects ``p`` regular samples, which the masters gather up to the
    root.  (2) The root sorts the samples and picks ``p - 1`` regular
    pivots.  (3) The pivots are scattered down to the workers, which cut
    their lists with them.  (4, 5) The block ``j`` of each worker goes to
    worker ``j``, which merges what it receives.  As in the experiments of
    the thesis, this last exchange is done directly between the workers
    (with :func:`sgl_gps`) instead of through the masters.  ``2L + 1``
    supersteps on a machine with ``L`` levels.
    """
    p = bsp_p()
    tiers = _tiers(ordered=False)

    def keys_of(xs: Sequence[T]) -> List[Any]:
        return list(xs) if key is None else [key(x) for x in xs]

    local = parfun(lambda l: sorted(l, key=key), v)

    def regular_samples(l: List[T]) -> List[T]:
        return [l[(k * len(l)) // p] for k in range(p)] if len(l) >= p else list(l)

    # steps 1 and 2: the samples go up to the root, which picks the pivots
    samples: Par[Any] = parfun(regular_samples, local)
    for tier in tiers:
        parts = _gather(tier, samples)
        samples = parfun(lambda ps: [x for _, xs in (ps or []) for x in xs], parts)

    def pick(pid: int) -> Callable[[List[T]], Any]:
        if pid != 0:  # processor 0 is the root: the leader of the whole machine
            return lambda xs: None
        return lambda xs: sorted(xs, key=key)[p + p // 2 - 1 :: p][: p - 1]  # type: ignore[type-var,arg-type]

    # step 3: the pivots go down to everybody
    pivots = bcast_multi(0, apply(mkpar(pick), samples))

    # steps 4 and 5: direct exchange of the blocks, then merge
    def split(pid: int, data: Any) -> List[Any]:
        l, pvts = data
        keys = keys_of(l)
        bounds = [0] + [bisect_right(keys, k) for k in keys_of(pvts)] + [len(l)]
        blocks = [l[bounds[i] : bounds[i + 1]] or None for i in range(len(bounds) - 1)]
        return blocks + [None] * (p - len(blocks))

    def assemble(pid: int, blocks: List[Any]) -> List[T]:
        return list(_merge(*[b for b in blocks if b is not None], key=key))

    return sgl_gps(split, assemble, parfun(lambda l, pvts: (l, pvts), local, pivots))


def _dh_sequential(oplus: Op, otimes: Op, data: List[T]) -> List[T]:
    """``dh`` of a list whose length is a power of two (bottom-up)."""
    n = len(data)
    if n < 1 or n & (n - 1):
        raise ValueError(f"dh: the length of a list must be a power of two, got {n}")
    width = 1
    while width < n:
        result = list(data)
        for start in range(0, n, 2 * width):
            for k in range(start, start + width):
                u, w = data[k], data[k + width]
                result[k] = oplus(u, w)
                result[k + width] = otimes(u, w)
        data = result
        width *= 2
    return data


def dh(oplus: Op, otimes: Op, v: Par[Sequence[T]]) -> Par[List[T]]:
    """Distributable homomorphism (the butterfly) on a distributed list.

    ``dh`` transforms a list ``[x_1, ..., x_n]`` whose length is a power of
    two into the list ``[y_1, ..., y_n]`` of the same length such that, if
    ``u`` is ``dh`` of the first half and ``w`` is ``dh`` of the second
    half, ``y_i = u_i oplus w_i`` in the first half of the result and
    ``y_{i + n/2} = u_i otimes w_i`` in the second half (``dh`` of one
    element is this element).  It expresses a class of divide-and-conquer
    algorithms, the Fast Fourier Transform for instance.

    Algorithm 5 of the thesis: each worker computes ``dh`` of its own
    elements; then, ``log2(p)`` times, the workers exchange their lists two
    by two (worker ``i`` with ``i + 1``, then with ``i + 2``, ``i + 4``...)
    and combine them element by element, with ``oplus`` for the worker of
    the first half and ``otimes`` for the other.  As in the experiments of
    the thesis the exchange is direct, and each of these supersteps is
    restricted to the lowest level that contains the pairs of workers.

    The number of processors must be a power of two and every processor
    must hold the same number of elements, a power of two.
    """
    p = bsp_p()
    if p & (p - 1):
        raise ValueError(f"dh: the number of processors must be a power of two, got {p}")
    data = parfun(lambda l: _dh_sequential(oplus, otimes, list(l)), v)

    def step(half: int, data: Par[List[T]]) -> Par[List[T]]:
        level = max(bsp_distance(pid, pid ^ half) for pid in range(p))

        def combine(pid: int) -> Callable[[List[T], Callable[[int], Any]], List[T]]:
            def with_partner(mine: List[T], f: Callable[[int], Any]) -> List[T]:
                other = f(pid ^ half)
                if len(other) != len(mine):
                    raise ValueError("dh: the processors must hold the same number of elements")
                if pid & half:  # second half: mine is w, the partner has u
                    return [otimes(u, w) for u, w in builtins.zip(other, mine)]
                return [oplus(u, w) for u, w in builtins.zip(mine, other)]

            return with_partner

        to_partner = mkpar(lambda pid: lambda l: lambda dst: l if dst == pid ^ half else None)
        received = put(apply(to_partner, data), level=level)
        return apply(mkpar(combine), data, received)

    half = 1
    while half < p:
        data = step(half, data)
        half *= 2
    return data
