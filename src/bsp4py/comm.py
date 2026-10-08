"""Parallel functions with communications (port of the module ``Comm``).

Shifts, total exchange, scatter, gather, broadcasts, scans (parallel
prefixes), folds (parallel reductions), and the GPS function, a simplified
``put``.  All of them are written with the primitives only.

Adaptations to Python:

* the functions specialised for OCaml lists, arrays and strings
  (``scatter_list``, ``scatter_array``, ``scatter_string``...) accept any
  Python sequence that supports ``len`` and slicing (``list``, ``tuple``,
  ``str``, ``bytes``, NumPy arrays); the three names are kept as aliases;
* ``scan_logp`` is a corrected version of the original (which is marked "to
  be verified" in BSML 0.5 and applies the operation in the wrong order for
  non commutative operations).

Additions: ``fold_multi`` and ``bcast_multi`` are the versions of the fold
and of the broadcast for multi-level machines (several computers with
several cores each, for example): they follow the tree of the groups of
processors to send as little as possible through the slow links.

``sgl_gps`` is not in BSML 0.5 either.  It is the GPS function that Chong
Li defined in BSML for the SGL model, in Section 6.2 of his PhD thesis.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from itertools import accumulate
from typing import Any, Callable, List, Optional, TypeVar

from ._core import (
    InvalidProcessor,
    Par,
    apply,
    bsp_groups,
    bsp_levels,
    bsp_p,
    dumps,
    loads,
    mkpar,
    put,
    within_bounds,
)
from .base import applyat, parfun
from .utils import natmod

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
C = TypeVar("C")

__all__ = [
    "BcastError",
    "GatherError",
    "ScatterError",
    "bcast_direct",
    "bcast_multi",
    "bcast_totex",
    "bcast_totex_array",
    "bcast_totex_gen",
    "bcast_totex_list",
    "bcast_totex_string",
    "fold_array_direct",
    "fold_array_logp",
    "fold_direct",
    "fold_list_direct",
    "fold_list_logp",
    "fold_logp",
    "fold_multi",
    "fold_wide",
    "gather",
    "gather_list",
    "scan_array_direct",
    "scan_array_logp",
    "scan_direct",
    "scan_list_direct",
    "scan_list_logp",
    "scan_logp",
    "scan_wide",
    "scan_wide_direct",
    "scan_wide_logp",
    "scatter",
    "scatter_array",
    "scatter_list",
    "scatter_string",
    "sgl_gps",
    "shift",
    "shift_left",
    "shift_right",
    "total_exchange",
    "totex",
]

Op = Callable[[T, T], T]


class ScatterError(InvalidProcessor):
    """Raised by ``scatter`` when the root is not a valid processor."""


class GatherError(InvalidProcessor):
    """Raised by ``gather`` when the destination is not a valid processor."""


class BcastError(InvalidProcessor):
    """Raised by the broadcasts when the root is not a valid processor."""


# --------------------------------------------------------------------------
# Shifts and total exchange
# --------------------------------------------------------------------------


def shift(dec: int, datas: Par[T]) -> Par[T]:
    """Shift the values from processor to processor: the value of processor
    ``i`` goes to processor ``(i + dec) mod p``.  One superstep; the cost is
    ``n*g + l`` where ``n`` is the size of the biggest value."""
    p = bsp_p()
    mkmsg = mkpar(
        lambda pid: lambda data: lambda dst: (data,) if dst == natmod(dec + pid, p) else None
    )
    received = put(apply(mkmsg, datas))
    return apply(mkpar(lambda pid: lambda f: f(natmod(pid - dec, p))[0]), received)


def shift_right(vec: Par[T]) -> Par[T]:
    """``<v_0, ..., v_{p-1}>`` becomes ``<v_{p-1}, v_0, ..., v_{p-2}>``."""
    return shift(1, vec)


def shift_left(vec: Par[T]) -> Par[T]:
    """``<v_0, ..., v_{p-1}>`` becomes ``<v_1, ..., v_{p-1}, v_0>``."""
    return shift(-1, vec)


def totex(vv: Par[T]) -> Par[Callable[[int], T]]:
    """Total exchange: ``totex(<v_0, ..., v_{p-1}>)`` is a vector of
    functions ``<f_0, ..., f_{p-1}>`` such that ``f_i(j) = v_j``."""
    received = put(parfun(lambda v: lambda dst: (v,), vv))
    return parfun(lambda f: lambda j: f(j)[0], received)


def total_exchange(vec: Par[T]) -> Par[List[T]]:
    """Total exchange: every processor gets the list ``[v_0, ..., v_{p-1}]``."""
    p = bsp_p()
    return parfun(lambda f: [f(j) for j in range(p)], totex(vec))


# --------------------------------------------------------------------------
# Scatter and gather
# --------------------------------------------------------------------------


def scatter(partition: Callable[[T, int], Optional[U]], root: int, v: Par[T]) -> Par[Optional[U]]:
    """Scatter the value held by processor ``root``.

    ``partition(value, pid)`` is the part of the value that is sent to
    processor ``pid`` (``None`` to send nothing; the processor then gets
    ``None``).  Raises :class:`ScatterError` if ``root`` is not a valid
    processor number.  One superstep."""
    if not within_bounds(root):
        raise ScatterError(root)
    mkmsg = mkpar(
        lambda pid: (
            (lambda x: lambda dst: partition(x, dst))
            if pid == root
            else (lambda x: lambda dst: None)
        )
    )
    msg = put(apply(mkmsg, v))
    return parfun(lambda f: f(root), msg)


def _cut(x: Any, i: int, p: int) -> Any:
    """The ``i``-th of the ``p`` consecutive parts of the sequence ``x``.

    The sizes of the parts differ by at most one, the bigger parts first."""
    quotient, remainder = divmod(len(x), p)
    if i < remainder:
        start = i * (quotient + 1)
        return x[start : start + quotient + 1]
    start = remainder * (quotient + 1) + (i - remainder) * quotient
    return x[start : start + quotient]


def _paste(parts: List[Any]) -> Any:
    """Concatenate the parts produced by :func:`_cut`."""
    first = parts[0]
    if isinstance(first, str):
        return "".join(parts)
    if isinstance(first, (bytes, bytearray)):
        return type(first)(b"").join(parts)
    if isinstance(first, list):
        return [x for part in parts for x in part]
    if isinstance(first, tuple):
        return tuple(x for part in parts for x in part)
    if type(first).__module__.split(".")[0] == "numpy":
        import numpy

        return numpy.concatenate(parts)
    return reduce(lambda a, b: a + b, parts)


def scatter_list(root: int, vl: Par[Sequence[T]]) -> Par[Sequence[T]]:
    """Scatter the sequence held by processor ``root`` in ``p`` consecutive
    parts of (almost) equal sizes: processor ``i`` gets the ``i``-th part."""
    p = bsp_p()
    return scatter(lambda x, i: _cut(x, i, p), root, vl)  # type: ignore[return-value]


scatter_array = scatter_list
scatter_string = scatter_list


def gather(dst: int, vv: Par[T]) -> Par[Callable[[int], Optional[T]]]:
    """Gather the values ``v_0, ..., v_{p-1}`` on processor ``dst``.

    The result is a vector of functions; at processor ``dst`` the function
    ``f`` is such that ``f(i) = v_i``, at the other processors ``f(i)`` is
    ``None``.  Raises :class:`GatherError` if ``dst`` is not a valid
    processor number.  One superstep."""
    if not within_bounds(dst):
        raise GatherError(dst)
    mkmsg = mkpar(lambda pid: lambda v: lambda dest: (v,) if dest == dst else None)
    received = put(apply(mkmsg, vv))

    def unwrap(f: Callable[[int], Any]) -> Callable[[int], Any]:
        def get(i: int) -> Any:
            message = f(i)
            return None if message is None else message[0]

        return get

    return parfun(unwrap, received)


def gather_list(dst: int, vv: Par[T]) -> Par[List[T]]:
    """Gather the values on processor ``dst``: it gets the list
    ``[v_0, ..., v_{p-1}]`` and the other processors get ``[]``."""
    p = bsp_p()
    procs_at_dst = mkpar(lambda i: list(range(p)) if i == dst else [])
    return parfun(lambda f, l: [f(i) for i in l], gather(dst, vv), procs_at_dst)


# --------------------------------------------------------------------------
# Broadcasts
# --------------------------------------------------------------------------


def bcast_direct(root: int, vv: Par[T]) -> Par[T]:
    """Direct broadcast: ``<v_0, ..., v_{p-1}>`` becomes ``<v_root, ...,
    v_root>``.  Raises :class:`BcastError` if ``root`` is not a valid
    processor number.  One superstep; the cost is ``size*(p-1)*g + l`` where
    ``size`` is the size of the value ``v_root``."""
    if not within_bounds(root):
        raise BcastError(root)
    mkmsg = mkpar(lambda pid: lambda v: lambda dst: (v,) if pid == root else None)
    return parfun(lambda f: f(root)[0], put(apply(mkmsg, vv)))


def bcast_totex_gen(
    partition: Callable[[T, int], Optional[U]],
    paste: Callable[[Callable[[int], Optional[U]]], V],
    root: int,
    vv: Par[T],
) -> Par[V]:
    """Two-phase ("total exchange") broadcast of the value at ``root``.

    First the value is scattered using ``partition``, then the parts are
    totally exchanged and pasted with ``paste``, which gets a function from
    processor numbers to parts.  Two supersteps; for large values this is
    faster than :func:`bcast_direct` (cost ``2*size*g + 2*l``)."""
    if not within_bounds(root):
        raise BcastError(root)
    phase1 = scatter(partition, root, vv)
    phase2 = totex(phase1)
    return parfun(paste, phase2)


def bcast_totex_list(root: int, vl: Par[Sequence[T]]) -> Par[Sequence[T]]:
    """Two-phase broadcast of a sequence (list, tuple, string, bytes, NumPy
    array) held by processor ``root``."""
    p = bsp_p()
    return bcast_totex_gen(
        lambda x, i: _cut(x, i, p),
        lambda f: _paste([f(i) for i in range(p)]),
        root,
        vl,
    )


bcast_totex_array = bcast_totex_list
bcast_totex_string = bcast_totex_list


def bcast_totex(root: int, vv: Par[T]) -> Par[T]:
    """Two-phase broadcast of a value of any type: the value is serialised
    and the resulting bytes are broadcast with ``bcast_totex_string``."""
    if not within_bounds(root):
        raise BcastError(root)

    def nothing(v: Any) -> bytes:
        return b""

    serialise: Par[Callable[[Any], bytes]] = mkpar(lambda pid: dumps if pid == root else nothing)
    return parfun(loads, bcast_totex_string(root, apply(serialise, vv)))


# --------------------------------------------------------------------------
# Scans (parallel prefixes)
# --------------------------------------------------------------------------


def scan_direct(op: Op, vv: Par[T]) -> Par[T]:
    """Parallel prefix: if ``op`` is associative, ``scan_direct(op,
    <v_0, ..., v_{p-1}>)`` is ``<s_0, ..., s_{p-1}>`` where ``s_i = v_0 op
    v_1 op ... op v_i``.  One superstep; the communication cost is
    ``(p-1)*n*g + l`` where ``n`` is the size of the values."""
    mkmsg = mkpar(lambda pid: lambda v: lambda dst: None if dst < pid else (v,))
    received = put(apply(mkmsg, vv))
    return apply(
        mkpar(lambda pid: lambda f: reduce(op, [f(j)[0] for j in range(pid + 1)])),
        received,
    )


def scan_logp(op: Op, vec: Par[T]) -> Par[T]:
    """Same result as :func:`scan_direct` in ``ceil(log2 p)`` supersteps,
    each one of communication cost ``n*g + l``."""
    p = bsp_p()

    def step(n: int, vec: Par[T]) -> Par[T]:
        # each processor sends its value n processors further and combines
        # its own value with the one it receives from n processors before
        def message(pid: int) -> Callable[[T], Callable[[int], Any]]:
            return lambda v: lambda dst: (v,) if dst == pid + n else None

        def combine(pid: int) -> Callable[[T, Callable[[int], Any]], T]:
            if pid >= n:
                return lambda v, f: op(f(pid - n)[0], v)
            return lambda v, f: v

        received = put(apply(mkpar(message), vec))
        return apply(mkpar(combine), vec, received)

    n = 1
    while n < p:
        vec = step(n, vec)
        n *= 2
    return vec


def scan_wide(
    scan: Callable[[Op, Par[T]], Par[T]],
    seq_scan: Callable[[Op, C], C],
    last_element: Callable[[C], T],
    map_: Callable[[Callable[[T], T], C], C],
    op: Op,
    vv: Par[C],
) -> Par[C]:
    """Parallel prefix over a vector of collections of values.

    ``scan`` is the parallel scan used on one value per processor,
    ``seq_scan(op, collection)`` the sequential scan, ``last_element`` gives
    the last element of a collection and ``map_(f, collection)`` maps a
    function over a collection.  Every local collection must be non empty."""
    local_scan = parfun(lambda c: seq_scan(op, c), vv)
    last_elements = parfun(last_element, local_scan)
    values_to_add = shift_right(scan(op, last_elements))
    pop = applyat(0, lambda x: lambda y: y, lambda x: lambda y: op(x, y), values_to_add)
    return parfun(map_, pop, local_scan)


def scan_wide_direct(
    seq_scan: Callable[[Op, C], C],
    last_element: Callable[[C], T],
    map_: Callable[[Callable[[T], T], C], C],
    op: Op,
    vv: Par[C],
) -> Par[C]:
    """:func:`scan_wide` using :func:`scan_direct` as parallel scan."""
    return scan_wide(scan_direct, seq_scan, last_element, map_, op, vv)


def scan_wide_logp(
    seq_scan: Callable[[Op, C], C],
    last_element: Callable[[C], T],
    map_: Callable[[Callable[[T], T], C], C],
    op: Op,
    vv: Par[C],
) -> Par[C]:
    """:func:`scan_wide` using :func:`scan_logp` as parallel scan."""
    return scan_wide(scan_logp, seq_scan, last_element, map_, op, vv)


def _scan_list(scan: Callable[[Op, Par], Par], op: Op, vl: Par) -> Par:
    return scan_wide(
        scan,
        lambda op, l: list(accumulate(l, op)),
        lambda l: l[-1],
        lambda f, l: [f(x) for x in l],
        op,
        vl,
    )


def scan_list_direct(op: Op, vl: Par[Sequence[T]]) -> Par[List[T]]:
    """Parallel prefix of a distributed list (non empty on every processor):
    the concatenation of the result is the prefix of the concatenation of
    the argument."""
    return _scan_list(scan_direct, op, vl)


def scan_list_logp(op: Op, vl: Par[Sequence[T]]) -> Par[List[T]]:
    """Same as :func:`scan_list_direct` using :func:`scan_logp`."""
    return _scan_list(scan_logp, op, vl)


scan_array_direct = scan_list_direct
scan_array_logp = scan_list_logp


# --------------------------------------------------------------------------
# Folds (parallel reductions)
# --------------------------------------------------------------------------


def fold_direct(op: Op, vec: Par[T]) -> Par[T]:
    """Parallel reduction: every processor gets ``v_0 op v_1 op ... op
    v_{p-1}``.  One superstep (a total exchange)."""
    return parfun(lambda l: reduce(op, l), total_exchange(vec))


def fold_wide(
    par_fold: Callable[[Op, Par[T]], Par[T]],
    local_fold: Callable[[Op, C], T],
    op: Op,
    vec: Par[C],
) -> Par[T]:
    """Parallel reduction of a vector of collections: each collection is
    first reduced with ``local_fold(op, collection)``, then the partial
    results are reduced with the parallel fold ``par_fold``."""
    local_folded = parfun(lambda c: local_fold(op, c), vec)
    return par_fold(op, local_folded)


def fold_logp(op: Op, vec: Par[T]) -> Par[T]:
    """Same result as :func:`fold_direct`, computed with :func:`scan_logp`
    followed by a broadcast of the value of the last processor."""
    return bcast_totex(bsp_p() - 1, scan_logp(op, vec))


def fold_list_direct(op: Op, vl: Par[Sequence[T]]) -> Par[T]:
    """Parallel reduction of a distributed list (non empty on every
    processor)."""
    return fold_wide(fold_direct, reduce, op, vl)


def fold_list_logp(op: Op, vl: Par[Sequence[T]]) -> Par[T]:
    """Same as :func:`fold_list_direct` using :func:`fold_logp`."""
    return fold_wide(fold_logp, reduce, op, vl)


fold_array_direct = fold_list_direct
fold_array_logp = fold_list_logp


# --------------------------------------------------------------------------
# Multi-level machines: following the tree of the groups of processors
# --------------------------------------------------------------------------


def _leaders() -> List[List[int]]:
    """``leaders[k][pid]``: the leader (smallest processor) of the group of
    level ``k`` that contains ``pid``, for ``k`` from 0 to the number of
    levels."""
    table = []
    for level in range(bsp_levels() + 1):
        row = [0] * bsp_p()
        for group in bsp_groups(level):
            for pid in group:
                row[pid] = group[0]
        table.append(row)
    return table


def fold_multi(op: Op, vec: Par[T]) -> Par[T]:
    """Parallel reduction that follows the levels of the machine.

    Same result as :func:`fold_direct`: every processor gets ``v_0 op v_1 op
    ... op v_{p-1}``.  But instead of a total exchange between all the
    processors, the values go up the tree of the groups: in each group of
    level 1 the processors send their value to the leader of the group,
    which reduces them; the leaders of the level-1 groups do the same inside
    their level-2 group, and so on; the leaders of the biggest groups
    exchange their partial results; then the result goes down the same way.

    On a machine with ``L`` levels this takes ``2L - 1`` supersteps, of
    levels ``1, ..., L-1, L, L-1, ..., 1``, instead of one global superstep,
    but only one value per group of level ``L - 1`` crosses the slowest
    links, instead of one per processor.  It pays off when these links are
    much slower than the others and the values are big; ``bsp4py.probe`` and
    ``global_stats`` tell.  On a flat machine it is :func:`fold_direct`.

    ``op`` must be associative.  The values are combined in the order of
    the groups: this is the order of the processors when the groups are
    made of consecutive processors (always the case with a shape, like
    ``2x4``), otherwise ``op`` must also be commutative.
    """
    levels = bsp_levels()
    leaders = _leaders()
    p = bsp_p()

    def reducer(sources: List[int]) -> Callable[[Callable[[int], Any]], T]:
        return lambda f: reduce(op, [f(source)[0] for source in sources])

    def up(level: int, partial: Par[Any]) -> Par[Any]:
        """The leaders of the groups of the level below send their partial
        result to the leader of their group (to all the leaders, at the
        last level), which reduces what it receives."""
        below, here = leaders[level - 1], leaders[level]
        top = level == levels

        def send(pid: int) -> Callable[[Any], Callable[[int], Any]]:
            if below[pid] != pid:
                return lambda x: lambda dst: None
            if top:
                return lambda x: lambda dst: (x,) if below[dst] == dst else None
            return lambda x: lambda dst: (x,) if dst == here[pid] else None

        def combine(pid: int) -> Callable[[Callable[[int], Any]], Any]:
            if below[pid] != pid or not (top or here[pid] == pid):
                return lambda f: None
            return reducer(sorted({below[j] for j in range(p) if here[j] == here[pid]}))

        received = put(apply(mkpar(send), partial), level=level)
        return apply(mkpar(combine), received)

    def down(level: int, result: Par[Any]) -> Par[Any]:
        """The leader of each group sends the result to the leaders of the
        groups of the level below."""
        below, here = leaders[level - 1], leaders[level]

        def send(pid: int) -> Callable[[Any], Callable[[int], Any]]:
            if here[pid] != pid:
                return lambda x: lambda dst: None
            return lambda x: lambda dst: (x,) if below[dst] == dst and dst != pid else None

        def receive(pid: int) -> Callable[[Any, Callable[[int], Any]], Any]:
            if here[pid] == pid:
                return lambda x, f: x
            if below[pid] == pid:
                return lambda x, f: f(here[pid])[0]
            return lambda x, f: None

        received = put(apply(mkpar(send), result), level=level)
        return apply(mkpar(receive), result, received)

    value: Par[Any] = vec
    for level in range(1, levels + 1):
        value = up(level, value)
    for level in range(levels - 1, 0, -1):
        value = down(level, value)
    return value


def bcast_multi(root: int, vv: Par[T]) -> Par[T]:
    """Broadcast that follows the levels of the machine.

    Same result as :func:`bcast_direct`: ``<v_root, ..., v_root>``.  The
    value goes down the tree of the groups: ``root`` sends it to one
    processor of each of the other groups of level ``L - 1``; then, inside
    each of these groups, the processor that has the value sends it to one
    processor of each of the other groups of level ``L - 2``; and so on down
    to the processors.

    On a machine with ``L`` levels this takes ``L`` supersteps, of levels
    ``L, L-1, ..., 1``, instead of one global superstep, but the value
    crosses the slowest links once per group of level ``L - 1`` instead of
    once per processor.  On a flat machine it is :func:`bcast_direct`.
    Raises :class:`BcastError` if ``root`` is not a valid processor number.
    """
    if not within_bounds(root):
        raise BcastError(root)
    levels = bsp_levels()
    # holders[k][pid]: the processor of the level-k group of pid that gets
    # the value first: root in its own groups, the leader in the others
    holders = []
    for level in range(levels + 1):
        row = [0] * bsp_p()
        for group in bsp_groups(level):
            holder = root if root in group else group[0]
            for pid in group:
                row[pid] = holder
        holders.append(row)

    def down(level: int, value: Par[Any]) -> Par[Any]:
        """In each group of the given level, the processor that has the
        value sends it to the holders of the groups of the level below."""
        below, here = holders[level - 1], holders[level]

        def send(pid: int) -> Callable[[Any], Callable[[int], Any]]:
            if here[pid] != pid:
                return lambda x: lambda dst: None
            return lambda x: lambda dst: (x,) if below[dst] == dst and dst != pid else None

        def receive(pid: int) -> Callable[[Any, Callable[[int], Any]], Any]:
            if here[pid] == pid:
                return lambda x, f: x
            if below[pid] == pid:
                return lambda x, f: f(here[pid])[0]
            return lambda x, f: None

        received = put(apply(mkpar(send), value), level=level)
        return apply(mkpar(receive), value, received)

    value: Par[Any] = vv
    for level in range(levels, 0, -1):
        value = down(level, value)
    return value


# --------------------------------------------------------------------------
# GPS: all-to-all communication described by two sequential functions
# --------------------------------------------------------------------------


def sgl_gps(
    split: Callable[[int, T], Sequence[U]],
    assemble: Callable[[int, List[U]], V],
    indata: Par[T],
) -> Par[V]:
    """General communication in the style of SGL (gather-process-scatter).

    ``sgl_gps(split, assemble, <x_0, ..., x_{p-1}>)`` is
    ``<y_0, ..., y_{p-1}>`` where

    * ``split(i, x_i)`` is the sequence of the ``p`` values that processor
      ``i`` sends: the ``j``-th one goes to processor ``j``;
    * ``assemble(j, received)`` is the value ``y_j`` that processor ``j``
      builds from the list of the ``p`` values it received:
      ``received[i]`` comes from processor ``i``.

    In other words, if the ``split(i, x_i)`` are seen as the rows of a
    ``p`` x ``p`` matrix, processor ``j`` assembles the ``j``-th column.

    It is a simplified form of ``put``: the program only says, with two
    ordinary sequential functions, how each processor splits its data and
    how it assembles what it receives (the meta-data of the communication),
    separately from the data, the third argument.  The program reads as if
    the data were gathered on a master, rearranged and scattered again (an
    SGL program ``G; P; S``), but the processors exchange it directly.  One
    superstep, like ``put``: its cost is the one of the h-relation formed by
    the values sent.  A value that is ``None`` is not transmitted (and is
    received as ``None``).

    This is a port of ``sgl_gps`` as defined in BSML by Chong Li in Section
    6.2 ("Simplifying BSML's put", Listing 5) of his PhD thesis *Un modèle
    de transition logico-matérielle pour la simplification de la
    programmation parallèle* (Université Paris-Est, 2013,
    https://theses.hal.science/tel-00952082); see also the paper *GPS:
    Towards Simplified Communication on SGL Model* (IEEE Xplore, document
    6969454).  OCaml type: ``(int -> 'a -> 'b list) -> (int -> 'b list ->
    'c) -> 'a par -> 'c par``.  As everywhere in this package the two
    functions take their arguments together, ``split(pid, value)``, instead
    of being curried; ``split`` may return any sequence of length ``p`` (the
    thesis notes that arrays should be allowed besides lists).
    """
    p = bsp_p()
    procs_list = list(range(p))

    def parts(pid: int, value: T) -> List[U]:
        result = list(split(pid, value))
        if len(result) != p:
            raise ValueError(
                f"sgl_gps: split must return one value per processor ({p}), "
                f"got {len(result)} on processor {pid}"
            )
        return result

    splitted = apply(mkpar(lambda pid: lambda value: parts(pid, value)), indata)
    exchange = put(parfun(lambda l: lambda dst: l[dst], splitted))
    permuted = parfun(lambda f: [f(src) for src in procs_list], exchange)
    return apply(mkpar(lambda pid: lambda received: assemble(pid, received)), permuted)
