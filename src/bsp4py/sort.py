"""Parallel sorting (port of the module ``Sort`` of BSML).

The algorithm is the BSP regular sampling sort (PSRS) described in
K. R. Sujithan, "Towards a Scalable Parallel Object Database", PRG-TR-17-96,
Oxford University, as implemented in BSML 0.5.

Adaptation to Python: the order is given like for ``sorted``, by an optional
``key`` function, instead of a comparison function, and the same function
handles lists and any other sequence.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from heapq import merge
from typing import Any, Callable, List, Optional, TypeVar

from ._core import BspError, Par, apply, bsp_p, mkpar
from .base import parfun, put_list, put_one
from .comm import bcast_totex, total_exchange

T = TypeVar("T")

__all__ = [
    "RegularSamplingSortError",
    "regular_sampling_sort",
    "regular_sampling_sort_array",
    "regular_sampling_sort_list",
]


class RegularSamplingSortError(BspError, ValueError):
    """Raised when there are fewer than ``p*p`` elements to sort."""


def regular_sampling_sort(
    vec: Par[Sequence[T]], *, key: Optional[Callable[[T], Any]] = None
) -> Par[List[T]]:
    """Sort a distributed sequence with the regular sampling BSP algorithm.

    ``vec`` holds one sequence per processor.  The result holds one sorted
    list per processor and all the elements of processor ``i`` are before
    the elements of processor ``i+1`` in the order: the concatenation of the
    lists is the sorted concatenation of the argument.  The sort is stable.

    The total number of elements ``n`` must be at least ``p*p``, otherwise
    :class:`RegularSamplingSortError` is raised.  The algorithm ensures that
    at the end each processor holds at most ``2*n/p`` elements (when the
    elements are distinct).  Five supersteps.
    """
    p = bsp_p()

    def keys_of(xs: Sequence[T]) -> List[Any]:
        return list(xs) if key is None else [key(x) for x in xs]

    locally_sorted = parfun(lambda l: sorted(l, key=key), vec)
    local_lengths = total_exchange(parfun(len, vec))

    def total(lengths: List[int]) -> int:
        n = sum(lengths)
        if n < p * p:
            raise RegularSamplingSortError(
                f"regular sampling sort needs at least p*p = {p * p} elements, got {n}"
            )
        return n

    global_length = parfun(total, local_lengths)
    sample_stride = parfun(lambda n: n // (p * p), global_length)

    # Each processor contributes a number of samples proportional to the
    # number of elements it holds, p*p samples in total.
    def samples_per_processor(n: int, lengths: List[int]) -> List[int]:
        return [int((length * p * p) / (n - 0.5)) for length in lengths]

    approximation = parfun(samples_per_processor, global_length, local_lengths)
    number_of_samples = apply(
        mkpar(lambda pid: lambda ap: ap[pid] + (1 if pid < p * p - sum(ap) else 0)),
        approximation,
    )

    # Superstep: the regular samples are sent to processor 0, which sorts
    # them and picks p-1 regular pivots: the samples of ranks p + p/2,
    # 2p + p/2, ...  (This is the choice of the array version of BSML 0.5;
    # its list version starts one sample earlier, which with 2 or 3
    # processors selects one of the local minima and puts almost nothing on
    # the first processor.)
    sent_samples = put_one(
        parfun(
            lambda nos, stride, ls: (0, ls[::stride][: max(nos, 0)]),
            number_of_samples,
            sample_stride,
            locally_sorted,
        )
    )
    sorted_samples = parfun(lambda ll: sorted((x for l in ll for x in l), key=key), sent_samples)
    regular_pivots = bcast_totex(
        0,
        apply(
            mkpar(
                lambda pid: (
                    (lambda ss: ss[p + p // 2 - 1 :: p][: p - 1]) if pid == 0 else (lambda ss: ss)
                )
            ),
            sorted_samples,
        ),
    )

    # Superstep: each processor cuts its sorted list with the pivots and
    # sends the i-th block to processor i, which merges what it receives.
    def partition(pivots: List[T], ls: List[T]) -> List[Any]:
        keys = keys_of(ls)
        bounds = [0] + [bisect_right(keys, k) for k in keys_of(pivots)] + [len(ls)]
        blocks = [(i, ls[bounds[i] : bounds[i + 1]]) for i in range(len(bounds) - 1)]
        return [(i, block) for i, block in blocks if block]

    distributed_partitions = put_list(parfun(partition, regular_pivots, locally_sorted))
    return parfun(lambda blocks: list(merge(*blocks, key=key)), distributed_partitions)


regular_sampling_sort_list = regular_sampling_sort
regular_sampling_sort_array = regular_sampling_sort
