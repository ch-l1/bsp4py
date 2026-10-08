"""Parallel sample-sort written with the GPS function of SGL.

This is a port of Listing 6 of Chong Li's PhD thesis (Section 6.2.3,
"Implementing Tiskin-McColl parallel sample-sort with GPS function"), and of
Listing 7, the same algorithm written with ``put`` (from L. Gesbert's PhD
thesis), to compare the two styles.  As in the original, every processor must
hold at least p elements, and at least 2.

    python examples/sample_sort_gps.py [n]
    python -m bsp4py.run -n 4 examples/sample_sort_gps.py [n]
    mpiexec -n 4 python examples/sample_sort_gps.py [n]
"""

import random
import sys
from bisect import bisect_left

import bsp4py
from bsp4py import bsp_p, mkpar, parfun, print_once, procs, proj, put, sgl_gps, to_list

# === Auxiliary functions ===


def ocaml_mod(a, b):
    """The remainder of OCaml: it has the sign of ``a``."""
    return a % b if a >= 0 else -(-a % b)


def extract_n(n, length, l):
    """The n-1 elements that cut the list ``l`` of ``length`` elements in n
    parts of equal sizes."""
    return [x for i, x in enumerate(l) if ocaml_mod(n * i - 1, length) >= length - n]


def slice_p(l, pivots):
    """Cut the sorted list ``l`` at the pivots: ``len(pivots) + 1`` lists,
    the elements of the i-th one are smaller than the i-th pivot."""
    slices, start = [], 0
    for pivot in pivots:
        end = bisect_left(l, pivot, start)
        slices.append(l[start:end])
        start = end
    slices.append(l[start:])
    return slices


def merge(l1, l2):
    """Merge two sorted lists."""
    merged, i, j = [], 0, 0
    while i < len(l1) and j < len(l2):
        if l1[i] < l2[j]:
            merged.append(l1[i])
            i += 1
        else:
            merged.append(l2[j])
            j += 1
    return merged + l1[i:] + l2[j:]


def p_merge(p, ll):
    """Merge ``p`` sorted lists, two halves at a time."""
    if not ll:
        return []
    if len(ll) == 1:
        return ll[0]
    return merge(p_merge(p // 2, ll[: p // 2]), p_merge(p - p // 2, ll[p // 2 :]))


def gather_list(parlist):
    """Concatenation of the lists of a parallel vector (a ``proj``)."""
    f = proj(parlist)
    return [x for pid in procs() for x in f(pid)]


# === Parallel Sorting by Regular Sampling ===


def regular_sample_sort(lvlengths, lv):
    """With the GPS function (Listing 6)."""
    p = bsp_p()
    locsort = parfun(sorted, lv)
    regsampl = parfun(lambda l, length: extract_n(p, length, l), locsort, lvlengths)
    glosampl = sorted(gather_list(regsampl))
    pivots = extract_n(p, p * (p - 1), glosampl)
    split = lambda pid, send_raw: slice_p(send_raw, pivots)
    assemble = lambda pid, recv_raw: p_merge(p, recv_raw)
    return sgl_gps(split, assemble, locsort)


def psrs(lvlengths, lv):
    """With put (Listing 7)."""
    p = bsp_p()
    locsort = parfun(sorted, lv)
    regsampl = parfun(lambda l, length: extract_n(p, length, l), locsort, lvlengths)
    glosampl = sorted(gather_list(regsampl))
    pivots = extract_n(p, p * (p - 1), glosampl)
    comm = parfun(lambda l: slice_p(l, pivots), locsort)
    recv = put(parfun(lambda slices: lambda dst: slices[dst], comm))
    return parfun(lambda f: p_merge(p, [f(src) for src in range(p)]), recv)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    p = bsp_p()

    def random_block(pid):
        rng = random.Random(pid)
        return [rng.randrange(1_000_000) for _ in range(n // p + (1 if pid < n % p else 0))]

    data = mkpar(random_block)
    lengths = parfun(len, data)

    results = {}
    for sort in (regular_sample_sort, psrs):
        bsp4py.reset_stats()
        bsp4py.start_timing()
        result = sort(lengths, data)
        bsp4py.stop_timing()
        supersteps = bsp4py.stats().supersteps
        seconds = max(to_list(bsp4py.get_cost()))
        results[sort.__name__] = to_list(result)
        print_once(f"{sort.__name__}: {supersteps} supersteps, {seconds:.3f} s")

    blocks = results["regular_sample_sort"]
    flat = [x for block in blocks for x in block]
    assert flat == sorted(x for block in to_list(data) for x in block)
    assert blocks == results["psrs"]
    print_once(f"sorted {len(flat)} numbers on {p} processors; elements per processor:")
    print_once([len(block) for block in blocks])
