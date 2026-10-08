"""Approximation of pi by numerical integration of 4 / (1 + x*x) on [0, 1].

The n intervals are shared between the processors, each processor computes
its partial sum, and the partial sums are added with a parallel fold.

    python examples/pi.py [n]
    mpiexec -n 4 python examples/pi.py [n]
"""

import sys

from bsp4py import bsp_p, fold_direct, mkpar, print_once, proj

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
p = bsp_p()
h = 1.0 / n


def partial_sum(pid):
    """Sum over the intervals pid, pid + p, pid + 2p, ..."""
    total = 0.0
    for i in range(pid, n, p):
        x = h * (i + 0.5)
        total += 4.0 / (1.0 + x * x)
    return total * h


partial_sums = mkpar(partial_sum)  # local computations, no communication
total = fold_direct(lambda a, b: a + b, partial_sums)  # one superstep
pi = proj(total)(0)  # the same value on every processor
print_once(f"pi is approximately {pi:.12f} ({n} intervals, {p} processors)")
