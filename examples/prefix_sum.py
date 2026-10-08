"""Parallel prefix sums of a distributed list, and their BSP cost.

python examples/prefix_sum.py
mpiexec -n 4 python examples/prefix_sum.py
"""

import bsp4py
from bsp4py import bsp_p, print_once, scan_list_direct, scan_list_logp, skeleton

n = 5 * bsp_p() + 3
numbers = skeleton.make(lambda k: k + 1, n)  # the distributed list [1, 2, ..., n]

for scan in (scan_list_direct, scan_list_logp):
    bsp4py.reset_stats()
    sums = scan(lambda a, b: a + b, numbers)
    supersteps = bsp4py.stats().supersteps
    result = skeleton.extract(sums)
    assert result == [k * (k + 1) // 2 for k in range(1, n + 1)]
    print_once(f"{scan.__name__}: {supersteps} superstep(s), last sums: {result[-3:]}")
