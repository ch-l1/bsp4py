"""Parallel sorting by regular sampling (PSRS).

python examples/sort.py [n]
mpiexec -n 4 python examples/sort.py [n]
"""

import random
import sys

import bsp4py
from bsp4py import bsp_p, mkpar, parfun, print_once, regular_sampling_sort, to_list

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
p = bsp_p()


def random_block(pid):
    """Each processor draws its own part of the data."""
    rng = random.Random(pid)
    return [rng.random() for _ in range(n // p + (1 if pid < n % p else 0))]


data = mkpar(random_block)

bsp4py.start_timing()
result = regular_sampling_sort(data)
bsp4py.stop_timing()

sizes = to_list(parfun(len, result))
seconds = max(to_list(bsp4py.get_cost()))
print_once(f"sorted {sum(sizes)} numbers on {p} processors in {seconds:.3f} s")
print_once("elements per processor:", sizes)

# check: locally sorted, and the last element of a processor is not greater
# than the first element of the next one
bounds = to_list(parfun(lambda l: (l[0], l[-1], l == sorted(l)) if l else None, result))
bounds = [b for b in bounds if b is not None]
assert all(ok for _, _, ok in bounds)
assert all(bounds[i][1] <= bounds[i + 1][0] for i in range(len(bounds) - 1))
print_once("the result is sorted")
