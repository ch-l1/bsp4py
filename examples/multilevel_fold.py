"""Flat or multi-level?  Predicted and measured cost of two reductions.

On a machine with several levels (a few computers with several cores each,
for example) the links between the groups are much slower than the links
inside a group.  This program runs the same reduction with two algorithms,

* ``fold_direct``: one global superstep, every processor sends its value to
  all the others, and
* ``fold_multi``: the values follow the tree of the groups, one value per
  group crosses the slow links, at the price of more supersteps,

and for each one prints what the cost model needs (the supersteps of each
level and the bytes exchanged at each distance), the time predicted from the
parameters measured by ``bsp4py.probe``, and the time measured.

    BSP4PY_SHAPE=2x4 python examples/multilevel_fold.py          # simulation
    python -m bsp4py.run --shape 2x4 examples/multilevel_fold.py # local processes
    mpiexec -n 8 -hosts a,b python examples/multilevel_fold.py  # 2 computers

With MPI on several computers the structure is found automatically.  On one
computer the groups given by a shape are virtual: all the links have the
same speed and the multi-level algorithm has nothing to gain.  The last
lines show what the model predicts if the outermost links were slower.

Arguments: the size of the values (number of floats, default 100000).
"""

import sys
from array import array
from dataclasses import replace
from statistics import median

import bsp4py
from bsp4py import bsp_levels, bsp_p, fold_direct, fold_multi, mkpar, print_once, to_list
from bsp4py.probe import probe

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
REPEATS = 15

# Each processor holds a vector; the reduction keeps the one that starts with
# the biggest number (cheap to compute: the time goes in the communications).
# The vectors are arrays of floats, which are serialised as raw bytes like
# the messages of the benchmark; lists of Python floats would take several
# times longer to serialise, a local computation that g does not count.
vectors = mkpar(lambda pid: array("d", [float((7 * pid) % bsp_p())] + [0.5] * (n - 1)))
biggest = lambda a, b: a if a[0] >= b[0] else b


def measure(fold):
    """Counters of one run, and median time of several runs (seconds)."""
    bsp4py.reset_stats()
    result = fold(biggest, vectors)
    counters = bsp4py.global_stats()
    times = []
    for _ in range(REPEATS):
        bsp4py.start_timing()
        fold(biggest, vectors)
        bsp4py.stop_timing()
        times.append(max(to_list(bsp4py.get_cost())))
    return result, counters, median(times)


def describe(counters):
    levels = [step.level for step in counters.steps]
    volume = [sum(step.h[d] or 0 for step in counters.steps) for d in range(bsp_levels())]
    return f"supersteps of levels {levels}, bytes at distances 1..{bsp_levels()}: {volume}"


parameters = probe()
print_once(f"{bsp4py.backend_name()} backend, {bsp_p()} processors, shape {bsp4py.bsp_shape()}")
for level in parameters.levels:
    print_once(f"  level {level.level}: g = {level.g:.2e} s/byte, l = {level.l:.2e} s")

# what the model says for a machine whose outermost links are 20 times slower
slower = replace(
    parameters,
    levels=parameters.levels[:-1]
    + (replace(parameters.levels[-1], g=20 * parameters.levels[-1].g),),
)

results = []
for fold in (fold_direct, fold_multi):
    result, counters, seconds = measure(fold)
    results.append(to_list(result))
    print_once(f"{fold.__name__}: {describe(counters)}")
    print_once(
        f"  predicted {1e3 * parameters.communication_time(counters):.3f} ms, "
        f"measured {1e3 * seconds:.3f} ms; "
        f"with outermost links 20 times slower the model predicts "
        f"{1e3 * slower.communication_time(counters):.3f} ms"
    )

assert results[0] == results[1] and all(v == results[0][0] for v in results[0])
print_once("both algorithms give the same result on every processor")
