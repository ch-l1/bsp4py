"""Test configuration.

The tests only observe parallel vectors through ``proj`` (``to_list``), so
every assertion is about a global value.  This makes the whole suite a valid
SPMD program.  It can be run sequentially (``pytest``), where each test is
run for several numbers of simulated processors, or in parallel, where every
process runs the same tests in the same order::

    python -m bsp4py.run -n 4 -m pytest -p no:cacheprovider
    mpiexec -n 4 python -m pytest -p no:cacheprovider
"""

import os

import pytest

import bsp4py
from bsp4py import _core

BACKEND = os.environ.get("BSP4PY_BACKEND", "auto").lower()
if BACKEND == "auto":
    BACKEND = _core._auto_backend()
IS_PARALLEL = BACKEND != "seq"

if IS_PARALLEL:
    bsp4py.init(BACKEND)
    PROCESSORS = [bsp4py.bsp_p()]
else:
    PROCESSORS = [1, 2, 3, 4, 5, 8]

seq_only = pytest.mark.skipif(IS_PARALLEL, reason="specific to the sequential backend")

# Structures of multi-level machines: shapes, or the paths of the processors
# for a machine that is not regular.  A parallel session has the structure it
# was started with.
IRREGULAR = [(0, 0), (1, 0), (0, 1), (1, 1), (1, 2)]  # groups {0, 2} and {1, 3, 4}
MACHINES = (
    [None]
    if IS_PARALLEL
    else ["4", "1", "2x2", "2x3", "3x2", "1x3", "3x1", "2x2x2", "3x1x2", IRREGULAR]
)


@pytest.fixture(params=PROCESSORS, autouse=True, ids=lambda n: f"p={n}")
def p(request):
    """Run each test for every number of processors."""
    if IS_PARALLEL:
        bsp4py.reset_stats()
    else:
        bsp4py.init("seq", p=request.param)
    return request.param


@pytest.fixture(params=MACHINES, ids=lambda m: "current" if m is None else str(m).replace(" ", ""))
def machine(request, p):
    """Run a test for several structures of machine (with the sequential
    backend; use it with the ``single`` marker)."""
    if request.param is None:
        return bsp4py.bsp_shape()
    if isinstance(request.param, str):
        bsp4py.init("seq", shape=request.param)
    else:
        bsp4py.init("seq", topology=request.param)
    return request.param


def pytest_collection_modifyitems(config, items):
    """Tests marked ``single`` do not depend on the number of processors:
    they are run once instead of once per value of ``p``."""
    kept, dropped = [], []
    for item in items:
        callspec = getattr(item, "callspec", None)
        repeated = callspec is not None and callspec.params.get("p") != PROCESSORS[0]
        (dropped if item.get_closest_marker("single") and repeated else kept).append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept
