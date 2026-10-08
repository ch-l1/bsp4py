"""bsp4py: Bulk Synchronous Parallel programming for Python, in the style of BSML.

bsp4py is a library for programming BSP (Bulk Synchronous Parallel)
algorithms in a functional style.  A program manipulates *parallel vectors*
(one value per processor) with four primitives:

* ``mkpar(f)``      -- ``<f(0), ..., f(p-1)>``
* ``apply(vf, vv)`` -- ``<f_0(v_0), ..., f_{p-1}(v_{p-1})>``
* ``put(vf)``       -- global communication: ``f_i(j)`` is sent by ``i`` to ``j``
* ``proj(v)``       -- the function ``i -> v_i``

plus ``bsp_p()``, the number of processors.  The same program runs
sequentially (simulation, the default) or in parallel, with local processes
or with MPI::

    python program.py                        # sequential simulation, BSP4PY_P processors
    python -m bsp4py.run -n 8 program.py     # 8 local processes, nothing to install
    mpiexec -n 8 python program.py           # 8 MPI processes (needs mpi4py)

The core of the package is a port of the OCaml library BSML 0.5 (Bulk
Synchronous Parallel ML) by F. Loulergue et al. (https://bsml-lang.github.io);
the functions keep their BSML names.  Everything is available from the
top-level package; the sub-modules mirror the modules of the OCaml library:
``bsp4py.base``, ``bsp4py.comm``, ``bsp4py.sort``, ``bsp4py.skeleton`` and
``bsp4py.utils``.  Added to BSML: the GPS function of the SGL model
(``sgl_gps``) and multi-level machines (``put(vf, level=k)``, ``fold_multi``,
``bsp4py.probe``).
"""

from . import (  # noqa: F401  (_mp reads the launcher's environment)
    _mp,
    base,
    comm,
    skeleton,
    sort,
    utils,
)
from ._core import (
    BackendError,
    BspError,
    InvalidProcessor,
    NestingError,
    Par,
    Stats,
    Superstep,
    TimerFailure,
    abort,
    apply,
    backend_name,
    bsp_distance,
    bsp_g,
    bsp_group,
    bsp_groups,
    bsp_l,
    bsp_leader,
    bsp_levels,
    bsp_p,
    bsp_r,
    bsp_shape,
    bsp_topology,
    get_cost,
    global_stats,
    init,
    mkpar,
    proj,
    put,
    reset_stats,
    start_timing,
    stats,
    stop_timing,
    within_bounds,
)
from .base import *
from .comm import *
from .sort import *
from .utils import *

__version__ = "0.1.0"

__all__ = (
    [
        "BackendError",
        "BspError",
        "InvalidProcessor",
        "NestingError",
        "Par",
        "Stats",
        "Superstep",
        "TimerFailure",
        "abort",
        "apply",
        "backend_name",
        "bsp_distance",
        "bsp_g",
        "bsp_group",
        "bsp_groups",
        "bsp_l",
        "bsp_leader",
        "bsp_levels",
        "bsp_p",
        "bsp_r",
        "bsp_shape",
        "bsp_topology",
        "get_cost",
        "global_stats",
        "init",
        "mkpar",
        "proj",
        "put",
        "reset_stats",
        "start_timing",
        "stats",
        "stop_timing",
        "within_bounds",
        "base",
        "comm",
        "sort",
        "skeleton",
        "utils",
    ]
    + base.__all__
    + comm.__all__
    + sort.__all__
    + utils.__all__
)
