"""Core of bsp4py: parallel vectors and the four primitives.

This module is a port of the core of the OCaml library BSML 0.5
(``src/interface/bsmlsig.ml``, ``src/seq/sequential.ml`` and
``src/par/generic/parallel.ml``).

Three backends implement the same interface:

* ``seq`` -- a sequential simulator.  A parallel vector is a list of ``p``
  values and every primitive is executed for all processors in turn.  It is
  the reference semantics and the natural mode for development, testing and
  interactive use.
* ``mp`` and ``mpi`` -- parallel (SPMD) executions.  The whole program runs
  in ``p`` processes, a parallel vector only holds the local value, and
  ``put``/``proj`` exchange (pickled) messages.  With ``mpi`` the processes
  are started by an MPI launcher and communicate with ``MPI_Alltoall``
  (mpi4py); with ``mp`` they are started by ``python -m bsp4py.run`` and
  communicate through ``multiprocessing.connection`` (see ``_mp.py``).
"""

from __future__ import annotations

import os
import pickle
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar

from ._topology import Path, Shape, Topology, parse_shape

T = TypeVar("T")
U = TypeVar("U")
T_co = TypeVar("T_co", covariant=True)

__all__ = [
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
]


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class BspError(Exception):
    """Base class of all the exceptions raised by bsp4py."""


class InvalidProcessor(BspError, IndexError):
    """Raised when asked for a processor id that is not in ``0 .. bsp_p()-1``.

    In particular it is raised by the functions that :func:`proj` and
    :func:`put` return.
    """

    def __init__(self, pid: Any) -> None:
        super().__init__(f"invalid processor id: {pid!r}")
        self.pid = pid


class TimerFailure(BspError):
    """Raised by a meaningless call to :func:`start_timing`/:func:`stop_timing`."""


class NestingError(BspError):
    """Raised when a program tries to nest parallel vectors.

    BSML forbids parallel vectors of parallel vectors: the BSP model is flat
    and nesting would break the cost model.  OCaml BSML leaves this check to
    the programmer (or to a dedicated type system); this port checks it
    dynamically: calling a parallel primitive from inside a local computation
    (the function given to ``mkpar``, a function applied by ``apply``, a
    message function evaluated by ``put``) raises this exception.
    """


class BackendError(BspError):
    """Raised when the backend cannot be set up or is used inconsistently."""


# --------------------------------------------------------------------------
# Serialisation of messages
# --------------------------------------------------------------------------

try:  # closures and lambdas can be sent in messages when cloudpickle is there
    import cloudpickle as _pickler  # type: ignore
except ImportError:  # pragma: no cover - depends on the environment
    _pickler = pickle  # type: ignore


def dumps(value: Any) -> bytes:
    """Serialise a message (uses ``cloudpickle`` when it is installed)."""
    try:
        return _pickler.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except NestingError:
        raise
    except Exception as exc:  # pickle raises many different exception types
        hint = (
            ""
            if _pickler is not pickle
            else (" (install 'cloudpickle' to be able to send lambdas and closures)")
        )
        raise BspError(
            f"a value of type {type(value).__name__} cannot be sent in a "
            f"message because it cannot be serialised: {exc}{hint}"
        ) from exc


def loads(data: bytes) -> Any:
    """Inverse of :func:`dumps`."""
    return pickle.loads(data)


# --------------------------------------------------------------------------
# Local sections and the nesting check
# --------------------------------------------------------------------------

_tls = threading.local()


def _in_local() -> bool:
    return getattr(_tls, "depth", 0) > 0


class _LocalSection:
    """Context manager marking the evaluation of local (per-processor) code."""

    __slots__ = ()

    def __enter__(self) -> None:
        _tls.depth = getattr(_tls, "depth", 0) + 1

    def __exit__(self, *exc: object) -> None:
        _tls.depth -= 1


_local = _LocalSection()


def _require_global(name: str) -> None:
    if _in_local():
        raise NestingError(
            f"'{name}' was called inside a local computation: parallel "
            "primitives can only be used in the global (replicated) part of "
            "the program, parallel vectors cannot be nested"
        )


def _checked(value: T) -> T:
    """Check that the result of a local computation is not a parallel vector."""
    if isinstance(value, Par):
        raise NestingError(
            "a local computation returned a parallel vector: parallel vectors cannot be nested"
        )
    return value


# --------------------------------------------------------------------------
# Parallel vectors
# --------------------------------------------------------------------------


class Par(Generic[T_co]):
    """A parallel vector ``<v_0, ..., v_{p-1}>``: one value per processor.

    This is the abstract type ``'a par`` of BSML.  Parallel vectors are only
    created by :func:`mkpar`, :func:`apply` and :func:`put` and their
    contents are only observed with :func:`proj`; they have no other public
    operation, exactly like in BSML.
    """

    __slots__ = ("_backend", "_data")

    def __init__(self, backend: _Backend, data: Any) -> None:
        self._backend = backend
        self._data = data

    def __repr__(self) -> str:
        return self._backend.show(self)

    def __bool__(self) -> bool:
        raise TypeError("a parallel vector has no truth value; use proj() to observe it")

    def __iter__(self) -> Any:
        raise TypeError("a parallel vector is not iterable; use proj() or to_list()")

    def __len__(self) -> int:
        raise TypeError("a parallel vector has no len(); its size is bsp_p()")

    def __reduce__(self) -> Any:
        raise NestingError(
            "a parallel vector cannot be serialised or sent in a message: "
            "parallel vectors cannot be nested"
        )


class _Delivered(Generic[T]):
    """The function returned by ``proj`` and held in the result of ``put``.

    Calling it with a processor id gives the value associated with that
    processor; any other argument raises :class:`InvalidProcessor`.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Sequence[T]) -> None:
        self._values = values

    def __call__(self, pid: int) -> T:
        if isinstance(pid, bool) or not isinstance(pid, int):
            try:  # accept numpy integers and the like
                pid = pid.__index__()
            except AttributeError:
                raise InvalidProcessor(pid) from None
        if not 0 <= pid < len(self._values):
            raise InvalidProcessor(pid)
        return self._values[pid]

    def __repr__(self) -> str:
        return "<fun: pid -> value>"

    def __reduce__(self) -> Any:
        return (_Delivered, (list(self._values),))


# --------------------------------------------------------------------------
# Statistics (BSP cost accounting)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Superstep:
    """Cost counters of one superstep.

    ``level`` is the level of the superstep: the level of the groups inside
    which the processors communicated and synchronised (the number of levels
    of the machine for an ordinary, global, superstep).

    ``h`` gives the size of the communication in bytes for each distance:
    ``h[d - 1]`` is the maximum over the processors of the number of bytes
    sent to, or received from, processors at distance ``d`` (``None`` when
    it was not measured).  On a machine with two levels ``h[0]`` is what
    went between processors of the same group and ``h[1]`` what went from
    one group to another.
    """

    level: int
    h: Tuple[Optional[int], ...]


@dataclass
class Stats:
    """Counters of the BSP cost of what was executed so far.

    ``supersteps`` is the number of synchronisation barriers, i.e. the number
    of calls to ``put`` and ``proj``.

    ``h_relations`` has one entry per superstep: the size *h* in bytes of the
    communication, i.e. the maximum over the processors of the number of
    bytes sent or received (messages from a processor to itself are not
    counted).  ``steps`` has one :class:`Superstep` per superstep, with its
    level and the same measure for each distance.

    With the ``seq`` backend these are the exact values (``None`` when
    message copying is disabled).  With the parallel backends each process
    only knows its own traffic, so they are the bytes sent and received by
    the calling process; :func:`global_stats` gives the values for the whole
    machine.
    """

    supersteps: int = 0
    h_relations: List[Optional[int]] = field(default_factory=list)
    steps: List[Superstep] = field(default_factory=list)

    def copy(self) -> "Stats":
        return Stats(self.supersteps, list(self.h_relations), list(self.steps))


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


class _Backend:
    name = "?"
    p = 0
    g: Optional[float] = None
    l: Optional[float] = None
    r: Optional[float] = None

    topology: Topology

    def __init__(self) -> None:
        self.stats = Stats()
        self._timer_running = False
        self._timer_value = 0.0

    # -- to be provided by the backends ------------------------------------
    def mkpar(self, f: Callable[[int], Any]) -> Par:
        raise NotImplementedError

    def apply(self, vf: Par, vs: Sequence[Par]) -> Par:
        raise NotImplementedError

    def put(self, vf: Par, level: int) -> Par:
        raise NotImplementedError

    def detect_topology(self) -> Optional[Topology]:
        """Structure of the machine found by the backend itself, if any."""
        return None

    def proj(self, v: Par) -> _Delivered:
        raise NotImplementedError

    def show(self, v: Par) -> str:
        raise NotImplementedError

    def wtime(self) -> float:
        raise NotImplementedError

    def abort(self, code: int) -> None:
        raise NotImplementedError

    # -- timing ------------------------------------------------------------
    def start_timing(self) -> None:
        if self._timer_running:
            raise TimerFailure("Timer is already running")
        self._timer_value = self.wtime()
        self._timer_running = True

    def stop_timing(self) -> None:
        if not self._timer_running:
            raise TimerFailure("Timer was not started!")
        self._timer_value = self.wtime() - self._timer_value
        self._timer_running = False


class _SequentialBackend(_Backend):
    """Sequential simulation: a parallel vector is a list of ``p`` values."""

    name = "seq"

    def __init__(self, p: int, copy_messages: bool = True) -> None:
        super().__init__()
        if p < 1:
            raise BackendError(f"the number of processors must be >= 1, got {p}")
        self.p = p
        self.copy_messages = copy_messages

    def mkpar(self, f: Callable[[int], Any]) -> Par:
        with _local:
            data = [_checked(f(i)) for i in range(self.p)]
        return Par(self, data)

    def apply(self, vf: Par, vs: Sequence[Par]) -> Par:
        fs = vf._data
        with _local:
            data = [_checked(fs[i](*[v._data[i] for v in vs])) for i in range(self.p)]
        return Par(self, data)

    def _transmit(self, value: Any) -> tuple[Any, int]:
        """Simulate the transport of a message: returns (copy, size)."""
        if not self.copy_messages:
            _checked(value)
            return value, 0
        data = dumps(value)
        return loads(data), len(data)

    def _record(self, level: int, sent: List[List[int]], received: List[List[int]]) -> None:
        """Count a superstep: ``sent[d - 1][i]`` is the number of bytes that
        processor ``i`` sent to processors at distance ``d``."""
        self.stats.supersteps += 1
        if self.copy_messages:
            totals = [
                max(sum(row[i] for row in sent), sum(row[i] for row in received))
                for i in range(self.p)
            ]
            per_distance: Tuple[Optional[int], ...] = tuple(
                max(max(s), max(r)) for s, r in zip(sent, received)
            )
            self.stats.h_relations.append(max(totals))
        else:
            per_distance = (None,) * self.topology.levels
            self.stats.h_relations.append(None)
        self.stats.steps.append(Superstep(level, per_distance))

    def put(self, vf: Par, level: int) -> Par:
        p, topology = self.p, self.topology
        with _local:
            # outgoing[i][j]: message from processor i to processor j of its group
            outgoing = [{j: f(j) for j in topology.group(i, level)} for i, f in enumerate(vf._data)]
        inbox: List[List[Any]] = [[None] * p for _ in range(p)]
        sent = [[0] * p for _ in range(topology.levels)]
        received = [[0] * p for _ in range(topology.levels)]
        for i in range(p):
            distances = topology.distances(i)
            for j, message in outgoing[i].items():
                if message is None:  # None means "no message"
                    continue
                inbox[j][i], size = self._transmit(message)
                if i != j:
                    sent[distances[j] - 1][i] += size
                    received[distances[j] - 1][j] += size
        self._record(level, sent, received)
        return Par(self, [_Delivered(inbox[j]) for j in range(p)])

    def proj(self, v: Par) -> _Delivered:
        p, topology = self.p, self.topology
        values = []
        sent = [[0] * p for _ in range(topology.levels)]
        received = [[0] * p for _ in range(topology.levels)]
        for i, value in enumerate(v._data):
            copy, size = self._transmit(value)
            values.append(copy)
            distances = topology.distances(i)
            for j in range(p):
                if j != i:
                    sent[distances[j] - 1][i] += size
                    received[distances[j] - 1][j] += size
        self._record(topology.levels, sent, received)
        return _Delivered(values)

    def show(self, v: Par) -> str:
        return "<" + ", ".join(repr(x) for x in v._data) + ">"

    def wtime(self) -> float:
        return time.perf_counter()

    def abort(self, code: int) -> None:
        sys.exit(code)


def _drain_output() -> None:
    """Flush the standard streams before MPI_Abort.

    MPI launchers kill the processes as soon as one of them aborts and may
    drop the output they have not forwarded yet; the short pause leaves them
    the time to forward the error message that was just written.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(0.2)


class _SpmdBackend(_Backend):
    """Common part of the parallel backends (SPMD execution).

    The whole program runs in ``p`` processes.  A parallel vector only holds
    the value of the local processor, ``mkpar`` and ``apply`` compute
    locally, and ``put`` and ``proj`` are built on one operation that the
    sub-classes provide: a total exchange of buffers.
    """

    pid = 0

    def _alltoall(self, buffers: List[Optional[bytes]], level: int) -> List[Optional[bytes]]:
        """Send ``buffers[j]`` to each processor ``j`` (``None``: nothing)
        and return the buffers received, indexed by source processor.  Only
        the processors of the group of the given level take part: the
        buffers for the others are ``None``."""
        raise NotImplementedError

    def mkpar(self, f: Callable[[int], Any]) -> Par:
        with _local:
            data = _checked(f(self.pid))
        return Par(self, data)

    def apply(self, vf: Par, vs: Sequence[Par]) -> Par:
        with _local:
            data = _checked(vf._data(*[v._data for v in vs]))
        return Par(self, data)

    def _exchange(self, buffers: List[Optional[bytes]], level: int) -> List[Any]:
        me = self.pid
        distances = self.topology.distances(me)
        sent = [0] * self.topology.levels
        received = [0] * self.topology.levels
        for j, b in enumerate(buffers):
            if b is not None and j != me:
                sent[distances[j] - 1] += len(b)
        incoming = self._alltoall(buffers, level)
        for i, b in enumerate(incoming):
            if b is not None and i != me:
                received[distances[i] - 1] += len(b)
        self.stats.supersteps += 1
        self.stats.h_relations.append(max(sum(sent), sum(received)))
        self.stats.steps.append(Superstep(level, tuple(max(s, r) for s, r in zip(sent, received))))
        return [None if b is None else loads(b) for b in incoming]

    def put(self, vf: Par, level: int) -> Par:
        f = vf._data
        buffers: List[Optional[bytes]] = [None] * self.p
        with _local:
            messages = {j: f(j) for j in self.topology.group(self.pid, level)}
        for j, message in messages.items():
            if message is not None:
                buffers[j] = dumps(message)
        return Par(self, _Delivered(self._exchange(buffers, level)))

    def proj(self, v: Par) -> _Delivered:
        buffer = dumps(v._data)
        return _Delivered(self._exchange([buffer] * self.p, self.topology.levels))

    def show(self, v: Par) -> str:
        return f"<... processor {self.pid}: {v._data!r} ...>"


class _MPIBackend(_SpmdBackend):
    """SPMD execution on top of MPI (mpi4py)."""

    name = "mpi"

    def __init__(self) -> None:
        super().__init__()
        try:
            from mpi4py import MPI
        except Exception as exc:
            raise BackendError(
                "the 'mpi' backend needs mpi4py and a working MPI library "
                f"(pip install 'bsp4py[mpi]'): {exc}"
            ) from exc
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.p = self.comm.Get_size()
        self.pid = self.comm.Get_rank()
        self._subcomms: dict = {}
        self._install_excepthook()

    def detect_topology(self) -> Optional[Topology]:
        """Two levels when the processes run on several computers: the
        computers, and the processes of each computer."""
        names = self.comm.allgather(self.MPI.Get_processor_name())
        return Topology.from_hosts(names)

    def _install_excepthook(self) -> None:
        # An uncaught exception on one process must stop the whole machine,
        # otherwise the other processes wait forever at the next barrier.
        if self.p == 1 or getattr(sys.excepthook, "_bsp4py", False):
            return
        previous = sys.excepthook
        comm = self.comm

        def hook(exc_type: Any, exc: Any, tb: Any) -> None:
            previous(exc_type, exc, tb)
            _drain_output()
            comm.Abort(1)

        hook._bsp4py = True  # type: ignore[attr-defined]
        sys.excepthook = hook

    def _alltoall(self, buffers: List[Optional[bytes]], level: int) -> List[Optional[bytes]]:
        if level == self.topology.levels:
            return self.comm.alltoall(buffers)
        # a superstep of a lower level only involves the processes of the
        # group: it uses a communicator of its own, so that the other groups
        # are neither waited for nor disturbed
        group = self.topology.group(self.pid, level)
        key = (id(self.topology), level)
        if key not in self._subcomms:
            self._subcomms[key] = self.comm.Split(color=group[0], key=self.pid)
        received = self._subcomms[key].alltoall([buffers[j] for j in group])
        incoming: List[Optional[bytes]] = [None] * self.p
        for j, buffer in zip(group, received):
            incoming[j] = buffer
        return incoming

    def wtime(self) -> float:
        return self.MPI.Wtime()

    def abort(self, code: int) -> None:
        _drain_output()
        self.comm.Abort(code)
        sys.exit(code)  # pragma: no cover - Abort does not return


# --------------------------------------------------------------------------
# Choice of the backend
# --------------------------------------------------------------------------

_backend: Optional[_Backend] = None

# Environment variables set by the usual MPI launchers for their children.
_MPI_LAUNCHER_VARIABLES = (
    "PMI_RANK",  # MPICH (hydra), Intel MPI, MS-MPI, Slurm (pmi2)
    "PMI_SIZE",
    "PMIX_RANK",  # Open MPI >= 4, Slurm (pmix)
    "OMPI_COMM_WORLD_SIZE",  # Open MPI
    "MPI_LOCALRANKID",  # Intel MPI
)


def _launched_by_mpi() -> bool:
    return any(name in os.environ for name in _MPI_LAUNCHER_VARIABLES)


def _auto_backend() -> str:
    """Backend selected by ``"auto"``: the one of the launcher that started
    the program, the sequential simulator when there is none."""
    from . import _mp

    if _mp.LAUNCH is not None:
        return "mp"
    return "mpi" if _launched_by_mpi() else "seq"


def _env_float(name: str) -> Optional[float]:
    value = os.environ.get(name)
    return None if value is None else float(value)


def _resolve_topology(
    backend: _Backend,
    p: Optional[int],
    shape: Optional[Shape],
    topology: Optional[Sequence[Sequence[int]]],
) -> Topology:
    """Structure of the machine: the one that is asked for (argument of
    ``init`` or environment variable ``BSP4PY_SHAPE``), else the one found by
    the backend, else a flat machine.  ``p`` is the number of processors
    when it is already known."""
    try:
        if topology is not None:
            if shape is not None:
                raise ValueError("give either a shape or a topology, not both")
            result = Topology(topology)
        else:
            origin = "shape"
            if shape is None and "BSP4PY_SHAPE" in os.environ:
                shape, origin = os.environ["BSP4PY_SHAPE"], "BSP4PY_SHAPE"
            sizes = None if shape is None else parse_shape(shape)
            if sizes is not None:
                result = Topology.from_shape(sizes)
            elif p is None:
                raise ValueError("the number of processors is not known")
            elif shape is None:
                return backend.detect_topology() or Topology.flat(p)
            else:
                return Topology.flat(p)
            if p is not None and result.p != p:
                raise ValueError(
                    f"{origin}={shape!r} describes {result.p} processors but there are {p}"
                )
    except ValueError as error:
        raise BackendError(f"invalid structure of the machine: {error}") from None
    if p is not None and result.p != p:
        raise BackendError(f"the topology describes {result.p} processors but there are {p}")
    return result


def init(
    backend: Optional[str] = None,
    p: Optional[int] = None,
    *,
    shape: Optional[Shape] = None,
    topology: Optional[Sequence[Sequence[int]]] = None,
    copy_messages: Optional[bool] = None,
    g: Optional[float] = None,
    l: Optional[float] = None,
    r: Optional[float] = None,
) -> None:
    """Choose and set up the BSP machine.

    Calling :func:`init` is optional: the first use of the library calls it with no
    argument.  It can be called again later (typically with the ``seq``
    backend, to simulate another number of processors); the parallel vectors
    created before the call can then no longer be used.

    :param backend: ``"seq"`` (sequential simulation), ``"mp"`` (parallel
        execution by local processes, started with ``python -m bsp4py.run``),
        ``"mpi"`` (parallel execution with mpi4py) or ``"auto"``.  Defaults
        to the environment variable ``BSP4PY_BACKEND``, or else to ``"auto"``,
        which selects the backend of the launcher that started the program
        (``python -m bsp4py.run``: ``"mp"``; ``mpiexec``, ``mpirun``,
        ``srun``...: ``"mpi"``) and ``"seq"`` when there is none.
    :param p: number of simulated processors of the ``seq`` backend.
        Defaults to the environment variable ``BSP4PY_P``, or else to 4.  With
        the parallel backends the number of processors is the number of
        processes started by the launcher and ``p``, when given, must be
        equal to it.
    :param shape: structure of a multi-level machine: the numbers of
        sub-groups from the outermost level to the processors, as a text or a
        list.  ``"2x4"`` (or ``[2, 4]``) is 2 groups of 4 processors, for
        example 2 computers with 4 cores each; ``"2x2x4"`` has three levels;
        ``"flat"`` is the ordinary flat machine.  Defaults to the environment
        variable ``BSP4PY_SHAPE``.  With the ``seq`` backend the shape also
        gives the number of processors; with the parallel backends the
        processes are grouped in the order of their numbers and the shape
        must describe as many processors as there are processes.  Without a
        shape, the ``mpi`` backend finds the structure itself when the
        processes run on several computers (two levels: the computers, and
        the processes of each one); every other machine is flat.
    :param topology: for the machines that are not regular, the path of
        every processor (see :func:`bsp_topology`), instead of a shape.
    :param copy_messages: with the ``seq`` backend, whether the values that
        go through ``put`` and ``proj`` are serialised and deserialised as
        they are in a real execution (default: true, or the environment
        variable ``BSP4PY_COPY``).  This keeps the simulation faithful: no
        aliasing between processors, unserialisable messages are detected,
        message sizes are measured.  Disable it for speed.
    :param g, l, r: BSP parameters of the machine made available through
        :func:`bsp_g`, :func:`bsp_l` and :func:`bsp_r` (default: the
        environment variables ``BSP4PY_G``, ``BSP4PY_L`` and ``BSP4PY_R``).
    """
    global _backend
    _require_global("init")
    kind = (backend or os.environ.get("BSP4PY_BACKEND") or "auto").lower()
    if kind == "auto":
        kind = _auto_backend()
    if kind in ("seq", "sequential"):
        described = topology is not None or shape is not None or "BSP4PY_SHAPE" in os.environ
        if p is None and "BSP4PY_P" in os.environ:
            p = int(os.environ["BSP4PY_P"])
        if p is None and not (described and _describes_processors(shape, topology)):
            p = 4
        if copy_messages is None:
            copy_messages = os.environ.get("BSP4PY_COPY", "1").lower() not in (
                "0",
                "false",
                "no",
                "off",
            )
        structure = _resolve_topology(_Backend(), p, shape, topology)
        new: _Backend = _SequentialBackend(structure.p, copy_messages)
        new.topology = structure
    elif kind in ("mpi", "mp"):
        if kind == "mpi":
            new = _MPIBackend()
        else:
            from . import _mp

            new = _mp.MpBackend()
        if p is not None and p != new.p:
            raise BackendError(f"init(p={p}) but the program runs on {new.p} process(es)")
        new.topology = _resolve_topology(new, new.p, shape, topology)
    else:
        raise BackendError(f"unknown backend {kind!r} (expected 'seq', 'mp', 'mpi' or 'auto')")
    new.g = g if g is not None else _env_float("BSP4PY_G")
    new.l = l if l is not None else _env_float("BSP4PY_L")
    new.r = r if r is not None else _env_float("BSP4PY_R")
    _backend = new


def _describes_processors(
    shape: Optional[Shape], topology: Optional[Sequence[Sequence[int]]]
) -> bool:
    """Whether the description of the machine gives the number of processors
    (a ``"flat"`` shape does not)."""
    if topology is not None:
        return True
    if shape is None:
        shape = os.environ.get("BSP4PY_SHAPE", "")
    try:
        return parse_shape(shape) is not None
    except ValueError:
        return True  # reported later, with a proper message


def _get() -> _Backend:
    if _backend is None:
        init()
    assert _backend is not None
    return _backend


def _vector(value: Any, primitive: str, position: str) -> Par:
    """Check that an argument is a parallel vector of the current machine."""
    if not isinstance(value, Par):
        raise TypeError(
            f"{primitive}: the {position} must be a parallel vector, got "
            f"{type(value).__name__} (use replicate() to make a parallel "
            "vector from a value)"
        )
    if value._backend is not _backend:
        raise BackendError(
            f"{primitive}: the {position} is a parallel vector created "
            "before the last call to init()"
        )
    return value


# --------------------------------------------------------------------------
# Machine parameters
# --------------------------------------------------------------------------


def backend_name() -> str:
    """Name of the backend in use: ``"seq"``, ``"mp"`` or ``"mpi"``."""
    return _get().name


def bsp_p() -> int:
    """Number *p* of processors of the parallel machine."""
    return _get().p


def bsp_g() -> Optional[float]:
    """BSP parameter *g* of the machine, the cost of communication.

    It is the value given to :func:`init` or in the environment variable
    ``BSP4PY_G``, in the unit you chose (``None`` when it was not given);
    ``python -m bsp4py.probe`` measures it.
    """
    return _get().g


def bsp_l() -> Optional[float]:
    """BSP parameter *l* of the machine, the cost of a synchronisation barrier.

    It is the value given to :func:`init` or in the environment variable
    ``BSP4PY_L``, in the unit you chose (``None`` when it was not given);
    ``python -m bsp4py.probe`` measures it.
    """
    return _get().l


def bsp_r() -> Optional[float]:
    """BSP parameter *r* of the machine, the speed of the processors.

    It is the value given to :func:`init` or in the environment variable
    ``BSP4PY_R``, in the unit you chose (``None`` when it was not given);
    ``python -m bsp4py.probe`` measures it.
    """
    return _get().r


def within_bounds(n: int) -> bool:
    """``True`` when ``n`` is a processor id, i.e. ``0 <= n < bsp_p()``."""
    return 0 <= n < _get().p


# --------------------------------------------------------------------------
# Multi-level structure of the machine
# --------------------------------------------------------------------------


def _pid(pid: int) -> int:
    if isinstance(pid, bool) or not isinstance(pid, int) or not 0 <= pid < _get().p:
        raise InvalidProcessor(pid)
    return pid


def bsp_levels() -> int:
    """Number of levels of the machine: 1 for a flat BSP machine, 2 for
    several computers with several cores each, and so on.

    The groups of level ``k`` gather processors that are close to each
    other: on a machine with two levels, the level-1 groups are the
    computers (their processors communicate through the memory) and the only
    level-2 group is the whole machine (through the network).  The level-0
    groups are the processors themselves.  See ``init`` for how the
    structure is given or found.
    """
    return _get().topology.levels


def bsp_topology() -> List[Path]:
    """The path of every processor in the tree of the groups: for each
    processor, the indices of the groups it belongs to, from the outermost
    level to the processor itself.  On 2 computers of 2 cores:
    ``[(0, 0), (0, 1), (1, 0), (1, 1)]``.  On a flat machine:
    ``[(0,), (1,), ...]``."""
    return list(_get().topology.paths)


def bsp_shape() -> str:
    """Short description of the structure, like ``"2x4"`` (``"8"`` when
    the machine is flat)."""
    return _get().topology.describe()


def bsp_groups(level: int) -> List[List[int]]:
    """The groups of processors of the given level (0 to ``bsp_levels()``),
    each one as the sorted list of its processors."""
    return [list(group) for group in _get().topology.groups(level)]


def bsp_group(pid: int, level: int) -> List[int]:
    """The processors of the group of the given level that contains ``pid``."""
    return list(_get().topology.group(_pid(pid), level))


def bsp_leader(pid: int, level: int) -> int:
    """The smallest processor of the group of ``pid`` at the given level,
    used as the representative of the group by the multi-level algorithms."""
    return _get().topology.leader(_pid(pid), level)


def bsp_distance(i: int, j: int) -> int:
    """Distance between two processors: the level of the smallest group
    that contains both (0 for a processor and itself)."""
    return _get().topology.distance(_pid(i), _pid(j))


# --------------------------------------------------------------------------
# The four primitives
# --------------------------------------------------------------------------


def mkpar(f: Callable[[int], T]) -> Par[T]:
    """Parallel vector creation: ``mkpar(f) = <f(0), ..., f(p-1)>``.

    ``f(i)`` is evaluated by processor ``i``, in the local computation phase
    of the current superstep.
    """
    _require_global("mkpar")
    if not callable(f):
        raise TypeError(
            f"mkpar: expected a function from processor ids to values, got {type(f).__name__}"
        )
    return _get().mkpar(f)


def apply(vf: Par[Callable[..., U]], *vs: Par[Any]) -> Par[U]:
    """Pointwise parallel application.

    ``apply(<f_0, ..., f_{p-1}>, <v_0, ..., v_{p-1}>)`` is
    ``<f_0(v_0), ..., f_{p-1}(v_{p-1})>``.  No communication is involved.

    Python functions are not curried, so ``apply`` accepts several vectors
    of arguments: ``apply(vf, v, w)`` is ``<f_0(v_0, w_0), ...>`` (this is
    ``apply2`` of the BSML standard library).
    """
    _require_global("apply")
    backend = _get()
    _vector(vf, "apply", "first argument")
    if not vs:
        raise TypeError("apply: at least one parallel vector of arguments is needed")
    for v in vs:
        _vector(v, "apply", "argument")
    return backend.apply(vf, vs)


def put(vf: Par[Callable[[int], T]], level: Optional[int] = None) -> Par[Callable[[int], T]]:
    """Global communication and synchronisation barrier.

    The argument is a vector of functions ``<f_0, ..., f_{p-1}>`` where
    ``f_i(j)`` is the message that processor ``i`` sends to processor ``j``.
    The result is a vector of functions ``<g_0, ..., g_{p-1}>`` where
    ``g_j(i) = f_i(j)`` is the message received by processor ``j`` from
    processor ``i``.

    ``None`` means "no message": when ``f_i(j)`` is ``None`` nothing is
    transmitted and ``g_j(i)`` is ``None``.  This ends the current superstep.

    On a multi-level machine ``level`` restricts the superstep to the groups
    of that level (1 to ``bsp_levels()``): the processors only communicate,
    and synchronise, with the processors of their own group.  ``f_i(j)`` is
    only asked for the processors ``j`` of the group of ``i``, and ``g_j(i)``
    is ``None`` for the processors ``i`` outside the group of ``j``.  A
    superstep of a low level is cheaper than a global one: it uses the fast
    links inside the groups and does not wait for the other groups.  By
    default the superstep is global, as in BSML.
    """
    _require_global("put")
    backend = _get()
    _vector(vf, "put", "argument")
    if level is None:
        level = backend.topology.levels
    else:
        backend.topology._check(level, lowest=1)
    return backend.put(vf, level)


def proj(v: Par[T]) -> Callable[[int], T]:
    """Projection, the dual of :func:`mkpar`: makes a vector global.

    ``proj(<v_0, ..., v_{p-1}>)`` is a function ``f`` such that
    ``f(i) = v_i``; it raises :class:`InvalidProcessor` for any other
    argument.  This ends the current superstep.
    """
    _require_global("proj")
    backend = _get()
    _vector(v, "proj", "argument")
    return backend.proj(v)


# --------------------------------------------------------------------------
# Miscellaneous
# --------------------------------------------------------------------------


def abort(code: int, message: str) -> None:
    """Print ``message`` and abort the whole computation with exit ``code``."""
    backend = _get()
    print(message, file=sys.stderr, flush=True)
    backend.abort(code)


def start_timing() -> None:
    """Start the timer."""
    _require_global("start_timing")
    _get().start_timing()


def stop_timing() -> None:
    """Stop the timer."""
    _require_global("stop_timing")
    _get().stop_timing()


def get_cost() -> Par[float]:
    """Vector of the time elapsed, on each processor, between the calls to
    :func:`start_timing` and :func:`stop_timing` (in seconds)."""
    backend = _get()
    if backend._timer_running:
        raise TimerFailure("Timer is still running")
    elapsed = backend._timer_value
    return mkpar(lambda pid: elapsed)


def _replicated(v: Par[T]) -> T:
    """The value of a vector that holds the same value on every processor.

    For the library functions that end by sending their result to all the
    processors: the result is then known everywhere and can be returned as
    an ordinary (global) value, without a ``proj``.  No communication.  The
    caller guarantees that the values are the same; nothing is checked.
    """
    backend = _get()
    _vector(v, "replicated", "argument")
    if isinstance(backend, _SpmdBackend):
        return v._data
    return v._data[0]


def stats() -> Stats:
    """BSP cost counters since the last :func:`init` or :func:`reset_stats`.

    With the parallel backends the sizes of the communications are the ones
    seen by the calling process; see :func:`global_stats`.
    """
    return _get().stats.copy()


def global_stats() -> Stats:
    """BSP cost counters of the whole machine.

    Same as :func:`stats`, but with the parallel backends the size of each
    communication is the maximum over all the processors, as the BSP cost
    model wants, instead of the traffic of the calling process.  The
    processors exchange their counters to compute it: this takes one
    superstep, which is not in the result but is counted afterwards.
    """
    _require_global("global_stats")
    backend = _get()
    mine = backend.stats.copy()
    counters = backend.proj(backend.mkpar(lambda pid: _get().stats.copy()))
    everyone = [counters(pid).steps[: mine.supersteps] for pid in range(backend.p)]

    def biggest(values: Any) -> Optional[int]:
        measured = [v for v in values if v is not None]
        return max(measured) if measured else None

    steps = [
        Superstep(step.level, tuple(biggest(column) for column in zip(*(s.h for s in same))))
        for step, same in zip(mine.steps, zip(*everyone))
    ]
    totals = [
        biggest(counters(pid).h_relations[index] for pid in range(backend.p))
        for index in range(mine.supersteps)
    ]
    return Stats(mine.supersteps, totals, steps)


def reset_stats() -> None:
    """Reset the counters returned by :func:`stats`."""
    _get().stats = Stats()
