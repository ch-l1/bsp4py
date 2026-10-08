"""Very often used functions (port of the module ``Base`` of BSML).

Everything here is written with the four primitives only, exactly like in
the OCaml library, so it runs unchanged on every backend.

Adaptations to Python:

* OCaml functions are curried, Python functions are not.  ``parfun`` and
  ``apply`` therefore accept any number of vectors: ``parfun(f, v, w)`` is
  ``parfun2 f v w``.  The names ``parfun2`` ... ``apply4`` are kept so that
  BSML programs translate mechanically.
* OCaml uses ``Some v`` / ``None`` in messages.  In Python a message is any
  value and ``None`` means "no message".
* ``procs`` and ``this`` are functions (``procs()``, ``this()``).
"""

from __future__ import annotations

import io
import sys
from collections.abc import Iterable, Mapping
from typing import Any, Callable, List, Tuple, TypeVar, Union

from ._core import Par, apply, bsp_p, mkpar, proj, put
from .utils import natmod

T = TypeVar("T")
U = TypeVar("U")

__all__ = [
    "apply2",
    "apply3",
    "apply4",
    "applyat",
    "applyif",
    "bsml_print",
    "get_list",
    "get_one",
    "mask",
    "parfun",
    "parfun2",
    "parfun3",
    "parfun4",
    "parprint",
    "print_once",
    "procs",
    "proj_list_pids",
    "put_list",
    "put_one",
    "replicate",
    "this",
    "to_list",
]


def _write(text: str) -> None:
    """Write on the standard output in one go, so that the lines printed by
    different processes are not mixed up."""
    sys.stdout.write(text)
    sys.stdout.flush()


def replicate(x: T) -> Par[T]:
    """``replicate(x)`` is the parallel vector ``<x, ..., x>``."""
    return mkpar(lambda pid: x)


def parfun(f: Callable[..., U], *vs: Par[Any]) -> Par[U]:
    """``parfun(f, <x_0, ..., x_{p-1}>) = <f(x_0), ..., f(x_{p-1})>``.

    With several vectors, ``parfun(f, v, w, ...)`` applies ``f`` pointwise to
    the values of all of them: ``<f(v_0, w_0, ...), ...>``.  This is the
    Python counterpart of the local sections ``<< f $v$ $w$ >>`` of BSML 0.5;
    use :func:`this` to get the processor id: ``parfun(f, this(), v)``.
    """
    if not vs:
        raise TypeError("parfun: at least one parallel vector is needed")
    return apply(replicate(f), *vs)


def parfun2(f: Callable[[Any, Any], U], v1: Par, v2: Par) -> Par[U]:
    """Same as :func:`parfun` with a function of arity 2."""
    return parfun(f, v1, v2)


def parfun3(f: Callable[[Any, Any, Any], U], v1: Par, v2: Par, v3: Par) -> Par[U]:
    """Same as :func:`parfun` with a function of arity 3."""
    return parfun(f, v1, v2, v3)


def parfun4(f: Callable[[Any, Any, Any, Any], U], v1: Par, v2: Par, v3: Par, v4: Par) -> Par[U]:
    """Same as :func:`parfun` with a function of arity 4."""
    return parfun(f, v1, v2, v3, v4)


def apply2(vf: Par[Callable[[Any, Any], U]], v1: Par, v2: Par) -> Par[U]:
    """Same as ``apply`` with a vector of functions of arity 2."""
    return apply(vf, v1, v2)


def apply3(vf: Par[Callable[[Any, Any, Any], U]], v1: Par, v2: Par, v3: Par) -> Par[U]:
    """Same as ``apply`` with a vector of functions of arity 3."""
    return apply(vf, v1, v2, v3)


def apply4(
    vf: Par[Callable[[Any, Any, Any, Any], U]], v1: Par, v2: Par, v3: Par, v4: Par
) -> Par[U]:
    """Same as ``apply`` with a vector of functions of arity 4."""
    return apply(vf, v1, v2, v3, v4)


def mask(pred: Callable[[int], Any], v1: Par[T], v2: Par[T]) -> Par[T]:
    """Vector holding the value of ``v1`` at the processors ``i`` such that
    ``pred(i)`` is true and the value of ``v2`` at the others."""
    return apply(mkpar(lambda i: lambda x, y: x if pred(i) else y), v1, v2)


def applyif(
    pred: Callable[[int], Any],
    f1: Callable[[T], U],
    f2: Callable[[T], U],
    v: Par[T],
) -> Par[U]:
    """Apply ``f1`` at the processors ``i`` such that ``pred(i)`` is true and
    ``f2`` at the others."""
    return apply(mkpar(lambda i: f1 if pred(i) else f2), v)


def applyat(n: int, f1: Callable[[T], U], f2: Callable[[T], U], v: Par[T]) -> Par[U]:
    """Apply ``f1`` at processor ``n`` and ``f2`` at the others."""
    return applyif(lambda i: i == n, f1, f2, v)


def procs() -> List[int]:
    """The list of the processor ids: ``[0, ..., p-1]``."""
    return list(range(bsp_p()))


def this() -> Par[int]:
    """The vector ``<0, ..., p-1>``: each processor holds its own id."""
    return mkpar(lambda pid: pid)


def bsml_print(printer: Callable[[T], Any], at_pid: int, vec: Par[T]) -> Par[None]:
    """Call ``printer`` on the value held by processor ``at_pid``."""

    def nothing(x: Any) -> None:
        return None

    def run(x: T) -> None:
        printer(x)

    return apply(mkpar(lambda pid: run if pid == at_pid else nothing), vec)


def parprint(vec: Par[T], fmt: Callable[[T], str] = str) -> Par[None]:
    """Print the vector, one line per processor.

    Each line starts with the number of the processor; ``fmt`` turns a value
    into text.  For example ``parprint(this())`` gives with 4 processors::

        Process 0 : 0
        Process 1 : 1
        Process 2 : 2
        Process 3 : 3

    (With the ``mpi`` backend every process prints its own line and the
    order of the lines is not specified.)
    """

    def show(pid: int, data: T) -> None:
        _write(f"Process {pid} : {fmt(data)}\n")

    return parfun(show, this(), vec)


def print_once(*args: Any, **kwargs: Any) -> None:
    """Like ``print``, but prints only once whatever the backend.

    With the ``mpi`` backend the global part of the program is executed by
    every process, so a plain ``print`` there prints *p* times; this
    function prints on processor 0 only.  (Not in OCaml BSML.)
    """
    if "file" in kwargs:
        mkpar(lambda pid: print(*args, **kwargs) if pid == 0 else None)
        return
    buffer = io.StringIO()
    print(*args, file=buffer, **{k: v for k, v in kwargs.items() if k != "flush"})
    text = buffer.getvalue()
    mkpar(lambda pid: _write(text) if pid == 0 else None)


def get_one(datas: Par[T], srcs: Par[int]) -> Par[T]:
    """``get_one(<x_0, ..., x_{p-1}>, <i_0, ..., i_{p-1}>)`` is
    ``<x_{i_0}, ..., x_{i_{p-1}}>``: each processor fetches the value of
    another one.  Processor numbers are taken modulo *p*.  Two supersteps."""
    p = bsp_p()
    pids = parfun(lambda i: natmod(i, p), srcs)
    ask = put(parfun(lambda i: lambda dst: True if dst == i else None, pids))
    reply = put(parfun(lambda f, d: lambda dst: (d,) if f(dst) else None, ask, datas))
    return parfun(lambda f, i: f(i)[0], reply, pids)


def get_list(datas: Par[T], lsrcs: Par[List[int]]) -> Par[List[T]]:
    """Like :func:`get_one` with a list of sources per processor.  The order
    of the elements of each result list is the order of the processor
    numbers in the argument list."""
    p = bsp_p()
    lpids = parfun(lambda l: [natmod(i, p) for i in l], lsrcs)
    ask = put(parfun(lambda l: lambda dst: True if dst in l else None, lpids))
    reply = put(parfun(lambda f, d: lambda dst: (d,) if f(dst) else None, ask, datas))
    return parfun(lambda f, l: [f(i)[0] for i in l], reply, lpids)


def _received(f: Callable[[int], Any], p: int) -> List[Any]:
    """Messages of a delivery function, ordered by source processor."""
    return [m[0] for m in map(f, range(p)) if m is not None]


def put_one(dst_and_datas: Par[Tuple[int, T]]) -> Par[List[T]]:
    """Each processor holds a pair ``(dst, v)``: ``v`` is sent to processor
    ``dst``.  If ``dst`` is not a valid processor number the pair is
    ignored.  Each processor gets the list of the values it received,
    ordered by source processor."""
    p = bsp_p()
    msgs = parfun(
        lambda pair: lambda dst: (pair[1],) if dst == pair[0] else None,
        dst_and_datas,
    )
    return parfun(lambda f: _received(f, p), put(msgs))


def put_list(
    ldst_and_datas: Par[Union[Iterable[Tuple[int, T]], Mapping[int, T]]],
) -> Par[List[T]]:
    """Each processor holds an association list of pairs ``(dst, v)`` (or a
    dictionary): ``v`` is sent to processor ``dst``.  Invalid processor
    numbers are ignored.  If two pairs have the same destination only the
    first one is considered.  Each processor gets the list of the values it
    received, ordered by source processor."""
    p = bsp_p()

    def messages(pairs: Any) -> Callable[[int], Any]:
        table: dict = {}
        items = pairs.items() if isinstance(pairs, Mapping) else pairs
        for dst, value in items:
            table.setdefault(dst, value)
        return lambda dst: (table[dst],) if dst in table else None

    return parfun(lambda f: _received(f, p), put(parfun(messages, ldst_and_datas)))


def proj_list_pids(v: Par[T]) -> List[Tuple[int, T]]:
    """The list ``[(0, v_0), ..., (p-1, v_{p-1})]``.  One superstep."""
    f = proj(v)
    return [(i, f(i)) for i in procs()]


def to_list(v: Par[T]) -> List[T]:
    """The list ``[v_0, ..., v_{p-1}]`` of the values of the vector.  One
    superstep."""
    f = proj(v)
    return [f(i) for i in procs()]
