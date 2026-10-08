"""Small sequential helpers (port of ``Bsmlutils`` / ``Tools`` of BSML)."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any, Callable, List, Optional, Tuple, TypeVar

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")

__all__ = ["compose", "curry", "filtermap", "from_to", "identity", "natmod", "uncurry"]


def natmod(i: int, m: int) -> int:
    """``i`` modulo ``m``, always in ``0 .. m-1`` (also for negative ``i``)."""
    return (m + (i % m)) % m


def from_to(n1: int, n2: int) -> List[int]:
    """The list ``[n1, n1+1, ..., n2]`` (both bounds included)."""
    return list(range(n1, n2 + 1))


def filtermap(pred: Callable[[T], Any], f: Callable[[T], U], xs: Iterable[T]) -> List[U]:
    """``[f(x) for x in xs if pred(x)]``."""
    return [f(x) for x in xs if pred(x)]


def identity(x: T) -> T:
    """The identity function (``id`` in BSML)."""
    return x


def compose(f: Callable[[U], V], g: Callable[[T], U]) -> Callable[[T], V]:
    """Function composition: ``compose(f, g)(x) = f(g(x))``."""
    return lambda x: f(g(x))


def _arity(f: Callable[..., Any]) -> int:
    """Number of positional parameters without default value of ``f``."""
    try:
        parameters = inspect.signature(f).parameters.values()
    except (TypeError, ValueError):
        raise TypeError(
            f"curry: cannot find the number of arguments of {f!r}, give it explicitly"
        ) from None
    count = 0
    for parameter in parameters:
        if parameter.kind is parameter.VAR_POSITIONAL:
            raise TypeError(
                f"curry: {f!r} takes any number of arguments, give the arity explicitly"
            )
        if (
            parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            and parameter.default is parameter.empty
        ):
            count += 1
    return count


def curry(f: Callable[..., Any], arity: Optional[int] = None) -> Callable[[Any], Any]:
    """Curried version of a function of several arguments.

    ``curry(f)(x)(y)(z) = f(x, y, z)``: the result takes the arguments of
    ``f`` one at a time, like an OCaml function.  This gives the functions
    expected by the BSML idioms that rely on partial application::

        apply(apply(replicate(curry(f)), v), w)   # OCaml: apply (apply (replicate f) v) w
        adders = parfun(curry(operator.add, 2), v)   # a vector of partial applications

    ``arity`` is the number of arguments of ``f``; by default it is the
    number of its positional parameters that have no default value.  It must
    be given for the functions that have no signature (some built-ins) or
    that take any number of arguments.  Each partial application is a new
    function: it can be applied several times.
    """
    n = _arity(f) if arity is None else arity
    if n < 1:
        raise ValueError(f"curry: the arity must be at least 1, got {n}")

    def partial(args: Tuple[Any, ...]) -> Callable[[Any], Any]:
        def step(x: Any) -> Any:
            arguments = (*args, x)
            return f(*arguments) if len(arguments) == n else partial(arguments)

        return step

    return partial(())


def uncurry(f: Callable[[Any], Any]) -> Callable[..., Any]:
    """Inverse of :func:`curry`: ``uncurry(f)(x, y, z) = f(x)(y)(z)``.

    Turns a curried function (``lambda x: lambda y: ...``) into a function
    of several arguments, as expected by the variadic ``parfun`` and
    ``apply``.  At least one argument must be given.
    """

    def uncurried(first: Any, *others: Any) -> Any:
        result = f(first)
        for x in others:
            result = result(x)
        return result

    return uncurried
