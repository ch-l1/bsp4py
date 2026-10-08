import operator

import pytest

from bsp4py import (
    apply,
    compose,
    curry,
    filtermap,
    from_to,
    identity,
    mkpar,
    natmod,
    parfun,
    put,
    replicate,
    this,
    to_list,
    uncurry,
)


def test_helpers():
    assert [natmod(i, 3) for i in range(-4, 5)] == [2, 0, 1, 2, 0, 1, 2, 0, 1]
    assert from_to(2, 5) == [2, 3, 4, 5] and from_to(3, 2) == []
    assert filtermap(lambda x: x % 2, lambda x: x * 10, range(6)) == [10, 30, 50]
    assert identity("x") == "x"
    assert compose(lambda x: x + 1, lambda x: x * 2)(5) == 11


def test_curry():
    f = lambda x, y, z: (x, y, z)
    g = curry(f)
    assert g(1)(2)(3) == (1, 2, 3)
    partial = g(1)(2)  # a partial application can be used several times
    assert partial(3) == (1, 2, 3) and partial("z") == (1, 2, "z")
    assert curry(lambda x: -x)(4) == -4
    assert curry(lambda x, y=10: x + y)(1) == 11  # defaults are not curried
    assert curry(lambda x, y=10: x + y, 2)(1)(2) == 3
    assert curry(max, 3)(1)(5)(2) == 5
    assert curry(operator.add)(1)(2) == 3


def test_curry_errors():
    with pytest.raises(TypeError):
        curry(lambda *args: args)
    with pytest.raises(ValueError):
        curry(lambda: 0)
    with pytest.raises(ValueError):
        curry(operator.add, 0)


def test_uncurry():
    g = lambda x: lambda y: lambda z: (x, y, z)
    assert uncurry(g)(1, 2, 3) == (1, 2, 3)
    assert uncurry(g)(1)(2)(3) == (1, 2, 3)
    assert uncurry(curry(operator.sub))(5, 3) == 2
    assert curry(uncurry(g), 3)(1)(2)(3) == (1, 2, 3)


def test_curried_bsml_programs(p):
    v, w = mkpar(lambda i: 10 * i), this()
    # the OCaml definitions, word for word:
    #   let parfun f v = apply (replicate f) v
    #   let parfun2 f v w = apply (parfun f v) w
    ml_parfun = lambda f, v: apply(replicate(f), v)
    ml_parfun2 = lambda f, v, w: apply(ml_parfun(f, v), w)
    expected = [10 * i - i for i in range(p)]
    assert to_list(ml_parfun2(lambda x: lambda y: x - y, v, w)) == expected
    assert to_list(ml_parfun2(curry(operator.sub), v, w)) == expected
    # a vector of partial applications, applied twice
    subtract_from = parfun(curry(operator.sub), v)
    assert to_list(apply(subtract_from, w)) == expected
    assert to_list(apply(subtract_from, v)) == [0] * p
    # and curried functions with the variadic parfun
    assert to_list(parfun(uncurry(lambda x: lambda y: x - y), v, w)) == expected
    # put (mkpar (fun i dst -> ...)), with a curried function of two arguments
    received = put(mkpar(curry(lambda i, dst: (i, dst))))
    assert to_list(parfun(lambda f: f(0), received)) == [(0, j) for j in range(p)]
