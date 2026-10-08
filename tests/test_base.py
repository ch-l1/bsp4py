import pytest
from conftest import seq_only

import bsp4py
from bsp4py import (
    apply2,
    apply3,
    apply4,
    applyat,
    applyif,
    bsml_print,
    bsp_p,
    get_list,
    get_one,
    mask,
    mkpar,
    parfun,
    parfun2,
    parfun3,
    parfun4,
    parprint,
    print_once,
    procs,
    proj_list_pids,
    put_list,
    put_one,
    replicate,
    this,
    to_list,
)


def test_replicate_procs_this(p):
    assert to_list(replicate("x")) == ["x"] * p
    assert procs() == list(range(p))
    assert to_list(this()) == list(range(p))
    assert proj_list_pids(mkpar(lambda i: -i)) == [(i, -i) for i in range(p)]


def test_parfun(p):
    v = mkpar(lambda i: i + 1)
    assert to_list(parfun(lambda x: x * 2, v)) == [2 * (i + 1) for i in range(p)]
    assert to_list(parfun(lambda x, y: (x, y), v, this())) == [(i + 1, i) for i in range(p)]
    assert to_list(parfun2(lambda x, y: x - y, v, v)) == [0] * p
    assert to_list(parfun3(lambda x, y, z: x + y + z, v, v, v)) == [3 * (i + 1) for i in range(p)]
    assert to_list(parfun4(lambda a, b, c, d: (a, b, c, d), v, v, v, this())) == [
        (i + 1, i + 1, i + 1, i) for i in range(p)
    ]
    with pytest.raises(TypeError):
        parfun(abs)


def test_applyn(p):
    v = this()
    assert to_list(apply2(mkpar(lambda i: lambda x, y: i + x + y), v, v)) == [
        3 * i for i in range(p)
    ]
    assert to_list(apply3(mkpar(lambda i: lambda x, y, z: i + x + y + z), v, v, v)) == [
        4 * i for i in range(p)
    ]
    assert to_list(apply4(mkpar(lambda i: lambda a, b, c, d: i + a + b + c + d), v, v, v, v)) == [
        5 * i for i in range(p)
    ]


def test_mask_applyif_applyat(p):
    a, b = replicate("a"), replicate("b")
    assert to_list(mask(lambda i: i % 2 == 0, a, b)) == [
        "a" if i % 2 == 0 else "b" for i in range(p)
    ]
    v = this()
    assert to_list(applyif(lambda i: i >= 2, lambda x: -x, lambda x: x, v)) == [
        -i if i >= 2 else i for i in range(p)
    ]
    for n in range(p):
        assert to_list(applyat(n, lambda x: "here", lambda x: x, v)) == [
            "here" if i == n else i for i in range(p)
        ]


def test_get_one(p):
    datas = mkpar(lambda i: f"data{i}")
    for dec in (-p - 1, -1, 0, 1, 2, p, 2 * p + 1):
        bsp4py.reset_stats()
        srcs = mkpar(lambda i: i + dec)
        assert to_list(get_one(datas, srcs)) == [f"data{(i + dec) % p}" for i in range(p)]
        assert bsp4py.stats().supersteps == 2 + 1
    # everybody asks processor 0; values may be None
    assert to_list(get_one(replicate(None), replicate(0))) == [None] * p


def test_get_list(p):
    datas = mkpar(lambda i: i * 10)
    lsrcs = mkpar(lambda i: [i + 1, i, i - 1, i + 1] if i % 2 == 0 else [])
    expected = [
        [((i + 1) % p) * 10, i * 10, ((i - 1) % p) * 10, ((i + 1) % p) * 10] if i % 2 == 0 else []
        for i in range(p)
    ]
    assert to_list(get_list(datas, lsrcs)) == expected


def test_put_one(p):
    # everybody sends its pid to processor (i * 2) mod p
    sent = mkpar(lambda i: ((i * 2) % bsp_p(), i))
    expected = [[i for i in range(p) if (i * 2) % p == j] for j in range(p)]
    assert to_list(put_one(sent)) == expected
    # invalid destinations are ignored, None is a value like any other
    sent = mkpar(lambda i: (0 if i % 2 == 0 else bsp_p() + i, None))
    expected = [[None for i in range(p) if i % 2 == 0] if j == 0 else [] for j in range(p)]
    assert to_list(put_one(sent)) == expected


def test_put_list(p):
    n = p
    sent = mkpar(
        lambda i: [(j, (i, j)) for j in range(i, n)] + [(i, "ignored"), (-1, "x"), (n, "y")]
    )
    expected = [[(i, j) for i in range(j + 1)] for j in range(p)]
    assert to_list(put_list(sent)) == expected
    # dictionaries are accepted too
    sent = mkpar(lambda i: {(i + 1) % n: i})
    assert to_list(put_list(sent)) == [[(j - 1) % p] for j in range(p)]


@seq_only
def test_printing(p, capsys):
    parprint(mkpar(lambda i: i * 1.5))
    assert capsys.readouterr().out == "".join(f"Process {i} : {i * 1.5}\n" for i in range(p))
    parprint(this(), fmt=lambda x: f"<{x}>")
    assert capsys.readouterr().out == "".join(f"Process {i} : <{i}>\n" for i in range(p))
    bsml_print(lambda x: print("value", x), p - 1, mkpar(lambda i: i))
    assert capsys.readouterr().out == f"value {p - 1}\n"
    print_once("a", 1, sep="-")
    assert capsys.readouterr().out == "a-1\n"
