import math
from functools import reduce
from itertools import accumulate

import pytest

import bsp4py
from bsp4py import (
    BcastError,
    GatherError,
    InvalidProcessor,
    ScatterError,
    bcast_direct,
    bcast_totex,
    bcast_totex_array,
    bcast_totex_gen,
    bcast_totex_list,
    bcast_totex_string,
    bsp_p,
    fold_array_direct,
    fold_array_logp,
    fold_direct,
    fold_list_direct,
    fold_list_logp,
    fold_logp,
    fold_wide,
    gather,
    gather_list,
    mkpar,
    parfun,
    replicate,
    scan_array_direct,
    scan_array_logp,
    scan_direct,
    scan_list_direct,
    scan_list_logp,
    scan_logp,
    scan_wide_direct,
    scan_wide_logp,
    scatter,
    scatter_array,
    scatter_list,
    scatter_string,
    shift,
    shift_left,
    shift_right,
    to_list,
    total_exchange,
    totex,
)


def supersteps(f):
    """Number of supersteps of the evaluation of f()."""
    bsp4py.reset_stats()
    f()
    return bsp4py.stats().supersteps


def concat(a, b):
    return a + b


def test_shift(p):
    v = mkpar(lambda i: f"v{i}")
    for dec in range(-2 * p - 1, 2 * p + 2):
        assert to_list(shift(dec, v)) == [f"v{(i - dec) % p}" for i in range(p)]
    assert to_list(shift_right(v)) == [f"v{(i - 1) % p}" for i in range(p)]
    assert to_list(shift_left(v)) == [f"v{(i + 1) % p}" for i in range(p)]
    assert to_list(shift_right(replicate(None))) == [None] * p
    assert supersteps(lambda: shift(3, v)) == 1


def test_total_exchange(p):
    v = mkpar(lambda i: (i, None))
    everything = [(i, None) for i in range(p)]
    assert to_list(total_exchange(v)) == [everything] * p
    f = totex(v)
    assert to_list(parfun(lambda g: [g(j) for j in range(bsp_p())], f)) == [everything] * p
    with pytest.raises(InvalidProcessor):
        parfun(lambda g: g(-1), f)
    assert supersteps(lambda: total_exchange(v)) == 1


@pytest.mark.parametrize("extra", [0, 1, 2])
@pytest.mark.parametrize("full", [0, 1, 3])
def test_scatter_sequences(p, extra, full):
    n = full * p + min(extra, p - 1)
    for sequence in (list(range(n)), tuple(range(n)), "x" * n, bytes(n)):
        for root in sorted({0, p - 1}):
            held = mkpar(lambda i: sequence if i == root else sequence[:0])
            parts = to_list(scatter_list(root, held))
            assert reduce(concat, parts, sequence[:0]) == sequence
            sizes = [len(part) for part in parts]
            assert max(sizes) - min(sizes) <= 1 and sizes == sorted(sizes, reverse=True)
            assert all(type(part) is type(sequence) for part in parts)
    assert scatter_array is scatter_list and scatter_string is scatter_list


def test_scatter(p):
    v = mkpar(lambda i: 100 * i)
    for root in range(p):
        # send "root value + destination" to the even processors only
        result = scatter(lambda x, dst: x + dst if dst % 2 == 0 else None, root, v)
        assert to_list(result) == [100 * root + i if i % 2 == 0 else None for i in range(p)]
    assert supersteps(lambda: scatter(lambda x, dst: x, 0, v)) == 1
    for root in (-1, p):
        with pytest.raises(ScatterError):
            scatter(lambda x, dst: x, root, v)


def test_gather(p):
    v = mkpar(lambda i: [i])
    n = bsp_p()
    for dst in range(p):
        f = gather(dst, v)
        table = to_list(parfun(lambda g: [g(i) for i in range(n)], f))
        assert table == [[[i] for i in range(p)] if j == dst else [None] * p for j in range(p)]
        assert to_list(gather_list(dst, v)) == [
            [[i] for i in range(p)] if j == dst else [] for j in range(p)
        ]
    assert supersteps(lambda: gather_list(0, v)) == 1
    for dst in (-1, p):
        with pytest.raises(GatherError):
            gather(dst, v)
        with pytest.raises(GatherError):
            gather_list(dst, v)


def test_bcast(p):
    v = mkpar(lambda i: {"pid": i, "data": list(range(i))})
    for root in range(p):
        expected = [{"pid": root, "data": list(range(root))}] * p
        assert to_list(bcast_direct(root, v)) == expected
        assert to_list(bcast_totex(root, v)) == expected
    assert to_list(bcast_direct(0, replicate(None))) == [None] * p
    assert supersteps(lambda: bcast_direct(0, v)) == 1
    assert supersteps(lambda: bcast_totex(0, v)) == 2
    for root in (-1, p):
        for bcast in (bcast_direct, bcast_totex, bcast_totex_list):
            with pytest.raises(BcastError):
                bcast(root, v)


def test_bcast_totex_sequences(p):
    for n in (0, 1, p - 1, p, 3 * p + 1):
        for sequence in (list(range(n)), tuple(range(n)), "ab" * n, bytes(range(n % 256))):
            for root in sorted({0, p // 2, p - 1}):
                held = mkpar(lambda i: sequence if i == root else None)
                assert to_list(bcast_totex_list(root, held)) == [sequence] * p
    assert bcast_totex_array is bcast_totex_list and bcast_totex_string is bcast_totex_list


def test_bcast_totex_gen(p):
    n = bsp_p()
    # broadcast an integer as n parts whose sum is the integer
    partition = lambda x, dst: x // n + (1 if dst < x % n else 0)
    paste = lambda f: sum(f(i) for i in range(n))
    v = mkpar(lambda i: 1000 + i)
    assert to_list(bcast_totex_gen(partition, paste, p - 1, v)) == [1000 + p - 1] * p


def test_bcast_numpy(p):
    numpy = pytest.importorskip("numpy")
    array = numpy.arange(3 * p + 2, dtype=float)
    held = mkpar(lambda i: array if i == 0 else None)
    for result in to_list(bcast_totex_array(0, held)):
        assert isinstance(result, numpy.ndarray) and (result == array).all()


def test_scan(p):
    # string concatenation is associative but not commutative
    v = mkpar(lambda i: chr(ord("a") + i))
    expected = list(accumulate(chr(ord("a") + i) for i in range(p)))
    assert to_list(scan_direct(concat, v)) == expected
    assert to_list(scan_logp(concat, v)) == expected
    numbers = mkpar(lambda i: i + 1)
    assert to_list(scan_logp(lambda a, b: a * b, numbers)) == [
        math.factorial(i + 1) for i in range(p)
    ]
    assert supersteps(lambda: scan_direct(concat, v)) == 1
    assert supersteps(lambda: scan_logp(concat, v)) == math.ceil(math.log2(p))


def test_scan_of_lists(p):
    def block(i):  # blocks of different sizes, never empty
        start = i * (i + 1) // 2
        return [str(k) + "," for k in range(start, start + i + 1)]

    v = mkpar(block)
    flat = [x for i in range(p) for x in block(i)]
    expected_flat = list(accumulate(flat))
    for scan in (scan_list_direct, scan_list_logp, scan_array_direct, scan_array_logp):
        result = to_list(scan(concat, v))
        assert [len(l) for l in result] == [i + 1 for i in range(p)]
        assert [x for l in result for x in l] == expected_flat


def test_scan_wide_on_tuples(p):
    v = mkpar(lambda i: (i, i))
    seq_scan = lambda op, t: tuple(accumulate(t, op))
    last = lambda t: t[-1]
    tmap = lambda f, t: tuple(f(x) for x in t)
    expected = list(accumulate(x for i in range(p) for x in (i, i)))
    for scan_wide in (scan_wide_direct, scan_wide_logp):
        result = to_list(scan_wide(seq_scan, last, tmap, lambda a, b: a + b, v))
        assert all(isinstance(t, tuple) for t in result)
        assert [x for t in result for x in t] == expected


def test_fold(p):
    v = mkpar(lambda i: chr(ord("a") + i))
    whole = "".join(chr(ord("a") + i) for i in range(p))
    assert to_list(fold_direct(concat, v)) == [whole] * p
    assert to_list(fold_logp(concat, v)) == [whole] * p
    assert supersteps(lambda: fold_direct(concat, v)) == 1
    assert supersteps(lambda: fold_logp(concat, v)) == math.ceil(math.log2(p)) + 2


def test_fold_of_lists(p):
    v = mkpar(lambda i: [f"{i}.{k} " for k in range(i + 1)])
    whole = "".join(f"{i}.{k} " for i in range(p) for k in range(i + 1))
    for fold in (fold_list_direct, fold_list_logp, fold_array_direct, fold_array_logp):
        assert to_list(fold(concat, v)) == [whole] * p
    # fold_wide with another local fold: number of elements
    count = fold_wide(fold_direct, lambda op, l: len(l), lambda a, b: a + b, v)
    assert to_list(count) == [p * (p + 1) // 2] * p
