import pytest

import bsp4py
from bsp4py import mkpar, skeleton, to_list


def sizes(v):
    return to_list(bsp4py.parfun(len, v))


@pytest.mark.parametrize("extra", [0, 1, 3])
def test_make_extract_length(p, extra):
    for n in sorted({0, 1, p - 1, p, 4 * p + min(extra, p - 1)}):
        v = skeleton.make(lambda k: k * k, n)
        assert skeleton.extract(v) == [k * k for k in range(n)]
        assert skeleton.length(v) == n
        blocks = sizes(v)
        assert sum(blocks) == n and max(blocks) - min(blocks) <= 1
        assert blocks == sorted(blocks, reverse=True)


def test_make_uses_no_communication(p):
    bsp4py.reset_stats()
    skeleton.make(lambda k: k, 10 * p)
    skeleton.par(list(range(10 * p)))
    assert bsp4py.stats().supersteps == 0


def test_par(p):
    for data in ([], ["a"], list(range(3 * p + 1)), tuple(range(2 * p)), "text" * p):
        v = skeleton.par(data)
        assert skeleton.extract(v) == list(data)
        assert sizes(v) == sizes(skeleton.make(lambda k: k, len(data)))


def test_map_zip(p):
    n = 3 * p + 2
    v = skeleton.make(lambda k: k, n)
    w = skeleton.make(lambda k: 10 * k, n)
    assert skeleton.extract(skeleton.map(lambda x: x + 1, v)) == [k + 1 for k in range(n)]
    assert skeleton.extract(skeleton.zip(lambda x, y: x + y, v, w)) == [11 * k for k in range(n)]
    assert skeleton.extract(skeleton.to_list(skeleton.par(tuple(range(n))))) == list(range(n))
    assert skeleton.to_array is skeleton.to_list


def test_index(p):
    # blocks of different sizes, some of them empty
    v = mkpar(lambda i: ["x"] * (i % 3))
    n = sum(i % 3 for i in range(p))
    assert skeleton.extract(skeleton.map_index(lambda k, x: (k, x), v)) == [
        (k, "x") for k in range(n)
    ]
    assert skeleton.extract(skeleton.zip_index(lambda k, x, y: (k, x + y), v, v)) == [
        (k, "xx") for k in range(n)
    ]


def test_zip_needs_the_same_distribution(p):
    v = mkpar(lambda i: [0] * (i + 1))
    w = mkpar(lambda i: [0] * (i + 2))
    with pytest.raises(ValueError):
        skeleton.zip(lambda x, y: x, v, w)
    with pytest.raises(ValueError):
        skeleton.zip_index(lambda k, x, y: x, v, w)


def test_shifts(p):
    for n in (p, p + 1, 3 * p + 2):
        v = skeleton.make(lambda k: k, n)
        left = skeleton.shift_left("end", v)
        right = skeleton.shift_right("begin", v)
        assert skeleton.extract(left) == list(range(1, n)) + ["end"]
        assert skeleton.extract(right) == ["begin"] + list(range(n - 1))
        assert sizes(left) == sizes(v) == sizes(right)
    bsp4py.reset_stats()
    skeleton.shift_left(0, skeleton.make(lambda k: k, p))
    assert bsp4py.stats().supersteps == 1


def test_shifts_need_non_empty_blocks(p):
    v = mkpar(lambda i: [])
    with pytest.raises(ValueError):
        skeleton.shift_left(0, v)
    with pytest.raises(ValueError):
        skeleton.shift_right(0, v)
