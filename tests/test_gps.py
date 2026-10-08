import importlib.util
import random
from pathlib import Path

import pytest

import bsp4py
from bsp4py import NestingError, bsp_p, mkpar, parfun, put, replicate, sgl_gps, this, to_list

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_sort_gps.py"


def load_example():
    spec = importlib.util.spec_from_file_location("sample_sort_gps", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_and_assemble(p):
    # processor i sends (i, j, x_i) to every processor j
    data = mkpar(lambda i: f"x{i}")
    result = sgl_gps(
        lambda i, x: [(i, j, x) for j in range(bsp_p())],
        lambda j, received: (j, received),
        data,
    )
    assert to_list(result) == [(j, [(i, j, f"x{i}") for i in range(p)]) for j in range(p)]


def test_one_superstep(p):
    bsp4py.reset_stats()
    sgl_gps(lambda i, x: [x] * bsp_p(), lambda j, received: received, this())
    assert bsp4py.stats().supersteps == 1


def test_columns_of_the_matrix(p):
    # the vectors split(i, x_i) are the rows of a p x p matrix:
    # processor j receives the j-th column
    rows = mkpar(lambda i: [10 * i + j for j in range(bsp_p())])
    columns = sgl_gps(lambda i, row: row, lambda j, column: column, rows)
    assert to_list(columns) == [[10 * i + j for i in range(p)] for j in range(p)]
    # doing it twice gives the matrix back
    again = sgl_gps(lambda i, row: row, lambda j, column: column, columns)
    assert to_list(again) == to_list(rows)


def test_same_as_put(p):
    # sgl_gps split assemble v, written with put as in the thesis (Listing 7)
    n = bsp_p()
    split = lambda i, x: [x * (j + 1) for j in range(n)]
    assemble = lambda j, received: sum(received) - j
    data = mkpar(lambda i: i + 1)
    comm = parfun(lambda i, x: split(i, x), this(), data)
    recv = put(parfun(lambda parts: lambda dst: parts[dst], comm))
    with_put = parfun(lambda j, f: assemble(j, [f(src) for src in range(n)]), this(), recv)
    assert to_list(sgl_gps(split, assemble, data)) == to_list(with_put)


def test_the_processor_number_is_given_to_both_functions(p):
    n = bsp_p()
    result = sgl_gps(
        lambda i, x: [i if j == (i + 1) % n else None for j in range(n)],  # to the right
        lambda j, received: (j, [x for x in received if x is not None]),
        replicate("unused"),
    )
    assert to_list(result) == [(j, [(j - 1) % p]) for j in range(p)]


def test_nothing_is_sent_for_none(p):
    bsp4py.reset_stats()
    result = sgl_gps(lambda i, x: [None] * bsp_p(), lambda j, received: received, this())
    assert to_list(result) == [[None] * p] * p
    assert bsp4py.stats().h_relations[0] == 0


def test_split_may_return_any_sequence(p):
    n = bsp_p()
    for sequence in (tuple, list, lambda xs: "".join(map(str, xs))):
        result = sgl_gps(
            lambda i, x: sequence([(i + j) % 10 for j in range(n)]),
            lambda j, received: list(received),
            this(),
        )
        expected = [[(i + j) % 10 for i in range(p)] for j in range(p)]
        if sequence not in (tuple, list):
            expected = [[str(x) for x in column] for column in expected]
        assert to_list(result) == expected


def test_split_must_return_one_value_per_processor(p):
    for wrong in (p - 1, p + 1):
        with pytest.raises(ValueError):
            sgl_gps(lambda i, x: [x] * wrong, lambda j, received: received, this())
    assert to_list(this()) == list(range(p))


def test_errors(p):
    with pytest.raises(TypeError):
        sgl_gps(lambda i, x: [x] * bsp_p(), lambda j, received: received, [1, 2])
    with pytest.raises(NestingError):  # split and assemble are local code
        sgl_gps(lambda i, x: [this()] * bsp_p(), lambda j, received: received, this())


# -- the parallel sample-sort of the thesis (examples/sample_sort_gps.py) -----


@pytest.mark.single
def test_extract_n():
    example = load_example()
    # n - 1 elements, regularly spaced, never the first one
    assert example.extract_n(4, 8, list(range(8))) == [2, 4, 6]
    assert example.extract_n(2, 8, list(range(8))) == [4]
    assert example.extract_n(3, 10, list(range(10))) == [3, 6]
    assert example.extract_n(1, 5, list(range(5))) == []
    for n in range(1, 9):
        for length in range(max(n, 2), 40):
            assert len(example.extract_n(n, length, list(range(length)))) == n - 1


@pytest.mark.single
def test_sequential_helpers():
    example = load_example()
    assert example.slice_p([1, 2, 3, 5, 5, 8], [3, 5]) == [[1, 2], [3], [5, 5, 8]]
    assert example.slice_p([], [3, 5]) == [[], [], []]
    assert example.merge([1, 4, 6], [2, 4, 9]) == [1, 2, 4, 4, 6, 9]
    lists = [[5, 9], [], [1, 7, 8], [2], [3, 3]]
    assert example.p_merge(5, lists) == sorted(x for l in lists for x in l)
    assert example.p_merge(1, [[1, 2]]) == [1, 2] and example.p_merge(0, []) == []


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("shape", ["even", "increasing", "duplicates"])
def test_sample_sort(p, seed, shape):
    example = load_example()
    rng = random.Random(100 * seed + p)
    sizes = [p + (30 * (i + 1) if shape == "increasing" else 60) for i in range(p)]
    top = 5 if shape == "duplicates" else 10_000
    blocks = [[rng.randrange(top) for _ in range(size)] for size in sizes]
    data = mkpar(lambda i: blocks[i])
    lengths = parfun(len, data)

    bsp4py.reset_stats()
    with_gps = example.regular_sample_sort(lengths, data)
    assert bsp4py.stats().supersteps == 2
    result = to_list(with_gps)
    assert [x for block in result for x in block] == sorted(x for b in blocks for x in b)
    assert result == to_list(example.psrs(lengths, data))
    if shape == "even":  # regular sampling: at most 2n/p elements per processor
        assert max(len(block) for block in result) <= 2 * sum(sizes) // p
