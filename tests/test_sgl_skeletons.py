import importlib.util
import random
from itertools import accumulate
from pathlib import Path

import pytest
from conftest import IRREGULAR

import bsp4py
from bsp4py import bsp_levels, bsp_p, mkpar, parfun, skeleton, to_list
from bsp4py.skeleton import _dh_sequential

pytestmark = pytest.mark.single  # the machines are given by the fixture "machine"

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "dh_applications.py"


def load_example():
    spec = importlib.util.spec_from_file_location("dh_applications", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def concat(a, b):
    return a + b


def levels_of(f):
    """Levels of the supersteps of the evaluation of f()."""
    bsp4py.reset_stats()
    f()
    return [step.level for step in bsp4py.stats().steps]


def letters(seed, smallest=0):
    """Blocks of letters of various sizes, one per processor."""
    rng = random.Random(seed)
    return [
        [chr(ord("a") + rng.randrange(26)) for _ in range(rng.randrange(smallest, 5))]
        for _ in range(bsp_p())
    ]


def follows_the_tree(machine):
    """Whether reduce and scan follow the levels (not on the irregular machine)."""
    return machine != IRREGULAR


# -- reduce -----------------------------------------------------------------------


def test_reduce(machine):
    levels = bsp_levels()
    for seed in range(4):
        blocks = letters(seed)
        blocks[seed % bsp_p()] = ["x", "y"]  # never empty everywhere
        v = mkpar(lambda i: blocks[i])
        # string concatenation is associative but not commutative
        assert skeleton.reduce(concat, v) == "".join(x for b in blocks for x in b)
    numbers = skeleton.make(lambda k: k + 1, 5 * bsp_p() + 2)
    assert skeleton.reduce(lambda a, b: a + b, numbers) == sum(range(1, 5 * bsp_p() + 3))
    expected = [*range(1, levels + 1), *range(levels - 1, 0, -1)]
    assert levels_of(lambda: skeleton.reduce(concat, v)) == (
        expected if follows_the_tree(machine) else [levels]
    )


def test_reduce_of_an_empty_list(machine):
    with pytest.raises(ValueError):
        skeleton.reduce(concat, mkpar(lambda i: []))
    assert skeleton.reduce(concat, mkpar(lambda i: ["a"] if i == bsp_p() - 1 else [])) == "a"


# -- scan -------------------------------------------------------------------------


def test_scan(machine):
    levels = bsp_levels()
    for seed in range(6):
        blocks = letters(seed)
        v = mkpar(lambda i: blocks[i])
        result = to_list(skeleton.scan(concat, v))
        flat = [x for b in blocks for x in b]
        assert [x for b in result for x in b] == list(accumulate(flat))
        assert [len(b) for b in result] == [len(b) for b in blocks]  # same distribution
    tiers = levels if follows_the_tree(machine) else 1
    up = list(range(1, levels + 1)) if follows_the_tree(machine) else [levels]
    assert levels_of(lambda: skeleton.scan(concat, v)) == up + up[::-1]
    assert len(up) == tiers


def test_scan_of_numbers_and_of_empty_lists(machine):
    n = 4 * bsp_p() + 3
    sums = skeleton.extract(skeleton.scan(lambda a, b: a + b, skeleton.make(lambda k: k, n)))
    assert sums == [k * (k + 1) // 2 for k in range(n)]
    assert to_list(skeleton.scan(concat, mkpar(lambda i: []))) == [[]] * bsp_p()
    # tuples are accepted, the result is made of lists
    result = to_list(skeleton.scan(lambda a, b: a * b, mkpar(lambda i: (2, 1))))
    assert [x for b in result for x in b] == [2 ** (k // 2 + 1) for k in range(2 * bsp_p())]


# -- sort -------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
def test_sort(machine, seed):
    p, levels = bsp_p(), bsp_levels()
    rng = random.Random(10 * seed + p)
    sizes = [rng.choice([0, 1, 3, 40, 200]) for _ in range(p)]
    blocks = [[rng.randrange(-50, 50) for _ in range(size)] for size in sizes]
    v = mkpar(lambda i: blocks[i])
    result = to_list(skeleton.sort(v))
    assert [x for b in result for x in b] == sorted(x for b in blocks for x in b)
    assert all(b == sorted(b) for b in result)
    expected = [*range(1, levels + 1), *range(levels, 0, -1), levels]
    assert levels_of(lambda: skeleton.sort(v)) == expected


def test_sort_with_a_key_is_stable(machine):
    rng = random.Random(bsp_p())
    words = [(rng.choice("abcde") * rng.randrange(1, 4), k) for k in range(30 * bsp_p())]
    key = lambda pair: len(pair[0])
    result = skeleton.extract(skeleton.sort(skeleton.par(words), key=key))
    assert result == sorted(words, key=key)  # sorted() is stable
    backwards = skeleton.extract(skeleton.sort(skeleton.par(words), key=lambda pair: -pair[1]))
    assert backwards == words[::-1]


def test_sort_is_balanced(machine):
    p = bsp_p()
    rng = random.Random(p)
    data = [rng.random() for _ in range(400 * p * p)]
    sizes = [len(b) for b in to_list(skeleton.sort(skeleton.par(data)))]
    assert sum(sizes) == len(data) and max(sizes) <= 1.5 * len(data) / p


# -- dh ---------------------------------------------------------------------------


def dh_reference(oplus, otimes, xs):
    """The definition of dh, word for word."""
    if len(xs) == 1:
        return xs
    half = len(xs) // 2
    u, w = dh_reference(oplus, otimes, xs[:half]), dh_reference(oplus, otimes, xs[half:])
    return [oplus(a, b) for a, b in zip(u, w)] + [otimes(a, b) for a, b in zip(u, w)]


def plus(a, b):
    return f"({a}+{b})"


def times(a, b):
    return f"({a}*{b})"


def test_dh_sequential():
    for n in (1, 2, 4, 8, 16, 64):
        xs = [str(k) for k in range(n)]
        assert _dh_sequential(plus, times, xs) == dh_reference(plus, times, xs)
    assert _dh_sequential(plus, times, ["a", "b"]) == ["(a+b)", "(a*b)"]
    for n in (0, 3, 6, 12):
        with pytest.raises(ValueError):
            _dh_sequential(plus, times, ["x"] * n)


def test_dh(machine):
    p, levels = bsp_p(), bsp_levels()
    if p & (p - 1):  # the number of processors must be a power of two
        with pytest.raises(ValueError):
            skeleton.dh(plus, times, skeleton.make(str, 4 * p))
        return
    for per_processor in (1, 2, 8):
        xs = [str(k) for k in range(p * per_processor)]
        result = skeleton.dh(plus, times, skeleton.par(xs))
        assert skeleton.extract(result) == dh_reference(plus, times, xs)
        assert to_list(parfun(len, result)) == [per_processor] * p
    used = levels_of(lambda: skeleton.dh(plus, times, skeleton.par(xs)))
    assert len(used) == p.bit_length() - 1  # log2(p) supersteps
    assert used == sorted(used) and all(1 <= level <= levels for level in used)


def test_dh_needs_lists_of_the_same_power_of_two_length(machine):
    if bsp_p() & (bsp_p() - 1):
        pytest.skip("the number of processors is not a power of two")
    with pytest.raises(ValueError):
        skeleton.dh(plus, times, mkpar(lambda i: ["x"] * 3))
    with pytest.raises(ValueError):
        skeleton.dh(plus, times, mkpar(lambda i: []))
    if bsp_p() > 1:
        with pytest.raises(ValueError):
            skeleton.dh(plus, times, mkpar(lambda i: ["x"] * (2 if i % 2 else 4)))
    assert to_list(bsp4py.this()) == list(range(bsp_p()))


# -- the applications of dh (examples/dh_applications.py) -------------------------


def test_fft_and_tridiagonal_solver(machine):
    p = bsp_p()
    if p & (p - 1):
        pytest.skip("the number of processors is not a power of two")
    example = load_example()
    rng = random.Random(p)
    for n in (p, 8 * p, 64 * p):
        signal = [rng.uniform(-1.0, 1.0) for _ in range(n)]
        direct = example.direct_fourier_transform(signal)
        assert max(abs(a - b) for a, b in zip(example.fft(signal), direct)) < 1e-9 * n
        system = example.random_system(n, rng)
        solution = example.tds(system)
        assert max(abs(a - b) for a, b in zip(solution, example.thomas(system))) < 1e-9


def test_bit_reversal_and_row_operations():
    example = load_example()
    assert example.bit_reversed(list(range(8))) == [0, 4, 2, 6, 1, 5, 3, 7]
    assert example.bit_reversed(["a"]) == ["a"] and example.bit_reversed([0, 1]) == [0, 1]
    # a big system: the row operations must not overflow
    rng = random.Random(1)
    system = example.random_system(4096, rng)
    solution = example.tds(system) if not bsp_p() & (bsp_p() - 1) else example.thomas(system)
    assert max(abs(a - b) for a, b in zip(solution, example.thomas(system))) < 1e-9
