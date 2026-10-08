import random

import pytest

from bsp4py import (
    RegularSamplingSortError,
    mkpar,
    parfun,
    regular_sampling_sort,
    regular_sampling_sort_array,
    regular_sampling_sort_list,
    skeleton,
    to_list,
)


def distribute(data, weights):
    """Cut the list ``data`` in blocks whose sizes are proportional to weights."""
    total = sum(weights)
    bounds = [0]
    for w in weights:
        bounds.append(bounds[-1] + (len(data) * w) // total)
    bounds[-1] = len(data)
    return [data[bounds[i] : bounds[i + 1]] for i in range(len(weights))]


def check_sorted(blocks, data, key=None):
    """The blocks are sorted and their concatenation is the sorted data."""
    assert [x for block in blocks for x in block] == sorted(data, key=key)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("shape", ["even", "increasing", "one", "last", "holes"])
def test_regular_sampling_sort(p, seed, shape):
    rng = random.Random(1000 * seed + p)
    n = rng.randrange(p * p, 40 * p * p + 2)
    data = [rng.randrange(-n, n) for _ in range(n)]
    weights = {
        "even": [1] * p,
        "increasing": [i + 1 for i in range(p)],
        "one": [1] + [0] * (p - 1),
        "last": [0] * (p - 1) + [1],
        "holes": [i % 2 for i in range(p)] if p > 1 else [1],
    }[shape]
    blocks = distribute(data, weights)
    result = regular_sampling_sort(mkpar(lambda i: blocks[i]))
    check_sorted(to_list(result), data)


def test_many_duplicates(p):
    rng = random.Random(p)
    data = [rng.randrange(3) for _ in range(20 * p * p)]
    check_sorted(to_list(regular_sampling_sort(skeleton.par(data))), data)
    data = [7] * (p * p)
    check_sorted(to_list(regular_sampling_sort(skeleton.par(data))), data)


def test_sorted_and_reversed_inputs(p):
    for data in (list(range(10 * p * p)), list(range(10 * p * p, 0, -1))):
        result = to_list(regular_sampling_sort(skeleton.par(data)))
        check_sorted(result, data)
        if p > 1:  # distinct values: at most 2n/p elements per processor
            assert max(len(block) for block in result) <= 2 * len(data) // p


def test_key_and_stability(p):
    rng = random.Random(p)
    words = [(rng.choice("abcde") * rng.randrange(1, 4), k) for k in range(12 * p * p)]
    key = lambda pair: len(pair[0])
    result = to_list(regular_sampling_sort(skeleton.par(words), key=key))
    check_sorted(result, words, key=key)  # sorted() is stable, so equal means stable
    result = to_list(regular_sampling_sort(skeleton.par(words), key=lambda pair: -pair[1]))
    assert [x for block in result for x in block] == words[::-1]


def test_other_sequences(p):
    data = tuple(random.Random(p).sample(range(1000), 5 * p * p))
    blocks = distribute(data, [1] * p)
    result = to_list(regular_sampling_sort(mkpar(lambda i: blocks[i])))
    check_sorted(result, data)
    assert all(isinstance(block, list) for block in result)
    assert regular_sampling_sort_list is regular_sampling_sort
    assert regular_sampling_sort_array is regular_sampling_sort


def test_not_enough_elements(p):
    data = list(range(p * p - 1))
    with pytest.raises(RegularSamplingSortError):
        regular_sampling_sort(skeleton.par(data))
    assert to_list(parfun(len, skeleton.par(data))) is not None  # still usable


def test_balance(p):
    # random data evenly distributed: every processor ends with about n/p elements
    rng = random.Random(p)
    n = 2000 * p * p
    data = [rng.random() for _ in range(n)]
    sizes = [len(block) for block in to_list(regular_sampling_sort(skeleton.par(data)))]
    assert sum(sizes) == n
    assert max(sizes) <= 1.5 * n / p and min(sizes) >= 0.5 * n / p
