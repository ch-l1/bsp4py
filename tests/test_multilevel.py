import pytest
from conftest import IRREGULAR, seq_only

import bsp4py
from bsp4py import (
    BackendError,
    BcastError,
    InvalidProcessor,
    Stats,
    Superstep,
    bcast_direct,
    bcast_multi,
    bsp_distance,
    bsp_group,
    bsp_groups,
    bsp_leader,
    bsp_levels,
    bsp_p,
    bsp_topology,
    fold_direct,
    fold_multi,
    global_stats,
    mkpar,
    parfun,
    put,
    this,
    to_list,
)
from bsp4py._topology import Topology, parse_shape
from bsp4py.probe import BspParameters, LevelParameters, probe

pytestmark = pytest.mark.single  # the machines are given by the fixture "machine"


def concat(a, b):
    return a + b


# -- the description of the machines -------------------------------------------


def test_parse_shape():
    assert parse_shape("2x4") == [2, 4] and parse_shape(" 2X3x4 ") == [2, 3, 4]
    assert (
        parse_shape("2*4") == [2, 4] and parse_shape([3, 1]) == [3, 1] and parse_shape("7") == [7]
    )
    assert parse_shape("flat") is None and parse_shape("") is None
    for wrong in ("2x", "axb", "2x0", "-1", [], [2, 0]):
        with pytest.raises(ValueError):
            parse_shape(wrong)


def test_topology():
    t = Topology.from_shape([2, 3])
    assert t.p == 6 and t.levels == 2
    assert t.paths == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert t.groups(0) == [[0], [1], [2], [3], [4], [5]]
    assert t.groups(1) == [[0, 1, 2], [3, 4, 5]] and t.groups(2) == [[0, 1, 2, 3, 4, 5]]
    assert t.group(4, 1) == [3, 4, 5] and t.leader(4, 1) == 3 and t.leader(4, 2) == 0
    assert [t.distance(1, j) for j in range(6)] == [1, 0, 1, 2, 2, 2]
    assert t.describe() == "2x3" and Topology.flat(5).describe() == "5"
    assert Topology.flat(3).paths == [(0,), (1,), (2,)]

    t = Topology.from_shape([2, 2, 2])
    assert t.levels == 3 and t.groups(1) == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert t.groups(2) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert [t.distance(0, j) for j in range(8)] == [0, 1, 2, 2, 3, 3, 3, 3]

    t = Topology(IRREGULAR)  # groups that are not made of consecutive processors
    assert t.groups(1) == [[0, 2], [1, 3, 4]] and t.leader(4, 1) == 1
    assert [t.distance(0, j) for j in range(5)] == [0, 2, 1, 2, 2]
    assert "2+3" in t.describe()

    for wrong in ([], [()], [(0,), (0, 1)], [(0, 0), (0, 0)]):
        with pytest.raises(ValueError):
            Topology(wrong)
    with pytest.raises(ValueError):
        t.groups(3)
    with pytest.raises(TypeError):
        t.groups("1")


def test_topology_from_the_names_of_the_computers():
    # what the MPI backend does with the names of the computers of the processes
    assert Topology.from_hosts(["a", "a", "a"]) is None  # one computer: a flat machine
    t = Topology.from_hosts(["a", "a", "b", "b", "b"])
    assert t.levels == 2 and t.groups(1) == [[0, 1], [2, 3, 4]]
    assert t.paths == [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
    t = Topology.from_hosts(["n2", "n1", "n2", "n1", "n3"])  # processes placed in turn
    assert t.groups(1) == [[0, 2], [1, 3], [4]] and t.distance(0, 2) == 1 and t.distance(0, 1) == 2


@seq_only
def test_init_with_a_shape(monkeypatch):
    bsp4py.init("seq", shape="2x4")
    assert (bsp_p(), bsp_levels(), bsp4py.bsp_shape()) == (8, 2, "2x4")
    assert bsp_topology()[5] == (1, 1) and bsp_groups(1) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    bsp4py.init("seq", p=6, shape=[3, 2])
    assert (bsp_p(), bsp_levels()) == (6, 2)
    bsp4py.init("seq", topology=IRREGULAR)
    assert (bsp_p(), bsp_levels()) == (5, 2) and bsp_groups(1) == [[0, 2], [1, 3, 4]]
    bsp4py.init("seq", p=3, shape="flat")
    assert (bsp_p(), bsp_levels(), bsp4py.bsp_shape()) == (3, 1, "3")
    bsp4py.init("seq", p=3)
    assert bsp_levels() == 1 and bsp_topology() == [(0,), (1,), (2,)]

    for wrong in (
        dict(p=6, shape="2x4"),
        dict(shape="2xfour"),
        dict(shape="2x4", topology=IRREGULAR),
        dict(p=4, topology=IRREGULAR),
        dict(topology=[(0,), (0,)]),
    ):
        with pytest.raises(BackendError):
            bsp4py.init("seq", **wrong)

    monkeypatch.setenv("BSP4PY_SHAPE", "3x2")
    bsp4py.init("seq")
    assert (bsp_p(), bsp_levels()) == (6, 2)
    bsp4py.init("seq", shape="2x2")  # the argument wins
    assert bsp_p() == 4
    with pytest.raises(BackendError):
        bsp4py.init("seq", p=4)  # 4 processors, but BSP4PY_SHAPE describes 6
    monkeypatch.setenv("BSP4PY_SHAPE", "flat")
    bsp4py.init("seq")
    assert (bsp_p(), bsp_levels()) == (4, 1)


# -- the structure seen by the programs ----------------------------------------


def test_structure(machine):
    p, levels = bsp_p(), bsp_levels()
    assert levels >= 1 and len(bsp_topology()) == p
    assert bsp_groups(0) == [[i] for i in range(p)]
    assert bsp_groups(levels) == [list(range(p))]
    for level in range(levels + 1):
        groups = bsp_groups(level)
        assert sorted(pid for group in groups for pid in group) == list(range(p))
        for group in groups:
            for pid in group:
                assert bsp_group(pid, level) == group and bsp_leader(pid, level) == group[0]
        if level:  # the groups of a level are made of groups of the level below
            below = bsp_groups(level - 1)
            assert all(any(set(small) <= set(group) for group in groups) for small in below)
    for i in range(p):
        for j in range(p):
            d = bsp_distance(i, j)
            assert d == bsp_distance(j, i) and (d == 0) == (i == j) and 0 <= d <= levels
            assert all((j in bsp_group(i, level)) == (level >= d) for level in range(levels + 1))
    # the structure can be used in local code
    assert to_list(mkpar(lambda i: bsp_leader(i, 1))) == [bsp_leader(i, 1) for i in range(p)]


def test_structure_errors(machine):
    for wrong in (-1, bsp_levels() + 1):
        with pytest.raises(ValueError):
            bsp_groups(wrong)
        with pytest.raises(ValueError):
            bsp_group(0, wrong)
    for wrong in (-1, bsp_p(), "0"):
        with pytest.raises(InvalidProcessor):
            bsp_group(wrong, 1)
        with pytest.raises(InvalidProcessor):
            bsp_distance(0, wrong)


# -- supersteps restricted to a level -------------------------------------------


def test_put_at_a_level(machine):
    p = bsp_p()
    for level in range(1, bsp_levels() + 1):
        received = put(mkpar(lambda i: lambda j: (i, j)), level=level)
        table = to_list(parfun(lambda g: [g(i) for i in range(bsp_p())], received))
        assert table == [
            [(i, j) if i in bsp_group(j, level) else None for i in range(p)] for j in range(p)
        ]
    # by default the superstep is global
    table = to_list(
        parfun(lambda g: [g(i) for i in range(bsp_p())], put(mkpar(lambda i: lambda j: i)))
    )
    assert table == [list(range(p))] * p


def test_messages_are_asked_for_the_group_only(machine):
    for level in range(1, bsp_levels() + 1):

        def message(i, level=level):
            def f(j):
                assert j in bsp_group(i, level)
                return j

            return f

        received = put(mkpar(message), level=level)
        assert to_list(parfun(lambda j, g: g(j), this(), received)) == list(range(bsp_p()))


def test_invalid_levels(machine):
    v = mkpar(lambda i: lambda j: None)
    for wrong in (0, -1, bsp_levels() + 1):
        with pytest.raises(ValueError):
            put(v, level=wrong)
    for wrong in ("1", 1.0, True):
        with pytest.raises(TypeError):
            put(v, level=wrong)
    assert to_list(this()) == list(range(bsp_p()))


def test_counters_per_level_and_distance(machine):
    p, levels = bsp_p(), bsp_levels()
    size = len(bsp4py._core.dumps(b"x" * 100))
    bsp4py.reset_stats()
    for level in range(1, levels + 1):
        put(mkpar(lambda i: lambda j: b"x" * 100), level=level)
    to_list(this())  # a global superstep
    counters = global_stats()
    assert counters.supersteps == levels + 1 == len(counters.steps) == len(counters.h_relations)
    assert [step.level for step in counters.steps] == [*range(1, levels + 1), levels]
    for level, step, total in zip(range(1, levels + 1), counters.steps, counters.h_relations):
        partners = [
            [sum(1 for j in bsp_group(i, level) if bsp_distance(i, j) == d) for i in range(p)]
            for d in range(1, levels + 1)
        ]
        assert step.h == tuple(size * max(row) for row in partners)
        assert all(h == 0 for h in step.h[level:])  # nothing goes further than the group
        assert total == size * max(sum(row[i] for row in partners) for i in range(p))
    # global_stats took a superstep, counted afterwards
    assert bsp4py.stats().supersteps == levels + 2


@seq_only
def test_global_stats_is_stats_in_the_simulator(machine):
    bsp4py.reset_stats()
    put(mkpar(lambda i: lambda j: "message"), level=1)
    to_list(mkpar(lambda i: [i] * i))
    local = bsp4py.stats()
    assert global_stats() == local
    assert isinstance(local.steps[0], Superstep) and local.steps[0].level == 1


# -- algorithms that follow the levels ------------------------------------------


def test_fold_multi(machine):
    p, levels = bsp_p(), bsp_levels()
    numbers = mkpar(lambda i: 3 * i + 1)
    add = lambda a, b: a + b
    assert to_list(fold_multi(add, numbers)) == to_list(fold_direct(add, numbers))

    bsp4py.reset_stats()
    fold_multi(add, numbers)
    steps = bsp4py.stats().steps
    assert [step.level for step in steps] == [*range(1, levels + 1), *range(levels - 1, 0, -1)]

    # with groups of consecutive processors the order of the values is kept
    if machine != IRREGULAR:
        letters = mkpar(lambda i: chr(ord("a") + i))
        whole = "".join(chr(ord("a") + i) for i in range(p))
        assert to_list(fold_multi(concat, letters)) == [whole] * p


def test_fold_multi_sends_less_through_the_slow_links(machine):
    levels = bsp_levels()
    values = mkpar(lambda i: b"x" * 1000)
    pick = lambda a, b: a

    def volume(fold):
        bsp4py.reset_stats()
        fold(pick, values)
        steps = global_stats().steps
        return [sum(step.h[d] for step in steps) for d in range(levels)]

    direct, multi = volume(fold_direct), volume(fold_multi)
    size = len(bsp4py._core.dumps((b"x" * 1000,)))
    subgroups = len(bsp_groups(levels - 1))
    # one value per group of the level below the top crosses the outermost links
    assert multi[-1] == size * (subgroups - 1) <= direct[-1]
    assert direct[-1] == size * max(
        sum(1 for j in range(bsp_p()) if bsp_distance(i, j) == levels) for i in range(bsp_p())
    )


def test_bcast_multi(machine):
    p, levels = bsp_p(), bsp_levels()
    v = mkpar(lambda i: {"from": i})
    for root in range(p):
        assert (
            to_list(bcast_multi(root, v)) == to_list(bcast_direct(root, v)) == [{"from": root}] * p
        )
    bsp4py.reset_stats()
    bcast_multi(p - 1, v)
    assert [step.level for step in bsp4py.stats().steps] == list(range(levels, 0, -1))
    for root in (-1, p):
        with pytest.raises(BcastError):
            bcast_multi(root, v)


# -- the cost model ---------------------------------------------------------------


def test_probe_measures_every_level(machine):
    parameters = probe(niters=2, maxn=16, maxh=2048, points=3)
    assert [level.level for level in parameters.levels] == list(range(1, bsp_levels() + 1))
    assert all(level.g >= 0.0 and level.l >= 0.0 for level in parameters.levels)
    if bsp_levels() == 1:
        assert parameters.levels == (LevelParameters(1, parameters.g, parameters.l),)
    for level in parameters.levels:  # nobody at this distance: nothing to measure
        if not any(bsp_distance(0, j) == level.level for j in range(bsp_p())) and bsp_p() == 1:
            assert level.g == 0.0


def test_predicted_time():
    levels = (LevelParameters(1, 1.0, 10.0), LevelParameters(2, 100.0, 1000.0))
    parameters = BspParameters(p=4, r=1e6, g=50.0, l=500.0, levels=levels)
    counters = Stats(
        supersteps=3,
        h_relations=[7, 9, 4],
        steps=[Superstep(1, (7, 0)), Superstep(2, (3, 6)), Superstep(1, (4, 0))],
    )
    # level 1: 7*1 + 10; level 2: 3*1 + 6*100 + 1000; level 1: 4*1 + 10
    assert parameters.communication_time(counters) == 17 + 1603 + 14
    assert parameters.flat_communication_time(counters) == 20 * 50.0 + 3 * 500.0
    # without the levels of the supersteps, or of the machine: the flat model
    assert parameters.communication_time(Stats(3, [7, 9, 4])) == 20 * 50.0 + 3 * 500.0
    flat = BspParameters(p=4, r=1e6, g=50.0, l=500.0)
    assert flat.communication_time(counters) == 20 * 50.0 + 3 * 500.0
    # counters of a machine with another number of levels
    other = Stats(1, [5], [Superstep(1, (5,))])
    assert parameters.communication_time(other) == 5 * 50.0 + 500.0
