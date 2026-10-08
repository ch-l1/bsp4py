import pickle

import pytest
from conftest import seq_only

import bsp4py
from bsp4py import (
    BackendError,
    BspError,
    InvalidProcessor,
    NestingError,
    TimerFailure,
    apply,
    bsp_p,
    mkpar,
    parfun,
    proj,
    put,
    replicate,
    this,
    to_list,
    within_bounds,
)


def test_bsp_p(p):
    assert bsp_p() == p
    assert within_bounds(0) and within_bounds(p - 1)
    assert not within_bounds(-1) and not within_bounds(p)


def test_mkpar_proj(p):
    f = proj(mkpar(lambda i: i * i))
    assert [f(i) for i in range(p)] == [i * i for i in range(p)]


@pytest.mark.parametrize("bad", [-1, "p", "0", 1.0, None])
def test_proj_invalid_processor(p, bad):
    f = proj(mkpar(lambda i: i))
    with pytest.raises(InvalidProcessor):
        f(p if bad == "p" else bad)


def test_proj_is_a_superstep(p):
    bsp4py.reset_stats()
    proj(this())
    assert bsp4py.stats().supersteps == 1


def test_apply(p):
    vf = mkpar(lambda i: lambda x: (i, x))
    vv = mkpar(lambda i: 10 * i)
    assert to_list(apply(vf, vv)) == [(i, 10 * i) for i in range(p)]


def test_apply_several_vectors(p):
    vf = mkpar(lambda i: lambda x, y, z: i + x * y - z)
    assert to_list(apply(vf, this(), this(), replicate(1))) == [i + i * i - 1 for i in range(p)]


def test_apply_is_not_a_superstep(p):
    bsp4py.reset_stats()
    apply(mkpar(lambda i: lambda x: x), this())
    assert bsp4py.stats().supersteps == 0


def test_put(p):
    # f_i(j) is sent by i to j; g_j(i) = f_i(j)
    sent = mkpar(lambda i: lambda j: ("from", i, "to", j))
    received = put(sent)
    table = to_list(parfun(lambda g: [g(i) for i in range(bsp_p())], received))
    assert table == [[("from", i, "to", j) for i in range(p)] for j in range(p)]


def test_put_none_means_no_message(p):
    sent = mkpar(lambda i: lambda j: i if (i + j) % 2 == 0 else None)
    received = put(sent)
    table = to_list(parfun(lambda g: [g(i) for i in range(bsp_p())], received))
    assert table == [[i if (i + j) % 2 == 0 else None for i in range(p)] for j in range(p)]


def test_put_is_a_superstep(p):
    bsp4py.reset_stats()
    put(mkpar(lambda i: lambda j: None))
    assert bsp4py.stats().supersteps == 1
    assert bsp4py.stats().h_relations == [0]


def test_put_delivery_function_invalid_processor(p):
    received = put(mkpar(lambda i: lambda j: i))
    with pytest.raises(InvalidProcessor):
        parfun(lambda g: g(bsp_p()), received)


def test_put_evaluates_messages_for_valid_destinations_only(p):
    def message(i):
        def f(j):
            assert 0 <= j < bsp_p()
            return j

        return f

    received = put(mkpar(message))
    assert to_list(parfun(lambda g: g(0), received)) == list(range(p))


def test_messages_are_values_of_any_type(p):
    values = [{"a": [1, 2.5, "x"]}, (1, 2), "text", b"bytes", 3 + 4j, frozenset({1})]
    sent = mkpar(lambda i: lambda j: values[(i + j) % len(values)])
    table = to_list(parfun(lambda g: [g(i) for i in range(bsp_p())], put(sent)))
    assert table == [[values[(i + j) % len(values)] for i in range(p)] for j in range(p)]


def test_functions_can_be_sent(p):
    pytest.importorskip("cloudpickle")
    sent = mkpar(lambda i: lambda j: lambda x: x + 100 * i + j)
    received = put(sent)
    result = parfun(lambda g: [g(i)(1) for i in range(bsp_p())], received)
    assert to_list(result) == [[1 + 100 * i + j for i in range(p)] for j in range(p)]


# -- nesting ----------------------------------------------------------------


def test_nesting_mkpar_in_mkpar(p):
    with pytest.raises(NestingError):
        mkpar(lambda i: mkpar(lambda j: j))


def test_nesting_returning_a_vector(p):
    v = this()
    with pytest.raises(NestingError):
        mkpar(lambda i: v)
    with pytest.raises(NestingError):
        replicate(v)


def test_nesting_proj_in_apply(p):
    v = this()
    with pytest.raises(NestingError):
        parfun(lambda x: proj(v)(0), v)


def test_nesting_put_in_put(p):
    v = mkpar(lambda i: lambda j: None)
    with pytest.raises(NestingError):
        put(mkpar(lambda i: lambda j: put(v)))


def test_nesting_vector_in_a_message(p):
    v = this()
    with pytest.raises(NestingError):
        put(mkpar(lambda i: lambda j: [v]))
    with pytest.raises(NestingError):
        pickle.dumps(v)


def test_global_functions_are_usable_after_a_nesting_error(p):
    with pytest.raises(NestingError):
        mkpar(lambda i: this())
    assert to_list(this()) == list(range(p))


def test_local_code_may_use_global_values(p):
    f = proj(mkpar(lambda i: 2 * i))  # a global function ...
    n = bsp_p()
    # ... used inside local computations, as well as bsp_p()
    assert to_list(mkpar(lambda i: f((i + 1) % n) + bsp_p())) == [
        2 * ((i + 1) % p) + p for i in range(p)
    ]


# -- misuse -----------------------------------------------------------------


def test_type_errors(p):
    v = this()
    with pytest.raises(TypeError):
        mkpar(3)
    with pytest.raises(TypeError):
        apply(lambda x: x, v)
    with pytest.raises(TypeError):
        apply(replicate(abs))
    with pytest.raises(TypeError):
        apply(replicate(abs), 1)
    with pytest.raises(TypeError):
        put(lambda j: None)
    with pytest.raises(TypeError):
        proj([1, 2])
    with pytest.raises(TypeError):
        bool(v)
    with pytest.raises(TypeError):
        iter(v)
    with pytest.raises(TypeError):
        len(v)


def test_exceptions_of_local_code_propagate(p):
    with pytest.raises(ZeroDivisionError):
        mkpar(lambda i: 1 // 0)
    assert to_list(this()) == list(range(p))


# -- timing -----------------------------------------------------------------


def test_timing(p):
    bsp4py.start_timing()
    with pytest.raises(TimerFailure):
        bsp4py.start_timing()
    with pytest.raises(TimerFailure):
        bsp4py.get_cost()
    bsp4py.stop_timing()
    with pytest.raises(TimerFailure):
        bsp4py.stop_timing()
    assert all(t >= 0.0 for t in to_list(bsp4py.get_cost()))


# -- sequential backend only --------------------------------------------------


@seq_only
def test_repr(p):
    assert repr(mkpar(lambda i: str(i))) == "<" + ", ".join(repr(str(i)) for i in range(p)) + ">"


@seq_only
def test_init_errors(p):
    with pytest.raises(BackendError):
        bsp4py.init("threads")
    with pytest.raises(BackendError):
        bsp4py.init("seq", p=0)


@seq_only
def test_vectors_do_not_survive_init(p):
    v = this()
    bsp4py.init("seq", p=p)
    with pytest.raises(BackendError):
        proj(v)
    with pytest.raises(BackendError):
        apply(replicate(abs), v)


@seq_only
def test_messages_are_copied(p):
    v = mkpar(lambda i: [i])
    f = proj(v)
    f(0).append("changed")
    assert proj(v)(0) == [0]

    shared = [0]
    received = put(mkpar(lambda i: lambda j: shared))
    parfun(lambda g: g(0).append(1), received)
    assert shared == [0]


@seq_only
def test_messages_are_not_copied_on_request(p):
    bsp4py.init("seq", p=p, copy_messages=False)
    v = mkpar(lambda i: [i])
    proj(v)(0).append("changed")
    assert proj(v)(0) == [0, "changed"]
    assert bsp4py.stats().h_relations == [None, None]


@seq_only
def test_h_relation(p):
    size = len(bsp4py._core.dumps(b"x" * 1000))
    bsp4py.reset_stats()
    proj(replicate(b"x" * 1000))  # everybody sends 1000 bytes to everybody
    put(mkpar(lambda i: lambda j: b"x" * 1000 if i == 0 else None))  # 0 sends to all
    put(mkpar(lambda i: lambda j: b"x" * 1000 if j == i else None))  # to oneself
    assert bsp4py.stats().h_relations == [(p - 1) * size, (p - 1) * size, 0]


@seq_only
def test_unserialisable_messages_are_detected(p):
    import threading

    with pytest.raises(BspError):
        proj(replicate(threading.Lock()))


@seq_only
def test_machine_parameters(p):
    assert bsp4py.bsp_g() is None
    bsp4py.init("seq", p=p, g=2.0, l=300.0, r=1e9)
    assert (bsp4py.bsp_g(), bsp4py.bsp_l(), bsp4py.bsp_r()) == (2.0, 300.0, 1e9)
    assert bsp4py.backend_name() == "seq"
