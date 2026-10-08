import pytest
from conftest import seq_only

import bsp4py
from bsp4py import Stats, bsp_p
from bsp4py.probe import BspParameters, _least_squares, _messages, main, probe

SMALL = dict(niters=2, maxn=32, maxh=4096, points=3)


def test_least_squares():
    xs = [0, 1, 2, 5, 10]
    g, l = _least_squares(xs, [3.0 + 0.5 * x for x in xs])
    assert g == pytest.approx(0.5) and l == pytest.approx(3.0)
    assert _least_squares([4, 4, 4], [1.0, 2.0, 3.0]) == (0.0, 2.0)


def test_messages_form_an_exact_h_relation(p):
    if p == 1:
        pytest.skip("no communication on one processor")
    for h in (0, 1, p - 2, p - 1, p, 5 * p + 3, 1000):
        messages = [_messages(p, h)(pid) for pid in range(p)]
        size = lambda m: 0 if m is None else len(m)
        assert all(messages[pid](pid) is None for pid in range(p))
        assert [sum(size(messages[pid](dst)) for dst in range(p)) for pid in range(p)] == [h] * p
        assert [sum(size(messages[pid](dst)) for pid in range(p)) for dst in range(p)] == [h] * p


def test_probe(p):
    parameters = probe(**SMALL)
    assert parameters.p == p == bsp_p()
    assert parameters.r > 0.0 and parameters.l >= 0.0 and parameters.g >= 0.0
    if p == 1:
        assert parameters.g == 0.0
    bsp4py.start_timing()  # the timer is left stopped
    bsp4py.stop_timing()


def test_probe_arguments(p):
    for wrong in (dict(niters=0), dict(maxn=8), dict(maxh=0), dict(points=1)):
        with pytest.raises(ValueError):
            probe(**{**SMALL, **wrong})


def test_parameters():
    parameters = BspParameters(p=4, r=1e6, g=2.0, l=10.0)
    # three supersteps, h unknown for one of them
    assert parameters.communication_time(Stats(3, [5, None, 7])) == 12 * 2.0 + 3 * 10.0
    assert parameters.environment() == "export BSP4PY_G=2.0 BSP4PY_L=10.0 BSP4PY_R=1000000.0"


@seq_only
def test_command_line(p, capsys):
    main(["--niters", "2", "--maxn", "16", "--maxh", "1024", "--points", "3"])
    output = capsys.readouterr().out
    assert f"p = {p}\n" in output and "g = " in output and "l = " in output
    assert "export BSP4PY_G=" in output and "sequential backend" in output
