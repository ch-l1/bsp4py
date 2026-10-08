"""Checks of the parallel backends, started from a sequential test session.

They run programs with the two launchers -- ``python -m bsp4py.run`` for the
``mp`` backend and ``mpiexec`` for the ``mpi`` backend -- and compare with
the sequential simulation.  The MPI ones are skipped when MPI (``mpiexec``
and mpi4py) is not available, and everything is skipped inside a parallel
session.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import IS_PARALLEL

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
MPIEXEC = os.environ.get("MPIEXEC") or shutil.which("mpiexec") or shutil.which("mpirun")
TIMEOUT = 300


def run(command, **env):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("BSP4PY_")}
    environment.update(env)
    return subprocess.run(
        list(map(str, command)),
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        env=environment,
        cwd=ROOT,
        check=False,
    )


def mpi_available():
    if IS_PARALLEL or MPIEXEC is None:
        return False
    try:
        check = run([MPIEXEC, "-n", "1", sys.executable, "-c", "from mpi4py import MPI"])
    except (OSError, subprocess.TimeoutExpired):
        return False
    return check.returncode == 0


pytestmark = pytest.mark.skipif(IS_PARALLEL, reason="already in a parallel session")
needs_mpi = pytest.mark.skipif(not mpi_available(), reason="MPI is not available")


@pytest.fixture(params=["mp", pytest.param("mpi", marks=needs_mpi)])
def launcher(request):
    """Function running ``python <arguments>`` on n processes of a backend."""
    backend = request.param

    def start(n, *arguments, shape=None):
        """``shape`` groups the processes like a multi-level machine."""
        if backend == "mp":
            # "python -m bsp4py.run -n N program" or "... -n N -m module"
            grouping = [] if shape is None else ["--shape", shape]
            return run([sys.executable, "-m", "bsp4py.run", "-n", n, *grouping, *arguments])
        environment = {} if shape is None else {"BSP4PY_SHAPE": shape}
        return run([MPIEXEC, "-n", n, sys.executable, *arguments], **environment)

    start.backend = backend
    return start


def write(tmp_path, source):
    program = tmp_path / "program.py"
    program.write_text(source)
    return program


# -- both launchers -----------------------------------------------------------


def test_the_test_suite_in_parallel(p, launcher):
    if p > 4:
        pytest.skip("kept short: the suite is run on 1 to 4 processes")
    result = launcher(p, "-m", "pytest", "tests", "-p", "no:cacheprovider")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count(" passed") == p and "failed" not in result.stdout


@pytest.mark.parametrize("example", ["hello.py", "pi.py", "prefix_sum.py"])
def test_same_output_as_the_simulation(p, launcher, example):
    simulated = run([sys.executable, EXAMPLES / example], BSP4PY_P=str(p), BSP4PY_BACKEND="seq")
    parallel = launcher(p, EXAMPLES / example)
    assert simulated.returncode == 0, simulated.stderr
    assert parallel.returncode == 0, parallel.stderr
    # the processes print their lines in any order
    assert sorted(parallel.stdout.splitlines()) == sorted(simulated.stdout.splitlines())


def test_sort_example(p, launcher):
    simulated = run([sys.executable, EXAMPLES / "sort.py", "20000"], BSP4PY_P=str(p))
    parallel = launcher(p, EXAMPLES / "sort.py", "20000")
    assert parallel.returncode == 0, parallel.stderr
    assert "the result is sorted" in parallel.stdout
    # same data, same algorithm: same distribution of the result
    assert parallel.stdout.splitlines()[1:] == simulated.stdout.splitlines()[1:]


def test_sample_sort_gps_example(p, launcher):
    simulated = run([sys.executable, EXAMPLES / "sample_sort_gps.py", "20000"], BSP4PY_P=str(p))
    parallel = launcher(p, EXAMPLES / "sample_sort_gps.py", "20000")
    assert parallel.returncode == 0, parallel.stderr
    lines = parallel.stdout.splitlines()
    assert lines[0].startswith("regular_sample_sort: 2 supersteps")
    assert lines[1].startswith("psrs: 2 supersteps")
    assert lines[2:] == simulated.stdout.splitlines()[2:] and len(lines) == 4


def test_an_error_on_one_process_stops_the_machine(p, launcher, tmp_path):
    program = write(
        tmp_path,
        "from bsp4py import *\n"
        "v = mkpar(lambda i: 1 // i)  # fails on processor 0 only\n"
        "print_once('not reached', to_list(v))\n",
    )
    result = launcher(p, program)
    assert result.returncode != 0
    assert "ZeroDivisionError" in result.stderr and "not reached" not in result.stdout
    assert result.stderr.count("Traceback") == 1


def test_probe(p, launcher):
    result = launcher(p, "-m", "bsp4py.probe", "--niters", "2", "--maxn", "16", "--maxh", "2048")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == f"Benchmark starts ({launcher.backend} backend)" and lines[1] == f"p = {p}"
    assert len(lines) == 6 and lines[5].startswith("export BSP4PY_G=")


def test_big_messages(p, launcher, tmp_path):
    # every processor sends 4 MB to every other one at the same time: more
    # than the connections can buffer
    program = write(
        tmp_path,
        "from bsp4py import *\n"
        "size = 4_000_000\n"
        "data = mkpar(lambda i: bytes([i]) * size)\n"
        "lengths = parfun(lambda l: [(b[0], len(b)) for b in l], total_exchange(data))\n"
        "assert to_list(lengths) == [[(i, size) for i in procs()]] * bsp_p()\n"
        "print_once('exchanged')\n",
    )
    result = launcher(p, program)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "exchanged\n"


def test_supersteps_with_messages_of_all_sizes(p, launcher, tmp_path):
    # small and big messages in the same and in successive supersteps, in a
    # pattern that is different for every pair of processors
    program = write(
        tmp_path,
        "from bsp4py import *\n"
        "sizes = [None, 0, 7, 900, 1024, 1025, 70_000, 600_000]\n"
        "def size(step, src, dst):\n"
        "    return sizes[(step * 7 + src * 3 + dst * 5 + step * src) % len(sizes)]\n"
        "def message(step, src, dst):\n"
        "    n = size(step, src, dst)\n"
        "    return None if n is None else (step, src, dst, bytes(n))\n"
        "def check(step, dst, f):\n"
        "    for src in procs():\n"
        "        m, n = f(src), size(step, src, dst)\n"
        "        if n is None:\n"
        "            assert m is None\n"
        "        else:\n"
        "            assert m[:3] == (step, src, dst) and len(m[3]) == n\n"
        "    return True\n"
        "for step in range(40):\n"
        "    received = put(mkpar(lambda src: lambda dst: message(step, src, dst)))\n"
        "    ok = parfun(lambda dst, f: check(step, dst, f), this(), received)\n"
        "    if step % 10 == 9:\n"
        "        assert to_list(ok) == [True] * bsp_p()\n"
        "print_once('done')\n",
    )
    result = launcher(p, program)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "done\n"


# -- multi-level machines ---------------------------------------------------------


@pytest.mark.single
@pytest.mark.parametrize("shape, n", [("2x2", 4), ("2x2x2", 8), ("3x1", 3)])
def test_the_test_suite_on_a_multilevel_machine(launcher, shape, n):
    result = launcher(n, "-m", "pytest", "tests", "-p", "no:cacheprovider", shape=shape)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count(" passed") == n and "failed" not in result.stdout


@pytest.mark.single
def test_multilevel_example(launcher):
    example = EXAMPLES / "multilevel_fold.py"
    simulated = run([sys.executable, example, "3000"], BSP4PY_SHAPE="2x2").stdout.splitlines()
    result = launcher(4, example, "3000", shape="2x2")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == f"{launcher.backend} backend, 4 processors, shape 2x2"
    assert lines[1].startswith("  level 1: g = ") and lines[2].startswith("  level 2: g = ")
    # the supersteps and the sizes of the communications are those of the simulation
    counters = [line for line in lines if "supersteps of levels" in line]
    assert counters == [line for line in simulated if "supersteps of levels" in line]
    assert counters[0].startswith("fold_direct: supersteps of levels [2], ")
    assert counters[1].startswith("fold_multi: supersteps of levels [1, 2, 1], ")
    assert lines[-1] == "both algorithms give the same result on every processor"


@pytest.mark.single
def test_dh_applications_example(launcher):
    example = EXAMPLES / "dh_applications.py"
    for n, shape in ((4, "2x2"), (4, "flat"), (8, "2x2x2")):
        result = launcher(n, example, "64", shape=shape)
        assert result.returncode == 0, result.stderr
        assert result.stdout == (
            f"FFT of 64 numbers on {n} processors: same as the direct transform\n"
            "tridiagonal system of 64 equations: same solution as the direct method\n"
        )


@pytest.mark.single
def test_probe_on_a_multilevel_machine(launcher):
    result = launcher(
        4, "-m", "bsp4py.probe", "--niters", "2", "--maxn", "16", "--maxh", "2048", shape="2x2"
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[5].startswith("shape = 2x2")
    assert lines[6].startswith("level 1: g = ") and lines[7].startswith("level 2: g = ")
    assert len(lines) == 9 and lines[8].startswith("export BSP4PY_G=")


@pytest.mark.single
def test_a_shape_must_describe_the_processes(launcher, tmp_path):
    program = write(
        tmp_path, "import bsp4py\nbsp4py.print_once(bsp4py.bsp_p(), bsp4py.bsp_shape())\n"
    )
    if launcher.backend == "mp":
        result = launcher(3, program, shape="2x2")  # refused by the launcher
        assert result.returncode == 2 and "describes 4 processes, not 3" in result.stderr
        # without -n the shape gives the number of processes
        result = run([sys.executable, "-m", "bsp4py.run", "--shape=2x3", program])
        assert (result.returncode, result.stdout) == (0, "6 2x3\n")
        for arguments in (["--shape"], ["--shape", "2xa", program]):
            result = run([sys.executable, "-m", "bsp4py.run", *arguments])
            assert result.returncode == 2 and "usage:" in result.stderr
    else:
        result = launcher(3, program, shape="2x2")  # refused by the library
        assert result.returncode != 0 and "describes 4 processors but there are 3" in result.stderr
        result = launcher(4, program, shape="flat")
        assert (result.returncode, result.stdout) == (0, "4 4\n")


# -- the launcher of the mp backend -------------------------------------------


def bsp4py_run(*arguments, **env):
    return run([sys.executable, "-m", "bsp4py.run", *arguments], **env)


def test_arguments_and_backend(p, tmp_path):
    program = write(
        tmp_path,
        "import sys, bsp4py\n"
        "pids = bsp4py.to_list(bsp4py.this())\n"
        "bsp4py.print_once(bsp4py.backend_name(), bsp4py.bsp_p(), pids, sys.argv[1:])\n",
    )
    for spelling in (["-n", p], ["-np", p], [f"--np={p}"]):
        # options after the program are for the program, even -n
        result = bsp4py_run(*spelling, program, "a", "-n", "7", "--flag", BSP4PY_BACKEND="seq")
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"mp {p} {list(range(p))} ['a', '-n', '7', '--flag']\n"


@pytest.mark.single
def test_synonyms(tmp_path):
    program = write(
        tmp_path, "import bsp4py\nbsp4py.print_once(bsp4py.backend_name(), bsp4py.bsp_p())\n"
    )
    result = run([sys.executable, "-m", "bsp4py", "-n", "2", program])
    assert (result.returncode, result.stdout) == (0, "mp 2\n")


def test_exit_status(p, tmp_path):
    program = write(tmp_path, "import sys, bsp4py\nbsp4py.proj(bsp4py.this())\nsys.exit(3)\n")
    result = bsp4py_run("-n", p, program)
    assert result.returncode == 3 and result.stderr == ""
    program = write(
        tmp_path, "import bsp4py\nbsp4py.this()\nbsp4py.abort(5, 'stopping on purpose')\n"
    )
    result = bsp4py_run("-n", p, program)
    assert result.returncode == 5
    assert result.stderr.count("stopping on purpose") == p


def test_a_program_that_does_not_use_bsp4py(p, tmp_path):
    program = write(tmp_path, "import sys\nsys.stdout.write('hello\\n')\n")
    result = bsp4py_run("-n", p, program)
    assert (result.returncode, result.stdout) == (0, "hello\n" * p)


@pytest.mark.single
def test_processes_started_by_the_program_are_not_processors(tmp_path):
    program = write(
        tmp_path,
        "import subprocess, sys, bsp4py\n"
        "bsp4py.proj(bsp4py.this())\n"
        "code = 'import bsp4py; print(bsp4py.backend_name(), bsp4py.bsp_p())'\n"
        "out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)\n"
        "bsp4py.print_once(out.stdout.strip(), out.returncode)\n",
    )
    result = bsp4py_run("-n", 2, program, BSP4PY_P="3")
    assert (result.returncode, result.stdout) == (0, "seq 3 0\n"), result.stderr


@pytest.mark.single
def test_a_program_that_is_not_deterministic_is_stopped(tmp_path):
    # processor 1 finishes while the others wait for it at a barrier
    program = write(
        tmp_path,
        "import os, sys, bsp4py\n"
        "v = bsp4py.this()\n"
        "if bsp4py.proj(v)(0) == 0 and v._data == 1:\n"
        "    sys.exit(0)\n"
        "bsp4py.proj(v)\n"
        "print('not reached')\n",
    )
    start = time.monotonic()
    result = bsp4py_run("-n", 3, program)
    assert result.returncode != 0 and "not reached" not in result.stdout
    assert "the processors lost each other" in result.stderr
    assert time.monotonic() - start < 30


@pytest.mark.single
def test_busy_processes_are_stopped_when_one_fails(tmp_path):
    # processor 0 fails while processor 1 computes for a long time
    program = write(
        tmp_path,
        "import time, bsp4py\n"
        "v = bsp4py.this()\n"
        "if v._data == 1:\n"
        "    time.sleep(1000)\n"
        "if v._data == 0:\n"
        "    raise RuntimeError('failure of processor 0')\n"
        "bsp4py.proj(v)\n",
    )
    start = time.monotonic()
    result = bsp4py_run("-n", 3, program)
    assert time.monotonic() - start < 30
    assert result.returncode == 1 and "failure of processor 0" in result.stderr
    assert "bsp4py.run: processor 0 exited with status 1" in result.stderr


@pytest.mark.single
def test_only_some_processes_use_bsp4py(tmp_path):
    # the machine cannot be set up: the processes that wait for it are stopped
    program = write(
        tmp_path,
        "import os\n"
        "rank = os.environ.get('BSP4PY_RUN_RANK')\n"
        "import bsp4py\n"
        "if rank != '0':\n"
        "    bsp4py.bsp_p()\n"
        "    print('not reached')\n",
    )
    result = bsp4py_run("-n", 3, program)
    assert result.returncode != 0 and "not reached" not in result.stdout
    assert "the processors lost each other" in result.stderr


def test_no_thread_is_left_when_a_process_exits(p, tmp_path):
    # a thread still running when the interpreter shuts down may crash it:
    # the thread that sends the big messages must be over by then
    program = write(
        tmp_path,
        "import atexit, sys, threading\n"
        "def threads():\n"
        "    return sorted(t.name for t in threading.enumerate() if t.name.startswith('bsp4py'))\n"
        "# registered before bsp4py is used: called after bsp4py has closed the machine\n"
        "atexit.register(lambda: sys.stdout.write(f'at exit: {threads()}\\n'))\n"
        "import bsp4py\n"
        "sizes = bsp4py.proj(bsp4py.mkpar(lambda i: bytes(200_000)))\n"
        "assert all(len(sizes(i)) == 200_000 for i in range(bsp4py.bsp_p()))\n"
        "bsp4py.print_once(f'running: {threads()}')\n",
    )
    result = bsp4py_run("-n", p, program)
    assert result.returncode == 0, result.stderr
    running = "running: ['bsp4py-sender']" if p > 1 else "running: []"
    assert sorted(result.stdout.splitlines()) == ["at exit: []"] * p + [running]


@pytest.mark.single
@pytest.mark.parametrize(
    "source, status",
    [
        ("import bsp4py; bsp4py.proj(bsp4py.this())", 0),
        ("pass", 0),  # the launcher is still waiting for the processes to register
        ("import bsp4py; bsp4py.proj(bsp4py.this()); raise SystemExit(4)", 4),
    ],
)
def test_no_thread_is_left_when_the_launcher_returns(tmp_path, source, status):
    driver = write(
        tmp_path,
        "import sys, threading\n"
        "from bsp4py.run import launch\n"
        "status = launch(3, [sys.executable, '-c', sys.argv[1]])\n"
        "threads = sorted(t.name for t in threading.enumerate() if t.name.startswith('bsp4py'))\n"
        "print(status, threads)\n",
    )
    start = time.monotonic()
    result = run([sys.executable, driver, source])
    assert (result.returncode, result.stdout) == (0, f"{status} []\n"), result.stderr
    assert time.monotonic() - start < 30


@pytest.mark.single
def test_a_crash_of_a_process_is_reported(capsys):
    from bsp4py.run import _status

    # all the processors finished, one of them badly: on Windows with an
    # access violation, elsewhere killed by a signal
    assert _status({0: 0, 1: 0xC0000005, 2: 0}, set()) == 0xC0000005
    assert capsys.readouterr().err == "bsp4py.run: processor 1 exited with status 0xC0000005\n"
    assert _status({0: -11, 1: 0}, set()) == 128 + 11
    assert capsys.readouterr().err == "bsp4py.run: processor 0 was killed by signal 11\n"
    assert _status({0: 3, 1: 3}, set()) == 3  # an ordinary failure: the program said why
    assert capsys.readouterr().err == ""
    assert _status({0: 3, 1: -15}, {1}) == 3
    assert capsys.readouterr().err == (
        "bsp4py.run: processor 0 exited with status 3; the other processors were stopped\n"
    )


@pytest.mark.single
@pytest.mark.skipif(sys.platform == "win32", reason="no signals on Windows")
def test_a_process_killed_after_the_last_superstep(tmp_path):
    program = write(
        tmp_path,
        "import os, signal, bsp4py\n"
        "pid = bsp4py.this()._data\n"
        "bsp4py.proj(bsp4py.this())\n"
        "if pid == 1:\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n",
    )
    result = bsp4py_run("-n", 3, program)
    assert result.returncode == 128 + 9
    assert result.stderr == "bsp4py.run: processor 1 was killed by signal 9\n"


@pytest.mark.single
def test_command_line_errors(tmp_path):
    for arguments in (
        [],
        ["-n", "2"],
        ["-n"],
        ["-n", "0", "x.py"],
        ["-n", "two", "x.py"],
        ["--unknown", "x.py"],
        ["-m"],
        [tmp_path / "missing.py"],
    ):
        result = bsp4py_run(*arguments)
        assert result.returncode == 2 and "usage:" in result.stderr, arguments
    result = bsp4py_run("--help")
    assert result.returncode == 0 and "usage:" in result.stdout


@pytest.mark.single
def test_the_backend_needs_its_launcher():
    result = run([sys.executable, "-c", "import bsp4py; bsp4py.bsp_p()"], BSP4PY_BACKEND="mp")
    assert result.returncode != 0 and "python -m bsp4py.run" in result.stderr
