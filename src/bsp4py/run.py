"""Launcher of the ``mp`` backend: run a BSML program on local processes.

::

    python -m bsp4py.run -n 4 program.py [arguments]
    python -m bsp4py.run -n 4 -m package.module [arguments]

starts 4 Python processes that all execute the program and that form a BSP
machine of 4 processors (see ``_mp.py``).  It plays the role of ``mpiexec``
for the ``mpi`` backend, with nothing to install: only the standard library
is used.  ``python -m bsp4py`` and the command ``bsp4py`` are synonyms.

The exit status is the one of the program.  If a process fails, the others
are stopped and the status of the failed process is returned.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Connection, Listener
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._mp import ENV_ADDRESS, ENV_AUTHKEY, ENV_RANK, ENV_SIZE, EXIT_PEER_LOST, FAMILY
from ._topology import parse_shape

__all__ = ["launch", "main"]

USAGE = """\
usage: python -m bsp4py.run [-n P] [--shape S] program.py [arguments]
       python -m bsp4py.run [-n P] [--shape S] -m module [arguments]

Run a BSML program on P local processes (one BSP processor per process).

options:
  -n P, -np P, --np P   number of processes (default: the number of CPUs)
  --shape S             group the processes like a multi-level machine:
                        "2x4" is 2 groups of 4 processes (and sets P to 8)
  -m module             run a module, like "python -m module"
  -h, --help            show this message
"""

#: Seconds left to the processes to stop by themselves once one has failed,
#: then to obey when they are asked to stop.
GRACE = 1.0


class _Registrar(threading.Thread):
    """Collects the addresses of the processes and gives them to all.

    The processes register when they first use bsp4py, which may be at any
    time, or never: the launcher does not wait for them, this thread does.
    """

    def __init__(self, listener: Listener, size: int, authkey: bytes) -> None:
        super().__init__(name="bsp4py-registrar", daemon=True)
        self.listener = listener
        self.size = size
        self.authkey = authkey
        self.lock = threading.Lock()
        self.pending: Dict[int, Tuple[Connection, Any]] = {}
        self.complete = False
        self.cancelled = False
        self.stopping = False

    def run(self) -> None:
        while True:
            try:
                connection = self.listener.accept()
            except (AuthenticationError, EOFError, ConnectionError):
                if self.stopping:
                    return
                continue  # not one of ours, or a process that was stopped meanwhile
            except Exception:  # the listener is closed
                return
            if self.stopping:  # the connection of the launcher: the run is over
                connection.close()
                return
            try:
                rank, address = connection.recv()
            except (EOFError, OSError):
                connection.close()
                continue
            with self.lock:
                if self.cancelled:
                    connection.close()
                    continue
                self.pending[rank] = (connection, address)
                if len(self.pending) < self.size:
                    continue
                addresses = [self.pending[i][1] for i in range(self.size)]
                self._release(addresses)
                self.complete = True
                return

    def _release(self, addresses: Optional[List[Any]]) -> None:
        """Answer to the registered processes (``None``: hang up)."""
        for connection, _ in self.pending.values():
            try:
                if addresses is not None:
                    connection.send(addresses)
                connection.close()
            except OSError:
                pass
        self.pending.clear()

    def process_exited(self) -> None:
        """A process is over.  If the machine is not set up yet it will
        never be: the processes that wait for it are told so."""
        with self.lock:
            if not self.complete:
                self.cancelled = True
                self._release(None)

    def stop(self, timeout: float) -> None:
        """End of the run: make the thread return, and wait for it.

        When the processes did not all register, the thread is still waiting
        for a connection; the launcher makes one, which wakes it up.  This
        is done by another thread so that the launcher can never be blocked
        here.
        """
        self.process_exited()
        self.stopping = True
        if not self.is_alive():
            return
        waker = threading.Thread(target=self._wake, name="bsp4py-waker", daemon=True)
        waker.start()
        self.join(timeout)
        waker.join(timeout)

    def _wake(self) -> None:
        try:
            Client(self.listener.address, authkey=self.authkey).close()
        except (EOFError, OSError, AuthenticationError):
            pass


def launch(size: int, command: Sequence[str], shape: Optional[str] = None) -> int:
    """Run ``command`` in ``size`` processes that form a BSP machine and
    return the exit status of the run.  ``shape`` groups the processes like
    a multi-level machine (see ``bsp4py.init``)."""
    if size < 1:
        raise ValueError(f"the number of processes must be at least 1, got {size}")
    authkey = os.urandom(32)
    listener = Listener(family=FAMILY, backlog=size, authkey=authkey)
    registrar = _Registrar(listener, size, authkey)
    registrar.start()

    processes: List[subprocess.Popen] = []
    waiters: List[threading.Thread] = []
    exits: queue.SimpleQueue[Tuple[int, int]] = queue.SimpleQueue()
    codes: Dict[int, int] = {}
    stopped: set = set()  # processes stopped by the launcher

    def stop(force: bool) -> None:
        for rank, process in enumerate(processes):
            if rank not in codes and process.poll() is None:
                stopped.add(rank)
                try:
                    process.kill() if force else process.terminate()
                except OSError:
                    pass

    def on_sigterm(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    previous_handler = None
    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.signal(signal.SIGTERM, on_sigterm)
    interrupted = False
    try:
        environment = dict(os.environ)
        # the processes find their backend by themselves, and the processes
        # they may start do not inherit a backend that is not theirs
        environment.pop("BSP4PY_BACKEND", None)
        if shape is not None:
            environment["BSP4PY_SHAPE"] = shape
        environment.update(
            {
                ENV_SIZE: str(size),
                ENV_ADDRESS: str(listener.address),
                ENV_AUTHKEY: authkey.hex(),
            }
        )
        for rank in range(size):
            environment[ENV_RANK] = str(rank)
            process = subprocess.Popen(
                list(command),
                env=environment,
                stdin=None if rank == 0 else subprocess.DEVNULL,  # like mpiexec
            )
            processes.append(process)
            waiter = threading.Thread(
                target=lambda rank=rank, process=process: exits.put((rank, process.wait())),
                name=f"bsp4py-wait-{rank}",
                daemon=True,
            )
            waiter.start()
            waiters.append(waiter)

        deadline: Optional[float] = None  # when to stop the processes that remain
        force = False
        while len(codes) < len(processes):
            try:
                rank, code = exits.get(timeout=0.1)
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    stop(force)
                    force, deadline = True, time.monotonic() + GRACE
                continue
            codes[rank] = code
            registrar.process_exited()
            if code != 0 and deadline is None:
                deadline = time.monotonic() + GRACE
    except KeyboardInterrupt:
        interrupted = True
        stop(force=False)
        limit = time.monotonic() + GRACE
        while any(p.poll() is None for p in processes) and time.monotonic() < limit:
            time.sleep(0.02)
        stop(force=True)
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)
        stop(force=True)  # nothing to stop, unless the launcher itself failed
        # The threads of the launcher are daemon threads, so that they can
        # never prevent it from leaving.  But none must still be running when
        # the interpreter shuts down, which kills them wherever they are and
        # is not safe (CPython can crash at that point): they are all waited
        # for here.
        limit = time.monotonic() + GRACE
        for waiter in waiters:
            waiter.join(max(0.0, limit - time.monotonic()))
        registrar.stop(GRACE)
        listener.close()

    if interrupted:
        return 130
    return _status(codes, stopped)


def _status(codes: Dict[int, int], stopped: set) -> int:
    """Exit status of the run, with a message when the machine broke down."""
    failures = [
        (rank, code)
        for rank, code in sorted(codes.items())
        if code not in (0, EXIT_PEER_LOST) and rank not in stopped
    ]
    lost = [rank for rank, code in codes.items() if code == EXIT_PEER_LOST]
    if failures:
        rank, code = failures[0]
        # killed by a signal, or (Windows) by an exception such as 0xC0000005,
        # an access violation: said even when the other processors finished
        crashed = code < 0 or code > 255
        if lost or stopped or crashed:
            if code < 0:
                what = f"was killed by signal {-code}"
            elif crashed:
                what = f"exited with status 0x{code:08X}"
            else:
                what = f"exited with status {code}"
            others = "; the other processors were stopped" if lost or stopped else ""
            print(f"bsp4py.run: processor {rank} {what}{others}", file=sys.stderr)
        return code if code > 0 else 128 - code
    if lost or stopped:
        print(
            "bsp4py.run: the processors lost each other: one of them finished before the "
            "others, which were stopped (the global part of a BSML program must do the "
            "same thing on every processor)",
            file=sys.stderr,
        )
        return EXIT_PEER_LOST
    return 0


def _parse(arguments: List[str]) -> Tuple[int, Optional[str], List[str]]:
    """Number of processes, shape and command to run, from the command line."""
    size: Optional[int] = None
    shape: Optional[str] = None
    position = 0

    def result(command: List[str]) -> Tuple[int, Optional[str], List[str]]:
        sizes = None if shape is None else parse_shape(shape)
        if sizes is None:
            return size or os.cpu_count() or 1, shape, command
        processes = 1
        for n in sizes:
            processes *= n
        if size is not None and size != processes:
            raise ValueError(f"--shape {shape} describes {processes} processes, not {size}")
        return processes, shape, command

    while position < len(arguments):
        argument = arguments[position]
        if argument in ("-h", "--help"):
            print(USAGE, end="")
            raise SystemExit(0)
        if argument == "--shape" or argument.startswith("--shape="):
            if "=" in argument:
                shape = argument.split("=", 1)[1]
                position += 1
            elif position + 1 >= len(arguments):
                raise ValueError("--shape needs a shape, like 2x4")
            else:
                shape = arguments[position + 1]
                position += 2
        elif argument in ("-n", "-np", "--np"):
            if position + 1 >= len(arguments):
                raise ValueError(f"{argument} needs a number of processes")
            size = _size(arguments[position + 1])
            position += 2
        elif argument.startswith("--np="):
            size = _size(argument[len("--np=") :])
            position += 1
        elif argument == "-m":
            if position + 1 >= len(arguments):
                raise ValueError("-m needs the name of a module")
            return result([sys.executable, "-m", *arguments[position + 1 :]])
        elif argument.startswith("-") and argument != "-":
            raise ValueError(f"unknown option {argument}")
        else:
            if not os.path.exists(argument):
                raise ValueError(f"no such file: {argument}")
            return result([sys.executable, *arguments[position:]])
    raise ValueError("no program to run")


def _size(text: str) -> int:
    try:
        size = int(text)
    except ValueError:
        raise ValueError(f"the number of processes must be an integer, got {text!r}") from None
    if size < 1:
        raise ValueError(f"the number of processes must be at least 1, got {size}")
    return size


def main(argv: Optional[List[str]] = None) -> int:
    """Command line of the launcher; returns the exit status."""
    try:
        size, shape, command = _parse(sys.argv[1:] if argv is None else list(argv))
    except ValueError as error:
        print(f"bsp4py.run: {error}\n\n{USAGE}", end="", file=sys.stderr)
        return 2
    return launch(size, command, shape)


if __name__ == "__main__":
    sys.exit(main())
