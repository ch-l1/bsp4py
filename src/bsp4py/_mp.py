"""The ``mp`` backend: parallel execution by local processes, without MPI.

The launcher ``python -m bsp4py.run -n P program.py`` (see ``run.py``) starts
``P`` Python processes that all execute the program, like ``mpiexec`` does.
The first time the program uses bsp4py, each process connects to all the
others with ``multiprocessing.connection`` (Unix domain sockets on Linux and
macOS, named pipes on Windows); ``put`` and ``proj`` then exchange messages
on these connections.  Only the standard library is needed.

Setting up the machine::

    launcher                         process i
    --------                         ---------
    listens on an address A
    starts the P processes, giving
    them i, P, A and a secret key
                                     listens on an address A_i
                    <--------------  connects to A, sends (i, A_i)
    waits for the P addresses
    sends [A_0, ..., A_{P-1}]  ---->
                                     connects to A_j for each j < i
                                     accepts the connections of each j > i

Every connection is authenticated with the secret key drawn by the launcher.
"""

from __future__ import annotations

import atexit
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Connection, Listener
from typing import List, NoReturn, Optional, Tuple

from ._core import BackendError, _SpmdBackend

#: Environment variables through which the launcher describes the machine.
ENV_RANK = "BSP4PY_RUN_RANK"
ENV_SIZE = "BSP4PY_RUN_SIZE"
ENV_ADDRESS = "BSP4PY_RUN_ADDRESS"
ENV_AUTHKEY = "BSP4PY_RUN_AUTHKEY"

#: Kind of address used by the connections.
FAMILY = "AF_PIPE" if sys.platform == "win32" else "AF_UNIX"

#: Messages of at most this number of bytes are sent by the main thread,
#: without the help of the sender thread.  This cannot block: when a
#: processor starts a superstep, each of its connections holds at most the
#: message of the previous superstep (the other end may not have read it
#: yet, but it has to before it can go further), and a Unix socket always
#: has room for two messages this small.  If the previous message was a big
#: one, the send may wait until the other end reads it, which it does
#: without waiting for this processor.  Not used on Windows, where the
#: behaviour of named pipes has not been checked.
SMALL = 0 if sys.platform == "win32" else 1024

#: Exit status of a process that stops because another processor is gone.
EXIT_PEER_LOST = 86

#: Seconds left to the sender thread to finish when the process exits.
CLOSE_TIMEOUT = 1.0


@dataclass(frozen=True)
class Launch:
    """What the launcher tells to each process."""

    rank: int
    size: int
    address: str
    authkey: bytes


def _read_environment() -> Optional[Launch]:
    """Read *and remove* the description of the machine left by the launcher.

    The variables are removed so that the processes that the program may
    start itself do not take themselves for processors of the machine.
    """
    values = [os.environ.pop(name, None) for name in (ENV_RANK, ENV_SIZE, ENV_ADDRESS, ENV_AUTHKEY)]
    if any(value is None for value in values):
        return None
    rank, size, address, authkey = values
    assert rank is not None and size is not None and address is not None and authkey is not None
    return Launch(int(rank), int(size), address, bytes.fromhex(authkey))


#: Description of the machine when the program was started by the launcher.
LAUNCH = _read_environment()


def _machine_lost() -> NoReturn:
    """Stop this process: another processor is gone, the machine is broken.

    The launcher reports the error; the process leaves at once, like after
    an ``MPI_Abort``, because nothing can be computed any more.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # the streams may be closed already
            pass
    os._exit(EXIT_PEER_LOST)


class _Machine:
    """The connections of this process with the other processors."""

    def __init__(self, launch: Launch) -> None:
        self.pid = launch.rank
        self.p = launch.size
        self.connections: List[Optional[Connection]] = [None] * self.p
        try:
            self._connect(launch)
        except (EOFError, OSError, AuthenticationError):
            _machine_lost()
        # Messages are sent by a second thread while the main thread
        # receives: with blocking sends only, two processors that send big
        # messages to each other would wait for each other forever.
        self._outbox: queue.SimpleQueue[Optional[Tuple[List[Optional[bytes]], List[int]]]] = (
            queue.SimpleQueue()
        )
        self._sent = threading.Event()
        self._send_failed = False
        self._closed = False
        self._sender: Optional[threading.Thread] = None
        if self.p > 1:
            self._sender = threading.Thread(
                target=self._send_loop, name="bsp4py-sender", daemon=True
            )
            self._sender.start()
        atexit.register(self.close)

    def _connect(self, launch: Launch) -> None:
        key = launch.authkey
        with Listener(family=FAMILY, backlog=self.p, authkey=key) as listener:
            with Client(launch.address, authkey=key) as launcher:
                launcher.send((self.pid, listener.address))
                addresses = launcher.recv()
            for j in range(self.pid):
                connection = Client(addresses[j], authkey=key)
                connection.send(self.pid)
                self.connections[j] = connection
            for _ in range(self.pid + 1, self.p):
                connection = listener.accept()
                self.connections[connection.recv()] = connection

    def close(self) -> None:
        """Stop the sender thread and close the connections; called when the
        process exits.

        The sender is a daemon thread, so that it never keeps alive a process
        whose program is over.  But a daemon thread that is still running when
        the interpreter shuts down is killed wherever it is, which is not
        safe (CPython can crash at that point): the thread is asked to
        return, and is waited for, while the interpreter is still whole.
        """
        if self._closed:
            return
        self._closed = True
        sender, self._sender = self._sender, None
        if sender is not None:
            self._outbox.put(None)
            sender.join(CLOSE_TIMEOUT)
            if sender.is_alive():
                # it is sending to a processor that does not read (the
                # program was interrupted): the connections are left to it
                return
        for connection in self.connections:
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass

    def _send_loop(self) -> None:
        while True:
            work = self._outbox.get()
            if work is None:  # the process exits
                return
            buffers, peers = work
            try:
                self._send(buffers, peers)
            except (OSError, ValueError):
                self._send_failed = True
            finally:
                self._sent.set()

    def _send(self, buffers: List[Optional[bytes]], peers: List[int]) -> None:
        for j in peers:
            connection = self.connections[j]
            assert connection is not None
            # an empty message stands for "nothing" (a serialised value is
            # never empty)
            connection.send_bytes(buffers[j] or b"")

    def alltoall(self, buffers: List[Optional[bytes]], peers: List[int]) -> List[Optional[bytes]]:
        """Exchange with the processors ``peers`` (in increasing order): the
        other processors of the group that performs the superstep."""
        incoming: List[Optional[bytes]] = [None] * self.p
        incoming[self.pid] = buffers[self.pid]
        if not peers:
            return incoming
        if self._closed:
            raise BackendError("the process is exiting: the machine is closed")
        small = SMALL > 0 and all(b is None or len(b) <= SMALL for b in buffers)
        try:
            if small:  # cannot block: sent at once, without the other thread
                self._send(buffers, peers)
            else:
                self._sent.clear()
                self._outbox.put((buffers, peers))
            for j in peers:
                connection = self.connections[j]
                assert connection is not None
                incoming[j] = connection.recv_bytes() or None
        except (EOFError, OSError):
            _machine_lost()
        if not small:
            self._sent.wait()
            if self._send_failed:
                _machine_lost()
        return incoming


_machine: Optional[_Machine] = None


class MpBackend(_SpmdBackend):
    """SPMD execution by the local processes started by ``bsp4py.run``."""

    name = "mp"

    def __init__(self) -> None:
        super().__init__()
        global _machine
        if _machine is None:
            if LAUNCH is None:
                raise BackendError(
                    "the 'mp' backend needs its launcher: start the program with "
                    "'python -m bsp4py.run -n P program.py'"
                )
            _machine = _Machine(LAUNCH)
        self.machine = _machine
        self.p = _machine.p
        self.pid = _machine.pid

    def _alltoall(self, buffers: List[Optional[bytes]], level: int) -> List[Optional[bytes]]:
        peers = [j for j in self.topology.group(self.pid, level) if j != self.pid]
        return self.machine.alltoall(buffers, peers)

    def wtime(self) -> float:
        return time.perf_counter()

    def abort(self, code: int) -> None:
        sys.exit(code)
