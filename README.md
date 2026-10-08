# bsp4py

**Bulk Synchronous Parallel programming for Python, in the style of BSML.**

bsp4py is a library for programming
[BSP](https://en.wikipedia.org/wiki/Bulk_synchronous_parallel) (Bulk
Synchronous Parallel) algorithms: a parallel program is written as an ordinary
sequential-looking program that manipulates *parallel vectors* with four
primitives. Programs are deterministic, deadlock-free, and their cost can be
read from their structure.

Its core is a port to Python of the OCaml library
[BSML 0.5](https://bsml-lang.github.io) (Bulk Synchronous Parallel ML) by
Frédéric Loulergue and his co-authors: the primitives, the standard library
(`Base`, `Comm`, `Sort`), and the skeletons on distributed structures. To it
are added the GPS function of the SGL model, a simplified communication
primitive, and multi-level machines (several computers with several cores
each), with a cost model that can be checked against measurements. The same
program runs

- **sequentially**, in a simulator of a BSP machine with any number of
  processors (ideal for learning, developing, debugging and testing),
- **in parallel on one computer**, with local processes started by
  `python -m bsp4py.run` (nothing to install: only the standard library is used),
- **in parallel with MPI**, through [mpi4py](https://mpi4py.readthedocs.io),
  on one computer or on a cluster.

Status: version 0.1, alpha. The API follows the OCaml library closely (the
functions keep their BSML names) and is covered by tests on the three
backends, but it has not been used in production.

## Installation

Python 3.9 or later. From a clone of the repository:

```sh
pip install .            # simulator and local processes, no dependency
pip install ".[mpi]"     # + mpi4py and cloudpickle, to run with MPI
```

The MPI backend needs an MPI library (MPICH, Open MPI, Intel MPI, MS-MPI...).
`cloudpickle` is optional: it allows messages to contain lambdas and closures.

| Backend | Started with | Needs | Tested on |
|---|---|---|---|
| `seq`, simulator | `python program.py` | nothing | Linux, macOS, Windows |
| `mp`, local processes | `python -m bsp4py.run -n 8 program.py` | nothing | Linux, macOS, Windows |
| `mpi` | `mpiexec -n 8 python program.py` | MPI, mpi4py | Linux, MPICH |

Nothing in the package is specific to a system: the `mp` backend uses
`multiprocessing.connection`, which provides Unix sockets on Linux and macOS
and named pipes on Windows. The test suite runs on the simulator and on local
processes on the three systems at every push (Python 3.9 and 3.13); the MPI
backend has only been run on Linux, with MPICH.

## A first program

```python
from bsp4py import bsp_p, mkpar, parfun, parprint, print_once, proj, put, this

# mkpar: one value per processor, computed from the processor number
squares = mkpar(lambda pid: pid * pid)
parprint(squares)

# parfun: apply a function on every processor (no communication)
doubled = parfun(lambda x: 2 * x, squares)

# put: each processor sends its value to its right neighbour (one superstep)
p = bsp_p()
messages = parfun(lambda pid, x: lambda dst: x if dst == (pid + 1) % p else None, this(), doubled)
received = put(messages)
from_the_left = parfun(lambda pid, f: f((pid - 1) % p), this(), received)

# proj: back to an ordinary value, known by everybody (one superstep)
f = proj(from_the_left)
print_once([f(pid) for pid in range(p)])
```

```console
$ python first.py                    # simulation with 4 processors
Process 0 : 0
Process 1 : 1
Process 2 : 4
Process 3 : 9
[18, 0, 2, 8]
$ BSP4PY_P=8 python first.py                # simulation with 8 processors
$ python -m bsp4py.run -n 8 first.py        # 8 local processes
$ mpiexec -n 8 python first.py            # 8 MPI processes
```

In an interactive session parallel vectors are displayed like in the BSML
toplevel:

```pycon
>>> from bsp4py import *
>>> mkpar(lambda pid: pid * pid)
<0, 1, 4, 9>
>>> scan_direct(lambda a, b: a + b, this())
<0, 1, 3, 6>
```

More programs are in [`examples/`](examples).

## The model in one minute

A BSP machine is a set of `p` processors with local memories, a network, and a
global synchronisation barrier. A computation is a sequence of *supersteps*:
the processors compute locally, send messages, and the barrier delivers them.

A BSML program has two levels:

- the **global** level is the program you write, an ordinary Python program;
- the **local** level is what runs on each processor: the functions you give
  to the primitives.

A *parallel vector* `<v_0, ..., v_{p-1}>` (type `Par`) holds one local value
per processor. There are exactly four ways to use it:

| Primitive | Meaning | Cost |
|---|---|---|
| `mkpar(f)` | `<f(0), ..., f(p-1)>` | local computation |
| `apply(vf, vv)` | `<f_0(v_0), ..., f_{p-1}(v_{p-1})>` | local computation |
| `put(vf)` | `f_i(j)` is the message sent by `i` to `j`; the result is `<g_0, ..., g_{p-1}>` with `g_j(i) = f_i(j)` | communication + barrier |
| `proj(v)` | the global function `i -> v_i` | communication + barrier |

plus `bsp_p()`, the number of processors. In `put`, `None` means "no message".

Parallel vectors cannot be nested: local code cannot use the primitives. This
is what keeps the cost model simple (the number of supersteps of a program is
the number of `put` and `proj` it executes).

## From OCaml BSML to Python

Programs translate almost mechanically. The differences come from the host
language:

| OCaml BSML | Python | |
|---|---|---|
| `'a par` | `Par[T]` | generic type, usable with type checkers |
| `bsp_p` | `bsp_p()` | a function, like `procs()` and `this()` |
| `mkpar (fun i -> e)` | `mkpar(lambda i: e)` | |
| `apply vf vv` | `apply(vf, vv)` | |
| `apply2 vf v w` | `apply(vf, v, w)` or `apply2(vf, v, w)` | functions of several arguments instead of curried functions |
| `parfun2 f v w` | `parfun(f, v, w)` or `parfun2(f, v, w)` | |
| `apply (apply vf v) w` | `apply(apply(vf, v), w)` | the curried style works too, see below |
| `<< f $v$ $w$ >>` | `parfun(f, v, w)` | local sections of BSML 0.5 |
| `<< f $this$ $v$ >>` | `parfun(f, this(), v)` | |
| `put (mkpar (fun i dst -> ...))` | `put(mkpar(lambda i: lambda dst: ...))` | |
| `Some v` / `None` in messages | `v` / `None` | |
| `proj v i` | `proj(v)(i)` | |
| `Invalid_processor` | `InvalidProcessor` | |
| comparison function of `regular_sampling_sort` | `key=` argument, like `sorted` | |
| `scatter_list`, `scatter_array`, `scatter_string` | one function for all Python sequences | the three names exist |

### Curried functions

OCaml functions are curried, and BSML programs use it: `parfun2 f v w` is
defined as `apply (parfun f v) w`. In this package the helpers that take
several vectors (`parfun`, `parfun2`..., `apply` with several vectors,
`apply2`...) expect ordinary Python functions of several arguments, which is
what existing Python functions are. The curried style is available as well,
because `apply` with one vector is exactly the `apply` of BSML:

```python
import operator
from bsp4py import apply, curry, mkpar, parfun, replicate, this, uncurry

v, w = mkpar(lambda i: 10 * i), this()

# curried functions written with nested lambdas ...
vf = replicate(lambda x: lambda y: x - y)
apply(apply(vf, v), w)                         # <0, 9, 18, 27>

# ... or obtained from ordinary functions with curry
subtract_from = parfun(curry(operator.sub), v)  # a vector of partial applications
apply(subtract_from, w)                        # <0, 9, 18, 27>

# and the other way round, a curried function given to the variadic parfun
parfun(uncurry(lambda x: lambda y: x - y), v, w)  # <0, 9, 18, 27>
```

`curry(f)(x)(y)(z) = f(x, y, z)` and `uncurry(g)(x, y, z) = g(x)(y)(z)`.

## What is in the package

Everything is available from the top-level package (`from bsp4py import ...`)
except the skeletons, which are in `bsp4py.skeleton` because of their names
(`map`, `zip`...).

- **Core** – `Par`, `bsp_p`, `mkpar`, `apply`, `put`, `proj`, `within_bounds`,
  `abort`, `start_timing`, `stop_timing`, `get_cost`, `bsp_g`, `bsp_l`,
  `bsp_r`, `init`; cost counters: `stats`, `global_stats`, `reset_stats`;
  structure of multi-level machines: `bsp_levels`, `bsp_topology`,
  `bsp_shape`, `bsp_groups`, `bsp_group`, `bsp_leader`, `bsp_distance`.
- **`bsp4py.base`** – `replicate`, `parfun` (`parfun2`...`parfun4`), `apply2`...
  `apply4`, `mask`, `applyat`, `applyif`, `procs`, `this`, `bsml_print`,
  `parprint`, `get_one`, `get_list`, `put_one`, `put_list`, `proj_list_pids`,
  `to_list`.
- **`bsp4py.comm`** – `shift`, `shift_left`, `shift_right`, `totex`,
  `total_exchange`, `scatter` (`scatter_list`...), `gather`, `gather_list`,
  `bcast_direct`, `bcast_totex` (`bcast_totex_gen`, `bcast_totex_list`...),
  `scan_direct`, `scan_logp`, `scan_wide` (`scan_wide_direct`,
  `scan_list_direct`...), `fold_direct`, `fold_logp`, `fold_wide`
  (`fold_list_direct`...), `fold_multi` and `bcast_multi` for multi-level
  machines, and `sgl_gps`, the GPS function of the SGL model: a simplified
  `put` (see below).
- **`bsp4py.sort`** – `regular_sampling_sort` (parallel sorting by regular
  sampling).
- **`bsp4py.skeleton`** – distributed lists: `make`, `par`, `extract`, `length`,
  `map`, `zip`, `map_index`, `zip_index`, `shift_left`, `shift_right`; and the
  data-parallel skeletons of SGL, which follow the levels of the machine:
  `reduce`, `scan`, `sort`, `dh` (see below).
- **`bsp4py.utils`** – `curry`, `uncurry`, `natmod`, `from_to`, `filtermap`,
  `compose`, `identity`.
- **`bsp4py.probe`** – measure of the BSP parameters of the machine (see below);
  not imported by `from bsp4py import *`.

Every function has a docstring: `help(bsp4py.scan_direct)`.

### Additions of the Python version

- **Dynamic check of nesting.** OCaml BSML relies on the programmer (or on a
  dedicated type system) to avoid nested parallel vectors. Here, using a
  primitive inside local code, returning a vector from local code or putting a
  vector in a message raises `NestingError`.
- **Cost counters.** `bsp4py.stats()` gives the number of supersteps executed
  and, for each of them, the size *h* of the communication in bytes;
  `bsp4py.reset_stats()` resets them. With the simulator *h* is the exact
  h-relation of the superstep; with MPI each process only knows its own
  traffic.
- **`print_once`**, a `print` that prints once whatever the backend (see
  below).
- **`to_list(v)`**, the list `[v_0, ..., v_{p-1}]`.
- **`sgl_gps`**, which is not in BSML 0.5: it comes from Chong Li's PhD thesis
  on SGL (see below).
- **Multi-level machines**: the structure of the machine, supersteps
  restricted to a level, cost counters and benchmark per level (see below).

### Differences with BSML 0.5

- `scan_logp` is a corrected version: the original is marked "to be verified"
  in the sources of BSML 0.5 and gives wrong results with operations that are
  not commutative.
- `fold_array_direct` uses the direct fold (in BSML 0.5 it uses `fold_logp`).
- `gather` and `gather_list` raise `GatherError` for an invalid processor, as
  documented (BSML 0.5 declares the exception but does not raise it).
- `parprint(vec, fmt=str)` takes a function that formats a value instead of a
  function that prints it.
- In `bsp4py.skeleton`, `make`, `par`, `length` and `extract` use fewer
  supersteps than the original, with the same results.
- `bsmlprobe` is adapted rather than ported: it measures *g* in seconds per
  byte and *l* in seconds instead of flops (see below), and the
  `~/.bsmllibrc` file is replaced by the arguments of `init` and the
  environment variables `BSP4PY_G`, `BSP4PY_L`, `BSP4PY_R`.
- Not ported: the functions for backward compatibility with BSMLlib 0.1
  (`Bsmlbckcomp`), the camlp4 syntax extension (replaced by the variadic
  `parfun`), the TCP backend (use MPI, or the `mp` backend on one computer),
  and the custom toplevel (the Python REPL does the job).

## Skeletons: map, reduce, scan, sort, dh

`bsp4py.skeleton` works on *distributed lists*: parallel vectors that hold
one list per processor and stand for the concatenation of these lists.

```python
from bsp4py import skeleton

xs = skeleton.make(lambda k: k * k, 1000)        # [0, 1, 4, ...], evenly distributed
ys = skeleton.map(lambda x: x + 1, xs)           # element by element, no communication
skeleton.reduce(lambda a, b: a + b, ys)          # 332834500, an ordinary value
zs = skeleton.scan(lambda a, b: a + b, ys)       # prefix sums, same distribution
ws = skeleton.sort(skeleton.par([5, 3, 9, 1]))   # sorted, and ordered between processors
skeleton.extract(ws)                             # [1, 3, 5, 9], back to an ordinary list
```

`reduce`, `scan`, `sort` and `dh` are the data-parallel skeletons that Chong
Li programmed with the SGL model (Section 5.2 of his thesis). An SGL program
is recursive: a master makes its children compute, gathers their results,
computes, and scatters data back to them. Here the leader of each group of
processors plays the master of the group, so these skeletons follow the levels
of the machine (next section) with supersteps restricted to each level; on a
flat machine they are the algorithms of the thesis for one master and `p`
workers.

| Skeleton | Result | Supersteps on `L` levels |
|---|---|---|
| `reduce(op, v)` | `x_0 op x_1 op ... op x_{n-1}`, as a value | `2L - 1` |
| `scan(op, v)` | the distributed list of the prefixes | `2L` |
| `sort(v, key=None)` | the sorted distributed list (regular sampling) | `2L + 1` |
| `dh(oplus, otimes, v)` | the "distributable homomorphism" (butterfly) | `log2(p)` |

`dh` expresses a class of divide-and-conquer algorithms: the result for a list
is obtained from the results `u` and `w` for its two halves, as `u_i oplus w_i`
in the first half and `u_i otimes w_i` in the second.
[`examples/dh_applications.py`](examples/dh_applications.py) ports the two
applications of the thesis, the Fast Fourier Transform and the solution of a
tridiagonal system of equations, and checks them against direct computations.
`dh` needs a number of processors that is a power of two, each holding the
same number of elements, a power of two.

Differences with the thesis: the last exchange of `sort` and the exchanges of
`dh` go directly between the workers, as in the experiments of the thesis,
instead of through the masters; `reduce` sends its result back down so that it
is known everywhere; in the tridiagonal solver one of the four row operations
is written in an equivalent form that does not overflow on large systems; and
when the groups of a machine are not made of consecutive processors `reduce`
and `scan` use the machine as a flat one, to keep the order of the elements.

## GPS: simplified communication

`put` is the general communication primitive of BSML, and the hardest one to
use: the program has to build a function from destinations to messages on
every processor, and gets a function from senders to messages. `sgl_gps` is a
simpler form proposed by Chong Li for the SGL (Scatter-Gather Language) model,
in which a master only communicates with its workers through `scatter`,
`pardo` and `gather` (on a flat machine: `mkpar`, `apply` and `proj`). GPS
stands for gather-process-scatter: the program reads like the data were
gathered, rearranged and scattered again, but it is exchanged directly between
the processors, in one superstep.

```python
sgl_gps(split, assemble, indata)
```

- `split(pid, value)` gives the `p` values that processor `pid` sends, one per
  destination, from its local value;
- `assemble(pid, received)` builds the new local value of processor `pid` from
  the list of the `p` values it received, one per sender;
- `indata` is the parallel vector of the data.

The two functions are ordinary sequential functions: this is the whole
description of the communication. For example the last step of a parallel
sample-sort, once every processor has sorted its list and the pivots are
known, is to send to processor `j` the elements that belong to the `j`-th
interval and to merge what arrives:

```python
split = lambda pid, sorted_list: slice_at(sorted_list, pivots)     # p slices
assemble = lambda pid, slices: merge_all(slices)
result = sgl_gps(split, assemble, locally_sorted)
```

[`examples/sample_sort_gps.py`](examples/sample_sort_gps.py) is the complete
program, a port of the sample-sort of the thesis below, next to the same
algorithm written with `put`. The function is a port of the BSML definition
given in Section 6.2 of Chong Li's PhD thesis, *Un modèle de transition
logico-matérielle pour la simplification de la programmation parallèle*
(Université Paris-Est, 2013, [tel-00952082](https://theses.hal.science/tel-00952082));
see also the paper [*GPS: Towards Simplified Communication on SGL
Model*](https://ieeexplore.ieee.org/document/6969454/).

## Choosing the machine

Nothing to do in general: the first use of the library selects the backend of
the launcher that started the program, and the sequential simulator when
there is none. An MPI launcher is recognised by the environment variables that
MPICH, Open MPI and Intel MPI set for the processes they start (only MPICH has
been tested so far); if yours is not recognised, set `BSP4PY_BACKEND=mpi`.

| Environment variable | `init` argument | |
|---|---|---|
| `BSP4PY_BACKEND` | `backend` | `seq`, `mp`, `mpi` or `auto` (default) |
| `BSP4PY_P` | `p` | number of simulated processors (default 4) |
| `BSP4PY_SHAPE` | `shape` | structure of a multi-level machine, like `2x4` (see below) |
| `BSP4PY_COPY` | `copy_messages` | `0` to stop copying messages in the simulator |
| `BSP4PY_G`, `BSP4PY_L`, `BSP4PY_R` | `g`, `l`, `r` | BSP parameters returned by `bsp_g()`... |

```python
import bsp4py

bsp4py.init("seq", p=16)   # simulate 16 processors
```

## Running on local processes

```console
$ python -m bsp4py.run -n 4 program.py arg1 arg2      # a script and its arguments
$ python -m bsp4py.run -n 4 -m package.module         # a module, like "python -m"
```

starts 4 Python processes that all execute the program and form a BSP machine
of 4 processors, like `mpiexec` does for MPI. `-n` defaults to the number of
CPUs; `python -m bsp4py` and the command `bsp4py` are synonyms. `--shape 2x4`
groups the processes like a multi-level machine (see below).

- The processes connect to each other (each one with all the others) the first
  time the program uses bsp4py, with `multiprocessing.connection`; the
  connections are authenticated with a key drawn by the launcher.
- The standard input goes to processor 0; the standard output and error of all
  the processes go to the terminal.
- The exit status is the one of the program. If a process fails, the launcher
  stops the others, says which processor failed and returns its status.
- If the processes stop agreeing (one finishes while the others wait for it at
  a barrier), the launcher stops them and says so instead of waiting forever.

This backend is meant for one computer: it removes the need for MPI when all
you want is to use the cores of your machine. On the 2-core machine used for
the development it gives the same speed-up as MPI on local computations and
the same cost per byte exchanged; a superstep without data costs somewhat more
(about 30 µs against 15 µs with 2 processes). For several machines, use MPI.

## Multi-level machines

A BSP machine is flat: `p` processors, all at the same distance from each
other. The machines people have are usually not: a few computers, each with
several cores. MPI already uses the fastest link for each pair of processes
(shared memory inside a computer, the network between two computers), so
programs run as fast as they can without any change. What a flat model cannot
do is *tell the cost*: one `g` and one `l` do not describe a machine whose
links differ by orders of magnitude, and they do not show how much of the
traffic of an algorithm goes through the slow links.

So this package lets a machine have **levels**. The processors are the leaves
of a tree of groups; the groups of level 1 gather the processors that are
closest to each other, the groups of level 2 gather groups of level 1, and so
on up to the whole machine. The *distance* between two processors is the level
of the smallest group that contains both. With 2 computers of 4 cores:

| | level-1 groups | level-2 group | distance 0–1 | distance 0–5 |
|---|---|---|---|---|
| `2x4` | `{0,1,2,3}`, `{4,5,6,7}` | `{0,...,7}` | 1 (same computer) | 2 (network) |

Any number of levels is possible (`2x2x4`: 2 racks of 2 computers of 4 cores),
and the groups need not have the same size. A flat machine has one level.

**Where the structure comes from**

- With MPI on several computers it is found automatically: two levels, the
  computers and the processes of each computer.
- Otherwise it is given: `BSP4PY_SHAPE=2x4`, `python -m bsp4py.run --shape 2x4`,
  or `bsp4py.init(shape="2x4")`; `bsp4py.init(topology=[...])` takes the path of
  every processor for machines that are not regular. On one computer such
  groups are virtual, which is enough to develop and to count costs. The
  shape also overrides what MPI finds (`BSP4PY_SHAPE=flat` for a flat machine).

**What programs can do with it**

```python
from bsp4py import bsp_levels, bsp_groups, bsp_distance, bsp_leader, put, fold_multi, bcast_multi

bsp_levels()            # 2
bsp_groups(1)           # [[0, 1, 2, 3], [4, 5, 6, 7]]
bsp_distance(0, 5)      # 2
bsp_leader(6, 1)        # 4: the smallest processor of the group of 6

put(messages, level=1)  # a superstep inside the groups of level 1
```

- `put(vf, level=k)` is a superstep **restricted to the groups of level `k`**:
  the processors only communicate, and synchronise, with the processors of
  their own group. It is cheaper than a global superstep: it only uses the
  fast links and does not wait for the other groups. Without `level`, `put`
  and `proj` are global, as in BSML.
- `fold_multi` and `bcast_multi` are the reduction and the broadcast that
  follow the tree of the groups: one value per group crosses the slowest
  links, instead of one per processor, at the price of `2L - 1` (resp. `L`)
  supersteps on a machine with `L` levels. On a flat machine they are
  `fold_direct` and `bcast_direct`.

**Verifying the cost of an algorithm**

- `bsp4py.stats()` records for every superstep its level and the bytes exchanged
  at each distance; `bsp4py.global_stats()` gives these counters for the whole
  machine with the parallel backends.
- `bsp4py.probe` measures a `g` and an `l` per level. A superstep of level `k`
  that exchanges `h_d` bytes at each distance `d` is predicted to take
  `sum(h_d * g_d) + l_k`; `parameters.communication_time(counters)` adds this
  up for a program.

[`examples/multilevel_fold.py`](examples/multilevel_fold.py) does it for the
two reductions:

```console
$ python -m bsp4py.run --shape 2x2 examples/multilevel_fold.py
mp backend, 4 processors, shape 2x2
  level 1: g = 2.43e-09 s/byte, l = 4.26e-05 s
  level 2: g = 2.25e-09 s/byte, l = 1.16e-04 s
fold_direct: supersteps of levels [2], bytes at distances 1..2: [800091, 1600182]
  predicted 5.665 ms, measured 7.490 ms; with outermost links 20 times slower the model predicts 74.119 ms
fold_multi: supersteps of levels [1, 2, 1], bytes at distances 1..2: [1600182, 800091]
  predicted 5.894 ms, measured 5.176 ms; with outermost links 20 times slower the model predicts 40.121 ms
both algorithms give the same result on every processor
```

Here the two groups are virtual, on one computer: both levels have the same
`g` and the two algorithms cost about the same. The counters show what
changes on a real cluster: `fold_multi` sends half as many bytes at distance
2, and the last figures are what the model predicts when these links are 20
times slower.

Limits of this first version: it has only been run with virtual groups on one
computer, so the automatic detection with MPI on several computers and the
quality of the predictions there are not verified; `g` is measured with
messages of raw bytes, and messages made of many small Python objects take
longer to serialise, which is a local computation that the model does not
count; only the reduction and the broadcast have a multi-level version so far.

## Measuring the BSP parameters

In the BSP model a machine is described by *p*, the number of processors, *r*,
their speed, *g*, the cost of communication, and *l*, the cost of a
synchronisation barrier; a superstep where each processor computes for at
most *w* and sends or receives at most *h* costs `w + h*g + l`.
`bsp4py.probe`, an adaptation of the `bsmlprobe` program of BSML, measures them.
Run it the way you run your programs:

```console
$ mpiexec -n 4 python -m bsp4py.probe          # or: python -m bsp4py.run -n 4 -m bsp4py.probe
Benchmark starts (mpi backend)
p = 4
r = 31.534 Mflops/s
g = 4.956e-09 s/byte   (0.156 flops/byte)
l = 8.130e-05 s        (2563.5 flops)
export BSP4PY_G=4.956...e-09 BSP4PY_L=8.130...e-05 BSP4PY_R=31534...
```

- *g* is in seconds per byte of (serialised) message, the unit of
  `bsp4py.stats()`: the slope of the line fitted through the times of supersteps
  that exchange from 0 to 1 MiB per processor.
- *l* is in seconds: the time of a superstep that exchanges nothing.
- *r* is the speed of the Python interpreter on the kernel of `bsmlprobe`, in
  flops per second. `g*r` and `l*r`, in parentheses, are the values in flops
  that `bsmlprobe` reports.

The last line sets the values returned by `bsp_g()`, `bsp_l()` and `bsp_r()`.
From a program:

```python
import bsp4py
from bsp4py.probe import probe

parameters = probe()            # BspParameters(p=..., r=..., g=..., l=...)
bsp4py.reset_stats()
...                             # the program to study
seconds = parameters.communication_time(bsp4py.stats())   # sum(h)*g + supersteps*l
```

Keep in mind that this is a model: the measures vary from run to run and with
the load of the machine, the cost of communication is not exactly linear in
the size of the messages, and with MPI `bsp4py.stats()` only counts the traffic
of the calling process. Expect an order of magnitude, not a precise timing.

## Writing programs that also run in parallel

With the `mp` and `mpi` backends the program is executed by every process
(SPMD); a parallel vector only holds the local value and `put` and `proj`
exchange messages. The result is the same as in the simulator provided the
program follows the rules of BSML:

1. **The global part of the program must be deterministic**: every process
   must take the same decisions. Seed random generators with the same value
   everywhere if you use them at the global level; read files and command
   lines identically on every process. Values obtained with `proj` are the
   same everywhere by construction.
2. **Local code must not modify global state.** In the simulator all the
   "processors" share one Python interpreter, in parallel they do not.
3. **Messages must be serialisable** with `pickle` (or `cloudpickle` when it is
   installed, which adds lambdas and closures). The simulator serialises the
   messages too, so that problems show up before going parallel and a
   processor cannot modify the data of another one.
4. **Use `print_once`, `parprint` or `bsml_print` to print.** A plain `print`
   at the global level is executed by every process.

If a process fails with an exception, the whole machine is stopped (with MPI
the library installs an exception hook that calls `MPI_Abort`; with local
processes the launcher stops the other processes).

A word on performance: the local computations run at the speed of Python, so
the benefit of the parallel execution depends on what the local functions do
(NumPy and other compiled code work well); the primary goals of this library
are the BSML programming model, teaching and prototyping.

## Tests

```sh
pip install -e ".[mpi,test]" numpy
python -m pytest                                          # simulator; also starts parallel runs
python -m bsp4py.run -n 4 -m pytest -p no:cacheprovider     # the same tests on local processes
mpiexec -n 4 python -m pytest -p no:cacheprovider         # the same tests as an MPI program
```

The tests observe vectors only through `proj`, so the test suite itself is a
BSML program: it runs unchanged on the three backends. The first command also
starts the suite and the examples with both launchers (the MPI runs are
skipped when MPI is not available).

## Credits and licence

BSML is the work of F. Loulergue and O. Ballereau, W. Bousdira, L. Gesbert,
F. Gava, G. Hains, G. Petiot and J. Tesson (with code from X. Leroy's
OCamlMPI). See https://bsml-lang.github.io for the OCaml library, its manual
and the publications about BSML.

The GPS function and the skeletons `reduce`, `scan`, `sort` and `dh` come
from Chong Li's PhD thesis on the SGL model, supervised by Gaétan Hains
(Sections 6.2 and 5.2; reference below). `sgl_gps`,
`examples/sample_sort_gps.py` and `examples/dh_applications.py` are ports of
its listings and algorithms; see also the papers
[*GPS: Towards Simplified Communication on SGL Model*](https://ieeexplore.ieee.org/document/6969454/)
and [IEEE Xplore 6341490](https://ieeexplore.ieee.org/document/6341490) on the
skeletons.

The multi-level machines draw on two lines of work on hierarchical BSP
machines, themselves built on L. G. Valiant's Multi-BSP model:

- **SGL**, the Scatter-Gather Language of Chong Li and Gaétan Hains: a
  machine is a tree of masters and workers, and its cost model has a `g` and
  an `l` for each level. Chong Li, *Un modèle de transition logico-matérielle
  pour la simplification de la programmation parallèle*, PhD thesis,
  Université Paris-Est, 2013
  ([tel-00952082](https://theses.hal.science/tel-00952082)).
- **Multi-ML**, the extension of BSML to Multi-BSP machines by Victor
  Allombert, supervised by Frédéric Gava and Julien Tesson. Victor Allombert,
  *Functional abstraction for programming multi-level architectures:
  formalisation and implementation*, PhD thesis, Université Paris-Est, 2017
  ([tel-01693568](https://theses.hal.science/tel-01693568)).

bsp4py takes from them the tree of groups of processors and the cost
parameters per level. It is a much simpler design, not an implementation of
either: the program stays a flat BSML program, in which supersteps can be
restricted to a level; there are no master processes, multi-functions or
tree-distributed values, and no static typing of the levels.

This Python version is derived from the sources of BSML 0.5 and is
distributed under the same licence, the GNU Lesser General Public License
version 2.1 (see [`LICENSE`](LICENSE)). It is not affiliated with the authors
of BSML.
