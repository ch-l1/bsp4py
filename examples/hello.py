"""First steps with BSML: parallel vectors and the four primitives.

python examples/hello.py                 # simulation, 4 processors
BSP4PY_P=6 python examples/hello.py        # simulation, 6 processors
mpiexec -n 6 python examples/hello.py    # 6 MPI processes
"""

from bsp4py import apply, bsp_p, mkpar, parfun, parprint, print_once, proj, put, this

print_once(f"The BSP machine has {bsp_p()} processors")

# mkpar: one value per processor, computed from the processor number
squares = mkpar(lambda pid: pid * pid)
parprint(squares)

# apply: a vector of functions applied pointwise to a vector of values
add_pid = mkpar(lambda pid: lambda x: x + pid)
parprint(apply(add_pid, squares), fmt=lambda x: f"pid*pid + pid = {x}")

# parfun: the same function everywhere, on one or several vectors
pairs = parfun(lambda pid, sq: (pid, sq), this(), squares)

# put: each processor sends a message to its right neighbour
to_the_right = parfun(
    lambda pid, sq: lambda dst: f"{sq} from {pid}" if dst == (pid + 1) % bsp_p() else None,
    this(),
    squares,
)
received = put(to_the_right)
parprint(
    parfun(lambda pid, f: f((pid - 1) % bsp_p()), this(), received), fmt=lambda m: f"received {m}"
)

# proj: from a parallel vector back to an ordinary (global) function
f = proj(squares)
print_once("Sum of the squares:", sum(f(pid) for pid in range(bsp_p())))
