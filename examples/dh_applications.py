"""Two applications of the dh skeleton: FFT and a tridiagonal system solver.

``skeleton.dh`` is the "distributable homomorphism" (the butterfly pattern
of divide-and-conquer).  These are the two program examples of Section
5.2.2.1 of Chong Li's PhD thesis:

* the Fast Fourier Transform of a list of ``n = 2^m`` numbers, and
* the solution of a tridiagonal system of ``n = 2^m`` equations ``A.x = b``.

Both are checked against a direct sequential computation.

    python examples/dh_applications.py [n]                     # 4 simulated processors
    python -m bsp4py.run --shape 2x2 examples/dh_applications.py [n]
    mpiexec -n 4 python examples/dh_applications.py [n]

The number of processors must be a power of two, at most ``n``.
"""

import cmath
import random
import sys

from bsp4py import bsp_p, print_once, skeleton

# -- Fast Fourier Transform -------------------------------------------------------


def bit_reversed(xs):
    """The list in bit-reversed order of the indices: dh splits a list in its
    two halves, the FFT in its elements of even and of odd indices."""
    n = len(xs)
    bits = n.bit_length() - 1
    return [xs[int(format(i, f"0{bits}b")[::-1], 2) if bits else 0] for i in range(n)]


def fft(xs):
    """``[sum(x_k * w**(k*i) for k in range(n)) for i in range(n)]`` where
    ``w`` is the n-th root of unity ``exp(2*pi*sqrt(-1)/n)``.

    The elements are triples (value, position, length of the current list):
    combining two lists of length n1, the element of position i gives
    ``u + w**i * v`` at position i and ``u - w**i * v`` at position i + n1,
    with w the root of unity of order 2*n1.
    """

    def root(i, n):
        return cmath.exp(2j * cmath.pi * i / n)

    def oplus(u, v):
        (x1, i, n1), (x2, _, _) = u, v
        return (x1 + root(i, 2 * n1) * x2, i, 2 * n1)

    def otimes(u, v):
        (x1, i, n1), (x2, _, _) = u, v
        return (x1 - root(i, 2 * n1) * x2, i + n1, 2 * n1)

    triples = skeleton.par([(complex(x), 0, 1) for x in bit_reversed(xs)])
    return skeleton.extract(skeleton.map(lambda t: t[0], skeleton.dh(oplus, otimes, triples)))


def direct_fourier_transform(xs):
    n = len(xs)
    return [
        sum(x * cmath.exp(2j * cmath.pi * k * i / n) for k, x in enumerate(xs)) for i in range(n)
    ]


# -- Tridiagonal system solver ------------------------------------------------------
#
# A row is (a1, a2, a3, a4): in a sub-system, a1 multiplies the unknown just
# before the sub-system, a2 the unknown of the row, a3 the unknown just after
# the sub-system, and a4 is the right-hand side.  Initially: the lower
# diagonal, the diagonal, the upper diagonal and b.  The four operations are
# row operations of a Gaussian elimination.


def star(a, b):
    c = a[1] / b[0]
    return (a[0], a[2] - c * b[1], -c * b[2], a[3] - c * b[3])


def diamond(a, b):
    c = a[2] / b[1]
    return (a[0] - c * b[0], a[1], -c * b[2], a[3] - c * b[3])


def circle(a, b):
    c = a[2] / b[1]
    return (a[0], a[1] - c * b[0], -c * b[2], a[3] - c * b[3])


def bullet(a, b):
    # The thesis gives (a1, -(a2/b1)*b2, a3 - b3*(a2/b1), a4 - b4*a2/b1): the
    # row a from which the first unknown of b is eliminated.  This is the
    # same equation multiplied by -b1/a2, i.e. the row b from which that
    # unknown is eliminated; it keeps the coefficients of b at their scale,
    # where the original form overflows for systems of a few hundred rows.
    c = b[0] / a[1]
    return (-c * a[0], b[1], b[2] - c * a[2], b[3] - c * a[3])


def tds(rows):
    """Solution ``x`` of the tridiagonal system given as the list of its rows
    ``(lower, diagonal, upper, right-hand side)``, with 0 for the lower
    element of the first row and for the upper element of the last one.

    The elements are triples (row, first row, last row of the sub-system the
    row is part of): two neighbouring sub-systems are combined using the
    last row of the first one and the first row of the second one.
    """

    def oplus(u, v):
        (a1, f1, l1), (_, f2, l2) = u, v
        special = star(l1, f2)
        return (diamond(a1, special), diamond(f1, special), bullet(circle(l1, f2), l2))

    def otimes(u, v):
        (_, f1, l1), (a2, f2, l2) = u, v
        special = circle(l1, f2)
        return (bullet(special, a2), diamond(f1, star(l1, f2)), bullet(special, l2))

    triples = skeleton.par([(row, row, row) for row in rows])
    solved = skeleton.extract(skeleton.map(lambda t: t[0], skeleton.dh(oplus, otimes, triples)))
    return [row[3] / row[1] for row in solved]


def thomas(rows):
    """Direct sequential solution of a tridiagonal system (Thomas algorithm)."""
    n = len(rows)
    c, d = [0.0] * n, [0.0] * n
    for i, (lower, diagonal, upper, rhs) in enumerate(rows):
        pivot = diagonal - (lower * c[i - 1] if i else 0.0)
        c[i] = upper / pivot
        d[i] = (rhs - (lower * d[i - 1] if i else 0.0)) / pivot
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = d[i] - (c[i] * x[i + 1] if i < n - 1 else 0.0)
    return x


def random_system(n, rng):
    """A diagonally dominant tridiagonal system."""
    rows = []
    for i in range(n):
        lower = rng.uniform(0.5, 1.5) if i > 0 else 0.0
        upper = rng.uniform(0.5, 1.5) if i < n - 1 else 0.0
        rows.append((lower, 4.0 + rng.random(), upper, rng.uniform(-10.0, 10.0)))
    return rows


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    rng = random.Random(2013)

    signal = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    error = max(abs(a - b) for a, b in zip(fft(signal), direct_fourier_transform(signal)))
    assert error < 1e-9 * n
    print_once(f"FFT of {n} numbers on {bsp_p()} processors: same as the direct transform")

    system = random_system(n, rng)
    error = max(abs(a - b) for a, b in zip(tds(system), thomas(system)))
    assert error < 1e-9
    print_once(f"tridiagonal system of {n} equations: same solution as the direct method")
