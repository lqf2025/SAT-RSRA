import numpy as np
from random import randint
import math, random
import itertools
import matplotlib.pyplot as plt


def generateclause(n, m):  # variable indices are 1-based in [1, n], requires n >= 3
    c = np.zeros((m, 3), dtype=int)
    for i in range(m):
        a1 = randint(1, n)
        while True:
            a2 = randint(1, n)
            if a1 != a2:
                break
        while True:
            a3 = randint(1, n)
            if a3 != a1 and a3 != a2:
                break
        c[i][0] = a1
        c[i][1] = a2
        c[i][2] = a3
    return c


def printclause(clausein, m):
    clause = clausein.copy()
    for i in range(m):
        string = "clause " + str(i) + str(":") + str(clause[i][0]) + "," + str(clause[i][1]) + "," + str(clause[i][2]) + "  "
        print(string, end=" ")
        if (i % 7 == 6):
            print("")
    print("")


def build_matrix_from_clauses(clauses):
    """
    Build the linear system A x = 1 (mod 2) from 3-uniform hyperedges.
    Input clauses use 1-based variable indices.
    Returns:
      A: (m, n) binary incidence matrix (0/1), dtype=int8
      b: (m,) all-ones RHS vector, dtype=int8
    """
    if len(clauses) == 0:
        return np.zeros((0, 0), dtype=np.int8), np.zeros((0,), dtype=np.int8)

    n = max(max(c) for c in clauses)
    m = len(clauses)
    A = np.zeros((m, n), dtype=np.int8)
    for i, c in enumerate(clauses):
        for v in c:
            A[i, v - 1] = 1
    b = np.ones((m,), dtype=np.int8)
    return A, b


def rref_gf2(A_in, b_in):
    """
    Reduced row echelon form of [A | b] over GF(2).
    Returns:
      A_rref, b_rref, pivot_cols, row_of_pivot
    """
    A = A_in.copy() % 2
    b = b_in.copy() % 2
    m, n = A.shape

    row = 0
    pivot_cols = []
    row_of_pivot = {}

    for col in range(n):
        sel = None
        for r in range(row, m):
            if A[r, col] == 1:
                sel = r
                break
        if sel is None:
            continue

        if sel != row:
            A[[sel, row]] = A[[row, sel]]
            b[[sel, row]] = b[[row, sel]]

        for r in range(m):
            if r != row and A[r, col] == 1:
                A[r, :] ^= A[row, :]
                b[r] ^= b[row]

        pivot_cols.append(col)
        row_of_pivot[col] = row
        row += 1
        if row == m:
            break

    return A % 2, b % 2, pivot_cols, row_of_pivot


def solve_linear_mod2(A, b):
    """
    Solve A x = b (mod 2).
    Returns:
      A basis matrix L whose columns span the solution space's homogeneous part.
      (The particular solution is not returned in this implementation.)
    """
    m, n = A.shape
    A_r, b_r, pivots, row_of_pivot = rref_gf2(A, b)

    for r in range(m):
        if np.all(A_r[r, :] == 0) and b_r[r] == 1:
            return None

    pivot_set = set(pivots)
    free_cols = [j for j in range(n) if j not in pivot_set]

    basis = []
    for f in free_cols:
        v = np.zeros((n,), dtype=np.int8)
        v[f] = 1
        for p in pivots:
            r = row_of_pivot[p]
            v[p] = int(A_r[r, f] % 2)
        basis.append(v)

    if len(basis) == 0:
        L = np.zeros((n, 0), dtype=np.int8)
    else:
        L = np.stack(basis, axis=1)

    return L % 2


def verify(m, clausesin, p2in):
    """Sanity check for the generated p2 against the clause constraints."""
    clauses = clausesin.copy()
    p2 = p2in.copy()
    s = 0
    for i in range(m):
        l1 = clauses[i][0]
        l2 = clauses[i][1]
        l3 = clauses[i][2]
        s = s + np.sum((p2[l1 - 1, :] + p2[l2 - 1, :] + p2[l3 - 1, :]) % 2)
    if s == 0:
        print("verified")
    else:
        print("not verified")


def generaterandom(n, m):
    """Generate random clauses and a corresponding GF(2) solution subspace representation."""
    clauses = generateclause(n, m)

    n = int(len(np.unique(clauses)))
    replace_dict = dict(zip(np.unique(clauses), range(1, n + 1)))

    result = clauses.copy()
    for old, new in replace_dict.items():
        result[clauses == old] = new
    clauses = result

    A, b = build_matrix_from_clauses(clauses)
    p2 = solve_linear_mod2(A, b)
    # verify(m, clauses, p2)
    return clauses, p2

