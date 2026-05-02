import numpy as np
from random import randint
import math, random
import itertools
import matplotlib.pyplot as plt

def planted_eps_1in3_clauses(n, eps, delta=0):
    # Generates a planted epsilon-1-in-3 SAT instance near the critical density.
    if eps > 0.2726:
        rcrit = 1.0 / (12.0 * eps * (1.0 - eps))
    else:
        rcrit = 0.6

    rng = np.random.default_rng()

    # critical density and m
    m = int(round(rcrit * (1.0 + delta) * n))

    x_star = rng.integers(0, 2, size=n, dtype=np.uint8)
    clauses = np.empty((m, 3), dtype=np.int32)

    for t in range(m):
        while True:
            vars_ = rng.choice(n, 3, replace=False).astype(np.int32) + 1
            neg  = (rng.random(3) < eps).astype(np.int32)   # Key: do not use uint8

            lits = vars_ * (1 - 2 * neg)                    # Results in +/- i
            vals = (x_star[vars_ - 1].astype(np.int32) ^ neg)  # Optional: unify to int32

            ok = (vals.sum() == 1)
            if ok:
                clauses[t] = lits
                break

    return x_star, clauses

def relabel_dimacs_clauses(clauses):
    # Renumbers variables in clauses to contiguous integers starting from 1.
    C = np.asarray(clauses, dtype=np.int32)
    vars_abs = np.unique(np.abs(C))
    idx = np.searchsorted(vars_abs, np.abs(C))  # 0..n_used-1
    out = np.sign(C).astype(np.int32) * (idx + 1)
    return out

def build_matrix_from_clauses(clauses, n=None):
    # Constructs the linear system (matrix A and vector b) for the mod-2 relaxation of the problem.
    C = np.asarray(clauses, dtype=np.int64)
    m = C.shape[0]
    if n is None:
        n = int(np.max(np.abs(C)))
    # variable indices in [0..n-1]
    idx = np.abs(C).astype(np.int64) - 1  # (m,3)

    A = np.zeros((m, n), dtype=np.uint8)

    # set A[row, idx]=1 for each of the 3 vars (XOR-accumulate to be safe if duplicates ever occur)
    rows = np.repeat(np.arange(m, dtype=np.int64), 3)
    cols = idx.reshape(-1)
    np.bitwise_xor.at(A, (rows, cols), 1)   # toggles 0/1

    # b = 1 XOR (#negated literals mod 2)
    neg_parity = (C < 0).sum(axis=1) & 1
    b = (1 ^ neg_parity).astype(np.uint8)

    return A, b

def rref_gf2(A_in, b_in):
    # Computes the Row Reduced Echelon Form (RREF) of the augmented matrix [A|b] over GF(2).
    A = (A_in.copy() & 1).astype(np.uint8, copy=False)
    b = (b_in.copy() & 1).astype(np.uint8, copy=False)
    m, n = A.shape

    row = 0
    pivots = []
    row_of_pivot = {}

    for col in range(n):
        # find pivot
        sel = None
        for r in range(row, m):
            if A[r, col]:
                sel = r
                break
        if sel is None:
            continue

        # swap
        if sel != row:
            A[[sel, row]] = A[[row, sel]]
            b[[sel, row]] = b[[row, sel]]

        # eliminate all other rows
        for r in range(m):
            if r != row and A[r, col]:
                A[r] ^= A[row]
                b[r] ^= b[row]

        pivots.append(col)
        row_of_pivot[col] = row
        row += 1
        if row == m:
            break

    return A, b, pivots, row_of_pivot

def affine_solutions_mod2(A, b):
    # Solves Ax=b (mod 2) returning a particular solution T and nullspace basis L.
    A = (np.asarray(A) & 1).astype(np.uint8, copy=False)
    b = (np.asarray(b).ravel() & 1).astype(np.uint8, copy=False)
    m, n = A.shape
    assert b.shape[0] == m

    A_r, b_r, pivots, row_of_pivot = rref_gf2(A, b)

    # inconsistency check: 0...0 | 1
    for r in range(m):
        if not A_r[r].any() and b_r[r]:
            return None, None

    pivot_set = set(pivots)
    free_cols = [j for j in range(n) if j not in pivot_set]

    # particular solution T: set free vars = 0, pivot vars = b
    T = np.zeros(n, dtype=np.uint8)
    for p in pivots:
        T[p] = b_r[row_of_pivot[p]]

    # nullspace basis L: one basis vector per free var
    k = len(free_cols)
    L = np.zeros((n, k), dtype=np.uint8)

    for idx, f in enumerate(free_cols):
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        for p in pivots:
            r = row_of_pivot[p]
            # from RREF row: x_p XOR (sum A_r[r,free]*x_free) = 0
            # only x_f=1 => x_p = A_r[r,f]
            v[p] = A_r[r, f]
        L[:, idx] = v

    return T, L

def verify(A, b, T, L):
    # Verifies that the computed solutions satisfy the linear system Ax=b and AL=0.
    A = (np.asarray(A) & 1).astype(np.uint8)
    b = (np.asarray(b).ravel() & 1).astype(np.uint8)
    T = (np.asarray(T).ravel() & 1).astype(np.uint8)
    L = (np.asarray(L) & 1).astype(np.uint8)

    ok_T = np.all(((A @ T) & 1) == b)
    ok_L = np.all(((A @ L) & 1) == 0)  # L may have 0 columns

    print(f"Verification: T={ok_T}, L={ok_L}")
    return ok_T and ok_L

def generaterandom(n, eposilon):
    # Orchestrates the generation of a random problem instance and solves its linear relaxation.
    
    ans, clauses = planted_eps_1in3_clauses(n, eposilon)
    clauses = relabel_dimacs_clauses(clauses)
    A, b = build_matrix_from_clauses(clauses)
    T, L = affine_solutions_mod2(A, b)
    return T, L, clauses, ans, A, b