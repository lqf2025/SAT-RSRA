import numpy as np
from random import randint
import math,random
import itertools
import matplotlib.pyplot as plt
def generateclause(n,m):#序号范围[1,n]要求n>=3
    c=np.zeros((m,3),dtype=int)
    for i in range(m):
        a1=randint(1,n)
        while(1):
            a2=randint(1,n)
            if(a1!=a2):
                break
        while(1):
            a3=randint(1,n)
            if(a3!=a1 and a3!=a2):
                break
        c[i][0]=a1
        c[i][1]=a2
        c[i][2]=a3
    return c
def printclause(clausein,m):
    clause=clausein.copy()
    for i in range(m):
        string="clause "+str(i)+str(":")+str(clause[i][0])+","+str(clause[i][1])+","+str(clause[i][2])+"  "
        print(string, end=" ")
        if(i%7==6):
            print("") 
    print("") 
def build_matrix_from_clauses(clauses):
    """
    clauses: list of triples (1-based variable indices)
    returns: A (m x n) numpy array dtype=int (entries 0/1), b (length m, all 1s)
    """
    if len(clauses) == 0:
        return np.zeros((0,0), dtype=np.int8), np.zeros((0,), dtype=np.int8)
    n = max(max(c) for c in clauses)
    m = len(clauses)
    A = np.zeros((m, n), dtype=np.int8)
    for i, c in enumerate(clauses):
        for v in c:
            A[i, v-1] = 1  # v is 1-based in input
    b = np.ones((m,), dtype=np.int8)
    return A, b
def rref_gf2(A_in, b_in):
    """
    Compute RREF of [A | b] over GF(2).
    Returns:
      A_rref, b_rref, pivot_cols (list of pivot column indices), row_of_pivot (dict col->row)
    """
    A = A_in.copy() % 2
    b = b_in.copy() % 2
    m, n = A.shape
    row = 0
    pivot_cols = []
    row_of_pivot = {}
    for col in range(n):
        # find pivot row with a 1 in this column at or below `row`
        sel = None
        for r in range(row, m):
            if A[r, col] == 1:
                sel = r
                break
        if sel is None:
            continue
        # swap sel and row if different
        if sel != row:
            A[[sel, row]] = A[[row, sel]]
            b[[sel, row]] = b[[row, sel]]
        # eliminate other rows' entries in this column
        for r in range(m):
            if r != row and A[r, col] == 1:
                # row r <- row r + row (mod 2)
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
      x0: one particular solution (numpy array length n) or None if no solution
      L: numpy array n x k whose columns form a basis for nullspace (over GF(2))
    """
    m, n = A.shape
    A_r, b_r, pivots, row_of_pivot = rref_gf2(A, b)
    # check inconsistency: a zero row in A but b==1 -> no solution
    for r in range(m):
        if np.all(A_r[r, :] == 0) and b_r[r] == 1:
            return None, None  # no solution

    pivot_set = set(pivots)
    free_cols = [j for j in range(n) if j not in pivot_set]

    # construct particular solution x0 by setting free vars = 0
    x0 = np.zeros((n,), dtype=np.int8)
    for p in pivots:
        r = row_of_pivot[p]
        # row equation: x_p + sum_{free} A_r[r,free]*x_free = b_r[r]
        # with x_free = 0 => x_p = b_r[r]
        x0[p] = int(b_r[r] % 2)

    # construct nullspace basis: for each free var f, set x_free[f]=1, others free=0,
    # and compute pivot entries x_p = sum(A_r[r, free]*x_free) (mod 2)
    basis = []
    for f in free_cols:
        v = np.zeros((n,), dtype=np.int8)
        v[f] = 1
        for p in pivots:
            r = row_of_pivot[p]
            # pivot variable value for this homogeneous solution:
            # x_p = sum_{free} A_r[r,free] * x_free (mod2)
            # here only free f might be 1
            v[p] = int(A_r[r, f] % 2)
        basis.append(v)
    if len(basis) == 0:
        L = np.zeros((n, 0), dtype=np.int8)  # zero columns
    else:
        L = np.stack(basis, axis=1)  # n x k
    return  L % 2
def verify(m,clausesin,p2in):
    clauses=clausesin.copy()
    p2=p2in.copy()
    s=0
    for i in range(m):
        l1=clauses[i][0]
        l2=clauses[i][1]
        l3=clauses[i][2]
        s=s+np.sum((p2[l1-1,:]+p2[l2-1,:]+p2[l3-1,:])%2)
    if(s==0):
        print("verified")
    else:
        print("not verified")
def generaterandom(n,m):   
    clauses=generateclause(n,m)
    n=int(len(np.unique(clauses)))
    for i in range(n):
        replace_dict = dict(zip(np.unique(clauses), range(1,n+1)))
    result=clauses.copy()
    for old, new in replace_dict.items():
        result[clauses == old] = new
    clauses=result
    A,b=build_matrix_from_clauses(clauses)
    p2=solve_linear_mod2(A,b)
    #verify(m,clauses,p2)
    return clauses,p2
#generaterandom(100,63)
