import numpy as np
import random
from pysat.solvers import Solver
from parfor import parfor

# Adjust import based on your actual filename
import generators as generator 

# ============================================================
# CNF Encoding for Exact-1-of-3 Constraint
# ============================================================
def add_exact1_clause(s, a, b, c):
    # Adds the 5-clause CNF encoding enforcing that exactly one of the three literals is true.
    
    s.add_clause([ a,  b,  c])     # At least one is true
    s.add_clause([-a, -b, -c])     # Not all three are true
    s.add_clause([ a, -b, -c])     # Not (0, 1, 1)
    s.add_clause([-a,  b, -c])     # Not (1, 0, 1)
    s.add_clause([-a, -b,  c])     # Not (1, 1, 0)

def solve_stats(name, clauses):
    # Solves the instance using a specific CDCL solver and retrieves conflict and propagation statistics.
    
    s = Solver(name=name)
    m = clauses.shape[0]
    for i in range(m):
        a, b, c = int(clauses[i, 0]), int(clauses[i, 1]), int(clauses[i, 2])
        add_exact1_clause(s, a, b, c)
    s.solve()
    st = s.accum_stats()
    s.delete()
    return float(st["conflicts"]), float(st["propagations"])

# ============================================================
# Instance Generation and Solver Benchmarking
# ============================================================
def generatecase(n, eps):
    # Generates a random instance and collects performance metrics from four different SAT solvers.
    # You have: T, L, clauses, ans, A, b
    T, L, clauses, ans, A, b = generator.generaterandom(n, eps)
    clauses = np.asarray(clauses, dtype=np.int32)

    n_used = int(np.max(np.abs(clauses)))     
    m = int(clauses.shape[0])                 
    k_free = int(L.shape[1])                  

    cmini, pmini = solve_stats("minisat22",  clauses)
    ccad,  pcad  = solve_stats("cadical195", clauses)
    cglu,  pglu  = solve_stats("glucose42",  clauses)
    clin,  plin  = solve_stats("lingeling",  clauses)

    return n_used, m, k_free, cmini, ccad, cglu, clin, pmini, pcad, pglu, plin

# ============================================================
# Parallel Execution Wrapper
# ============================================================
def singlesize(n, eps, copy, disable=False):
    # Executes parallel simulations for a fixed problem size to gather statistical data.
    @parfor(range(copy), disable=disable)
    def singlecase(i):
        return generatecase(n, eps)

    out = np.asarray(singlecase)
    return out

# ============================================================
# Main Data Collection Loop
# ============================================================
def main(eps, begin, end, copy, disable=False):
    # Orchestrates the data collection process across a range of problem sizes and saves the results.
    ns = np.arange(begin, end, dtype=int)
    G = len(ns)

    # 11 metrics: n_used, m, k_free, 4*conflicts, 4*propagations
    n_used = np.empty((G, copy), dtype=int)
    m_arr  = np.empty((G, copy), dtype=int)
    k_free = np.empty((G, copy), dtype=int)

    cmini = np.empty((G, copy), dtype=float)
    ccad  = np.empty((G, copy), dtype=float)
    cglu  = np.empty((G, copy), dtype=float)
    clin  = np.empty((G, copy), dtype=float)

    pmini = np.empty((G, copy), dtype=float)
    pcad  = np.empty((G, copy), dtype=float)
    pglu  = np.empty((G, copy), dtype=float)
    plin  = np.empty((G, copy), dtype=float)

    for gi, n in enumerate(ns):
        out = singlesize(int(n), eps, copy, disable=disable)  # (copy,11)

        n_used[gi, :] = out[:, 0]
        m_arr[gi, :]  = out[:, 1]
        k_free[gi, :] = out[:, 2]

        cmini[gi, :]  = out[:, 3]
        ccad[gi, :]   = out[:, 4]
        cglu[gi, :]   = out[:, 5]
        clin[gi, :]   = out[:, 6]

        pmini[gi, :]  = out[:, 7]
        pcad[gi, :]   = out[:, 8]
        pglu[gi, :]   = out[:, 9]
        plin[gi, :]   = out[:, 10]

        print(f"n={n} | Progress: {gi+1}/{G}")

    np.savez_compressed(
        f"uniclassicalmix{eps}.npz",
        eps=float(eps),
        ns=ns,
        copy=int(copy),
        n_used=n_used,
        m=m_arr,
        k_free=k_free,
        cmini=cmini, ccad=ccad, cglu=cglu, clin=clin,
        pmini=pmini, pcad=pcad, pglu=pglu, plin=plin,
    )


if __name__ == "__main__":
    copy  = 5000
    begin = 10
    end   = 125
    eps   = 0.07

    main(eps, begin, end, copy, disable=False)