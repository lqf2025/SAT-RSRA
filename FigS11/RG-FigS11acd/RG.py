import numpy as np
import generator
import random
from pysat.solvers import Solver
from parfor import parfor


def count_models_inplace(s: Solver, limit=None) -> int:
    # Enumerates and counts satisfying assignments for the solver instance (note: modifies solver state).
    
    cnt = 0
    for _ in s.enum_models():
        cnt += 1
        if limit is not None and cnt >= limit:
            break
    return cnt

def forms2(clausesin):
    # Initializes a SAT solver with pairwise exclusion constraints (at most one).
    clauses = clausesin.copy()
    s = Solver(name='minisat22')
    m = np.shape(clauses)[0]
    for i in range(m):
        a, b, c = int(clauses[i][0]), int(clauses[i][1]), int(clauses[i][2])
        s.add_clause([-a, -b])
    return s

def forms(clausesin):
    # Initializes a SAT solver with Exact Cover (1-in-3) constraints.
    
    clauses = clausesin.copy()
    s = Solver(name="minisat22")
    m = np.shape(clauses)[0]
    for i in range(m):
        a, b, c = int(clauses[i][0]), int(clauses[i][1]), int(clauses[i][2])
        s.add_clause([a, b, c])
        s.add_clause([-a, -b, -c])
        s.add_clause([a, -b, -c])
        s.add_clause([-a, b, -c])
        s.add_clause([-a, -b, c])
    return s


def generatecase(n, k):
    # Generates a solvable instance and returns dimensions and model counts for both constraint types.
    m = int(np.floor(n * k))
    if random.random() < n * k - m:
        m += 1

    while True:
        clauses, p2 = generator.generaterandom(n, m)
        s = forms(clauses)
        if s.solve():
            # Create fresh solvers for counting to avoid state contamination
            s1 = forms(clauses)
            dim = int(np.shape(p2)[1])
            model_cnt = int(count_models_inplace(s1))
            s2 = forms2(clauses)
            model_cnt2 = int(count_models_inplace(s2))
            return dim, model_cnt, model_cnt2


def collect(k, trials=10000, n_min=10, n_max=20, disable=False):
    # Runs parallel simulations to gather model count statistics across problem sizes.
    n_list = np.arange(n_min, n_max, dtype=int)

    # Separate storage for dimensions and model counts
    dim_samples = np.empty((len(n_list), trials), dtype=np.int64)
    count_samples = np.empty((len(n_list), trials), dtype=np.int64)
    count_samples2 = np.empty((len(n_list), trials), dtype=np.int64)

    for ni, n in enumerate(n_list):

        @parfor(range(trials), disable=disable)
        def single(i, n=n, k=k):
            dim, cnt, cnt2 = generatecase(n, k)
            return int(dim), int(cnt), int(cnt2)

        arr = np.asarray(single, dtype=np.int64)  # shape (trials, 3)
        dim_samples[ni, :] = arr[:, 0]
        count_samples[ni, :] = arr[:, 1]
        count_samples2[ni, :] = arr[:, 2]

        print(f"n={n} | Progress: {ni+1}/{len(n_list)}", flush=True)

    out = f"RGdata/RG{k}.npz"
    np.savez_compressed(
        out,
        k=float(k),
        n_list=n_list,
        trials=int(trials),
        dim_samples=dim_samples,
        count_samples=count_samples,
        count_samples2=count_samples2,
    )
    print(f"[Saved] {out}", flush=True)


if __name__ == "__main__":
    collect(0.75)
    collect(0.725)
    collect(0.7)
    collect(0.675)
    collect(0.65)
    collect(0.626)
    collect(0.6)
    collect(0.575)
    collect(0.55)