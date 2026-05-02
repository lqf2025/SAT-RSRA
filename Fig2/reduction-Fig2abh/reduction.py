import numpy as np
from random import randint
import math, random
import generator
from pysat.solvers import Solver
from parfor import parfor

def remove_duplicates(lst):
    """Return a list with duplicates removed while preserving first-occurrence order."""
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def replace_elements(matrix, elements_to_zero):
    """Replace all occurrences of specified elements in a 2D list with zeros."""
    target_set = set(elements_to_zero)
    return [
        [0 if elem in target_set else elem for elem in row]
        for row in matrix
    ]

def greedy(clausesin):
    """Run a greedy elimination heuristic on clauses and return the resulting retained-variable count."""
    clauses = clausesin.copy()
    n = np.max(clauses)
    print(n)
    listdelete = []
    while (np.max(clauses) != 0):
        flattened = clauses.ravel()
        freq = np.bincount(flattened, minlength=n + 1)[1:n + 1]
        min_val = min(freq[freq != 0])
        min_indices = [i for i, val in enumerate(freq) if val == min_val]
        p = random.choice(min_indices) + 1
        rows_with_p = np.any(clauses == p, axis=1)
        elements = clauses[rows_with_p, :]
        list = elements.flatten()
        list = remove_duplicates(list)
        print(p, list)
        clauses = np.array(replace_elements(clauses, list))
        listdelete.append(p)
    return n - len(listdelete)

def generatecase(n, k):
    """Generate a random instance and return summary statistics for the produced clauses and reduction."""
    m = int(np.floor(n * k))
    if (random.random() < n * k - m):
        m = m + 1
    clauses, p2 = generator.generaterandom(n, m)
    return len(np.unique(clauses)), np.shape(p2)[1], greedy(clauses)

def forms(clausesin):
    """Build a SAT solver instance encoding the clause constraints in CNF form."""
    clauses = clausesin.copy()
    s = Solver(name='minisat22')
    m = np.shape(clauses)[0]
    for i in range(m):
        a, b, c = int(clauses[i][0]), int(clauses[i][1]), int(clauses[i][2])
        s.add_clause([a, b, c])
        s.add_clause([-a, -b, -c])
        s.add_clause([a, b, -c])
        s.add_clause([-a, b, c])
        s.add_clause([a, -b, c])
    return s

def count(n, k):
    """Estimate satisfiability probability by Monte Carlo sampling of random CNF instances."""
    @parfor(range(5000), disable=True)
    def single2(i):
        m = int(np.floor(n * k))
        if (random.random() < n * k - m):
            m = m + 1
        clauses = generator.generateclause(n, m)
        s = forms(clauses)
        if (s.solve() == True):
            return 1
        return 0
    p = sum(np.array(single2)) / 5000
    return p

if __name__ == '__main__':
    krange = 0.1 + np.array(range(100)) * 0.9 / 100
    nlist = np.zeros((67, 100))
    klist = np.zeros((67, 100))
    slist = np.zeros((67, 100))
    glist = np.zeros((67, 100))
    for pos in range(100):
        k = krange[pos]
        for n in range(10, 70):
            print(k, n)
            @parfor(range(5000), disable=True)
            def single(i):
                a1, a2, a3 = generatecase(n, k)
                return a1, a2, a3
            single = np.array(single)
            nlist[n - 3][pos] = sum(single[:, 0])
            klist[n - 3][pos] = sum(single[:, 1])
            glist[n - 3][pos] = sum(single[:, 2])
    np.savez("reduction.npz", nlist=nlist, klist=klist, glist=glist)
