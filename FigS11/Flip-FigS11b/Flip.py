import generator
import numpy as np
import random
from pysat.solvers import Solver
from parfor import parfor

def generatecase(n, k):
    # Generates a guaranteed solvable problem instance based on the ratio k.
    
    m = int(np.floor(n * k))
    if(random.random() < n * k - m):
        m = m + 1
    while(1):
        clauses, p2 = generator.generaterandom(n, m)
        dimention = np.shape(p2)[1]
        s = forms(clauses)
        if(s.solve() == True):
            break
    return m, clauses, p2

def forms(clausesin):
    # Initializes the SAT solver with CNF clauses derived from the input constraints.
    clauses = clausesin.copy()
    s = Solver(name='minisat22')
    m = np.shape(clauses)[0]
    for i in range(m):
        a, b, c = int(clauses[i][0]), int(clauses[i][1]), int(clauses[i][2])
        s.add_clause([a, b, c])
        s.add_clause([-a, -b, -c])
        s.add_clause([a, -b, -c])
        s.add_clause([-a, b, -c])
        s.add_clause([-a, -b, c])
    return s

def calculateallzerorow(pin): 
    # Identifies rows in the matrix that contain only zeros.
    p = pin.copy()
    r = np.shape(p)[0]
    c = np.shape(p)[1]
    zeronum = []
    for i in range(r):
        count = 0
        for j in range(c):
            count = count + p[i][j]
        if(count == 0):
            zeronum = np.append(zeronum, i)
    return zeronum

def seek11(clausesin, stringin): 
    # Identifies indices of clauses where a conflict occurs based on the current solution string.
    string = stringin.copy()
    clauses = clausesin.copy()
    m = np.shape(clauses)[0]
    wrongplace = []
    for i in range(m):
        l1 = clauses[i][0] - 1
        l2 = clauses[i][1] - 1
        if(string[l1] * string[l2] == 1):
            wrongplace = np.append(wrongplace, i)
    return wrongplace

def verify(stringin, clausesin):
    # Verifies if the current solution satisfies the Exact Cover constraint for all clauses.
    
    string = stringin.copy()
    clauses = clausesin.copy()
    for i in range(int(np.shape(clauses)[0])):
        l1 = clauses[i][0] - 1
        l2 = clauses[i][1] - 1
        l3 = clauses[i][2] - 1
        if(string[l1] + string[l2] + string[l3] != 1):
            print("Verification failed.")

def randomsolver(initialstringin, p2in, clausesin, maxtime):
    # iteratively attempts to solve the problem by resolving conflicts via basis changes and bit flips.
    
    string = initialstringin.copy()
    p2 = p2in.copy()
    dim = np.shape(p2)[1]
    clauses = clausesin.copy()
    zerorow = calculateallzerorow(p2)
    for i in range(maxtime):
        plnum = seek11(clauses, string)
        E = len(plnum)
        if(E == 0):
            verify(string, clauses)
            return i
        
        pl = int(random.sample(list(plnum), 1)[0])
        num1 = clauses[pl][0] - 1
        num2 = clauses[pl][1] - 1
        if(num1 in zerorow and num2 not in zerorow):
            r = 2
        if(num1 not in zerorow and num2 in zerorow):
            r = 1
        if(num1 not in zerorow and num2 not in zerorow):
            r = random.randint(1, 2)
        snumber = clauses[pl][r] - 1 
        
        for i in range(dim):
            if(p2[snumber][i] == 1):
                string = (string + p2[:, i]) % 2
                part1 = range(0, i)
                part2 = range(i + 1, dim)
                p = np.append(part1, part2)
                p = np.append(p, i)
                p = p.astype('int')
                p2 = p2[:, p]
                for j in range(dim - 1):
                    if(p2[snumber][j] == 1):
                        p2[:, j] = (p2[:, j] + p2[:, dim - 1]) % 2
                break 
    return 0

def singlerun(n, k, maxtime):
    # Executes a single simulation run: generates a case and attempts to solve it within the time limit.
    m, clauses, p2 = generatecase(n, k)
    initialstring = np.ones(int(np.max(clauses)), dtype=int)
    stime = randomsolver(initialstring, p2, clauses, maxtime)
    return stime

if __name__ == '__main__':
    k = 0.65
    maxtime = 10000
    reps = 10000
    ns = np.arange(70, 125) 

    posslist = np.empty(len(ns), dtype=float)
    meanlist = np.empty(len(ns), dtype=float)

    all_runs = np.empty((len(ns), reps), dtype=float)

    nz_chunks = []
    nz_offsets = np.zeros(len(ns) + 1, dtype=np.int64)
    cursor = 0

    for idx, n in enumerate(ns):
        @parfor(range(reps), disable=False)
        def single(i):
            return singlerun(int(n), k, maxtime)

        arr = np.asarray(single, dtype=float)
        all_runs[idx, :] = arr

        nz = arr[arr != 0]
        zero_count = reps - nz.size

        posslist[idx] = zero_count / reps
        meanlist[idx] = float(nz.mean()) if nz.size else np.nan

        nz_chunks.append(nz)
        cursor += nz.size
        nz_offsets[idx + 1] = cursor

    nz_values = np.concatenate(nz_chunks) if cursor else np.array([], dtype=float)

    np.savez_compressed(
        f"Flipdata/Flip{k}_raw.npz",
        k=float(k),
        maxtime=int(maxtime),
        reps=int(reps),
        ns=ns,
        posslist=posslist,
        mean=meanlist,
        all_runs=all_runs,
        nz_values=nz_values,
        nz_offsets=nz_offsets,
    )