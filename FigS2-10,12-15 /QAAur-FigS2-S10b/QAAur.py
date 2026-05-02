import numpy as np
import generator
import random
from parfor import parfor

def calculatesquantumstate(p2in, s):
    # Computes the quantum state vector corresponding to the binary string s.
    p2 = p2in.copy()
    n = np.shape(p2)[0]
    col = np.shape(p2)[1]
    state = np.zeros((n), dtype=int)
    for i in range(col):
        if(int(s[i]) == 1):
            state = state + p2[:, i]
    state = (state + 1) % 2
    return state

def calculateenergy(clausesin, statein):
    # Calculates the energy of a specific quantum state based on the problem clauses.
    clauses = clausesin.copy()
    state = statein.copy()
    m = np.shape(clausesin)[0]
    energy = 0
    for i in range(m):
        l1 = clauses[i][0]
        l2 = clauses[i][1]
        energy = energy + state[l1-1] * state[l2-1]
    return energy

def convertbin(x, n):
    # Converts an integer to a binary array of fixed length n.
    o_bin = bin(x)[2:] 
    out_bin = o_bin.rjust(n, '0')
    table = np.zeros((n), dtype='int')
    for i in range(n):
        table[i] = int(out_bin[i])
    return table

def convert210(s2in):
    # Converts a binary array back to a decimal integer.
    s2 = s2in.copy()
    length = len(s2)
    ten = 0
    for i in range(length):
        ten = ten + s2[i] * 2**(length - 1 - i)
    return ten

def calculateenergymatrix(dim, clausesin, p2in):
    # Constructs the diagonal energy matrix (Hamiltonian) for the given clauses.
    
    p2 = p2in.copy()
    matrix = np.zeros((2**dim))
    clauses = clausesin.copy()
    for i in range(2**dim):
        s = convertbin(i, dim)
        state = calculatesquantumstate(p2, s)
        energy = calculateenergy(clauses, state)
        matrix[i] = energy
    return matrix

def calculate(v1, mat):
    # Computes the expectation value of an operator given a state vector.
    v = np.dot(np.array(mat), np.array(v1))
    E = np.inner(np.array(v1).conjugate(), v)
    return E

def calculatecorrectmatrix(dim, clausesin, p2in):
    # Identifies the solution states (zero energy) and returns a diagonal indicator matrix.
    p2 = p2in.copy()
    matrix = np.zeros((2**dim))
    clauses = clausesin.copy()
    for i in range(2**dim):
        s = convertbin(i, dim)
        state = calculatesquantumstate(p2, s)
        energy = calculateenergy(clauses, state)
        if(energy < 0.5):
            matrix[i] = 1
    return matrix

def calculateenergymatrixur(n, clausesin):
    # Calculates the energy matrix for Unrestricted (UR) instances.
    clauses = clausesin.copy()
    matrix = np.zeros((2**n, 2**n))
    m = np.shape(clauses)[0]
    for i in range(2**n):
        s = convertbin(i, n)
        energy = 0
        for j in range(m):
            l1 = clauses[j][0]
            l2 = clauses[j][1]
            l3 = clauses[j][2]
            energy = energy + (s[l1-1] + s[l2-1] + s[l3-1] - 1) * (s[l1-1] + s[l2-1] + s[l3-1] - 1)
        matrix[i][i] = energy
    return matrix

def calculatecorrectmatrixur(n, clausesin):
    # Identifies the correct solution states for Unrestricted (UR) instances.
    clauses = clausesin.copy()
    matrix = np.zeros((2**n, 2**n))
    m = np.shape(clauses)[0]
    for i in range(2**n):
        s = convertbin(i, n)
        energy = 0
        for j in range(m):
            l1 = clauses[j][0]
            l2 = clauses[j][1]
            l3 = clauses[j][2]
            energy = energy + (s[l1-1] + s[l2-1] + s[l3-1] - 1) * (s[l1-1] + s[l2-1] + s[l3-1] - 1)
        if(energy < 0.5):
            matrix[i][i] = 1
    return matrix

def generatecase(n, k):
    # Generates a random solvable problem instance.
    m = int(np.floor(n * k))
    if(random.random() < n * k - m):
        m = m + 1
    while(1):
        clauses, p2 = generator.generaterandom(n, m)
        dimention = np.shape(p2)[1]
        m2 = calculatecorrectmatrix(dimention, clauses, p2)
        if(np.max(m2) > 0.5):
            break
    return m, clauses, p2

def initial(t: int) -> np.ndarray:
    # Generates the initial annealing schedule parameters.
    
    i = np.arange(1, t + 1, dtype=np.float64)
    s = i / (t + 1.0)
    w = np.exp(-5.0 * s * (1.0 - s))
    cum = np.cumsum(w)
    cum /= cum[-1]

    vec = np.zeros((2 * t,), dtype=np.float64)
    vec[t:2*t] = cum              
    vec[:t] = 1.0 - vec[t:2*t]
    return vec

def QAOApossur(clausesin, n, t):
    # Simulates the QAOA process for Unrestricted (UR) instances and returns success probability.
    
    clauses = clausesin.copy()
    para = initial(t) * 0.3
    m1 = calculateenergymatrixur(n, clauses)
    m2 = calculatecorrectmatrixur(n, clauses)
    diag2 = np.diag(m2)
    initialstate = np.ones((2**n)) / np.sqrt(2**n)
    diag = np.diag(m1)
    for i in range(t):
        g = para[t+i]
        b = para[i]
        r = np.exp(1j * g * diag)
        initialstate = r * initialstate
        trans = np.array([[0, 1], [1, 0]])
        matrixsingle = np.cos(b) * np.eye(2) - np.sin(b) * trans * 1j
        tensor = initialstate.reshape([2] * n)
        for k in range(n):
            tensor = np.moveaxis(tensor, k, 0)
            reshaped = tensor.reshape(2, -1)
            reshaped = matrixsingle @ reshaped
            tensor = reshaped.reshape(tensor.shape)
            tensor = np.moveaxis(tensor, 0, k)
        initialstate = tensor.reshape(-1)
    return sum((initialstate * initialstate.conjugate()) * diag2).real

def slopeur(
    plist, k,
    trials=10000, n_lo=5, n_hi=16,
    save_prefix="QAAurdata/QAAur",
    save_single=False,      
    save_single2=True,     
    single2_dtype=np.float32
):
    # Runs parallel simulations across problem sizes to gather performance data.
    plist = list(plist)
    plen = len(plist)

    n_list = np.arange(n_lo, n_hi, dtype=int)
    nn = len(n_list)

    sloperecordur = np.zeros((nn, plen), dtype=float)

    single_all = np.empty((nn,), dtype=object) if save_single else None
    single2_all = np.empty((nn, trials, plen), dtype=single2_dtype) if save_single2 else None

    for ni, n in enumerate(n_list):
        print(f"n = {n}")

        @parfor(range(trials), disable=False)
        def single(i):
            m, clauses, p2 = generatecase(int(n), float(k))
            return clauses, p2

        single_arr = np.array(single, dtype=object)  # (trials,), each element: (clauses, p2)
        if save_single:
            single_all[ni] = single_arr

        for pos, p_val in enumerate(plist):
            p_val_local = p_val
            n_local = int(n)

            @parfor(range(trials), disable=False)
            def single2(i):
                clauses_i, p2_i = single_arr[i]
                return QAOApossur(clauses_i, n_local, p_val_local)

            s2 = np.asarray(single2, dtype=float)

            if save_single2:
                single2_all[ni, :, pos] = s2.astype(single2_dtype, copy=False)

            p_hat = float(np.mean(s2))
            p_hat = max(p_hat, 1e-12)
            sloperecordur[ni, pos] = -np.log(p_hat)

    out_name = f"{save_prefix}{k}.npz"

    save_dict = dict(
        k=float(k),
        trials=int(trials),
        n_list=n_list,
        plist=np.asarray(plist),
        sloperecordur=sloperecordur,
    )
    if save_single:
        save_dict["single_all"] = single_all  
    if save_single2:
        save_dict["single2_all"] = single2_all  

    np.savez_compressed(out_name, **save_dict)
    print(f"Saved: {out_name}")

if __name__ == '__main__':
    slopeur([150], 0.7)
    slopeur([150], 0.725)
    slopeur([150], 0.75)
    slopeur([150], 0.55)
    slopeur([150], 0.575)
    slopeur([150], 0.6)
    slopeur([150], 0.626)
    slopeur([150], 0.65)
    slopeur([150], 0.675)