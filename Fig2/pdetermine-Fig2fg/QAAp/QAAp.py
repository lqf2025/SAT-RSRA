import numpy as np
import generator
import random
from parfor import parfor

def calculatesquantumstate(p2in, s):
    """Map an RSRA subspace label s to the corresponding n-bit assignment vector."""
    p2 = p2in.copy()
    n = np.shape(p2)[0]
    col = np.shape(p2)[1]
    state = np.zeros((n), dtype=int)
    for i in range(col):
        if int(s[i]) == 1:
            state = state + p2[:, i]
    state = (state + 1) % 2
    return state

def calculateenergy(clausesin, statein):
    """Compute the 2-SAT energy (number of violated clauses) for a given assignment."""
    clauses = clausesin.copy()
    state = statein.copy()
    m = np.shape(clausesin)[0]
    energy = 0
    for i in range(m):
        l1 = clauses[i][0]
        l2 = clauses[i][1]
        energy = energy + state[l1 - 1] * state[l2 - 1]
    return energy

def convertbin(x, n):
    """Convert an integer x into an n-bit binary array (MSB first)."""
    o_bin = bin(x)[2:]
    out_bin = o_bin.rjust(n, '0')
    table = np.zeros((n), dtype='int')
    for i in range(n):
        table[i] = int(out_bin[i])
    return table

def convert210(s2in):
    """Convert an MSB-first bit array to its integer value."""
    s2 = s2in.copy()
    length = len(s2)
    ten = 0
    for i in range(length):
        ten = ten + s2[i] * 2 ** (length - 1 - i)
    return ten

def calculateenergymatrix(dim, clausesin, p2in):
    """Build the diagonal cost vector diag over all 2^dim RSRA-subspace labels."""
    p2 = p2in.copy()
    matrix = np.zeros((2 ** dim))
    clauses = clausesin.copy()
    for i in range(2 ** dim):
        s = convertbin(i, dim)
        state = calculatesquantumstate(p2, s)
        energy = calculateenergy(clauses, state)
        matrix[i] = energy
    return matrix

def calculate(v1, mat):
    """Compute the quadratic"""
    v = np.dot(np.array(mat), np.array(v1))
    E = np.inner(np.array(v1).conjugate(), v)
    return E

def calculatecorrectmatrix(dim, clausesin, p2in):
    """Build the {0,1} target mask over subspace labels indicating energy==0."""
    p2 = p2in.copy()
    matrix = np.zeros((2 ** dim))
    clauses = clausesin.copy()
    for i in range(2 ** dim):
        s = convertbin(i, dim)
        state = calculatesquantumstate(p2, s)
        energy = calculateenergy(clauses, state)
        if energy < 0.5:
            matrix[i] = 1
    return matrix



def generatecase(n, k):
    """Generate a random instance with at least one satisfying assignment in the RSRA subspace."""
    m = int(np.floor(n * k))
    if random.random() < n * k - m:
        m = m + 1
    while (1):
        clauses, p2 = generator.generaterandom(n, m)
        dimention = np.shape(p2)[1]
        m2 = calculatecorrectmatrix(dimention, clauses, p2)
        if np.max(m2) > 0.5:
            break
    return m, clauses, p2

def initial_fast(t: int) -> np.ndarray:
    """Compute the schedule-based 2t-parameter initialization vector using vectorized numpy operations."""
    i = np.arange(1, t + 1, dtype=np.float64)
    s = i / (t + 1.0)
    w = np.exp(-5.0 * s * (1.0 - s))
    cum = np.cumsum(w)
    cum /= cum[-1]

    vec = np.zeros((2 * t,), dtype=np.float64)
    vec[t:2*t] = cum
    vec[:t] = 1.0 - vec[t:2*t]
    return vec

def QAAposs(clausesin, p2in, t):
    """Simulate the QAA-style circuit on the RSRA subspace and return the success probability."""
    clauses = clausesin.copy()
    p2 = p2in.copy()
    para = initial_fast(t) * 0.3
    dimention = np.shape(p2)[1]
    diag = calculateenergymatrix(dimention, clauses, p2)
    diag2 = calculatecorrectmatrix(dimention, clauses, p2)
    initialstate = np.ones((2 ** dimention)) / np.sqrt(2 ** dimention)
    for i in range(t):
        g = para[t + i]
        b = para[i]
        r = np.exp(1j * g * diag)
        initialstate = np.multiply(r, initialstate)
        trans = np.array([[0, 1], [1, 0]])
        matrixsingle = np.cos(b) * np.eye(2) - np.sin(b) * trans * 1j
        tensor = initialstate.reshape([2] * dimention)
        for k in range(dimention):
            tensor = np.moveaxis(tensor, k, 0)
            reshaped = tensor.reshape(2, -1)
            reshaped = matrixsingle @ reshaped
            tensor = reshaped.reshape(tensor.shape)
            tensor = np.moveaxis(tensor, 0, k)
        initialstate = tensor.reshape(-1)
    return sum((initialstate * initialstate.conjugate()) * diag2).real


def slope_raw(plist, tryn, trials=10000, n_range=range(25, 45), k=0.626, dtype=np.float32):
    """Generate and save raw success-probability data over (n, p) using parallelized instance sampling."""
    plist = np.asarray(list(plist), dtype=int)
    ns = np.asarray(list(n_range), dtype=int)
    plen = len(plist)
    Nn = len(ns)

    raw = np.empty((Nn, trials, plen), dtype=dtype)

    for ni, n in enumerate(ns):

        @parfor(range(trials), disable=False)
        def one_trial(i):
            m, clauses, p2 = generatecase(int(n), float(k))
            out = np.empty(plen, dtype=np.float32)
            for pos, p in enumerate(plist):
                out[pos] = QAAposs(clauses, p2, int(p))
            return out

        res = np.stack(one_trial, axis=0)
        raw[ni, :res.shape[0], :] = res

    out_name = f"QAAptry" + str(tryn) + ".npz"
    np.savez_compressed(
        out_name,
        k=float(k),
        trials=int(trials),
        ns=ns,
        plist=plist,
        raw=raw,
    )

if __name__ == "__main__":
    slope_raw(range(1, 150), 3, trials=2500)
