import numpy as np
import generator
import random
from parfor import parfor
import math

_S_CACHE = {}
_STRIDE_CACHE = {}

def _get_S(dim: int) -> np.ndarray:
    """Return the MSB-first bit matrix of all basis strings of length dim."""
    if dim in _S_CACHE:
        return _S_CACHE[dim]
    K = 1 << dim
    idx = np.arange(K, dtype=np.uint32)
    shifts = np.arange(dim - 1, -1, -1, dtype=np.uint32)
    S = ((idx[None, :] >> shifts[:, None]) & 1).astype(np.int8, copy=False)
    _S_CACHE[dim] = S
    return S

def _get_strides(dim: int):
    """Return (step, block) pairs for in-place single-qubit X-mixer updates on an MSB-first state vector."""
    if dim in _STRIDE_CACHE:
        return _STRIDE_CACHE[dim]
    strides = []
    for q in range(dim):
        step = 1 << (dim - 1 - q)
        block = step << 1
        strides.append((step, block))
    _STRIDE_CACHE[dim] = strides
    return strides

def initial(t: int) -> np.ndarray:
    """Construct the schedule 2t-parameter initialization vector used by the optimizer."""
    ssum = 0.0
    vec = np.zeros((2 * t,), dtype=np.float32)
    for i in range(1, t + 1):
        s = i / (t + 1)
        ssum += math.exp(-5.0 * s * (1.0 - s))
        vec[i + t - 1] = ssum
    vec /= ssum
    for i in range(t):
        vec[i] = 1.0 - vec[i + t]
    return vec

def calculateenergymatrix(dim, clausesin, p2in):
    """Compute the diagonal cost (energy) values for all 2^dim assignments in the RSRA subspace."""
    clauses = np.asarray(clausesin, dtype=np.int32)
    p2 = np.asarray(p2in, dtype=np.int8)
    n, dim2 = p2.shape
    assert int(dim) == int(dim2)

    K = 1 << int(dim)
    S = _get_S(int(dim))
    state_bits = (p2 @ S + 1) & 1
    state_bits = state_bits.astype(np.int8, copy=False)

    l1 = clauses[:, 0] - 1
    l2 = clauses[:, 1] - 1

    diag = np.zeros((K,), dtype=np.int16)
    for a, b in zip(l1, l2):
        diag += (state_bits[a] * state_bits[b]).astype(np.int16, copy=False)

    return diag.astype(float, copy=False)

def calculatecorrectmatrix(dim, clausesin, p2in):
    """Return a {0,1} mask over all 2^dim assignments indicating energy==0 for the given clause set."""
    diag = calculateenergymatrix(dim, clausesin, p2in)
    return (diag < 0.5).astype(float, copy=False)

def generatecase(n, k):
    """Sample a random instance until the induced search space contains at least one satisfying assignment."""
    m = int(np.floor(n * k))
    if random.random() < n * k - m:
        m = m + 1
    while True:
        clauses, p2 = generator.generaterandom(n, m)
        dim = np.shape(p2)[1]
        m2 = calculatecorrectmatrix(dim, clauses, p2)
        if np.max(m2) > 0.5:
            break
    return m, clauses, p2

def QAOAposs(dimention, t, parain, diag, diag2):
    """Evaluate the QAOA success probability for one instance given diagonal cost and target mask."""
    para = np.asarray(parain, dtype=float)
    dim = int(dimention)
    K = 1 << dim

    state = (np.ones(K, dtype=np.complex128) / np.sqrt(K))
    strides = _get_strides(dim)
    diag = np.asarray(diag, dtype=float)
    diag2 = np.asarray(diag2, dtype=float)

    for i in range(t):
        g = para[t + i]
        b = para[i]

        state *= np.exp(1j * g * diag)

        c = np.cos(b)
        s = np.sin(b)
        u00 = c
        u01 = -1j * s
        u10 = -1j * s
        u11 = c

        for step, block in strides:
            v = state.reshape(-1, block)
            a = v[:, :step]
            b2 = v[:, step:]

            newa = u00 * a + u01 * b2
            newb = u10 * a + u11 * b2
            a[:] = newa
            b2[:] = newb

        state = state.reshape(K)

    prob = (state.real * state.real + state.imag * state.imag)
    return float(np.sum(prob * diag2))

def averageQAOAsuccess(single, parain, num):
    """Compute the mean success probability over a list of precomputed instances."""
    para = np.asarray(parain, dtype=float)
    t = int(len(para) / 2)

    @parfor(range(num), disable=True)
    def singlecase(i):
        dim = np.shape(single[i][1])[1]
        return QAOAposs(dim, t, para, single[i][2], single[i][3])

    return sum(singlecase) / num

def gradient2(single, layer, par):
    """Estimate the finite-difference gradient of the mean success probability with respect to parameters."""
    plist = []
    for i in range(2 * layer):
        temp = np.asarray(par, dtype=float).copy()
        temp[i] = temp[i] + 0.001
        plist.append(temp)

    @parfor(range(2 * layer), disable=True)
    def singlecase(i):
        return averageQAOAsuccess(single, plist[i], train)

    return singlecase

def slope(n, run):
    """Train fixed-parameter QAOA at multiple depths and record test success probabilities."""
    plist = range(1, 45)
    lr1 = 0.09
    b1 = 0.86
    b2 = 0.9997
    eposilon = 1e-8
    plen = len(plist)
    data1 = np.zeros((plen), dtype=float)

    @parfor(range(train), disable=True)
    def single0(i):
        m, clauses, p2 = generatecase(n, 0.626)
        dim = np.shape(p2)[1]
        diag = calculateenergymatrix(dim, clauses, p2)
        diag2 = calculatecorrectmatrix(dim, clauses, p2)
        return clauses, p2, diag, diag2

    @parfor(range(test), disable=True)
    def single1(i):
        m, clauses, p2 = generatecase(n, 0.626)
        dim = np.shape(p2)[1]
        diag = calculateenergymatrix(dim, clauses, p2)
        diag2 = calculatecorrectmatrix(dim, clauses, p2)
        return clauses, p2, diag, diag2

    for pos in range(plen):
        print(pos)
        layer = plist[pos]
        guess = initial(layer) * 0.1
        garafor = np.zeros((2 * layer), dtype=float)
        hfor = 0.0
        for ep in range(epoch):
            E0 = averageQAOAsuccess(single0, guess, train)
            gradient = (np.array(gradient2(single0, layer, guess)) - E0) / 0.001
            gradient = -gradient
            garafor = garafor * b1 + gradient * (1 - b1)
            gnorm2 = np.linalg.norm(gradient) ** 2
            hfor = b2 * hfor + (1 - b2) * gnorm2
            vhat = hfor / (1 - b2 ** (ep + 1))
            mhat = garafor / (1 - b1 ** (ep + 1))
            mov = -lr1 / np.sqrt(eposilon + vhat) * (b1 * mhat + (1 - b1) / (1 - b1 ** (ep + 1)) * gradient)
            guess = guess + mov

        data1[pos] = averageQAOAsuccess(single1, guess, test)

    np.savez(f'QAOAsingle/{str(n)}/QAOAsingle{str(n)} {str(run)}.npz', data1=data1)

if __name__ == '__main__':
    epoch = 40
    train = 500
    test = 2000
    nsize = 20
    slope(nsize, 0)
    slope(nsize, 1)
    slope(nsize, 2)
    slope(nsize, 3)
    slope(nsize, 4)
