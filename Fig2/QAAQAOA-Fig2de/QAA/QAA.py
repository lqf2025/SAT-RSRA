import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import expm
from parfor import parfor
import random

def calculatesquantumstate(p2in, s):
    # Computes the quantum state vector corresponding to the input configuration s.
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
    # Converts a decimal integer into a binary array of fixed length n.
    o_bin = bin(x)[2:] 
    out_bin = o_bin.rjust(n, '0')
    table = np.zeros((n), dtype='int')
    for i in range(n):
        table[i] = int(out_bin[i])
    return table

def convert210(s2in):
    # Converts a binary array representation back into a decimal integer.
    s2 = s2in.copy()
    length = len(s2)
    ten = 0
    for i in range(length):
        ten = ten + s2[i] * 2**(length - 1 - i)
    return ten

def calculateenergymatrix(dim, clausesin, p2in):
    # Generates the diagonal Hamiltonian matrix representing the problem energies.
    p2 = p2in.copy()
    matrix = np.zeros((2**dim))
    clauses = clausesin.copy()
    @parfor(range(2**dim), disable=True)
    def single(i):
        s = convertbin(i, dim)
        state = calculatesquantumstate(p2, s)
        energy = calculateenergy(clauses, state)
        return energy
    return np.array(single)

def calculate(v1, mat):
    # Computes the expectation value of the given matrix with respect to vector v1.
    v = np.dot(np.array(mat), np.array(v1))
    E = np.inner(np.array(v1).conjugate(), v)
    return E

def calculatecorrectmatrix(dim, clausesin, p2in):
    # Generates a diagonal matrix identifying solution states where energy is minimal.
    p2 = p2in.copy()
    matrix = np.zeros((2**dim))
    clauses = clausesin.copy()
    @parfor(range(2**dim), disable=True)
    def single(i):
        s = convertbin(i, dim)
        state = calculatesquantumstate(p2, s)
        energy = calculateenergy(clauses, state)
        if(energy < 0.5):
            return 1
        else:
            return 0
    return np.array(single)

def initial(t: int) -> np.ndarray:
    # Generates the annealing schedule parameters based on the total number of steps.
    i = np.arange(1, t + 1, dtype=np.float64)
    s = i / (t + 1.0)
    w = np.exp(-5.0 * s * (1.0 - s))
    cum = np.cumsum(w)
    cum /= cum[-1]

    vec = np.zeros((2 * t,), dtype=np.float64)
    vec[t:2*t] = cum              
    vec[:t] = 1.0 - vec[t:2*t]
    return vec

def QAAposs(t):
    # Simulates the Quantum Adiabatic Algorithm evolution to find the ground state probability and energy.
    para = initial(t) * 0.3
    initialstate = np.ones((2**dimention)) / np.sqrt(2**dimention)
    for i in range(t):
        g = para[t+i]
        b = para[i]
        r = np.exp(1j * g * diag)
        initialstate = np.multiply(r, initialstate)
        trans = np.array([[0, 1], [1, 0]])
        matrixsingle = np.cos(b) * np.eye(2) - np.sin(b) * trans * 1j
        tensor = initialstate.reshape([2] * dimention)
        for k in range(dimention):
            # Move kth dimension to first position
            tensor = np.moveaxis(tensor, k, 0)
            # Reshape for matrix multiplication
            reshaped = tensor.reshape(2, -1)
            # Apply single-qubit gate
            reshaped = matrixsingle @ reshaped
            # Restore shape and move dimension back
            tensor = reshaped.reshape(tensor.shape)
            tensor = np.moveaxis(tensor, 0, k)
        initialstate = tensor.reshape(-1)
    return sum((initialstate * initialstate.conjugate()) * diag2).real, sum((initialstate * initialstate.conjugate()) * diag).real

if __name__ == '__main__':
    n = 100
    m = 63
    filename = "PQC" + str(n) + "," + str(m) + ".npz"
    clauses = np.load(filename)['clauses']
    m = np.load(filename)['m']
    p2 = np.load(filename)['p2']
    
    diag = calculateenergymatrix(np.shape(p2)[1], clauses, p2)
    diag2 = calculatecorrectmatrix(np.shape(p2)[1], clauses, p2)
    dimention = np.shape(p2)[1]
    
    print(f"Matrix Dimension: {np.shape(p2)[0]}")
    print(f"M value: {m}")
    
    energylist = []
    posslist = []
    trange2 = 12
    for t in (np.array(range(trange2)) + 2):
        poss, e = QAAposs(t)
        print(f"Steps: {t} | Prob: {poss:.6f} | Energy: {e:.6f}")
        energylist.append(e)
        posslist.append(poss)
        
    np.savez("QAAdraw.npz", x=np.array(range(trange2))+2, energylist=energylist, possibility=posslist)