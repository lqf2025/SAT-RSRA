import numpy as np
import generator
import random
import dlx
from pysat.solvers import Solver
from parfor import parfor

def generatecase(n, k):
    # Generates a solvable SAT instance and collects performance stats from multiple solvers.
    
    m = int(np.floor(n * k))
    if(random.random() < n * k - m):
        m = m + 1
    while(1):
        clauses, p2 = generator.generaterandom(n, m)
        s = forms(clauses)
        if(s.solve() == True):
            s2 = forms2(clauses)
            s2.solve()
            s3 = forms3(clauses)
            s3.solve()
            s4 = forms4(clauses)
            s4.solve()
            break
    return int(np.shape(p2)[0]), m, clauses, p2, s.accum_stats()['conflicts'], s2.accum_stats()['conflicts'], s3.accum_stats()['conflicts'], s4.accum_stats()['conflicts'], s.accum_stats()['propagations'], s2.accum_stats()['propagations'], s3.accum_stats()['propagations'], s4.accum_stats()['propagations']

def forms(clausesin):
    # Initializes a Minisat22 solver with the problem clauses.
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

def forms2(clausesin):
    # Initializes a Cadical195 solver with the problem clauses.
    clauses = clausesin.copy()
    s = Solver(name='Cadical195')
    m = np.shape(clauses)[0]
    for i in range(m):
        a, b, c = int(clauses[i][0]), int(clauses[i][1]), int(clauses[i][2])
        s.add_clause([a, b, c])
        s.add_clause([-a, -b, -c])
        s.add_clause([a, -b, -c])
        s.add_clause([-a, b, -c])
        s.add_clause([-a, -b, c])
    return s

def forms3(clausesin):
    # Initializes a Glucose42 solver with the problem clauses.
    clauses = clausesin.copy()
    s = Solver(name='Glucose42')
    m = np.shape(clauses)[0]
    for i in range(m):
        a, b, c = int(clauses[i][0]), int(clauses[i][1]), int(clauses[i][2])
        s.add_clause([a, b, c])
        s.add_clause([-a, -b, -c])
        s.add_clause([a, -b, -c])
        s.add_clause([-a, b, -c])
        s.add_clause([-a, -b, c])
    return s

def forms4(clausesin):
    # Initializes a Lingeling solver with the problem clauses.
    clauses = clausesin.copy()
    s = Solver(name='Lingeling')
    m = np.shape(clauses)[0]
    for i in range(m):
        a, b, c = int(clauses[i][0]), int(clauses[i][1]), int(clauses[i][2])
        s.add_clause([a, b, c])
        s.add_clause([-a, -b, -c])
        s.add_clause([a, -b, -c])
        s.add_clause([-a, b, -c])
        s.add_clause([-a, -b, c])
    return s

#########################################

def s2(n, o1, o2):
    # Computes the bitwise index shift for the quantum state vector.
    return (2 * o1 + o2) ^ n

def matrix2(th, o1, o2):
    # Constructs the two-qubit rotation matrix based on parameters and control bits.
    if(o1 == 0 and o2 == 0):
        return np.eye(4) 
    s = np.sin(th / 2)**2
    c = np.cos(th / 2)**2
    matrix = np.eye(4) * c
    list = [s2(x, o1, o2) for x in range(4)]
    for i in range(4):
        matrix[i][list[i]] = s
    return matrix

def calculateaverage12(p2in, parameter, dim, num1, num2):
    # Calculates the expectation value for specific qubits (num1 corresponds to 0, num2 to 1).
    p2 = p2in.copy()
    r = parameter.copy()
    vec = np.zeros((4), dtype=complex)
    vec[3] = 1 # Initial state |11>
    for j in range(dim):
        q1 = p2[num1-1][j]
        q2 = p2[num2-1][j]
        mat = matrix2(r[j], q1, q2)
        vec = mat.dot(vec)
    pr = vec[3]
    return pr

def calculateenergy(p2in, parameter, dim, clauses, m):
    # Computes the total energy (cost function) of the system given the parameters.
    Energysum = 0
    p2 = p2in.copy()
    r = parameter.copy()
    for i in range(m):
        l1 = clauses[i][0]
        l2 = clauses[i][1]
        Energysum = Energysum + calculateaverage12(p2, r, dim, l1, l2)
    return Energysum.real

def mostcommonstate(n, p2in, parameter, dim):
    # Decodes the most probable bit string from the optimized quantum parameters.
    p2 = p2in.copy()
    r = parameter.copy()
    r = r / np.pi
    r = np.rint(r) % 2
    s = np.zeros((n), dtype=int)
    s = s + 1
    for i in range(dim):
        if(r[i] == 1):
            for j in range(n):
                s[j] = s[j] + p2[j][i]
    s = s % 2
    s = ''.join(str(k) for k in s)
    return s

def calculategradient(p2in, par, dim, clauses, m):
    # Estimates the gradient of the energy function using the parameter shift rule.
    p2 = p2in.copy()
    th = np.pi / 2
    gra = np.zeros((dim))
    for i in range(dim):
        tempar1 = par.copy()
        tempar1[i] = tempar1[i] + th
        E1 = calculateenergy(p2, tempar1, dim, clauses, m)
        tempar2 = par.copy()
        tempar2[i] = tempar2[i] - th
        E2 = calculateenergy(p2, tempar2, dim, clauses, m)
        gra[i] = (E1 - E2) / (2 * np.sin(th))
    return gra

###################################################

def VQEnAdam(clauses, initialparametersin, m, lr, maxepoch, p2in, dim, b1, b2, eposilon):
    # Optimizes parameters using the Nesterov-accelerated Adaptive Moment Estimation (NAdam) algorithm.
    
    para = initialparametersin.copy()
    p2 = p2in.copy()
    energylist = []
    garafor = np.zeros((dim)) # First moment estimate
    hfor = 0 # Second moment estimate
    for i in range(maxepoch):
        energy = calculateenergy(p2in, para, dim, clauses, m)
        energylist = np.append(energylist, energy)
        # print(f"NAdam Epoch: {i}, Energy: {energy:.5f}")
        gradient = calculategradient(p2, para, dim, clauses, m)
        # print(f"Gradient Norm: {np.linalg.norm(gradient):.5f}")
        if(np.linalg.norm(gradient) < 0.01):
            finalenergy = calculateenergy(p2in, para, dim, clauses, m)        
            return finalenergy
        
        
        
        garafor = garafor * b1 + gradient * (1 - b1)
        hfor = b2 * hfor + (1 - b2) * np.linalg.norm(gradient) * np.linalg.norm(gradient)
        vhat = hfor / (1 - b2**(i + 1))
        mhat = garafor / (1 - b1**(i + 1))
        mov = -lr * 1 / np.sqrt(eposilon + vhat) * (b1 * mhat + (1 - b1) / (1 - b1**(i + 1)) * gradient)
        # print(f"Update Step: {np.linalg.norm(mov):.5f}")
        para = para + mov
    finalenergy = calculateenergy(p2in, para, dim, clauses, m)      
    return finalenergy

def convertexactcover(clausesin, n, m):
    # Transforms the SAT clause set into an exact cover matrix format.
    
    clauses = clausesin.copy()
    convertedmatrix = np.zeros((n, m))
    for i in range(m):
        l1 = clauses[i][0] - 1
        l2 = clauses[i][1] - 1
        l3 = clauses[i][2] - 1
        convertedmatrix[l1][i] = 1
        convertedmatrix[l2][i] = 1
        convertedmatrix[l3][i] = 1
    return convertedmatrix

def singlesize(n, k):
    # Executes a batch of parallel simulations for a specific problem size n.
    @parfor(range(copy))
    def singlecase(i):
        learningadam = 1
        b1 = 0.9
        b2 = 0.999
        maxadamepoch = 150
        VQEsuccess = 0
        nr, m, clauses, p2, cmini, ccad, cglu, clin, pmini, pcad, pglu, plin = generatecase(n, k)
        mdlx = convertexactcover(clauses, nr, m)
        status, cdlx, pdlx = dlx.solve(mdlx)
        if(i < copy2): 
            dimention = np.shape(p2)[1] # Dimension
            initialparameters = np.random.rand(dimention) * 4 * np.pi
            paramove = initialparameters.copy()
            groundE = VQEnAdam(clauses, paramove, m, learningadam, maxadamepoch, p2, dimention, b1, b2, 1e-8)
            if(groundE < 0.95):
                VQEsuccess = 1
        return cmini, ccad, cglu, clin, pmini, pcad, pglu, plin, cdlx, pdlx, VQEsuccess
    out = np.asarray(singlecase)
    return out

def main(k):
    # Orchestrates data collection across problem sizes and saves the results to a file.
    ns = np.arange(begin, end) 
    G = len(ns)

    # 11 outputs stored as (G, copy)
    cmini = np.empty((G, copy), dtype=float)
    ccad  = np.empty((G, copy), dtype=float)
    cglu  = np.empty((G, copy), dtype=float)
    clin  = np.empty((G, copy), dtype=float)

    pmini = np.empty((G, copy), dtype=float)
    pcad  = np.empty((G, copy), dtype=float)
    pglu  = np.empty((G, copy), dtype=float)
    plin  = np.empty((G, copy), dtype=float)

    cdlx  = np.empty((G, copy), dtype=float)
    pdlx  = np.empty((G, copy), dtype=float)

    VQEsuccess = np.empty((G, copy), dtype=int)

    for gi, n in enumerate(ns):
        out = singlesize(int(n), k)  # (copy, 11)

        cmini[gi, :] = out[:, 0]
        ccad[gi, :]  = out[:, 1]
        cglu[gi, :]  = out[:, 2]
        clin[gi, :]  = out[:, 3]

        pmini[gi, :] = out[:, 4]
        pcad[gi, :]  = out[:, 5]
        pglu[gi, :]  = out[:, 6]
        plin[gi, :]  = out[:, 7]

        cdlx[gi, :]  = out[:, 8]
        pdlx[gi, :]  = out[:, 9]

        VQEsuccess[gi, :] = out[:, 10]

    np.savez_compressed(
        'unidata/uni' + str(k) + 'p.npz',
        k=float(k),
        ns=ns,
        copy=int(copy),
        copy2=int(copy2),
        cmini=cmini, ccad=ccad, cglu=cglu, clin=clin,
        pmini=pmini, pcad=pcad, pglu=pglu, plin=plin,
        cdlx=cdlx, pdlx=pdlx,
        VQEsuccess=VQEsuccess,
    )

if __name__ == '__main__':
    copy = 5000
    copy2 = 1000
    begin = 20
    end = 125
    k = 0.6

    small = 1e-10
    main(k)