import numpy as np
import generator
import random
from parfor import parfor

def s2(n, o1, o2):
    """Return the permuted basis index induced by the (o1,o2)-controlled update."""
    return (2 * o1 + o2) ^ n

def matrix2(th, o1, o2):
    """Construct the 4x4 two-bit transition matrix parameterized by th and (o1,o2)."""
    if (o1 == 0 and o2 == 0):
        return np.eye(4)
    s = np.sin(th / 2) ** 2
    c = np.cos(th / 2) ** 2
    matrix = np.eye(4) * c
    idx = [s2(x, o1, o2) for x in range(4)]
    for i in range(4):
        matrix[i][idx[i]] = s
    return matrix

def calculateaverage12(p2in, parameter, dim, num1, num2):
    """Compute the (num1,num2) pair contribution by sequentially applying dim local 4x4 updates."""
    vec = np.zeros((4), dtype=complex)
    vec[3] = 1
    for j in range(dim):
        q1 = p2in[num1 - 1][j]
        q2 = p2in[num2 - 1][j]
        mat = matrix2(parameter[j], q1, q2)
        vec = mat.dot(vec)
    return vec[3]

def calculateenergy(p2in, parameter, dim, clauses, m):
    """Compute the objective value by summing pair contributions over all clauses."""
    sE = 0.0
    for i in range(m):
        l1, l2 = clauses[i][0], clauses[i][1]
        sE += calculateaverage12(p2in, parameter, dim, l1, l2)
    return sE.real

def generatecase(n, k):
    """Generate a random instance (clauses, dim, p2) at size n with density parameter k."""
    m = int(np.floor(n * k))
    if random.random() < n * k - m:
        m += 1
    clauses, p2 = generator.generaterandom(n, m)
    return clauses, np.shape(p2)[1], p2

def calculatevariance(n, k):
    """Estimate the variance of a single-parameter gradient via symmetric finite differences."""
    while True:
        clause, dim, p2 = generatecase(n, k)
        if dim >= 1:
            break

    partial = random.randint(0, dim - 1)
    th = np.pi / 2
    sum1 = 0.0
    sum2 = 0.0
    varcopy = 100

    for _ in range(varcopy):
        params = np.random.rand(dim) * 4 * np.pi

        p1 = params.copy()
        p1[partial] += th
        E1 = calculateenergy(p2, p1, dim, clause, clause.shape[0])

        p2p = params.copy()
        p2p[partial] -= th
        E2 = calculateenergy(p2, p2p, dim, clause, clause.shape[0])

        gra = (E1 - E2) / (2 * np.sin(th))
        sum1 += gra
        sum2 += gra * gra

    return sum2 / varcopy - (sum1 / varcopy) ** 2

def savenpz(k):
    """Compute and save gradient-variance samples across n for a fixed clause density k."""
    randomcopy = 1000
    ns = np.arange(10, 30)
    vars_all = np.zeros((len(ns), randomcopy), dtype=float)

    for ni, n in enumerate(ns):
        n = int(n)

        @parfor(range(randomcopy), disable=False)
        def one(i):
            return calculatevariance(n, k)

        vals = np.asarray(one, dtype=float)
        vars_all[ni, :] = vals

        print("n =", n, "mean =", float(np.mean(vals)))

    np.savez_compressed(
        "BP" + str(k) + ".npz",
        k=float(k),
        ns=ns,
        vars_all=vars_all,
        randomcopy=int(randomcopy),
    )

if __name__ == '__main__':
    for k in [0.476, 0.526, 0.576, 0.626, 0.676, 0.726]:
        savenpz(k)
