import os
import math
import random
import numpy as np
import multiprocessing as mp
import generator

# ============================================================
# Configuration to prevent thread contention between Numpy/MKL and Multiprocessing.
# ============================================================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def get_workers(default=24):
    # Determines the appropriate number of worker processes based on SLURM variables.
    x = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_NTASKS")
    try:
        w = int(x) if x else default
    except Exception:
        w = default
    return max(1, w)

def make_chunks(seq, chunk_size):
    # Splits a sequence into smaller chunks for parallel processing.
    return [seq[i:i+chunk_size] for i in range(0, len(seq), chunk_size)]

NPROC = get_workers(default=24)

# ============================================================
# Global caches for bitstring matrices, initial states, and stride patterns.
# ============================================================
_S_CACHE = {}         # dim -> (S: int8[dim, 2^dim])
_STATE0_CACHE = {}    # dim -> uniform complex64 state
_STRIDE_CACHE = {}    # dim -> list[(step, block)] for q=0..dim-1 (MSB->LSB)

def _get_S(dim: int) -> np.ndarray:
    # Generates or retrieves the cached bitstring matrix for the given dimension.
    if dim in _S_CACHE:
        return _S_CACHE[dim]
    K = 1 << dim
    idx = np.arange(K, dtype=np.uint32)
    shifts = np.arange(dim - 1, -1, -1, dtype=np.uint32)  # MSB -> LSB
    S = ((idx[None, :] >> shifts[:, None]) & 1).astype(np.int8, copy=False)
    _S_CACHE[dim] = S
    return S

def _get_state0(dim: int) -> np.ndarray:
    # Generates or retrieves the uniform superposition initial state.
    if dim in _STATE0_CACHE:
        return _STATE0_CACHE[dim]
    K = 1 << dim
    st = (np.ones(K, dtype=np.complex64) / np.sqrt(np.float32(K))).astype(np.complex64, copy=False)
    _STATE0_CACHE[dim] = st
    return st

def _get_strides(dim: int):
    # Pre-computes stride patterns for efficient gate application.
    if dim in _STRIDE_CACHE:
        return _STRIDE_CACHE[dim]
    strides = []
    # Original implementation reshape([2]*dim) axis0 = MSB
    for q in range(dim):
        step = 1 << (dim - 1 - q)
        block = step << 1
        strides.append((step, block))
    _STRIDE_CACHE[dim] = strides
    return strides

# ============================================================
# Initial Parameter Generation
# ============================================================
def initial(t: int) -> np.ndarray:
    # Generates the initial annealing schedule parameters using a sigmoid-like curve.
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

# ============================================================
# Fast Diagonal and Mask Construction
# ============================================================
def build_diag_and_mask_fast(clauses_np: np.ndarray, p2_np: np.ndarray):
    # Pre-computes the diagonal Hamiltonian and solution mask from the SAT clauses.
    
    clauses = np.asarray(clauses_np, dtype=np.int32)
    p2 = np.asarray(p2_np, dtype=np.int8)
    n, dim = p2.shape
    K = 1 << dim

    # Optimize by only using variables involved in clauses
    vars_used = np.unique(clauses[:, :2].ravel()) - 1  # 0-index
    vars_used = vars_used.astype(np.int32, copy=False)
    p2_sub = p2[vars_used, :]  # (nv, dim)

    # S: (dim, K)
    S = _get_S(dim)  # int8

    # state_bits: (nv, K)
    state_bits = (p2_sub @ S + 1) & 1
    state_bits = state_bits.astype(np.int8, copy=False)

    # Map original variable indices to used variable positions
    inv = np.full((n,), -1, dtype=np.int32)
    inv[vars_used] = np.arange(len(vars_used), dtype=np.int32)

    diag = np.zeros((K,), dtype=np.int16)
    for (l1, l2) in clauses[:, :2]:
        a = inv[l1 - 1]
        b = inv[l2 - 1]
        diag += (state_bits[a] * state_bits[b]).astype(np.int16, copy=False)

    mask = (diag == 0).astype(np.float32)  # energy < 0.5 <=> energy==0
    diag = diag.astype(np.float32, copy=False)
    return dim, diag, mask

# ============================================================
# Problem Generation
# ============================================================
def generatecase_with_diags(n: int, k: float, seed: int):
    # Generates a random solvable SAT instance and its diagonal representation.
    random.seed(seed)
    np.random.seed(seed & 0xffffffff)

    m = int(np.floor(n * k))
    if random.random() < n * k - m:
        m += 1

    while True:
        clauses, p2 = generator.generaterandom(n, m)
        dim, diag, mask = build_diag_and_mask_fast(clauses, p2)
        if mask.max() > 0.5:
            return dim, diag, mask

# ============================================================
# QAOA Simulation (Fast Version)
# ============================================================
def QAOAposs_fast(dim: int, t: int, para: np.ndarray, diag: np.ndarray, mask: np.ndarray) -> float:
    # Simulates the QAOA circuit to calculate the success probability.
    
    K = 1 << dim
    state = _get_state0(dim).copy()  # complex64[K]
    strides = _get_strides(dim)

    for i in range(t):
        g = para[t + i]
        b = para[i]

        # Phase separator: state *= exp(i*g*diag)
        phase = np.exp(1j * (g * diag)).astype(np.complex64, copy=False)
        state *= phase

        # Mixer single-qubit gate: cos(b) I - i sin(b) X
        c = np.cos(b).astype(np.float32)
        s = np.sin(b).astype(np.float32)
        u00 = np.complex64(c)
        u01 = np.complex64(-1j * s)
        u10 = np.complex64(-1j * s)
        u11 = np.complex64(c)

        # Apply U on every qubit (MSB -> LSB) using block views
        for (step, block) in strides:
            v = state.reshape(-1, block)      # view
            a = v[:, :step]
            b2 = v[:, step:]

            # Temporary storage to avoid overwriting
            newa = u00 * a + u01 * b2
            newb = u10 * a + u11 * b2
            a[:] = newa
            b2[:] = newb

        # Restore flat shape (view updated in place)
        state = state.reshape(K)

    # prob = |state|^2
    prob = (state.real * state.real + state.imag * state.imag).astype(np.float32, copy=False)
    return float(np.sum(prob * mask))

# ============================================================
# Parallel Evaluation Utilities
# ============================================================
_GLOBAL_TRAINSET = None

def _worker_init(trainset):
    # Initializes the global dataset for worker processes.
    global _GLOBAL_TRAINSET
    _GLOBAL_TRAINSET = trainset

def _eval_one_theta(args):
    # Computes the average success probability for a given parameter set on a minibatch.
    theta, idxs = args
    t = len(theta) // 2
    ssum = 0.0
    for j in idxs:
        dim, diag, mask = _GLOBAL_TRAINSET[j]
        ssum += QAOAposs_fast(dim, t, theta, diag, mask)
    return ssum / max(len(idxs), 1)

def eval_one_case(case, para):
    # Wraps the QAOA evaluation for a single test case.
    dim, diag, mask = case
    t = len(para) // 2
    return QAOAposs_fast(dim, t, para, diag, mask)

_G_TESTSET = None
_G_GUESS = None

def _init_worker(testset, guess):
    # Initializes global test data and parameters for worker processes.
    global _G_TESTSET, _G_GUESS
    _G_TESTSET = testset
    _G_GUESS = guess

def _sum_chunk(chunk):
    # Evaluates a chunk of test cases and accumulates the results.
    s = 0.0
    c = 0
    for item in chunk:
        case = _G_TESTSET[item] if isinstance(item, (int, np.integer)) else item
        val = eval_one_case(case, _G_GUESS)
        s += float(val)
        c += 1
    return s, c

def eval_full_test_parallel(testset, guess, ctx, nproc=8, chunk_size=64):
    # Evaluates the full test set in parallel chunks.
    chunks = make_chunks(testset, chunk_size)
    with ctx.Pool(processes=nproc, initializer=_init_worker, initargs=(testset, guess)) as pool:
        total_s = 0.0
        total_c = 0
        for s, c in pool.imap_unordered(_sum_chunk, chunks, chunksize=1):
            total_s += s
            total_c += c
    return total_s / total_c

# ============================================================
# Main Training Loop
# ============================================================
def data(layer, k, tryn,
         epoch=40, train=2000, test=5000,
         batch_size=500,
         eps_fd=1e-3,
         lr1=0.09, b1=0.86, b2=0.9997, epsilon=1e-8,
         seed=0):
    # Orchestrates the training process using parallel gradient estimation and Adam optimization.
    
    
    rng = np.random.default_rng(seed + 10007 * tryn)
    P = 2 * layer
    success = np.zeros((40,), dtype=float)

    # Use fork if available (Linux), otherwise spawn
    ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")

    for n in range(5, 45):
        print("\n==============================")
        print(f"n = {n}, k = {k}, try = {tryn}", flush=True)

        # ---------- Generate Train/Test Data ----------
        trainset = []
        for i in range(train):
            dim, diag, mask = generatecase_with_diags(n, k, seed=10_000_000 + 131*n + 1009*tryn + i)
            trainset.append((dim, diag, mask))

        testset = []
        for i in range(test):
            dim, diag, mask = generatecase_with_diags(n, k, seed=20_000_000 + 131*n + 1009*tryn + i)
            testset.append((dim, diag, mask))

        # ---------- Initialize Parameters and Optimizer ----------
        guess = (initial(layer) * 0.1).astype(np.float32)
        garafor = np.zeros((P,), dtype=np.float32)  # First moment
        hfor = np.float32(0.0)                      # Second moment

        # ---------- Training Loop ----------
        B = min(batch_size, train)
        with ctx.Pool(processes=NPROC, initializer=_worker_init, initargs=(trainset,)) as pool:
            for it in range(epoch):
                # Minibatch indices
                idxs = rng.integers(0, train, size=B, dtype=np.int32).tolist()

                # Construct theta list: base + epsilon perturbations
                thetas = [guess]
                for j in range(P):
                    th = guess.copy()
                    th[j] += np.float32(eps_fd)
                    thetas.append(th)

                # Parallel evaluation of all thetas
                vals = list(pool.map(_eval_one_theta, [(th, idxs) for th in thetas], chunksize=1))
                E0 = float(vals[0])

                # Forward difference gradient estimation
                Eplus = np.asarray(vals[1:], dtype=np.float32)
                df = (Eplus - np.float32(E0)) / np.float32(eps_fd)
                gradient = -df

                # Adam-like optimization update
                garafor = garafor * np.float32(b1) + gradient * np.float32(1 - b1)
                gnorm2 = np.float32(np.dot(gradient, gradient))
                hfor = np.float32(b2) * hfor + np.float32(1 - b2) * gnorm2

                vhat = hfor / np.float32(1 - (b2 ** (it + 1)))
                mhat = garafor / np.float32(1 - (b1 ** (it + 1)))
                mov = -lr1 * mhat / np.sqrt(epsilon + vhat)

                guess = guess + mov

                print(f"  epoch {it:02d}: train(mb)={E0:.6f}  ||g||={float(np.sqrt(gnorm2)):.3g}", flush=True)

        # ---------- Testing ----------
        test_success = eval_full_test_parallel(testset, guess, ctx, nproc=NPROC, chunk_size=64)
        success[n - 5] = -np.log(max(test_success, 1e-12))

        print(f"  test_success = {test_success:.8f}")
        print(f"  -log(test_success) = {success[n-5]:.8f}", flush=True)

    out_name = f"QAOAs{k}try{tryn}.npz"
    np.savez(out_name, success=success)
    print(f"\n[saved] {out_name}", flush=True)


if __name__ == "__main__":
    epoch = 40
    train = 2000
    test = 5000
    k = 0.5
    layer = 150
    
    data(layer, k, tryn=2,
         epoch=epoch, train=train, test=test,
         batch_size=500,
         eps_fd=1e-3,
         lr1=0.09, b1=0.86, b2=0.9997, epsilon=1e-8,
         seed=0)