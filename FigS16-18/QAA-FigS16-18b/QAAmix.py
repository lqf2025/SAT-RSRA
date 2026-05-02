import os
# Reduce BLAS oversubscription (important on Slurm + multiprocessing)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import math
import time
import random
import numpy as np
import multiprocessing as mp

import generators  # your module


# -------------------------
# Utilities: CPU workers
# -------------------------
def get_workers(default=8):
    # Determines the appropriate number of worker processes based on environment variables.
    x = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_NTASKS")
    try:
        w = int(x) if x else default
    except Exception:
        w = default
    return max(1, w)

def _worker_init(base_seed: int):
    # Initializes the random seed for each worker process to ensure distinct randomness.
    pid = os.getpid()
    s = (base_seed + 1000003 * pid + int(time.time())) & 0xFFFFFFFF
    random.seed(s)
    np.random.seed(s)


# -------------------------
# Cache: S matrices per dim
# -------------------------
_S_CACHE = {}  # dim -> uint8 array (dim, K)

def get_S(dim: int) -> np.ndarray:
    # Generates or retrieves a cached matrix of all possible binary strings for a given dimension.
    S = _S_CACHE.get(dim, None)
    if S is not None:
        return S
    K = 1 << dim
    idx = np.arange(K, dtype=np.uint32)
    shifts = np.arange(dim - 1, -1, -1, dtype=np.uint32)
    S = ((idx[None, :] >> shifts[:, None]) & 1).astype(np.uint8)  # (dim, K)
    _S_CACHE[dim] = S
    return S


# -------------------------
# Fast build of diag/mask (vectorized)
# energy = sum_j state[l1-1] * state[l2-1]
# state = (p2 @ s + 1) mod 2
# -------------------------
def build_diag_mask_fast(p2_np: np.ndarray, clauses_np: np.ndarray, T0):
    # Constructs the diagonal energy vector and validity mask for the problem Hamiltonian.
    
    lits = np.asarray(clauses_np, dtype=np.int32)

    p2 = np.asarray(p2_np, dtype=np.uint8)
    n, dim = p2.shape
    S = get_S(dim)  # (dim, K)
    K = S.shape[1]

    lit1 = lits[:, 0]
    lit2 = lits[:, 1]

    # abs(lit) -> var index (0-based)
    v1 = np.abs(lit1) - 1
    v2 = np.abs(lit2) - 1

    # Only use variables that appear in clauses
    vars0 = np.unique(np.concatenate([v1, v2], axis=0))
    p2_sub = p2[vars0, :]  # (u, dim)

    # Affine offset: default to all 1s if T0 is None (equivalent to "+1" in old code)
    T = np.asarray(T0, dtype=np.uint8).reshape(-1)
    const = T[vars0].astype(np.uint8, copy=False)[:, None]  # (u,1)

    # state_sub: (u, K)
    state_sub = (p2_sub @ S + const) & 1  # uint8

    # Map var -> row in state_sub
    map_idx = -np.ones(n, dtype=np.int32)
    map_idx[vars0] = np.arange(vars0.size, dtype=np.int32)

    i1 = map_idx[v1]  # (m,)
    i2 = map_idx[v2]  # (m,)

    # Sign bits: lit<0 means negate -> xor 1
    s1 = (lit1 < 0).astype(np.uint8)[:, None]  # (m,1)
    s2 = (lit2 < 0).astype(np.uint8)[:, None]  # (m,1)

    # Literal truth values (m,K)
    val1 = state_sub[i1, :] ^ s1
    val2 = state_sub[i2, :] ^ s2

    # Violated iff val1==1 and val2==1 => val1 & val2
    diag_int = (val1 & val2).sum(axis=0).astype(np.int16)  # (K,)

    mask = (diag_int == 0).astype(np.float32)
    diag = diag_int.astype(np.float32)
    return diag, mask


# -------------------------
# Fast initial (vectorized numpy)
# -------------------------
def initial_fast(t: int) -> np.ndarray:
    # Generates the annealing schedule parameters using a sigmoid-like curve.
    
    i = np.arange(1, t + 1, dtype=np.float64)
    s = i / (t + 1.0)
    w = np.exp(-5.0 * s * (1.0 - s))          
    cum = np.cumsum(w)                        

    vec = np.zeros((2 * t,), dtype=np.float64)
    vec[t:2*t] = cum                           
    vec /= cum[-1]
    vec[:t] = 1.0 - vec[t:2*t]                 
    return vec


# -------------------------
# Mixer apply without moveaxis (stride/block)
# U = cos(b) I - i sin(b) X
# -------------------------
def apply_mixer_all_qubits_inplace(state: np.ndarray, b: float, dim: int):
    # Applies the mixer Hamiltonian evolution operator to the quantum state using strided access.
    
    c = np.float32(np.cos(b))
    s = np.complex64(-1j * np.sin(b))
    for q in range(dim):
        half = 1 << q
        block = half << 1
        x = state.reshape(-1, block)  # view
        a = x[:, :half]
        d = x[:, half:]
        tmp = a.copy()
        a[:] = c * tmp + s * d
        d[:] = c * d   + s * tmp


# -------------------------
# QAA prob from diag/mask (no recompute of diag/mask)
# -------------------------
def QAAposs_from_diag(diag: np.ndarray, mask: np.ndarray, t: int) -> float:
    # Simulates the Quantum Adiabatic Algorithm and calculates the final success probability.
    
    K = diag.size
    dim = int(round(math.log2(K)))
    # Parameters
    para = initial_fast(t) * np.float32(0.3)

    state = (np.ones(K, dtype=np.complex64) / np.sqrt(np.float32(K)))

    for i in range(t):
        g = np.float32(para[t + i])
        b = np.float32(para[i])

        # exp(i*g*diag) * state
        phase = np.exp((1j * g) * diag).astype(np.complex64)
        state *= phase

        # Mixer
        apply_mixer_all_qubits_inplace(state, float(b), dim)

    # Prob = |state|^2
    prob = (state.real * state.real + state.imag * state.imag).astype(np.float32)
    return float((prob * mask).sum(dtype=np.float64))


# -------------------------
# Generate a satisfiable case (CPU)
# - uses mask.max() > 0 to ensure at least one feasible bitstring
# -------------------------
def generatecase_fast(n: int, k: float):
    # Generates a random solvable problem instance and its Hamiltonian representation.
    
    while True:
        ret = generators.generaterandom(n, k)
        T, L, clauses = ret[0], ret[1], ret[2]
        diag, mask = build_diag_mask_fast(L, clauses, T0=T)
        return clauses, L, diag, mask, T


# -------------------------
# Batched worker task
# Each task handles B trials to reduce multiprocessing overhead
# -------------------------
def _run_batch(args):
    # Executes a batch of QAA simulations for a given problem configuration.
    n, k, layerlist, B = args
    L = len(layerlist)
    out = np.empty((B, L), dtype=np.float32)
    for i in range(B):
        _, _, diag, mask, _T = generatecase_fast(n, k)
        for j, t in enumerate(layerlist):
            out[i, j] = QAAposs_from_diag(diag, mask, int(t))
    return out


# -------------------------
# Main data routine (fast)
# -------------------------
def data_fast(
    layerlist, k,
    trials=10000, final=65,
    save_prefix="QAAmix",
    save_single2=True,
    n_workers=None,
    batch_size=8,
    base_seed=12345,
    checkpoint_every_n=False
):
    # Orchestrates the parallel data collection process across different problem sizes.
    
    layerlist = [int(x) for x in layerlist]
    L = len(layerlist)
    if L == 0:
        raise ValueError("layerlist is empty")

    n_list = np.arange(5, final, dtype=int)
    nn = len(n_list)

    success = np.zeros((nn, L), dtype=np.float64)
    single2_all = np.empty((nn, trials, L), dtype=np.float32) if save_single2 else None

    if n_workers is None:
        n_workers = get_workers(default=8)
    n_workers = max(1, int(n_workers))

    # Number of batches
    B = int(batch_size)
    n_batches = (trials + B - 1) // B

    print(f"Config: W={n_workers}, B={B}, T={trials}, Batches={n_batches}", flush=True)

    ctx = mp.get_context("fork")  # Linux cluster typically supports fork
    with ctx.Pool(processes=n_workers, initializer=_worker_init, initargs=(base_seed,)) as pool:
        for ni, n in enumerate(n_list):
            print(f"\nProcessing n={n}...", flush=True)

            # Prepare batch args; last batch may be shorter
            batch_args = []
            for bi in range(n_batches):
                this_B = B if (bi < n_batches - 1) else (trials - (n_batches - 1) * B)
                batch_args.append((int(n), float(k), layerlist, int(this_B)))

            # Choose a chunksize to reduce scheduler overhead
            chunksize = max(1, len(batch_args) // (4 * n_workers))

            # Run
            offset = 0
            # imap_unordered gives better load-balance
            for out_block in pool.imap_unordered(_run_batch, batch_args, chunksize=chunksize):
                bsz = out_block.shape[0]
                if save_single2:
                    single2_all[ni, offset:offset + bsz, :] = out_block
                offset += bsz

            assert offset == trials, (offset, trials)

            # Compute success for each layer
            s2 = single2_all[ni] if save_single2 else None
            if s2 is None:
                raise RuntimeError("save_single2 must be True in this implementation to compute success.")

            for lj in range(L):
                p_hat = float(np.mean(s2[:, lj], dtype=np.float64))
                p_hat = max(p_hat, 1e-12)
                success[ni, lj] = -np.log(p_hat)
                print(f"n={n} | t={layerlist[lj]} | -log(p)={success[ni, lj]:.4f} | p={p_hat:.4e}", flush=True)

            # Optional checkpoint each n (recommended on clusters)
            if checkpoint_every_n:
                out_name = f"{save_prefix}{k}_partial.npz"
                np.savez_compressed(
                    out_name,
                    k=float(k),
                    trials=int(trials),
                    n_list=n_list,
                    layerlist=np.asarray(layerlist, dtype=int),
                    success=success,
                    single2_all=single2_all if save_single2 else None
                )
                print(f"[Checkpoint] {out_name}", flush=True)

    # Final save
    out_name = f"{save_prefix}{k}.npz"
    save_dict = dict(
        k=float(k),
        trials=int(trials),
        n_list=n_list,
        layerlist=np.asarray(layerlist, dtype=int),
        success=success,
    )
    if save_single2:
        save_dict["single2_all"] = single2_all
    np.savez_compressed(out_name, **save_dict)
    print(f"\n[Saved] {out_name}", flush=True)


# -------------------------
# Example entry
# -------------------------
if __name__ == "__main__":
    # Start small first to validate correctness, then scale up
    data_fast([100], 0.07, trials=10000, final=40, n_workers=None, batch_size=8)