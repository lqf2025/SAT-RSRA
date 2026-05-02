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

import generator  # your module


# -------------------------
# Utilities: CPU workers
# -------------------------
def get_workers(default=8):
    x = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_NTASKS")
    try:
        w = int(x) if x else default
    except Exception:
        w = default
    return max(1, w)

def _worker_init(base_seed: int):
    # Different seed per process
    pid = os.getpid()
    s = (base_seed + 1000003 * pid + int(time.time())) & 0xFFFFFFFF
    random.seed(s)
    np.random.seed(s)


# -------------------------
# Cache: S matrices per dim
# -------------------------
_S_CACHE = {}  # dim -> uint8 array (dim, K)

def get_S(dim: int) -> np.ndarray:
    """Return S: (dim, K) each column is a dim-bitstring (MSB->LSB)."""
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
def build_diag_mask_fast(p2_np: np.ndarray, clauses_np: np.ndarray):
    """
    p2_np: (n, dim) 0/1
    clauses_np: (m,2) 1-based indices in [1..n]
    returns:
      diag: (K,) float32
      mask: (K,) float32, mask=1 <=> diag==0
    """
    clauses = np.asarray(clauses_np, dtype=np.int32)
    p2 = np.asarray(p2_np, dtype=np.uint8)
    n, dim = p2.shape
    S = get_S(dim)  # (dim, K)

    # Only variables appearing in clauses matter
    vars0 = np.unique(clauses) - 1  # 0-based
    p2_sub = p2[vars0, :]          # (u, dim)

    # state_sub = (p2_sub @ S + 1) mod 2   -> (u, K)
    # matmul result is integer; keep modulo 2 by &1
    state_sub = (p2_sub @ S + 1) & 1  # uint8

    # Map original variable index -> row in state_sub
    map_idx = -np.ones(n, dtype=np.int32)
    map_idx[vars0] = np.arange(vars0.size, dtype=np.int32)

    i1 = map_idx[clauses[:, 0] - 1]
    i2 = map_idx[clauses[:, 1] - 1]

    # diag[k] = sum_j state_sub[i1_j,k]*state_sub[i2_j,k]
    diag_int = (state_sub[i1] * state_sub[i2]).sum(axis=0).astype(np.int16)
    mask = (diag_int == 0).astype(np.float32)
    diag = diag_int.astype(np.float32)
    return diag, mask


# -------------------------
# Fast initial (vectorized numpy)
# -------------------------
def initial_fast(t: int) -> np.ndarray:
    i = np.arange(1, t + 1, dtype=np.float64)
    s = i / (t + 1.0)
    w = np.exp(-5.0 * s * (1.0 - s))          # float64，和你原版更一致
    cum = np.cumsum(w)                         # 长度 t

    vec = np.zeros((2 * t,), dtype=np.float64)
    vec[t:2*t] = cum                           # <-- 关键：从 t 开始写到 2t-1
    vec /= cum[-1]
    vec[:t] = 1.0 - vec[t:2*t]                 # 对应你原版 vec[i]=1-vec[i+t]
    return vec


# -------------------------
# Mixer apply without moveaxis (stride/block)
# U = cos(b) I - i sin(b) X
# -------------------------
def apply_mixer_all_qubits_inplace(state: np.ndarray, b: float, dim: int):
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
    """
    diag: (K,) float32
    mask: (K,) float32
    returns probability in [0,1]
    """
    K = diag.size
    dim = int(round(math.log2(K)))
    # parameters
    para = initial_fast(t) * np.float32(0.3)

    state = (np.ones(K, dtype=np.complex64) / np.sqrt(np.float32(K)))

    for i in range(t):
        g = np.float32(para[t + i])
        b = np.float32(para[i])

        # exp(i*g*diag) * state
        phase = np.exp((1j * g) * diag).astype(np.complex64)
        state *= phase

        # mixer
        apply_mixer_all_qubits_inplace(state, float(b), dim)

    # prob = |state|^2
    prob = (state.real * state.real + state.imag * state.imag).astype(np.float32)
    return float((prob * mask).sum(dtype=np.float64))


# -------------------------
# Generate a satisfiable case (CPU)
# - uses mask.max() > 0 to ensure at least one feasible bitstring
# -------------------------
def generatecase_fast(n: int, k: float):
    m = int(np.floor(n * k))
    if random.random() < n * k - m:
        m += 1
    while True:
        clauses, p2 = generator.generaterandom(n, m)
        dim = p2.shape[1]
        diag, mask = build_diag_mask_fast(p2, clauses)
        if mask.max() > 0.5:
            return clauses, p2, diag, mask


# -------------------------
# Batched worker task
# Each task handles B trials to reduce multiprocessing overhead
# -------------------------
def _run_batch(args):
    n, k, layerlist, B = args
    L = len(layerlist)
    out = np.empty((B, L), dtype=np.float32)
    for i in range(B):
        _, _, diag, mask = generatecase_fast(n, k)
        for j, t in enumerate(layerlist):
            out[i, j] = QAAposs_from_diag(diag, mask, int(t))
    return out


# -------------------------
# Main data routine (fast)
# -------------------------
def data_fast(
    layerlist, k,
    trials=10000, final=65,
    save_prefix="QAAdata/QAAsingles",
    save_single2=True,
    n_workers=None,
    batch_size=8,
    base_seed=12345,
    checkpoint_every_n=False
):
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

    # number of batches
    B = int(batch_size)
    n_batches = (trials + B - 1) // B

    print(f"[config] workers={n_workers}, batch_size={B}, trials={trials}, batches={n_batches}", flush=True)

    ctx = mp.get_context("fork")  # Linux cluster typically supports fork
    with ctx.Pool(processes=n_workers, initializer=_worker_init, initargs=(base_seed,)) as pool:
        for ni, n in enumerate(n_list):
            print(f"\n[n={n}] generating+evaluating...", flush=True)

            # prepare batch args; last batch may be shorter
            batch_args = []
            for bi in range(n_batches):
                this_B = B if (bi < n_batches - 1) else (trials - (n_batches - 1) * B)
                batch_args.append((int(n), float(k), layerlist, int(this_B)))

            # choose a chunksize to reduce scheduler overhead
            chunksize = max(1, len(batch_args) // (4 * n_workers))

            # run
            offset = 0
            # imap_unordered gives better load-balance
            for out_block in pool.imap_unordered(_run_batch, batch_args, chunksize=chunksize):
                bsz = out_block.shape[0]
                if save_single2:
                    single2_all[ni, offset:offset + bsz, :] = out_block
                offset += bsz

            assert offset == trials, (offset, trials)

            # compute success for each layer
            s2 = single2_all[ni] if save_single2 else None
            if s2 is None:
                raise RuntimeError("save_single2 must be True in this implementation to compute success.")

            for lj in range(L):
                p_hat = float(np.mean(s2[:, lj], dtype=np.float64))
                p_hat = max(p_hat, 1e-12)
                success[ni, lj] = -np.log(p_hat)
                print(f"n={n}  t={layerlist[lj]}  -log(p)={success[ni, lj]:.6g}  p={p_hat:.6g}", flush=True)

            # optional checkpoint each n (recommended on clusters)
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
                print(f"[checkpoint saved] {out_name}", flush=True)

    # final save
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
    print(f"\n[saved] {out_name}", flush=True)


# -------------------------
# Example entry
# -------------------------
if __name__ == "__main__":
    # Start small first to validate correctness, then scale up
    data_fast([150], 0.626, trials=10000, final=90, n_workers=None, batch_size=8)
