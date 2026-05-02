#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QAA parameter-optimization (40 layers) — fast CPU version.
"""

import os
import math
import time
import random
import numpy as np
import multiprocessing as mp

import generators  # your module

# ============================================================
# Configuration to prevent thread contention between Numpy/MKL and Multiprocessing.
# ============================================================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ============================================================
# Worker Count Helper
# ============================================================
def get_workers(default=24):
    # Determines the appropriate number of worker processes based on SLURM environment variables.
    x = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_NTASKS")
    try:
        w = int(x) if x else default
    except Exception:
        w = default
    return max(1, w)


# ============================================================
# Global Caches
# ============================================================
_S_CACHE = {}          # dim -> uint8[dim, 2^dim]  (MSB->LSB)
_STATE0_CACHE = {}     # dim -> complex64[2^dim]
_MIX_STRIDES = {}      # dim -> list[(half, block)] for q=0..dim-1 (LSB->MSB)

def get_S(dim: int) -> np.ndarray:
    # Generates or retrieves the cached bitstring matrix for the given dimension.
    S = _S_CACHE.get(dim, None)
    if S is not None:
        return S
    K = 1 << dim
    idx = np.arange(K, dtype=np.uint32)
    shifts = np.arange(dim - 1, -1, -1, dtype=np.uint32)  # MSB -> LSB
    S = ((idx[None, :] >> shifts[:, None]) & 1).astype(np.uint8, copy=False)
    _S_CACHE[dim] = S
    return S

def get_state0(dim: int) -> np.ndarray:
    # Generates or retrieves the uniform superposition initial state.
    st = _STATE0_CACHE.get(dim, None)
    if st is not None:
        return st
    K = 1 << dim
    st = (np.ones(K, dtype=np.complex64) / np.sqrt(np.float32(K))).astype(np.complex64, copy=False)
    _STATE0_CACHE[dim] = st
    return st

def get_mixer_strides(dim: int):
    # Pre-computes stride patterns for applying mixer gates efficiently.
    s = _MIX_STRIDES.get(dim, None)
    if s is not None:
        return s
    strides = []
    for q in range(dim):
        half = 1 << q
        block = half << 1
        strides.append((half, block))
    _MIX_STRIDES[dim] = strides
    return strides


# ============================================================
# Initial Parameter Schedule
# ============================================================
def initial(t: int) -> np.ndarray:
    # Generates the initial annealing schedule parameters using a sigmoid-like curve.
    
    i = np.arange(1, t + 1, dtype=np.float64)
    s = i / (t + 1.0)
    w = np.exp(-5.0 * s * (1.0 - s))
    cum = np.cumsum(w)
    vec = np.zeros((2 * t,), dtype=np.float32)
    vec[t:2*t] = cum.astype(np.float32, copy=False)
    vec /= np.float32(cum[-1])
    vec[:t] = 1.0 - vec[t:2*t]
    return vec


# ============================================================
# Fast Diagonal and Mask Construction
# ============================================================
def build_diag_mask_fast(p2_np: np.ndarray, clauses_np: np.ndarray, T0: np.ndarray):
    # Constructs the diagonal energy vector and validity mask for the problem Hamiltonian.
    
    lits = np.asarray(clauses_np, dtype=np.int32)
    p2 = np.asarray(p2_np, dtype=np.uint8)
    n, dim = p2.shape

    S = get_S(dim)  # (dim, K)
    K = S.shape[1]

    lit1 = lits[:, 0]
    lit2 = lits[:, 1]
    v1 = np.abs(lit1) - 1
    v2 = np.abs(lit2) - 1

    vars0 = np.unique(np.concatenate([v1, v2], axis=0))
    p2_sub = p2[vars0, :]  # (u, dim)

    T = np.asarray(T0, dtype=np.uint8).reshape(-1)
    const = T[vars0].astype(np.uint8, copy=False)[:, None]  # (u,1)

    state_sub = (p2_sub @ S + const) & 1  # (u, K) uint8

    map_idx = -np.ones(n, dtype=np.int32)
    map_idx[vars0] = np.arange(vars0.size, dtype=np.int32)
    i1 = map_idx[v1]
    i2 = map_idx[v2]

    s1 = (lit1 < 0).astype(np.uint8)[:, None]  # (m,1)
    s2 = (lit2 < 0).astype(np.uint8)[:, None]

    val1 = state_sub[i1, :] ^ s1  # (m,K)
    val2 = state_sub[i2, :] ^ s2

    diag_int = (val1 & val2).sum(axis=0).astype(np.int16, copy=False)  # (K,)
    mask = (diag_int == 0).astype(np.float32, copy=False)
    diag = diag_int.astype(np.float32, copy=False)
    return dim, diag, mask


# ============================================================
# Mixer Operator Application
# ============================================================
def apply_mixer_all_qubits_inplace(state: np.ndarray, b: np.float32, dim: int):
    # Applies the mixer Hamiltonian evolution operator to the quantum state using strided access.
    
    c = np.complex64(np.cos(float(b)))
    s = np.complex64(-1j * np.sin(float(b)))
    for half, block in get_mixer_strides(dim):
        v = state.reshape(-1, block)  # view
        a = v[:, :half]
        d = v[:, half:]
        tmp = a.copy()
        a[:] = c * tmp + s * d
        d[:] = c * d   + s * tmp


# ============================================================
# QAA Probability Calculation
# ============================================================
def QAA_prob_fast(dim: int, t: int, theta: np.ndarray, diag: np.ndarray, mask: np.ndarray) -> float:
    # Simulates the QAA circuit for a given set of parameters and computes success probability.
    K = diag.size
    state = get_state0(dim).copy()

    for i in range(t):
        g = np.float32(theta[t + i])
        b = np.float32(theta[i])

        # phase separator: state *= exp(i*g*diag)
        phase = np.exp(1j * (g * diag)).astype(np.complex64, copy=False)
        state *= phase

        # mixer
        apply_mixer_all_qubits_inplace(state, b, dim)

    prob = (state.real * state.real + state.imag * state.imag).astype(np.float32, copy=False)
    return float(np.sum(prob * mask, dtype=np.float64))


# ============================================================
# Case Generation
# ============================================================
def generatecase_with_diags(n: int, k: float, seed: int):
    # Generates a random solvable problem instance and its Hamiltonian representation.
    rng = np.random.default_rng(seed)
    random.seed(seed)
    while True:
        ret= generators.generaterandom(n, k)
        T, L, clauses = ret[0], ret[1], ret[2]
        dim, diag, mask = build_diag_mask_fast(L, clauses, T0=T)
        if mask.max() > 0.5:
            return dim, diag, mask


# ============================================================
# Parallel Evaluation Utilities
# ============================================================
_GLOBAL_CASESET = None

def _init_caseset(caseset):
    # Initializes the global training set for worker processes.
    global _GLOBAL_CASESET
    _GLOBAL_CASESET = caseset

def _eval_one_theta(args):
    # Computes the mean success probability for a parameter set over a batch of training cases.
    theta, idxs = args
    t = len(theta) // 2
    ssum = 0.0
    for j in idxs:
        dim, diag, mask = _GLOBAL_CASESET[j]
        ssum += QAA_prob_fast(dim, t, theta, diag, mask)
    return ssum / max(len(idxs), 1)

_G_TESTSET = None
_G_THETA = None

def _init_test(testset, theta):
    # Initializes the global test set and parameters for worker processes.
    global _G_TESTSET, _G_THETA
    _G_TESTSET = testset
    _G_THETA = theta

def _prob_range(arg):
    # Evaluates the success probability for a range of test indices.
    start, stop = arg
    theta = _G_THETA
    t = len(theta) // 2
    out = np.empty((stop - start,), dtype=np.float32)
    for i, idx in enumerate(range(start, stop)):
        dim, diag, mask = _G_TESTSET[idx]
        out[i] = np.float32(QAA_prob_fast(dim, t, theta, diag, mask))
    return start, out


# ============================================================
# Main Optimization Routine
# ============================================================
def data_qaa_opt(
    k: float,
    tryn: int,
    layer: int = 40,
    start_n: int = 5,
    final_n: int = 35,          
    epoch: int = 40,
    train: int = 2000,
    test: int = 5000,
    batch_size: int = 500,
    eps_fd: float = 1e-3,
    lr1: float = 0.09,
    b1: float = 0.86,
    b2: float = 0.9997,
    epsilon: float = 1e-8,
    seed: int = 0,
    nproc: int | None = None,
    chunk_size_test: int = 64,
    save_prefix: str = "QAAopt",
    save_single2: bool = True,
):
    # Orchestrates the parameter optimization process using parallel gradient estimation and Adam optimization.
    
    if nproc is None:
        nproc = get_workers(default=24)
    nproc = int(max(1, nproc))

    ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")

    n_list = np.arange(start_n, final_n, dtype=int)
    nn = len(n_list)

    success = np.zeros((nn,), dtype=np.float64)
    theta_all = np.zeros((nn, 2 * layer), dtype=np.float32)
    single2_all = np.empty((nn, test), dtype=np.float32) if save_single2 else None

    rng = np.random.default_rng(seed + 10007 * tryn)
    P = 2 * layer

    print(f"Config: k={k}, try={tryn}, layers={layer}, nproc={nproc}")
    print(f"Range: n={start_n}..{final_n-1}, epoch={epoch}, train={train}, test={test}")

    for ni, n in enumerate(n_list):
        print("\n==============================")
        print(f"n={n} | k={k} | try={tryn}", flush=True)

        # ----- Build Train/Test Sets -----
        trainset = []
        for i in range(train):
            ss = 10_000_000 + 131 * int(n) + 1009 * int(tryn) + i
            trainset.append(generatecase_with_diags(int(n), float(k), seed=ss))

        testset = []
        for i in range(test):
            ss = 20_000_000 + 131 * int(n) + 1009 * int(tryn) + i
            testset.append(generatecase_with_diags(int(n), float(k), seed=ss))

        # ----- Init Theta and Optimizer -----
        theta = (initial(layer) * np.float32(0.1)).astype(np.float32, copy=False)
        m1 = np.zeros((P,), dtype=np.float32)     
        v2 = np.float32(0.0)                      
        B = int(min(batch_size, train))

        # ----- Training Loop -----
        with ctx.Pool(processes=nproc, initializer=_init_caseset, initargs=(trainset,)) as pool:
            for it in range(epoch):
                idxs = rng.integers(0, train, size=B, dtype=np.int32).tolist()

                thetas = [theta]
                for j in range(P):
                    th = theta.copy()
                    th[j] += np.float32(eps_fd)
                    thetas.append(th)

                vals = pool.map(_eval_one_theta, [(th, idxs) for th in thetas], chunksize=1)
                E0 = float(vals[0])
                Eplus = np.asarray(vals[1:], dtype=np.float32)
                df = (Eplus - np.float32(E0)) / np.float32(eps_fd)

                # Maximize success => Gradient = -df
                grad = -df

                m1 = m1 * np.float32(b1) + grad * np.float32(1.0 - b1)
                gnorm2 = np.float32(np.dot(grad, grad))
                v2 = np.float32(b2) * v2 + np.float32(1.0 - b2) * gnorm2

                mhat = m1 / np.float32(1.0 - (b1 ** (it + 1)))
                vhat = v2 / np.float32(1.0 - (b2 ** (it + 1)))

                step = -np.float32(lr1) * mhat / np.sqrt(np.float32(epsilon) + vhat)
                theta = theta + step

                print(f"  Epoch {it:02d}: Train={E0:.6f} | |Grad|={float(np.sqrt(gnorm2)):.3g}", flush=True)

        # ----- Testing -----
        ranges = []
        for s in range(0, test, chunk_size_test):
            ranges.append((s, min(test, s + chunk_size_test)))

        probs = np.empty((test,), dtype=np.float32)
        with ctx.Pool(processes=nproc, initializer=_init_test, initargs=(testset, theta)) as pool:
            for start, out in pool.imap_unordered(_prob_range, ranges, chunksize=1):
                probs[start:start + out.size] = out

        mean_test = float(np.mean(probs, dtype=np.float64))
        mean_test = max(mean_test, 1e-12)

        success[ni] = -np.log(mean_test)
        theta_all[ni, :] = theta
        if save_single2:
            single2_all[ni, :] = probs

        print(f"  Test Success: {mean_test:.8f}")
        print(f"  NegLogProb: {success[ni]:.8f}", flush=True)

    out_name = f"{save_prefix}_k{k}_try{tryn}.npz"
    save_dict = dict(
        k=float(k),
        tryn=int(tryn),
        layer=int(layer),
        epoch=int(epoch),
        train=int(train),
        test=int(test),
        batch_size=int(batch_size),
        eps_fd=float(eps_fd),
        lr1=float(lr1), b1=float(b1), b2=float(b2), epsilon=float(epsilon),
        seed=int(seed),
        n_list=n_list,
        success=success,
        theta_all=theta_all,
    )
    if save_single2:
        save_dict["single2_all"] = single2_all

    np.savez_compressed(out_name, **save_dict)
    print(f"\n[Saved] {out_name}", flush=True)


# ============================================================
# Example Entry Point
# ============================================================
if __name__ == "__main__":
    data_qaa_opt(
        k=0.5,
        tryn=0,
        layer=40,
        start_n=5,
        final_n=30,
        epoch=40,
        train=500,
        test=2000,
        batch_size=500,
        eps_fd=1e-3,
        lr1=0.09, b1=0.86, b2=0.9997, epsilon=1e-8,
        seed=0,
        nproc=None,
        chunk_size_test=64,
        save_prefix="QAOAmix",
        save_single2=True,
    )