import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import numpy as np
from scipy.optimize import minimize
from functools import partial

from jax import config
config.update("jax_enable_x64", True)

def initial(t: int) -> np.ndarray:
    """Return the schedule-based 2t-parameter initialization vector (beta then gamma)."""
    i = np.arange(1, t + 1, dtype=np.float64)
    s = i / (t + 1.0)
    w = np.exp(-5.0 * s * (1.0 - s))
    cum = np.cumsum(w)
    cum /= cum[-1]
    vec = np.zeros((2 * t,), dtype=np.float64)
    vec[t:2*t] = cum
    vec[:t] = 1.0 - vec[t:2*t]
    return vec

def build_diag_jax(clauses, p2):
    """Precompute the diagonal cost diag and target mask diag2 for the RSRA subspace."""
    n, dim = p2.shape
    K = 1 << dim
    idx = np.arange(K, dtype=np.uint32)
    shifts = np.arange(dim - 1, -1, -1, dtype=np.uint32)
    S = ((idx[None, :] >> shifts[:, None]) & 1).astype(np.int8)
    state_bits = (p2 @ S + 1) & 1
    l1 = clauses[:, 0] - 1
    l2 = clauses[:, 1] - 1
    diag = np.zeros((K,), dtype=np.float64)
    for a, b in zip(l1, l2):
        diag += (state_bits[a] * state_bits[b]).astype(np.float64)
    diag2 = (diag == 0).astype(np.float64)
    return jnp.array(diag), jnp.array(diag2)

@partial(jit, static_argnums=(1,))
def get_expected_energy(para, dim, diag):
    """Return the expected energy under the QAOA state defined by parameters para."""
    t = para.size // 2
    K = 1 << dim
    state = jnp.ones(K, dtype=jnp.complex128) / jnp.sqrt(K)
    beta = para[:t]
    gamma = para[t:]
    for i in range(t):
        state = state * jnp.exp(1j * gamma[i] * diag)
        for q in range(dim):
            state = state.reshape(1 << q, 2, 1 << (dim - 1 - q))
            c = jnp.cos(beta[i])
            s = -1j * jnp.sin(beta[i])
            gate = jnp.array([[c, s], [s, c]], dtype=jnp.complex128)
            state = jnp.einsum('ab,cbd->cad', gate, state)
        state = state.ravel()
    prob = jnp.abs(state) ** 2
    return jnp.real(jnp.dot(prob, diag))

@partial(jit, static_argnums=(1,))
def get_success_prob(para, dim, diag, diag2):
    """Return (success probability, expected energy) for the QAOA state defined by para."""
    t = para.size // 2
    K = 1 << dim
    state = jnp.ones(K, dtype=jnp.complex128) / jnp.sqrt(K)
    beta = para[:t]
    gamma = para[t:]
    for i in range(t):
        state = state * jnp.exp(1j * gamma[i] * diag)
        for q in range(dim):
            state = state.reshape(1 << q, 2, 1 << (dim - 1 - q))
            c = jnp.cos(beta[i])
            s = -1j * jnp.sin(beta[i])
            gate = jnp.array([[c, s], [s, c]], dtype=jnp.complex128)
            state = jnp.einsum('ab,cbd->cad', gate, state)
        state = state.ravel()
    prob = jnp.abs(state) ** 2
    success = jnp.real(jnp.dot(prob, diag2))
    energy = jnp.real(jnp.dot(prob, diag))
    return success, energy

def optimize_qaoa(dim, diag, x0, maxiter=500, verbose=False, print_every=25):
    """Run BFGS to minimize expected energy using JAX value-and-grad."""
    def loss_fn(para):
        return get_expected_energy(para, dim, diag)

    vg_func = jit(value_and_grad(loss_fn))

    history = {"step": 0, "last_val": 0.0}

    def f_scipy(x):
        val, g = vg_func(x)
        history["last_val"] = float(val)
        return np.array(val, dtype=np.float64), np.array(g, dtype=np.float64)

    def callback_fn(_xk):
        history["step"] += 1
        if verbose and (history["step"] % print_every == 0):
            print(f"Iter {history['step']:04d} | E = {history['last_val']:.10f}")

    if verbose:
        print(f"Starting BFGS | dim={dim}, n_params={len(x0)}")

    res = minimize(
        fun=f_scipy,
        x0=x0,
        method="BFGS",
        jac=True,
        callback=callback_fn,
        options={"disp": False, "maxiter": int(maxiter)},
    )
    return res

if __name__ == "__main__":
    data = np.load("PQC100,63.npz", allow_pickle=True)
    clauses = data["clauses"]
    p2 = data["p2"]
    dim = p2.shape[1]

    diag, diag2 = build_diag_jax(clauses, p2)

    t = 1
    x0 = initial(t) * 0.5

    res = optimize_qaoa(dim, diag, x0, maxiter=500, verbose=False)

    poss, E = get_success_prob(res.x, dim, diag, diag2)
    print(f"Done | t={t} | E={float(E):.6f} | success={float(poss):.6f}")

    xs = np.array([t], dtype=float)
    energylist = np.array([float(E)], dtype=float)
    possibility = np.array([float(poss)], dtype=float)

    save_filename = f"QAOAdraw{t}.npz"
    np.savez(save_filename, x=xs, energylist=energylist, possibility=possibility)
    print(f"Saved: {save_filename}")
