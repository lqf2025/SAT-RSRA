import numpy as np
from boot import mean_boot, fit_boot
import os


# ============================================================
# A) Load single2_all
# ============================================================
def load_qaaur_from_single2(npz_path, keep_trials=None):
    d = np.load(npz_path, allow_pickle=True)

    if "single2_all" not in d.files:
        raise ValueError(f"{npz_path}: missing key 'single2_all'")

    S = np.asarray(d["single2_all"], dtype=float)  # (N, T, P)
    N, T, P = S.shape

    n_list = np.asarray(d["n_list"], dtype=int) if "n_list" in d.files else np.arange(N, dtype=int)
    plist  = np.asarray(d["plist"])             if "plist"  in d.files else np.arange(P)
    k      = float(d["k"])                      if "k"      in d.files else np.nan

    if keep_trials is not None:
        keep_trials = int(min(keep_trials, T))
        S = S[:, :keep_trials, :]
        T = keep_trials

    return {"k": k, "n_list": n_list, "plist": plist, "single2_all": S, "trials_used": T}


# ============================================================
# B) log-fit + R^2 on point means:
#   fit log(y) = alpha + beta*n  on ns>=fit_n_min
#   return (a_hat, b_hat, r2) where y ≈ a*b^n
# ============================================================
def _fit_log_ab_r2(n_list, y_point, fit_n_min=None, eps=1e-12):
    n = np.asarray(n_list, float)
    y = np.asarray(y_point, float)

    if fit_n_min is None:
        mask = np.ones_like(n, dtype=bool)
    else:
        mask = n >= float(fit_n_min)

    x = n[mask]
    yy = np.clip(y[mask], eps, None)
    z = np.log(yy)

    slope, intercept = np.polyfit(x, z, 1)  # z = intercept + slope*x
    z_pred = intercept + slope * x

    sse = float(np.sum((z - z_pred) ** 2))
    sst = float(np.sum((z - float(np.mean(z))) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan

    a_hat = float(np.exp(intercept))
    b_hat = float(np.exp(slope))
    return a_hat, b_hat, float(r2)


# ============================================================
# C) pointwise mean(p) + CI(p) from trials, then convert to 1/p
# ============================================================
def pointwise_p_mean_ci(X_nt, ci=0.95, B=2000, eps=1e-12):
    """
    X_nt: (N,T) trials of success probability p at each n.
    Returns:
      p_mean, p_lo, p_hi      (all length N)
      inv_mean, inv_lo, inv_hi for 1/p via monotone transform
    """
    X_nt = np.asarray(X_nt, float)
    N, T = X_nt.shape

    p_mean = np.empty(N, float)
    p_lo   = np.empty(N, float)
    p_hi   = np.empty(N, float)

    inv_mean = np.empty(N, float)
    inv_lo   = np.empty(N, float)
    inv_hi   = np.empty(N, float)

    for i in range(N):
        x = X_nt[i, :]
        x = x[np.isfinite(x)]

        mu = float(np.mean(x))
        lo, hi = mean_boot(x, ci=ci, B=B)

        # clip to keep log / inverse stable
        mu_c = float(np.clip(mu, eps, 1.0))
        lo_c = float(np.clip(lo, eps, 1.0))
        hi_c = float(np.clip(hi, eps, 1.0))

        p_mean[i] = mu_c
        p_lo[i]   = lo_c
        p_hi[i]   = hi_c

        # 1/p is monotone decreasing in p:
        inv_mean[i] = 1.0 / mu_c
        inv_lo[i]   = 1.0 / hi_c  # lower 1/p from upper p
        inv_hi[i]   = 1.0 / lo_c  # upper 1/p from lower p

    return p_mean, p_lo, p_hi, inv_mean, inv_lo, inv_hi


# ============================================================
# D) Main: summarize all (or selected) p and SAVE:
#   - pointwise mean/CI for p and for 1/p
#   - fit_boot CI for (a,b) on p, then transform to 1/p
#   - r2 from log-fit on point means
# ============================================================
def summarize_qaaur_single2_with_boot(
    npz_path,
    keep_trials=10000,
    p_indices="all",          # "all" or list[int]
    fit_n_min=9,           # e.g. 70; None means use all n
    ci=0.95,
    B_point=2000,
    B_fit=2000,
    eps=1e-12,
    out_path=None,
):
    dat = load_qaaur_from_single2(npz_path, keep_trials=keep_trials)
    k = float(dat["k"])
    n_list = np.asarray(dat["n_list"], int)
    plist = np.asarray(dat["plist"])
    S = np.asarray(dat["single2_all"], float)  # (N,T,P)
    N, T, P = S.shape

    if p_indices is None or p_indices == "all":
        p_idx = np.arange(P, dtype=int)
    else:
        p_idx = np.asarray(p_indices, dtype=int)
    M = p_idx.size

    # outputs (N,M)
    p_mean = np.empty((N, M), float)
    p_ci_lo = np.empty((N, M), float)
    p_ci_hi = np.empty((N, M), float)

    inv_mean = np.empty((N, M), float)
    inv_ci_lo = np.empty((N, M), float)
    inv_ci_hi = np.empty((N, M), float)

    # per-curve (M,)
    p_fit_b_hat = np.empty(M, float)
    p_fit_r2    = np.empty(M, float)
    p_fit_a_ci_lo = np.empty(M, float)
    p_fit_a_ci_hi = np.empty(M, float)
    p_fit_b_ci_lo = np.empty(M, float)
    p_fit_b_ci_hi = np.empty(M, float)

    inv_fit_b_hat = np.empty(M, float)
    inv_fit_r2    = np.empty(M, float)
    inv_fit_a_ci_lo = np.empty(M, float)
    inv_fit_a_ci_hi = np.empty(M, float)
    inv_fit_b_ci_lo = np.empty(M, float)
    inv_fit_b_ci_hi = np.empty(M, float)

    # fit mask
    if fit_n_min is None:
        mask_fit = np.ones(N, dtype=bool)
        n_fit = n_list
    else:
        mask_fit = n_list >= int(fit_n_min)
        n_fit = n_list[mask_fit]

    for t, j in enumerate(p_idx):
        X_nt = S[:, :, j]  # (N,T)

        # pointwise mean(p)+CI(p) and 1/p via monotone transform
        mu, lo, hi, inv_mu, inv_lo, inv_hi = pointwise_p_mean_ci(
            X_nt, ci=ci, B=B_point, eps=eps
        )
        p_mean[:, t] = mu
        p_ci_lo[:, t] = lo
        p_ci_hi[:, t] = hi

        inv_mean[:, t] = inv_mu
        inv_ci_lo[:, t] = inv_lo
        inv_ci_hi[:, t] = inv_hi

        # point log-fit on p_mean for (a_hat,b_hat,r2)
        a_hat, b_hat, r2 = _fit_log_ab_r2(n_list, mu, fit_n_min=fit_n_min, eps=eps)
        p_fit_b_hat[t] = b_hat
        p_fit_r2[t] = r2

        # fit_boot CI on p directly (uses your function)
        a_ci, b_ci = fit_boot(n_fit, X_nt[mask_fit, :], ci=ci, B=B_fit, eps=eps)
        a_lo, a_hi = float(a_ci[0]), float(a_ci[1])
        b_lo, b_hi = float(b_ci[0]), float(b_ci[1])

        p_fit_a_ci_lo[t] = a_lo
        p_fit_a_ci_hi[t] = a_hi
        p_fit_b_ci_lo[t] = b_lo
        p_fit_b_ci_hi[t] = b_hi

        # transform to 1/p: if p ≈ a*b^n then 1/p ≈ (1/a)*(1/b)^n
        inv_fit_b_hat[t] = 1.0 / b_hat
        inv_fit_r2[t] = r2  # same r2 in log-space

        inv_fit_a_ci_lo[t] = 1.0 / a_hi
        inv_fit_a_ci_hi[t] = 1.0 / a_lo
        inv_fit_b_ci_lo[t] = 1.0 / b_hi
        inv_fit_b_ci_hi[t] = 1.0 / b_lo

    if out_path is None:


        # rename prefix
        out_path = npz_path.replace("QAAurdata/QAAur", "QAAurrecover/recoverQAAur", 1)


    np.savez_compressed(
        out_path,
        k=k,
        n_list=n_list,
        plist=plist,
        p_idx=p_idx,
        trials_used=int(T),
        ci=float(ci),
        fit_n_min=(int(fit_n_min) if fit_n_min is not None else -1),

        # pointwise p
        p_mean=p_mean,
        p_ci_lo=p_ci_lo,
        p_ci_hi=p_ci_hi,

        # pointwise 1/p
        inv_mean=inv_mean,
        inv_ci_lo=inv_ci_lo,
        inv_ci_hi=inv_ci_hi,

        # fit on p
        p_fit_b_hat=p_fit_b_hat,
        p_fit_r2=p_fit_r2,
        p_fit_a_ci_lo=p_fit_a_ci_lo,
        p_fit_a_ci_hi=p_fit_a_ci_hi,
        p_fit_b_ci_lo=p_fit_b_ci_lo,
        p_fit_b_ci_hi=p_fit_b_ci_hi,

        # derived fit for 1/p
        inv_fit_b_hat=inv_fit_b_hat,
        inv_fit_r2=inv_fit_r2,
        inv_fit_a_ci_lo=inv_fit_a_ci_lo,
        inv_fit_a_ci_hi=inv_fit_a_ci_hi,
        inv_fit_b_ci_lo=inv_fit_b_ci_lo,
        inv_fit_b_ci_hi=inv_fit_b_ci_hi,
    )

    return out_path


# ============================================================
# Example
# ============================================================
if __name__ == "__main__":
    for  k in [0.55,0.575,0.6,0.626,0.65,0.675,0.7,0.725,0.75]:
        out = summarize_qaaur_single2_with_boot(
            "QAAurdata/QAAur"+str(k)+".npz",
            keep_trials=10000,
            p_indices="all",
            fit_n_min=9,      # or 70, etc.
            ci=0.95,
            B_point=2000,
            B_fit=2000,         
            eps=1e-12,
        )
        print("saved:", out)