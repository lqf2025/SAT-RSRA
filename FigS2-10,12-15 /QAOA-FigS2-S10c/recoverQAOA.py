import os
import glob
import numpy as np
import scipy.stats as stats  # Needed for t-distribution
import boot  # Keeps boot.py for parameter fitting

# ============================================================
# Fit: success(n) = -log p(n) = log(1/p) = log a + n log b
# Point estimate helper (retained for calculating a_hat, b_hat)
# ============================================================
def fit_ab_from_success(n_fit, success_fit):
    n_fit = np.asarray(n_fit, float)
    s_fit = np.asarray(success_fit, float)

    X = np.column_stack([np.ones_like(n_fit), n_fit])  # (N,2)
    beta, *_ = np.linalg.lstsq(X, s_fit, rcond=None)
    loga, logb = float(beta[0]), float(beta[1])
    a = float(np.exp(loga))
    b = float(np.exp(logb))

    # R2 on success (log y)
    s_pred = loga + logb * n_fit
    sse = float(np.sum((s_fit - s_pred) ** 2))
    sst = float(np.sum((s_fit - float(np.mean(s_fit))) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    return a, b, loga, logb, float(r2)


# ============================================================
# IO helpers
# ============================================================
def make_npz_paths(prefix, k="", try_min=0, try_max=2, fmt="{prefix}{k}try{t}.npz"):
    return [fmt.format(prefix=prefix, k=k, t=t) for t in range(int(try_min), int(try_max) + 1)]


def glob_npz_paths(prefix, k="", pattern="{prefix}{k}try*.npz"):
    return sorted(glob.glob(pattern.format(prefix=prefix, k=k)))


def load_success_arrays(npz_paths, key="success", strict=True):
    succ_list, loaded = [], []
    for p in npz_paths:
        if not os.path.exists(p):
            if strict:
                raise FileNotFoundError(f"Missing npz: {p}")
            continue
        d = np.load(p, allow_pickle=True)
        s = np.asarray(d[key], float)
        succ_list.append(s)
        loaded.append(p)
    if len(succ_list) == 0:
        raise RuntimeError("No valid npz files loaded.")
    return succ_list, loaded


def default_out_path(npz_paths):
    # Adjust output path generation as needed
    out1 = npz_paths[0].replace("QAOAdata/QAOAs", "QAOArecover/recoverQAOAs", 1)
    return out1.replace("try0", "", 1)


# ============================================================
# Main recover (T-Distribution Version for Pointwise CI)
# ============================================================
def recover_multi_try(
    npz_paths,
    n_start=5,
    fit_n_min=25,
    fit_n_max=45,
    ci=0.95,
    B_param=2000, # Still used for fitting
    eps=1e-12,
    strict=True,
    key="success",
    out_path=None,
):
    succ_list, loaded_paths = load_success_arrays(npz_paths, key=key, strict=strict)

    L = min(len(s) for s in succ_list)
    succ_list = [s[:L] for s in succ_list]
    n_list = np.arange(int(n_start), int(n_start) + L, dtype=int)

    # p_runs: (R,N)
    # success = -log(p) => p = exp(-success)
    p_runs = np.stack([np.clip(np.exp(-s), eps, 1.0) for s in succ_list], axis=0)
    R, N = p_runs.shape
    # mean p over tries
    p_mean = np.clip(p_runs.mean(axis=0), eps, 1.0)

    # ============================================================
    # 1. Pointwise CI using Student's T-Distribution
    #    Assumption: Data is normally distributed (or N is small)
    # ============================================================
    p_lo = np.zeros(N, dtype=float)
    p_hi = np.zeros(N, dtype=float)
    
    # Degrees of freedom = Sample Size - 1
    df = R - 1
    
    # Calculate Standard Error of Mean (SEM) = std / sqrt(R)
    # ddof=1 for sample standard deviation
    p_sem = stats.sem(p_runs, axis=0, ddof=1) 
    
    # T-score for two-tailed test
    # ppf: Percent Point Function (inverse of cdf)
    alpha = 1.0 - ci
    t_score = stats.t.ppf(1.0 - alpha / 2.0, df)
    
    # Calculate Margin of Error
    margin_error = t_score * p_sem
    
    # CI = Mean +/- Margin of Error
    p_lo = p_mean - margin_error
    p_hi = p_mean + margin_error
    
    # Clip probabilities to be safe [eps, 1.0]
    p_lo = np.clip(p_lo, eps, 1.0)
    p_hi = np.clip(p_hi, eps, 1.0)

    # transform to success (-log p) and y (1/p)
    # Note: lo/hi swaps when taking negative log or inverse
    s_mean = -np.log(p_mean)
    s_lo = -np.log(np.clip(p_hi, eps, 1.0))
    s_hi = -np.log(np.clip(p_lo, eps, 1.0))

    y_mean = 1.0 / p_mean
    y_lo = 1.0 / np.clip(p_hi, eps, 1.0)
    y_hi = 1.0 / np.clip(p_lo, eps, 1.0)

    # ============================================================
    # 2. Point estimate fit on s_mean (Time Complexity 1/p)
    #    Fits 1/p_mean ~ a * b^n
    # ============================================================
    mask_fit = (n_list >= int(fit_n_min)) & (n_list <= int(fit_n_max))
    n_fit = n_list[mask_fit].astype(float)
    s_fit = s_mean[mask_fit]
    a_hat, b_hat, loga_hat, logb_hat, r2_hat = fit_ab_from_success(n_fit, s_fit)

    # ============================================================
    # 3. Bootstrap CI for (a,b) over tries using boot.py
    #    (As requested, this part remains unchanged)
    # ============================================================
    p_sub = p_runs[:, mask_fit]  # shape (R, N_fit)
    
    # boot.fit_boot needs Y shape (N, T), here T=R, so transpose
    # n_fit shape: (N_fit,)
    # p_sub.T shape: (N_fit, R)
    a_dec_ci, b_dec_ci = boot.fit_boot(
        n_fit, p_sub.T, 
        ci=ci, B=B_param, eps=eps
    )
    
    # Invert CI: 
    # if interval is [lo, hi], then 1/interval is [1/hi, 1/lo]
    a_ci = (1.0 / a_dec_ci[1], 1.0 / a_dec_ci[0])
    b_ci = (1.0 / b_dec_ci[1], 1.0 / b_dec_ci[0])

    if out_path is None:
        out_path = default_out_path(loaded_paths)

    np.savez_compressed(
        out_path,
        loaded_paths=np.array(loaded_paths, dtype=object),
        n_list=n_list,
        n_start=int(n_start),
        R=int(R),
        ci=float(ci),
        B_param=int(B_param),
        fit_n_min=int(fit_n_min),
        fit_n_max=int(fit_n_max),
        method="t-distribution",  # Mark the method

        p_runs=p_runs,
        p_mean=p_mean,
        p_ci_lo=p_lo,
        p_ci_hi=p_hi,

        success_mean=s_mean,
        success_ci_lo=s_lo,
        success_ci_hi=s_hi,

        y_mean=y_mean,
        y_ci_lo=y_lo,
        y_ci_hi=y_hi,

        a_hat=float(a_hat),
        b_hat=float(b_hat),
        loga_hat=float(loga_hat),
        logb_hat=float(logb_hat),
        r2_hat=float(r2_hat),

        a_ci=np.array(a_ci, float),
        b_ci=np.array(b_ci, float),
    )

    print("============================================================")
    print(f"Saved: {out_path}")
    print(f"Loaded R={R} tries (T-Distribution CI). Fit range: [{fit_n_min},{fit_n_max}]")
    print(f"Fit y=1/p_mean = a*b^n:")
    print(f"  a_hat={a_hat:.6e}, b_hat={b_hat:.10f}, R2(log y)={r2_hat:.6f}")
    print(f"  a_CI{int(ci*100)}%=[{a_ci[0]:.6e},{a_ci[1]:.6e}]")
    print(f"  b_CI{int(ci*100)}%=[{b_ci[0]:.10f},{b_ci[1]:.10f}]")
    print("============================================================")

    return out_path


# ============================================================
# Example
# ============================================================
if __name__ == "__main__":
    prefix = "QAOAdata/QAOAs"
    
    #for k in [0.2, 0.4, 0.8, 0.9]:    
    for k in [0.55,0.575,0.6,0.626,0.65,0.675,0.7,0.725,0.75]: 
        npz_list = make_npz_paths(prefix, k=str(k), try_min=0, try_max=2, fmt="{prefix}{k}try{t}.npz")
        
        recover_multi_try(
            npz_list,
            n_start=5,
            fit_n_min=25,
            fit_n_max=45,
            ci=0.95,
            B_param=2000,
            eps=1e-12,
            strict=False,
            key="success",
            out_path=None,
        )