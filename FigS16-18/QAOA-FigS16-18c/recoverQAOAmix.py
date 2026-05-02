#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats  # Added for t-distribution
from boot import mean_boot, fit_boot

def _k_to_tag(k):
    # Make the string representation of k in the filename stable: 0.6260 -> "0.626"
    if isinstance(k, float):
        s = f"{k:.12g}"
    else:
        s = str(k)
    return s.rstrip("0").rstrip(".")

def _load_one_try(prefix, k, tr, eps=1e-12):
    """
    Load one file and return:
      ns: (N,)
      y : (N,)  where y = 1 / mean_p_over_test
      mean_p: (N,)
    """
    tag = _k_to_tag(k)
    fp = f"{prefix}_k{tag}_try{tr}.npz"
    d = np.load(fp, allow_pickle=True)

    # your optimizer script saves n_list; some other scripts may save ns
    if "n_list" in d.files:
        ns = np.asarray(d["n_list"], dtype=int)
    elif "ns" in d.files:
        ns = np.asarray(d["ns"], dtype=int)
    else:
        raise KeyError(f"{fp} has no 'n_list' or 'ns'.")

    if "single2_all" not in d.files:
        raise KeyError(f"{fp} has no 'single2_all'. Cannot reconstruct per-n mean success probability.")

    probs = np.asarray(d["single2_all"], dtype=float)  # (N, test)
    mean_p = np.mean(probs, axis=1)
    mean_p = np.clip(mean_p, eps, None)
    y = 1.0 / mean_p
    return ns, y, mean_p, fp

def _align_by_common_ns(ns_list, y_list):
    """
    Intersect ns across tries and align.
    Return: ns_common, Y (N, T)
    """
    ns_common = ns_list[0].copy()
    for ns in ns_list[1:]:
        ns_common = np.intersect1d(ns_common, ns)

    if ns_common.size == 0:
        raise ValueError("No common n values across tries.")

    T = len(y_list)
    Y = np.empty((ns_common.size, T), dtype=float)

    for j, (ns, y) in enumerate(zip(ns_list, y_list)):
        pos = np.searchsorted(ns, ns_common)
        # ns[pos] should equal ns_common if ns_common is subset of ns
        if not np.all(ns[pos] == ns_common):
            raise ValueError("Alignment failed: ns_common not a subset of one try's ns.")
        Y[:, j] = y[pos]

    return ns_common, Y

def _fit_log_ab_r2(ns, y_mean, fit_n_min, fit_n_max=None, eps=1e-12):
    """
    R^2 of linear regression on log(y_mean) over fit range.
    log y = alpha + beta n  =>  y = a*b^n with a=exp(alpha), b=exp(beta)
    """
    ns = np.asarray(ns, dtype=float)
    y = np.asarray(y_mean, dtype=float)

    mask = (ns >= float(fit_n_min))
    if fit_n_max is not None:
        mask &= (ns <= float(fit_n_max))

    x = ns[mask]
    yy = np.clip(y[mask], eps, None)
    z = np.log(yy)

    slope, intercept = np.polyfit(x, z, 1)
    z_pred = intercept + slope * x

    sse = float(np.sum((z - z_pred) ** 2))
    sst = float(np.sum((z - float(np.mean(z))) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan

    a_hat = float(np.exp(intercept))
    b_hat = float(np.exp(slope))
    return a_hat, b_hat, r2

def summarize_k_from_tries(
    k,
    tries=(0, 1, 2),
    prefix="QAOAout",     # prefix for input files
    ci=0.95,
    B_mean=2000,
    B_fit=2000,
    fit_n_min=70,
    fit_n_max=None,
    eps=1e-12,
    out_prefix="RecoverQAOAout" 
):
    """
    For fixed k, read multiple tries and produce:
      - pointwise y_mean and CI using Student's T-distribution on probabilities (1/mean(P))
      - fit CI (a,b) from boot.fit_boot on probabilities (inverted)
      - R^2 from log-mean fit over the fit range
    """
    ns_list, y_list, p_list, fps = [], [], [], []
    for tr in tries:
        ns, y, mean_p, fp = _load_one_try(prefix, k, tr, eps=eps)
        ns_list.append(ns)
        y_list.append(y)
        p_list.append(mean_p)
        fps.append(fp)

    ns_common, Y = _align_by_common_ns(ns_list, y_list)   # (N,T)  y=1/mean_p for each try
    _, P = _align_by_common_ns(ns_list, p_list)           # (N,T)  mean_p for each try

    # ============================================================
    # 1. Pointwise Mean + CI using Student's T-Distribution
    #    Calculate on P (probs), then invert to Y (complexity)
    # ============================================================
    N, T = P.shape
    
    # Mean of probabilities
    p_mean = np.mean(P, axis=1)
    
    # CI using T-distribution
    if T > 1:
        p_sem = stats.sem(P, axis=1, ddof=1)
        alpha = 1.0 - ci
        # Degrees of freedom = T - 1
        t_score = stats.t.ppf(1.0 - alpha / 2.0, T - 1)
        margin = t_score * p_sem
        p_lo = p_mean - margin
        p_hi = p_mean + margin
    else:
        p_lo = p_mean
        p_hi = p_mean

    # Clip probabilities
    p_mean = np.clip(p_mean, eps, 1.0)
    p_lo = np.clip(p_lo, eps, 1.0)
    p_hi = np.clip(p_hi, eps, 1.0)

    # Invert to Complexity domain (y = 1/p)
    # Note: 1/p_hi corresponds to y_lo, 1/p_lo corresponds to y_hi
    y_mean = 1.0 / p_mean
    y_lo_ci = 1.0 / p_hi
    y_hi_ci = 1.0 / p_lo

    # ============================================================
    # 2. Fit CI (a,b) using bootstrap on P (Probabilities)
    #    Fit p ~ a_dec * b_dec^n, then invert parameters
    # ============================================================
    mask_fit = (ns_common >= int(fit_n_min))
    if fit_n_max is not None:
        mask_fit &= (ns_common <= int(fit_n_max))

    ns_fit = ns_common[mask_fit]
    # Use P matrix for fitting to ensure consistency with "mean probability" approach
    P_fit = P[mask_fit, :]
    
    # keep only finite columns
    col_ok = np.all(np.isfinite(P_fit), axis=0)
    P_fit = P_fit[:, col_ok]

    # fit_boot returns decay parameters for probability
    a_dec_ci, b_dec_ci = fit_boot(ns_fit, P_fit, ci=ci, B=B_fit, eps=eps)

    # Invert parameters for complexity: a = 1/a_dec, b = 1/b_dec
    a_ci = (1.0 / a_dec_ci[1], 1.0 / a_dec_ci[0])
    b_ci = (1.0 / b_dec_ci[1], 1.0 / b_dec_ci[0])

    # ============================================================
    # 3. R2 from log-fit on point means
    # ============================================================
    # Use the new y_mean (derived from 1/mean(P))
    a_hat, b_hat, r2 = _fit_log_ab_r2(ns_common, y_mean, fit_n_min, fit_n_max, eps=eps)

    tag = _k_to_tag(k)
    out_file = f"{out_prefix}{tag}.npz"
    np.savez_compressed(
        out_file,
        k=float(k),
        tries=np.asarray(list(tries), dtype=int),
        files=np.asarray(fps, dtype=object),
        ns=ns_common,
        # per-try curves
        Y_by_try=Y,              # (N,T)  y=1/mean_p for each try
        Pmean_by_try=P,          # (N,T)  mean_p for each try
        # pointwise summary (T-dist derived)
        y_mean=y_mean,
        y_ci_lo=y_lo_ci,
        y_ci_hi=y_hi_ci,
        # fit summary
        fit_n_min=int(fit_n_min),
        fit_n_max=(-1 if fit_n_max is None else int(fit_n_max)),
        a_hat=float(a_hat),
        b_hat=float(b_hat),
        r2=float(r2),
        a_ci=np.asarray(a_ci, dtype=float),
        b_ci=np.asarray(b_ci, dtype=float),
        ci=float(ci),
        method="t-distribution-on-P",
        B_mean=int(B_mean),
        B_fit=int(B_fit),
    )
    return out_file

def print_three_ns_ranges(k, tries=(0, 1, 2), prefix="QAOAout"):
    tag = _k_to_tag(k)
    for tr in tries:
        fp = f"{prefix}_k{tag}_try{tr}.npz"
        d = np.load(fp, allow_pickle=True)

        if "n_list" in d.files:
            key = "n_list"
        elif "ns" in d.files:
            key = "ns"
        else:
            raise KeyError(f"{fp} has no 'n_list' or 'ns'. keys={d.files}")

        ns = np.asarray(d[key], dtype=int)
        print(f"try={tr} | file={fp} | key={key} | n_min={ns.min()} | n_max={ns.max()} | len={len(ns)}")
        print("  ns =", ns)

if __name__ == "__main__":
    for kin in [0.3, 0.5, 0.07]:
        out = summarize_k_from_tries(
            k=kin,
            tries=(0, 1, 2),
            prefix="QAOAmix",
            ci=0.95,
            B_mean=2000,
            B_fit=2000,
            fit_n_min=15,
            fit_n_max=None,   
            out_prefix="RecoverQAOAmix"
        )
        print("[saved]", out)