import numpy as np
import os
from boot import mean_boot, fit_boot 

# ============================================================
# 1. Helper Functions
# ============================================================
def _eps_to_tag(eps):
    # Formats the epsilon value into a clean string tag for file naming.
    if isinstance(eps, float):
        s = f"{eps:.12g}"
    else:
        s = str(eps)
    return s.rstrip("0").rstrip(".")

def load_concat_cdcl_parts_union_ns(eps, n_parts=1, part_start=1, pattern="eps{tag}_cdcl_stats.npz"):
    # Loads and concatenates data from multiple partial result files into a single unified dataset.
    tag = _eps_to_tag(eps)
    files = [pattern.format(tag=tag, p=p) if "{p}" in pattern else pattern.format(tag=tag) 
             for p in range(part_start, part_start + n_parts)]

    loaded = []
    for fp in files:
        if os.path.exists(fp):
            loaded.append(np.load(fp, allow_pickle=True))
    
    if not loaded:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    keys_to_merge = ["cmini","ccad","cglu","clin","pmini","pcad","pglu","plin"]
    
    ns_all_set = set()
    for d in loaded:
        for x in d["ns"]: ns_all_set.add(int(x))
    ns_all = np.array(sorted(ns_all_set), dtype=int)
    
    merged = {"ns": ns_all, "eps": float(eps)}

    for key in keys_to_merge:
        if key not in loaded[0]: continue
        
        rows_by_n = {int(n): [] for n in ns_all}
        for d in loaded:
            ns_part = np.asarray(d["ns"], dtype=int)
            a_part = np.asarray(d[key])
            if a_part.ndim == 1: a_part = a_part[:, None]
            
            for i, n in enumerate(ns_part):
                rows_by_n[int(n)].append(a_part[i, :])
        
        row_concat = []
        max_cols = 0
        for n in ns_all:
            parts = rows_by_n[int(n)]
            combined = np.concatenate(parts) if parts else np.array([])
            max_cols = max(max_cols, combined.size)
            row_concat.append(combined)
            
        out = np.full((ns_all.size, max_cols), np.nan)
        for i, row in enumerate(row_concat):
            out[i, :row.size] = row
        merged[key] = out
        
    return merged

# ============================================================
# 2. Bootstrap Analysis and Saving (Added R2 Calculation)
# ============================================================
def run_bootstrap_analysis_and_save(
    eps, 
    n_parts=1, 
    fit_n_min=70,
    B_samples=1000, 
    ci_level=0.95
):
    # Performs bootstrap analysis, exponential fitting, and R-squared calculation on the solver data.
    
    
    # A. Load and merge raw data
    pattern = f"uniclassicalmix{eps}.npz"
    data = load_concat_cdcl_parts_union_ns(eps, n_parts, pattern=pattern)
    ns = data["ns"]
    
    save_dict = {"ns": ns, "eps": eps}
    keys = [k for k in data.keys() if k not in ["ns", "eps"]]
    
    # Filter range for fitting
    mask_fit = ns >= fit_n_min
    ns_fit = ns[mask_fit]

    print(f"Bootstrapping (B={B_samples}, n>={fit_n_min})...")

    for key in keys:
        raw_Y = data[key]
        N_ns, T_trials = raw_Y.shape
        
        # 1. Compute pointwise means and CIs
        means = np.zeros(N_ns)
        lows = np.zeros(N_ns)
        highs = np.zeros(N_ns)
        
        for i in range(N_ns):
            valid_samples = raw_Y[i, :][np.isfinite(raw_Y[i, :])]
            if valid_samples.size > 0:
                means[i] = np.mean(valid_samples)
                l, h = mean_boot(valid_samples, ci=ci_level, B=B_samples)
                lows[i], highs[i] = l, h
            else:
                means[i] = lows[i] = highs[i] = np.nan
        
        # 2. Perform exponential fitting analysis (using fit_boot)
        
        fit_raw = raw_Y[mask_fit, :]
        clean_fit_raw = np.where(np.isfinite(fit_raw), fit_raw, np.nanmean(fit_raw))
        
        a_ci, b_ci = fit_boot(ns_fit, clean_fit_raw, ci=ci_level, B=B_samples)
        
        # --- Compute point estimates and R2 ---
        y_obs_log = np.log(np.clip(means[mask_fit], 1e-12, None))
        poly = np.polyfit(ns_fit, y_obs_log, 1)
        b_hat = np.exp(poly[0])
        a_hat = np.exp(poly[1])
        
        # Calculate R2 (in log-linear space)
        y_pred_log = np.polyval(poly, ns_fit)
        ss_res = np.sum((y_obs_log - y_pred_log) ** 2)
        ss_tot = np.sum((y_obs_log - np.mean(y_obs_log)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # 3. Store results
        save_dict[f"{key}_mean"] = means
        save_dict[f"{key}_low"] = lows
        save_dict[f"{key}_high"] = highs
        save_dict[f"{key}_b_hat"] = b_hat
        save_dict[f"{key}_b_ci"] = np.array(b_ci)
        save_dict[f"{key}_a_hat"] = a_hat
        save_dict[f"{key}_a_ci"] = np.array(a_ci)
        save_dict[f"{key}_r2"] = r2

        print(f"[{key:<6}] b={b_hat:.6f} | R2={r2:.4f} | CI=[{b_ci[0]:.6f}, {b_ci[1]:.6f}]")

    # B. Save to new npz file
    out_file = f"Recover{eps}uniclassical.npz"
    np.savez_compressed(out_file, **save_dict)
    print(f"Saved: {out_file}")

# ============================================================
# Execution
# ============================================================
if __name__ == "__main__":
    run_bootstrap_analysis_and_save(
        eps=0.07, 
        n_parts=1, 
        fit_n_min=70, 
        B_samples=2000
    )
    run_bootstrap_analysis_and_save(
        eps=0.3, 
        n_parts=1, 
        fit_n_min=70, 
        B_samples=2000
    )
    run_bootstrap_analysis_and_save(
        eps=0.5, 
        n_parts=1, 
        fit_n_min=70, 
        B_samples=2000
    )