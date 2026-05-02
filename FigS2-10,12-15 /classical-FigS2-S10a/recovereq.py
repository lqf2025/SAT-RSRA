import numpy as np
from boot import mean_boot, fit_boot

# Convert k to a string tag with appropriate formatting
def _k_to_tag(k):
    if isinstance(k, float):
        s = f"{k:.12g}"
    else:
        s = str(k)
    return s.rstrip("0").rstrip(".")

# Load and merge parts of the dataset into a unified structure
def load_concat_uni_parts_union_ns(
    k, n_parts, part_start=1,
    pattern="unidata/uni{tag}p{p}.npz",
    keys_to_merge=None,
    keep_cols_map=None,
):
    tag = _k_to_tag(k)
    files = [pattern.format(tag=tag, p=p) for p in range(part_start, part_start + n_parts)]

    loaded = []
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        loaded.append(d)

    # Default keys to merge if not provided
    if keys_to_merge is None:
        keys_to_merge = [
            "cmini", "ccad", "cglu", "clin", "cdlx",
            "pmini", "pcad", "pglu", "plin", "pdlx",
            "VQEsuccess"
        ]
    if keep_cols_map is None:
        keep_cols_map = {}

    merged = {"k": float(k)}

    # Merge 'ns' arrays across all files and get the unique union of ns
    all_ns = [np.asarray(d["ns"], dtype=int) for d in loaded]
    ns_union = np.unique(np.concatenate(all_ns))
    merged["ns"] = ns_union

    # Merge data for each key specified in keys_to_merge
    for key in keys_to_merge:
        max_copy = 0
        for d in loaded:
            if key not in d.files:
                continue
            a = np.asarray(d[key])
            if a.ndim == 2:
                c = a.shape[1]
                if key in keep_cols_map:
                    c = min(c, int(keep_cols_map[key]))
                max_copy = max(max_copy, c)

        if max_copy == 0:
            continue

        out = np.full((ns_union.size, max_copy), np.nan, dtype=float)

        # Fill in the merged data
        for d in loaded:
            if key not in d.files:
                continue
            a = np.asarray(d[key], dtype=float)
            ns = np.asarray(d["ns"], dtype=int)

            # Handle column restrictions (if any)
            if key in keep_cols_map:
                a = a[:, :min(a.shape[1], int(keep_cols_map[key]))]

            if a.shape[1] < max_copy:
                pad = np.full((a.shape[0], max_copy - a.shape[1]), np.nan, dtype=float)
                a = np.concatenate([a, pad], axis=1)

            pos = np.searchsorted(ns_union, ns)
            out[pos, :] = a

        merged[key] = out

    return merged

# Compute row-wise bootstrapped mean and confidence intervals
def _row_boot_mean_ci(x_row, ci=0.95, B=2000):
    x = np.asarray(x_row, dtype=float)
    x = x[np.isfinite(x)]  # Remove NaN values
    mu = float(np.mean(x))
    lo, hi = mean_boot(x, ci=ci, B=B)
    return mu, lo, hi

# Fit bootstrapped curve to the data
def _fit_boot_on_matrix(ns, Y, fit_n_min, fit_n_max=None, ci=0.95, B=2000, eps=1e-12):
    ns = np.asarray(ns, dtype=int)
    Y = np.asarray(Y, dtype=float)

    # Mask for fitting based on the minimum number of data points
    mask_fit = ns >= int(fit_n_min)
    if fit_n_max is not None:
        fit_n_max = int(fit_n_max)
        mask_fit &= (ns <= fit_n_max)

    ns_fit = ns[mask_fit]
    Y_fit = Y[mask_fit, :]

    # Remove columns with NaN values
    col_ok = np.all(np.isfinite(Y_fit), axis=0)
    Y_fit = Y_fit[:, col_ok]

    # Apply fit and return the parameters
    a_ci, b_ci = fit_boot(ns_fit, Y_fit, ci=ci, B=B, eps=eps)
    return a_ci, b_ci

# Perform log-linear regression on y_mean and calculate R^2
def _fit_log_ab_r2(ns, y_mean, fit_n_min, fit_n_max=None, eps=1e-12):
    ns = np.asarray(ns, dtype=float)
    y = np.asarray(y_mean, dtype=float)

    # Apply fit range mask
    mask = ns >= float(fit_n_min)
    if fit_n_max is not None:
        fit_n_max = float(fit_n_max)
        mask &= (ns <= fit_n_max)

    x = ns[mask]
    y = np.clip(y[mask], eps, None)  # Ensure no zero or negative values

    z = np.log(y)
    slope, intercept = np.polyfit(x, z, 1)
    z_pred = intercept + slope * x

    # Compute R^2 and model parameters
    sse = float(np.sum((z - z_pred) ** 2))
    sst = float(np.sum((z - float(np.mean(z))) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan

    a_hat = float(np.exp(intercept))
    b_hat = float(np.exp(slope))
    return a_hat, b_hat, r2

# Main function to summarize the data with bootstrap and fit confidence intervals
def summarize_k_with_bootstrap(
    k, n_parts,
    fit_n_min=70,
    fit_n_max=None,              # Optional: Upper bound for fitting range
    fit_n_min_map=None,          # Optional: Override fit_n_min for specific k
    fit_n_max_map=None,          # Optional: Override fit_n_max for specific k
    ci=0.95,
    B_mean=2000,
    B_fit=2000,
    keep_vqe_cols=1000,
    pattern="unidata/uni{tag}p{p}.npz",
    out_prefix="Classical_keq",
):
    # Override fit_n_min and fit_n_max if specified for this k
    k_tag = _k_to_tag(k)

    if fit_n_min_map is not None:
        if k_tag in fit_n_min_map:
            fit_n_min = int(fit_n_min_map[k_tag])
        elif float(k) in fit_n_min_map:
            fit_n_min = int(fit_n_min_map[float(k)])

    if fit_n_max_map is not None:
        if k_tag in fit_n_max_map:
            fit_n_max = int(fit_n_max_map[k_tag])
        elif float(k) in fit_n_max_map:
            fit_n_max = int(fit_n_max_map[float(k)])

    fit_n_min = int(fit_n_min)
    fit_n_max = None if fit_n_max is None else int(fit_n_max)

    # Load the data and concatenate parts
    data = load_concat_uni_parts_union_ns(
        k, n_parts,
        pattern=pattern,
        keep_cols_map={"VQEsuccess": int(keep_vqe_cols)},
    )

    ns = np.asarray(data["ns"], dtype=int)

    curve_names = [
        "cmini", "ccad", "cglu", "clin", "cdlx",
        "pmini", "pcad", "pglu", "plin", "pdlx",
    ]

    tag = _k_to_tag(k)
    out_file = f"{out_prefix}{tag}.npz"

    npz_dict = dict(
        k=float(k),
        ns=ns,
        ci=float(ci),
        fit_n_min=int(fit_n_min),
        fit_n_max=(None if fit_n_max is None else int(fit_n_max)),
    )

    # 1) Classical curves: calculate mean, CI, fit CI, and R2 for each curve
    for name in curve_names:
        if name not in data:
            continue
        Y = np.asarray(data[name], dtype=float)

        mu = np.zeros(ns.size, dtype=float)
        lo = np.zeros(ns.size, dtype=float)
        hi = np.zeros(ns.size, dtype=float)
        for i in range(ns.size):
            mu[i], lo[i], hi[i] = _row_boot_mean_ci(Y[i, :], ci=ci, B=B_mean)

        a_ci, b_ci = _fit_boot_on_matrix(
            ns, Y, fit_n_min, fit_n_max=fit_n_max, ci=ci, B=B_fit, eps=1e-12
        )

        _, _, r2 = _fit_log_ab_r2(ns, mu, fit_n_min, fit_n_max=fit_n_max, eps=1e-12)

        npz_dict[f"{name}__mean"] = mu
        npz_dict[f"{name}__mean_ci_lo"] = lo
        npz_dict[f"{name}__mean_ci_hi"] = hi
        npz_dict[f"{name}__a_ci"] = np.asarray(a_ci, dtype=float)
        npz_dict[f"{name}__b_ci"] = np.asarray(b_ci, dtype=float)
        npz_dict[f"{name}__r2"] = float(r2)

    # 2) VQE: calculate 1/p, CI, fit CI, and R2
    if "VQEsuccess" in data:
        V = np.asarray(data["VQEsuccess"], dtype=float)

        p_mu = np.zeros(ns.size, dtype=float)
        p_lo = np.zeros(ns.size, dtype=float)
        p_hi = np.zeros(ns.size, dtype=float)
        for i in range(ns.size):
            p_mu[i], p_lo[i], p_hi[i] = _row_boot_mean_ci(V[i, :], ci=ci, B=B_mean)

        eps = 1e-12
        a_ci_p, b_ci_p = _fit_boot_on_matrix(
            ns, V, fit_n_min, fit_n_max=fit_n_max, ci=ci, B=B_fit, eps=eps
        )
        _, _, r2_p = _fit_log_ab_r2(ns, p_mu, fit_n_min, fit_n_max=fit_n_max, eps=eps)

        npz_dict["VQE_p__mean"] = p_mu
        npz_dict["VQE_p__mean_ci_lo"] = p_lo
        npz_dict["VQE_p__mean_ci_hi"] = p_hi
        npz_dict["VQE_p__a_ci"] = np.asarray(a_ci_p, dtype=float)
        npz_dict["VQE_p__b_ci"] = np.asarray(b_ci_p, dtype=float)
        npz_dict["VQE_p__r2"] = float(r2_p)

        inv_mu = 1.0 / np.clip(p_mu, eps, 1.0)
        inv_lo = 1.0 / np.clip(p_hi, eps, 1.0)
        inv_hi = 1.0 / np.clip(p_lo, eps, 1.0)

        npz_dict["VQE_1_over_p__mean"] = inv_mu
        npz_dict["VQE_1_over_p__mean_ci_lo"] = inv_lo
        npz_dict["VQE_1_over_p__mean_ci_hi"] = inv_hi

    # Save the results to a compressed file
    np.savez_compressed(out_file, **npz_dict)
    return out_file


# Example usage: process different k values
for k in [0.55, 0.575, 0.6, 0.626, 0.65, 0.675, 0.7, 0.725, 0.75]:
    out_file = summarize_k_with_bootstrap(
        k=k,
        n_parts=3,
        fit_n_min=25,              # Default fitting range
        fit_n_max=44,              # Optional: Upper bound for fitting range
        out_prefix="unirecovereq/Classical_keq",
    )
    print(out_file)
