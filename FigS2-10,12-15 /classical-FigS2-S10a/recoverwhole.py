import numpy as np
from boot import mean_boot, fit_boot

def _k_to_tag(k):
    if isinstance(k, float):
        s = f"{k:.12g}"
    else:
        s = str(k)
    return s.rstrip("0").rstrip(".")

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

    if keys_to_merge is None:
        keys_to_merge = [
            "cmini","ccad","cglu","clin","cdlx",
            "pmini","pcad","pglu","plin","pdlx",
            "VQEsuccess"
        ]
    if keep_cols_map is None:
        keep_cols_map = {}

    merged = {"k": float(k)}

    all_ns = []
    for d in loaded:
        all_ns.append(np.asarray(d["ns"], dtype=int))
    ns_union = np.unique(np.concatenate(all_ns))
    merged["ns"] = ns_union

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

        for d in loaded:
            if key not in d.files:
                continue
            a = np.asarray(d[key], dtype=float)
            ns = np.asarray(d["ns"], dtype=int)

            if key in keep_cols_map:
                a = a[:, :min(a.shape[1], int(keep_cols_map[key]))]

            if a.shape[1] < max_copy:
                pad = np.full((a.shape[0], max_copy - a.shape[1]), np.nan, dtype=float)
                a = np.concatenate([a, pad], axis=1)

            pos = np.searchsorted(ns_union, ns)
            out[pos, :] = a

        merged[key] = out

    return merged

def _row_boot_mean_ci(x_row, ci=0.95, B=2000):
    x = np.asarray(x_row, dtype=float)
    x = x[np.isfinite(x)]
    mu = float(np.mean(x))
    lo, hi = mean_boot(x, ci=ci, B=B)
    return mu, lo, hi

def _fit_boot_on_matrix(ns, Y, fit_n_min, ci=0.95, B=2000, eps=1e-12):
    ns = np.asarray(ns, dtype=int)
    Y = np.asarray(Y, dtype=float)

    mask_fit = ns >= int(fit_n_min)
    ns_fit = ns[mask_fit]
    Y_fit = Y[mask_fit, :]

    col_ok = np.all(np.isfinite(Y_fit), axis=0)
    Y_fit = Y_fit[:, col_ok]

    a_ci, b_ci = fit_boot(ns_fit, Y_fit, ci=ci, B=B, eps=eps)
    return a_ci, b_ci

def _fit_log_ab_r2(ns, y_mean, fit_n_min, eps=1e-12):
    """
    在拟合区间 ns >= fit_n_min 上，对 log(y_mean) 做线性回归：
        log(y) = alpha + beta * n
    返回：a_hat = exp(alpha), b_hat = exp(beta), R2
    """
    ns = np.asarray(ns, dtype=float)
    y = np.asarray(y_mean, dtype=float)

    mask = ns >= float(fit_n_min)
    x = ns[mask]
    y = np.clip(y[mask], eps, None)

    z = np.log(y)
    slope, intercept = np.polyfit(x, z, 1)  # z = intercept + slope*x
    z_pred = intercept + slope * x

    sse = float(np.sum((z - z_pred) ** 2))
    sst = float(np.sum((z - float(np.mean(z))) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan

    a_hat = float(np.exp(intercept))
    b_hat = float(np.exp(slope))
    return a_hat, b_hat, r2


def summarize_k_with_bootstrap(
    k, n_parts,
    fit_n_min=70,
    ci=0.95,
    B_mean=2000,
    B_fit=2000,
    keep_vqe_cols=1000,
    pattern="unidata/uni{tag}p{p}.npz",
    out_prefix="Classical_k",
):
    data = load_concat_uni_parts_union_ns(
        k, n_parts,
        pattern=pattern,
        keep_cols_map={"VQEsuccess": int(keep_vqe_cols)},
    )

    ns = np.asarray(data["ns"], dtype=int)

    curve_names = [
        "cmini","ccad","cglu","clin","cdlx",
        "pmini","pcad","pglu","plin","pdlx",
    ]

    tag = _k_to_tag(k)
    out_file = f"{out_prefix}{tag}.npz"

    npz_dict = dict(
        k=float(k),
        ns=ns,
        ci=float(ci),
        fit_n_min=int(fit_n_min),
    )

    # 1) classical curves: mean CI + fit CI + R2 (log-fit on point means)
    for name in curve_names:
        if name not in data:
            continue
        Y = np.asarray(data[name], dtype=float)  # (N,T) with NaN padding

        mu = np.zeros(ns.size, dtype=float)
        lo = np.zeros(ns.size, dtype=float)
        hi = np.zeros(ns.size, dtype=float)
        for i in range(ns.size):
            mu[i], lo[i], hi[i] = _row_boot_mean_ci(Y[i, :], ci=ci, B=B_mean)

        a_ci, b_ci = _fit_boot_on_matrix(ns, Y, fit_n_min, ci=ci, B=B_fit, eps=1e-12)

        # ---- NEW: R2 from log-mean linear regression
        _, _, r2 = _fit_log_ab_r2(ns, mu, fit_n_min, eps=1e-12)

        npz_dict[f"{name}__mean"] = mu
        npz_dict[f"{name}__mean_ci_lo"] = lo
        npz_dict[f"{name}__mean_ci_hi"] = hi
        npz_dict[f"{name}__a_ci"] = np.asarray(a_ci, dtype=float)
        npz_dict[f"{name}__b_ci"] = np.asarray(b_ci, dtype=float)
        npz_dict[f"{name}__r2"] = float(r2)   # <--- NEW

    # 2) VQE: store 1/p with CI + fit CI + R2 (log-fit on point means of 1/p)
    if "VQEsuccess" in data:
        V = np.asarray(data["VQEsuccess"], dtype=float)  # (Nn, T) with NaN padding

        # ---- pointwise mean + CI on p
        p_mu = np.zeros(ns.size, dtype=float)
        p_lo = np.zeros(ns.size, dtype=float)
        p_hi = np.zeros(ns.size, dtype=float)
        for i in range(ns.size):
            p_mu[i], p_lo[i], p_hi[i] = _row_boot_mean_ci(V[i, :], ci=ci, B=B_mean)

        # ---- fit CI on p directly
        eps = 1e-12
        a_ci_p, b_ci_p = _fit_boot_on_matrix(ns, V, fit_n_min, ci=ci, B=B_fit, eps=eps)

        # ---- R2 on log(mean(p))
        _, _, r2_p = _fit_log_ab_r2(ns, p_mu, fit_n_min, eps=eps)

        npz_dict["VQE_p__mean"] = p_mu
        npz_dict["VQE_p__mean_ci_lo"] = p_lo
        npz_dict["VQE_p__mean_ci_hi"] = p_hi
        npz_dict["VQE_p__a_ci"] = np.asarray(a_ci_p, dtype=float)
        npz_dict["VQE_p__b_ci"] = np.asarray(b_ci_p, dtype=float)
        npz_dict["VQE_p__r2"] = float(r2_p)

        # ---- (optional) also store 1/p curves ONLY via monotone transform (no fit)
        inv_mu = 1.0 / np.clip(p_mu, eps, 1.0)
        inv_lo = 1.0 / np.clip(p_hi, eps, 1.0)  # inverse monotone
        inv_hi = 1.0 / np.clip(p_lo, eps, 1.0)

        npz_dict["VQE_1_over_p__mean"] = inv_mu
        npz_dict["VQE_1_over_p__mean_ci_lo"] = inv_lo
        npz_dict["VQE_1_over_p__mean_ci_hi"] = inv_hi

    np.savez_compressed(out_file, **npz_dict)
    return out_file

# for  k in [0.55,0.575,0.6,0.626,0.65,0.675,0.7,0.725,0.75]:
#     out_file = summarize_k_with_bootstrap(
#         k=k, n_parts=3,
#         fit_n_min=70, ci=0.95,
#         B_mean=2000, B_fit=2000,
#         out_prefix="unirecoverwhole/Classical_k",
#     )
#     print(out_file)
# for  k in [0.2,0.4,0.8]:
#     out_file = summarize_k_with_bootstrap(
#         k=k, n_parts=1,
#         fit_n_min=70, ci=0.95,
#         B_mean=2000, B_fit=2000,
#         out_prefix="unirecoverwhole/Classical_k",
#     )
#     print(out_file)
for k in [0.9]:
    out_file = summarize_k_with_bootstrap(
        k=k,
        n_parts=1,
        fit_n_min=20,  
        out_prefix="unirecoverwhole/Classical_k",
    )
    print(out_file)