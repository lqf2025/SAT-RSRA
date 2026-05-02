import os
import numpy as np
from boot import mean_boot, fit_boot


def _default_out_path(npz_path):
    """
    Generate the output path by changing the output directory to 'QAArecovereq' 
    and modifying the file name from 'QAAsingles' to 'recoverQAAsingles'.
    """
    # Get the file's directory and base name
    dname = os.path.dirname(npz_path)
    fname = os.path.basename(npz_path)

    # Modify the filename
    stem = fname[:-4] if fname.endswith(".npz") else fname
    stem = stem.replace("QAAsingles", "recoverQAAeq", 1)

    # Set the output directory to 'QAArecovereq'
    out_dir = "QAArecovereq"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)  # Create the directory if it doesn't exist
    
    # Return the full output file path
    return os.path.join(out_dir, stem + ".npz")


def _fit_logline_r2(n, z):
    """Fit a log-linear curve and calculate R2."""
    n = np.asarray(n, float)
    z = np.asarray(z, float)
    slope, intercept = np.polyfit(n, z, 1)
    z_pred = intercept + slope * n
    sse = float(np.sum((z - z_pred) ** 2))
    sst = float(np.sum((z - float(np.mean(z))) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    return float(intercept), float(slope), float(r2)


def recover_QAAsinglec(
    npz_path,
    fit_n_min=45,
    fit_n_max=69,
    keep_trials=None,
    ci=0.95,
    B_point=2000,
    B_fit=2000,
    eps=1e-12,
    out_path=None,
):
    """Recover QAA data, perform bootstrapping and fitting, and save to output path."""
    # Load the input data from 'QAAdata' folder
    d = np.load(npz_path, allow_pickle=True)

    n_list = np.asarray(d["n_list"], dtype=int)
    layerlist = np.asarray(d["layerlist"], dtype=int)
    single2_all = np.asarray(d["single2_all"], dtype=float)  # (Nn, T, L)
    Nn, T, L = single2_all.shape

    if keep_trials is not None:
        keep_trials = int(min(keep_trials, T))
        single2_all = single2_all[:, :keep_trials, :]
        T = keep_trials

    # Initialize arrays for results
    p_mean   = np.empty((Nn, L), float)
    p_ci_lo  = np.empty((Nn, L), float)
    p_ci_hi  = np.empty((Nn, L), float)

    inv_mean  = np.empty((Nn, L), float)
    inv_ci_lo = np.empty((Nn, L), float)
    inv_ci_hi = np.empty((Nn, L), float)

    inv_fit_a_hat   = np.empty(L, float)
    inv_fit_b_hat   = np.empty(L, float)
    inv_fit_r2      = np.empty(L, float)
    inv_fit_a_ci_lo = np.empty(L, float)
    inv_fit_a_ci_hi = np.empty(L, float)
    inv_fit_b_ci_lo = np.empty(L, float)
    inv_fit_b_ci_hi = np.empty(L, float)

    # Fit mask
    mask_fit = (n_list >= int(fit_n_min)) & (n_list <= int(fit_n_max))
    n_fit = n_list[mask_fit].astype(float)

    # Process each layer
    for lj in range(L):
        X_nt = np.asarray(single2_all[:, :, lj], float)  # (Nn, T)
        X_nt = np.clip(X_nt, 0.0, 1.0)

        # Pointwise mean(p) and CI(p), then compute 1/E[p] using monotone transform
        for i in range(Nn):
            x = X_nt[i, :]
            x = x[np.isfinite(x)]

            mu = float(np.mean(x))
            lo, hi = mean_boot(x, ci=ci, B=B_point)

            mu = float(np.clip(mu, eps, 1.0))
            lo = float(np.clip(lo, eps, 1.0))
            hi = float(np.clip(hi, eps, 1.0))

            p_mean[i, lj]  = mu
            p_ci_lo[i, lj] = lo
            p_ci_hi[i, lj] = hi

            inv_mean[i, lj]  = 1.0 / mu
            inv_ci_lo[i, lj] = 1.0 / hi
            inv_ci_hi[i, lj] = 1.0 / lo

        # Fit p(n) ≈ a*b^n  and 1/p ≈ (1/a)*(1/b)^n using bootstrapping
        a_ci, b_ci = fit_boot(n_fit, X_nt[mask_fit, :], ci=ci, B=B_fit, eps=eps)
        a_lo, a_hi = float(a_ci[0]), float(a_ci[1])
        b_lo, b_hi = float(b_ci[0]), float(b_ci[1])

        # Log-mean linear regression for p_fit
        p_fit = np.clip(p_mean[mask_fit, lj], eps, 1.0)
        z_p = np.log(p_fit)
        intercept, slope, r2 = _fit_logline_r2(n_fit, -z_p)  # log(1/p) = -log(p)

        inv_fit_a_hat[lj] = float(np.exp(intercept))
        inv_fit_b_hat[lj] = float(np.exp(slope))
        inv_fit_r2[lj] = r2

        # CI transform to 1/p parameters
        inv_fit_a_ci_lo[lj] = 1.0 / a_hi
        inv_fit_a_ci_hi[lj] = 1.0 / a_lo
        inv_fit_b_ci_lo[lj] = 1.0 / b_hi
        inv_fit_b_ci_hi[lj] = 1.0 / b_lo

    # Determine the output file path (use the default if not provided)
    if out_path is None:
        out_path = _default_out_path(npz_path)  # Output to 'QAArecovereq' folder

    # Save the results to the 'QAArecovereq' folder
    np.savez_compressed(
        out_path,
        src=npz_path,
        n_list=n_list,
        layerlist=layerlist,
        trials_used=int(T),
        ci=float(ci),
        fit_n_min=int(fit_n_min),
        fit_n_max=int(fit_n_max),

        p_mean=p_mean,
        p_ci_lo=p_ci_lo,
        p_ci_hi=p_ci_hi,

        inv_mean=inv_mean,
        inv_ci_lo=inv_ci_lo,
        inv_ci_hi=inv_ci_hi,

        inv_fit_a_hat=inv_fit_a_hat,
        inv_fit_b_hat=inv_fit_b_hat,
        inv_fit_r2=inv_fit_r2,
        inv_fit_a_ci_lo=inv_fit_a_ci_lo,
        inv_fit_a_ci_hi=inv_fit_a_ci_hi,
        inv_fit_b_ci_lo=inv_fit_b_ci_lo,
        inv_fit_b_ci_hi=inv_fit_b_ci_hi,
    )
    return out_path


if __name__ == "__main__":
    # Loop through various k values and process each corresponding file
    for k in [0.55, 0.575, 0.6, 0.626, 0.65, 0.675, 0.7, 0.725, 0.75]:
        # Process each file and save the results to the 'QAArecovereq' folder
        out = recover_QAAsinglec(
            "QAAdata/QAAsingles" + str(k) + ".npz",  # Input path from 'QAAdata' folder
            fit_n_min=25,
            fit_n_max=44,
            keep_trials=None,
            ci=0.95,
            B_point=2000,
            B_fit=2000,
        )
        print("saved:", out)
