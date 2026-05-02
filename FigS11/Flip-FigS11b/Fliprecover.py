import numpy as np
from boot import mean_boot, fit_boot  # Custom bootstrap library (no fixed seed)

# =========================
# Configuration
# =========================
ks = [0.55, 0.575, 0.6, 0.626, 0.65, 0.675, 0.7, 0.725, 0.75]

ALPHA = 0.05
CI = 1.0 - ALPHA

B_POINT = 2000   # Bootstrap iterations for pointwise CI (mean_boot)
B_FIT = 2000     # Bootstrap iterations for parameter CI (fit_boot)

EPS = 1e-15

# Fitting range configuration
FIT_N_MIN = 70
FIT_N_MAX = 125

OUT_NPZ = "RecoverFlipall.npz"


# =========================
# Helper Functions
# =========================
def _ensure_runs_shape(ns, all_runs):
    """Ensures data shape is (Nn, T), transposing automatically if input is (T, Nn)."""
    ns = np.asarray(ns)
    all_runs = np.asarray(all_runs)
    if all_runs.ndim != 2:
        raise ValueError(f"all_runs must be 2D, got shape {all_runs.shape}")

    Nn = len(ns)
    if all_runs.shape[0] == Nn:
        return all_runs
    if all_runs.shape[1] == Nn:
        return all_runs.T
    raise ValueError(f"all_runs shape {all_runs.shape} not compatible with len(ns)={Nn}")


def fit_ab_point(ns, y_mean, eps=1e-15):
    """Calculates point estimates for a and b using log-linear regression (y ~ a*b^n)."""
    n = np.asarray(ns, dtype=np.float64)
    y = np.asarray(y_mean, dtype=np.float64)
    y = np.clip(y, eps, None)

    slope, intercept = np.polyfit(n, np.log(y), 1)
    a_hat = float(np.exp(intercept))
    b_hat = float(np.exp(slope))
    return a_hat, b_hat


def r2_logspace(ns, y_mean, a_hat, b_hat, eps=1e-15):
    """Calculates the coefficient of determination R^2 in log space."""
    n = np.asarray(ns, dtype=np.float64)
    y = np.clip(np.asarray(y_mean, dtype=np.float64), eps, None)

    z = np.log(y)
    z_pred = np.log(a_hat) + n * np.log(b_hat)

    sse = float(np.sum((z - z_pred) ** 2))
    sst = float(np.sum((z - np.mean(z)) ** 2))
    return (1.0 - sse / sst) if sst > 0 else np.nan


def pick_fit_mask(ns, n_min=25, n_max=45):
    """Generates a boolean mask to select the data points within the fitting range [n_min, n_max]."""
    ns = np.asarray(ns, dtype=int)
    if n_min is None or n_max is None:
        return np.ones_like(ns, dtype=bool)
    return (ns >= int(n_min)) & (ns <= int(n_max))


# =========================
# Main Workflow
# =========================
def main():
    """Main routine: iterates through k values, computes means, CIs, fit parameters, and saves results."""
    ns_ref = None

    y_mean_all = []
    y_ci_all = []

    a_hat_all = []
    b_hat_all = []
    a_ci_all = []
    b_ci_all = []
    r2_all = []

    for k in ks:
        # Load raw data
        data = np.load(f"Flipdata/Flip{k}_raw.npz", allow_pickle=True)
        ns = np.asarray(data["ns"], dtype=int)
        all_runs = _ensure_runs_shape(ns, np.asarray(data["all_runs"], dtype=np.float64))

        if ns_ref is None:
            ns_ref = ns.copy()
        else:
            if not (ns.shape == ns_ref.shape and np.all(ns == ns_ref)):
                raise ValueError(f"ns differs across k. First ns={ns_ref}, but k={k} has ns={ns}")

        Nn, T = all_runs.shape
        fit_mask = pick_fit_mask(ns, FIT_N_MIN, FIT_N_MAX)
        ns_fit = ns[fit_mask]
        Y_fit = all_runs[fit_mask, :]

        # ---------- Point estimation: Mean + Bootstrap CI ----------
        mu = all_runs.mean(axis=1, dtype=np.float64)

        ci_lo = np.empty(Nn, dtype=np.float64)
        ci_hi = np.empty(Nn, dtype=np.float64)
        for i in range(Nn):
            lo, hi = mean_boot(all_runs[i, :], ci=CI, B=B_POINT)
            ci_lo[i] = lo
            ci_hi[i] = hi

        # ---------- Fit point estimates and R2 ----------
        a_hat, b_hat = fit_ab_point(ns_fit, mu[fit_mask], eps=EPS)
        r2 = r2_logspace(ns_fit, mu[fit_mask], a_hat, b_hat, eps=EPS)

        # ---------- Fit parameter CI ----------
        a_ci, b_ci = fit_boot(ns_fit, Y_fit, ci=CI, B=B_FIT, eps=EPS)

        # Collect results
        y_mean_all.append(mu)
        y_ci_all.append(np.stack([ci_lo, ci_hi], axis=1))

        a_hat_all.append(a_hat)
        b_hat_all.append(b_hat)
        a_ci_all.append([a_ci[0], a_ci[1]])
        b_ci_all.append([b_ci[0], b_ci[1]])
        r2_all.append(r2)

        print(f"[k={k}] done: a_hat={a_hat:.6e}, b_hat={b_hat:.6e}, R2={r2:.6f}")

    # Convert to arrays
    y_mean_all = np.asarray(y_mean_all, dtype=np.float64)
    y_ci_all = np.asarray(y_ci_all, dtype=np.float64)
    a_hat_all = np.asarray(a_hat_all, dtype=np.float64)
    b_hat_all = np.asarray(b_hat_all, dtype=np.float64)
    a_ci_all = np.asarray(a_ci_all, dtype=np.float64)
    b_ci_all = np.asarray(b_ci_all, dtype=np.float64)
    r2_all = np.asarray(r2_all, dtype=np.float64)

    # ---------- Save results ----------
    np.savez_compressed(
        OUT_NPZ,
        k_list=np.asarray(ks, dtype=np.float64),
        ns=np.asarray(ns_ref, dtype=np.int32),
        y_mean=y_mean_all,
        y_ci=y_ci_all,
        a_hat=a_hat_all,
        b_hat=b_hat_all,
        a_ci=a_ci_all,
        b_ci=b_ci_all,
        r2=r2_all,
        ci=float(CI),
        B_point=int(B_POINT),
        B_fit=int(B_FIT),
        fit_n_min=(None if FIT_N_MIN is None else int(FIT_N_MIN)),
        fit_n_max=(None if FIT_N_MAX is None else int(FIT_N_MAX)),
        eps=float(EPS),
    )
    print(f"\n[saved] {OUT_NPZ}")


if __name__ == "__main__":
    main()