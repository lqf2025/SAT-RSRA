import numpy as np
import os

# =========================
# Configuration
# =========================
ks = [0.55, 0.575, 0.6, 0.626, 0.65, 0.675, 0.7, 0.725, 0.75]
rg_algos = ["Sampling-RSRA", "Sampling-2SAT", "grover-RSRA"]
flip_file = "Flip-FigS11b/RecoverFilpall.npz"
models = {
    "Sampling-RSRA": ("directRSRA_ci_ave", "directRSRAb_ci", "directRSRAr2"),
    "grover-RSRA":     ("grover_ci_ave",     "groverb_ci",     "grover_r2"),
    "Sampling-2SAT":     ("twoSAT_ci_ave",     "twoSATb_ci",     "twoSAT_r2")
}

def fit_abn(n, y, eps=1e-300):
    # Fits an exponential model y = a * b^n using linear regression on log-transformed data.
    
    n = np.asarray(n, float)
    y = np.asarray(y, float)
    y = np.clip(y, eps, np.inf)
    Y = np.log(y)
    X = np.column_stack([np.ones_like(n), n])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    loga, logb = beta
    return float(np.exp(loga)), float(np.exp(logb))

def compute_r2(y, yfit):
    # Calculates the coefficient of determination (R^2) to assess the goodness of fit.
    ss_res = np.sum((y - yfit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot

# =========================
# Summary Printing Functions
# =========================
def print_unified_summary1():
    # Prints the summary table for RG-related algorithms including base parameters and confidence intervals.
    print(f"\n{'='*85}")
    print(f"{'Algorithm Scaling Parameters Summary':^85}")
    print(f"{'='*85}")
    header = f"{'Algorithm':<15} | {'m/n (k)':<8} | {'b_hat (Point)':<14} | {'95% CI (lo, hi)':<25} | {'R2':<8}"
    print(header)
    print(f"{'-'*85}")

    for model_name, (ave_key, bci_key, r2_key) in models.items():
        for k in ks:
            fname = f"RG-FigS11acd/RGrecover/recover_RG{k}.npz"
            try:
                data = np.load(fname, allow_pickle=True)
                n = data["n_list"]
                y = data[ave_key]

                # Perform point estimation fit
                a_hat, b_hat = fit_abn(n, y)
                y_fit = a_hat * (b_hat ** n)
                r2_hat = compute_r2(y, y_fit)

                # Retrieve confidence intervals
                b_ci = data[bci_key]  # [lo, hi]

                # Format output
                print(f"{model_name:<15} | {k:<8.3f} | {b_hat:<14.6f} | ({b_ci[0]:.6f}, {b_ci[1]:.6f}) | {r2_hat:.4f}")
            
            except FileNotFoundError:
                continue
        
        # Draw separator after each model block
        print(f"{'-'*85}")

def print_unified_summary():
    # Prints the summary table for the Flip-RSRA algorithm from the consolidated data file.
    print("=" * 85)
    print(f"{'Algorithm':<15} | {'m/n (k)':<8} | {'b (base)':<12} | {'95% CI (lo, hi)':<25} | {'R2':<8}")
    print("=" * 85)

    # Process Flip algorithm series
    print(f"\n{'--- Flip-RSRA Summary Data ---':^85}")
    if os.path.exists(flip_file):
        f_data = np.load(flip_file, allow_pickle=True)
        f_ks = f_data["k_list"]
        f_b_ci = f_data["b_ci"]
        f_r2 = f_data["r2"]

        for i, k in enumerate(f_ks):
            # Use b_hat if available, otherwise estimate geometric mean from CI
            b_val = f_data["b_hat"][i]
            lo, hi = f_b_ci[i, 0], f_b_ci[i, 1]
            r2 = f_r2[i]
            print(f"{'Flip-RSRA':<15} | {k:<8.3f} | {b_val:<12.6f} | ({lo:.6f}, {hi:.6f}) | {r2:.4f}")
    else:
        print(f"Skipping: {flip_file} (Not Found)")
    
    print("=" * 85)

if __name__ == "__main__":
    print_unified_summary1()
    print_unified_summary()