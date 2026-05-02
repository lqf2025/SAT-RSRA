import numpy as np
import os

def _k_to_tag(k):
    # Formats the k value into a clean string tag for file naming.
    s = f"{float(k):.12g}"
    return s.rstrip("0").rstrip(".")

def get_value(d, keys, default=np.nan):
    # Attempts to retrieve a value from a dictionary using a list of potential keys.
    for k in keys:
        if k in d.files:
            return d[k]
    return default

def print_mix_scaling_results(k_list):
    # Loads and prints the scaling complexity parameters for Classical, QAA, and QAOA mix instances.
    
    
    # Table formatting
    header_fmt = "{:<15} | {:<20} | {:>12} | {:>24} | {:>10}"
    row_fmt    = "{:<15} | {:<20} | {:>12.6f} | {:>24} | {:>10.4f}"
    
    header_width = 90

    print("\n" + "=" * header_width)
    print(f"{'Mix Instance Complexity Scaling Summary (Model: y = a * c^n)':^90}")
    print("=" * header_width)
    print(header_fmt.format("Ratio (m/n)", "Algorithm/Solver", "Base (c)", "95% CI", "R^2"))
    print("-" * header_width)

    for k in k_list:
        tag = _k_to_tag(k)
        
        # Define file paths
        f_cls  = f"Classical-FigS16-18a/Recover{tag}uniclassical.npz"
        f_qaa  = f"QAA-FigS16-18b/QAAmixrecover{tag}.npz"
        f_qaoa = f"QAOA-FigS16-18c/RecoverQAOAmix{tag}.npz"

        found_any = False

        # -------------------------------------------------
        # 1. Classical (CDCL/WalkSAT etc.)
        # -------------------------------------------------
        if os.path.exists(f_cls):
            found_any = True
            try:
                d = np.load(f_cls, allow_pickle=True)
                solvers = ["cmini","ccad","cglu","clin","pmini","pcad","pglu","plin"]
                
                # Try generic key first, then specific solvers
                if "b_hat" in d.files:
                     b = float(d["b_hat"])
                     ci = np.asarray(d["b_ci"]).reshape(-1)
                     r2 = float(get_value(d, ["r2", "r2_hat"], np.nan))
                     print(row_fmt.format(tag, "Classical (Unified)", b, f"[{ci[0]:.6f}, {ci[1]:.6f}]", r2))
                else:
                    for s in solvers:
                        if f"{s}_b_hat" in d.files:
                            b = float(d[f"{s}_b_hat"])
                            ci = np.asarray(d[f"{s}_b_ci"]).reshape(-1)
                            r2 = float(get_value(d, [f"{s}_r2", f"{s}_r2_hat"], np.nan))
                            print(row_fmt.format(tag, f"Classical ({s})", b, f"[{ci[0]:.6f}, {ci[1]:.6f}]", r2))
            except Exception as e:
                print(f"{tag:<15} | {'Error loading Classical':<20} | {str(e)}")

        # -------------------------------------------------
        # 2. QAA
        # -------------------------------------------------
        if os.path.exists(f_qaa):
            found_any = True
            try:
                d = np.load(f_qaa, allow_pickle=True)
                b_val = get_value(d, ["b_hat", "B_hat", "b_hat_succ"])
                b_ci  = get_value(d, ["b_ci", "B_ci", "b_ci_succ"])
                r2    = get_value(d, ["r2", "r2_hat", "inv_fit_r2"])

                if not np.isnan(b_val).all():
                    b = float(b_val)
                    ci = np.asarray(b_ci).reshape(-1)
                    r2_val = float(r2) if r2 is not np.nan else np.nan
                    print(row_fmt.format(tag, "QAA-Mix", b, f"[{ci[0]:.6f}, {ci[1]:.6f}]", r2_val))
            except Exception as e:
                print(f"{tag:<15} | {'Error loading QAA':<20} | {str(e)}")

        # -------------------------------------------------
        # 3. QAOA
        # -------------------------------------------------
        if os.path.exists(f_qaoa):
            found_any = True
            try:
                d = np.load(f_qaoa, allow_pickle=True)
                b_val = get_value(d, ["b_hat", "B_hat"])
                b_ci  = get_value(d, ["b_ci", "B_ci"])
                r2    = get_value(d, ["r2", "r2_hat"])

                if not np.isnan(b_val).all():
                    b = float(b_val)
                    ci = np.asarray(b_ci).reshape(-1)
                    r2_val = float(r2) if r2 is not np.nan else np.nan
                    print(row_fmt.format(tag, "QAOA-Mix", b, f"[{ci[0]:.6f}, {ci[1]:.6f}]", r2_val))
            except Exception as e:
                print(f"{tag:<15} | {'Error loading QAOA':<20} | {str(e)}")

        if found_any:
            print("-" * header_width)
        else:
            print(f"Warning: No files found for k={tag}.")
            print("-" * header_width)

if __name__ == "__main__":
    target_k_list = [0.07, 0.3, 0.5]
    print_mix_scaling_results(target_k_list)