import numpy as np
import os

# --- Configuration ---
#listk = [0.626, 0.55, 0.575, 0.6, 0.65, 0.675, 0.7, 0.725, 0.75]
listk = [0.2,0.4,0.8,0.9]

def get_geom_mid(lo, hi):
    """Geometric midpoint of a confidence interval."""
    return np.sqrt(lo * hi) if lo > 0 and hi > 0 else (lo + hi) / 2.0

# ==============================================================================
# Part 1: Aggregated Summary Table (Comparison of Main Algorithms)
# ==============================================================================

def print_full_summary(list_k):
    print("\n" + "="*85)
    print(f"{'Quantum':^85}")
    print("="*85)
    
    # Table Header
    h_fmt = "{:<12} | {:<6} | {:>10} | {:>10} | {:>10} | {:>8}"
    r_fmt = "{:<12} | {:<6} | {:>10.6f} | {:>10.6f} | {:>10.6f} | {:>8.4f}"
    
    print(h_fmt.format("Algorithm", "k", "b (base)", "CI_lo", "CI_hi", "R2"))
    print("-" * 85)

    for k in list_k:
        # 1) VQE-RSRA
        try:
            path = f"classical-FigS2-S10a/unirecoverwhole/Classical_k{k}.npz"
            d = np.load(path, allow_pickle=True)
            b_ci_p = d["VQE_p__b_ci"]
            r2 = d["VQE_p__r2"]
            b_inv = 1.0 / get_geom_mid(b_ci_p[0], b_ci_p[1])
            lo_inv = 1.0 / b_ci_p[1]
            hi_inv = 1.0 / b_ci_p[0]
            print(r_fmt.format("VQE", str(k), b_inv, lo_inv, hi_inv, r2))
        except: pass

        # 2) QAA (Layer 150)
        try:
            path = f"QAA-FigS2-S10b/QAArecoverwhole/recoverQAA{k}.npz"
            d = np.load(path, allow_pickle=True)
            idx = np.where(d["layerlist"] == 150)[0][0]
            lo, hi = d["inv_fit_b_ci_lo"][idx], d["inv_fit_b_ci_hi"][idx]
            b = get_geom_mid(lo, hi)
            r2 = d["inv_fit_r2"][idx]
            print(r_fmt.format("QAA", str(k), b, lo, hi, r2))
        except: pass

        # 3) QAAur (Original)
        try:
            path = f"QAAur-FigS2-S10b/QAAurrecover/recoverQAAur{k}.npz"
            d = np.load(path, allow_pickle=True)
            lo, hi = d["inv_fit_b_ci_lo"][0], d["inv_fit_b_ci_hi"][0]
            b = get_geom_mid(lo, hi)
            r2 = d["inv_fit_r2"][0]
            print(r_fmt.format("QAAur", str(k), b, lo, hi, r2))
        except: pass

        # 4) QAOA
        try:
            path = f"QAOA-FigS2-S10c/QAOArecover/recoverQAOAs{k}.npz"
            d = np.load(path, allow_pickle=True)
            b = d["b_hat"]
            lo, hi = d["b_ci"]
            r2 = d["r2_hat"]
            print(r_fmt.format("QAOA", str(k), b, lo, hi, r2))
        except: pass

        # 5) QAOAur
        try:
            path = f"QAOAur-FigS2-S10c/QAOAurrecover/recoverQAOAur{k}.npz"
            d = np.load(path, allow_pickle=True)
            b = d["b_hat"]
            lo, hi = d["b_ci"]
            r2 = d["r2_hat"]
            print(r_fmt.format("QAOAur", str(k), b, lo, hi, r2))
        except: pass

        print("-" * 85)


# ==============================================================================
# Part 2: Detailed Classical Breakdown (Single Big Table)
# ==============================================================================

def print_detailed_classical_breakdown(list_k, digits=4):
    print("\n" + "="*85)
    print(f"{'Classical and VQE':^85}")
    print("="*85)

    # Define solvers to extract
    solvers = [
        "cmini", "ccad", "cglu", "clin", "cdlx",
        "pmini", "pcad", "pglu", "plin", "pdlx",
        "VQE_1_over_p"
    ]

    # Table Layout
    # k(8) | Solver(15) | b_mid(10) | CI_lo(10) | CI_hi(10) | R2(8)
    h_fmt = "{:<8} | {:<15} | {:>10} | {:>10} | {:>10} | {:>8}"
    r_fmt = "{:<8} | {:<15} | {:>10} | {:>10} | {:>10} | {:>8}"

    print(h_fmt.format("k", "Solver", "b_mid", "CI_lo", "CI_hi", "R2"))
    print("-" * 85)

    def _fmt(x): return f"{float(x):.{digits}f}"
    eps = 1e-300

    for k in list_k:
        npz_file = f"classical-FigS2-S10a/unirecoverwhole/Classical_k{k}.npz"
        
        if not os.path.exists(npz_file):
            # Print a placeholder if file is missing
            print(f"{str(k):<8} | {'(File Missing)':<15} | {'-':>10} | {'-':>10} | {'-':>10} | {'-':>8}")
            print("-" * 85)
            continue

        d = np.load(npz_file, allow_pickle=True)

        for name in solvers:
            # --- VQE Handling ---
            if name == "VQE_1_over_p":
                b_key = "VQE_p__b_ci"
                r2_key = "VQE_p__r2"
                if b_key not in d.files: continue
                
                b_ci_p = np.asarray(d[b_key], dtype=float)
                # Invert logic: 1/p
                b_ci = np.array([1.0 / max(b_ci_p[1], eps), 1.0 / max(b_ci_p[0], eps)])
                b_mid = get_geom_mid(b_ci[0], b_ci[1])
                
                r2 = float(d[r2_key]) if r2_key in d.files else None
                solver_label = "VQE"

            # --- Classical Solver Handling ---
            else:
                b_key = f"{name}__b_ci"
                r2_key = f"{name}__r2"
                if b_key not in d.files: continue

                b_ci = np.asarray(d[b_key], dtype=float)
                b_mid = get_geom_mid(b_ci[0], b_ci[1])
                
                r2 = float(d[r2_key]) if r2_key in d.files else None
                solver_label = name

            # Format R2
            r2s = f"{r2:.4f}" if r2 is not None else "NA"

            # Print Row
            print(r_fmt.format(
                str(k), 
                solver_label, 
                _fmt(b_mid), 
                _fmt(b_ci[0]), 
                _fmt(b_ci[1]), 
                r2s
            ))
        
        # Separator between different k values for readability
        print("-" * 85)


if __name__ == "__main__":
    # 1. Comparison Table
    print_full_summary(listk)

    # 2. Detailed Big Table for Classical Solvers
    print_detailed_classical_breakdown(listk, digits=4)