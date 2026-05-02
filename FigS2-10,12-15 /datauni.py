import numpy as np

def print_fit_summary_from_npz(npz_file, curve_names=None, fit_n_min=None, digits=4):
    """
    Prints a summary of fitting results from an NPZ file, including geometric means and R² values.
    """
    d = np.load(npz_file, allow_pickle=True)
    ns = np.asarray(d["ns"], dtype=int)
    k = float(d["k"]) if "k" in d.files else None

    if fit_n_min is None:
        fit_n_min = int(d["fit_n_min"]) if "fit_n_min" in d.files else int(ns.min())

    if curve_names is None:
        curve_names = [
            "cmini","ccad","cglu","clin","cdlx",
            "pmini","pcad","pglu","plin","pdlx",
            "VQE_1_over_p",  # Special handling for the 1/p curve (fit comes from VQE_p)
        ]

    def _geom_mid(ci_pair):
        """Computes the geometric midpoint of a given CI pair (low, high)."""
        lo, hi = float(ci_pair[0]), float(ci_pair[1])
        return float(np.sqrt(lo * hi))

    def _fmt(x):
        """Formats the given number to the specified number of decimal places (default 6)."""
        return f"{float(x):.{digits}f}"

    # Print header for the summary table
    print("\n=== Fit summary (from saved CI; mid = geometric mean of CI ends) ===")
    if k is not None and np.isfinite(k):
        print(f"file={npz_file} | k={k:g} | fit_n_min={fit_n_min}")
    else:
        print(f"file={npz_file} | fit_n_min={fit_n_min}")

    # Print the table header for results
    header = "name | b_mid [b_lo, b_hi] | R2"
    print(header)
    print("-" * len(header))

    eps = 1e-300  # Prevent issues with very small values

    for name in curve_names:
        # Process VQE data and display 1/p using p-fit
        if name == "VQE_1_over_p":
            b_key_p = "VQE_p__b_ci"
            r2_key_p = "VQE_p__r2"

            # Skip if the key for VQE data is missing
            if b_key_p not in d.files:
                continue

            b_ci_p = np.asarray(d[b_key_p], dtype=float)  # Get b CI for VQE (p)
            # Calculate 1/p CI = [1/b_hi, 1/b_lo]
            b_ci_inv = np.array([1.0 / max(b_ci_p[1], eps), 1.0 / max(b_ci_p[0], eps)], dtype=float)
            b_mid_inv = _geom_mid(b_ci_inv)

            # R2 value for VQE data
            r2 = float(d[r2_key_p]) if (r2_key_p in d.files) else None
            r2s = "NA" if r2 is None else f"{r2:.6f}"

            # Print result for VQE 1/p curve
            print(f"VQE (1/p) | {_fmt(b_mid_inv)} [{_fmt(b_ci_inv[0])}, {_fmt(b_ci_inv[1])}] | {r2s}")
            continue

        # Process normal curves: use name__b_ci and name__r2
        b_key = f"{name}__b_ci"
        r2_key = f"{name}__r2"

        # Skip if the current curve's fitting results are missing
        if b_key not in d.files:
            continue

        b_ci = np.asarray(d[b_key], dtype=float)
        b_mid = _geom_mid(b_ci)

        # Retrieve R2 value for the current curve
        r2 = float(d[r2_key]) if (r2_key in d.files) else None
        r2s = "NA" if r2 is None else f"{r2:.6f}"

        # Print result for the current curve
        print(f"{name} | {_fmt(b_mid)} [{_fmt(b_ci[0])}, {_fmt(b_ci[1])}] | {r2s}")

# Example usage
if __name__ == "__main__":
    for k in [0.55, 0.575, 0.6, 0.626, 0.65, 0.675, 0.7, 0.725, 0.75]:
        npz = f"classical-FigS2-S10a/unirecoverwhole/Classical_k{k}.npz"
        #npz = f"classical-FigS2-S10a/unirecovereq/Classical_keq{k}.npz"
        print_fit_summary_from_npz(npz, digits=4)  # Adjust digits as needed

    # for k in [0.2,0.4,0.8,0.9]:
    #     npz = f"unirecoverwhole/Classical_k{k}.npz"
    #     print_fit_summary_from_npz(npz, digits=4)
    
