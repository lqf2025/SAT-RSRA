import numpy as np
import matplotlib.pyplot as plt
from boot import fit_boot  # Import the bootstrap fitter.

def plot_b_ci_band_simplified(paths, ci=0.95, B=2000):
    """Load multiple raw datasets, fit b in mean(n)≈a b^n for each layer, and save (b_hat, CI) versus layer."""
    datas = [np.load(p, allow_pickle=True) for p in paths]
    ns = datas[0]["ns"]
    plist = datas[0]["plist"]
    raw_all = np.concatenate([d["raw"] for d in datas], axis=1)

    Nn, Total_Trials, P = raw_all.shape
    b_hats, b_los, b_his = [], [], []

    print(f"Start analysis: {Total_Trials} trials, {P} layer settings...")

    for pi in range(P):
        Y_p = raw_all[:, :, pi]
        _, b_ci = fit_boot(ns, Y_p, ci=ci, B=B)

        means = Y_p.mean(axis=1)
        coeffs = np.polyfit(ns, np.log(np.clip(means, 1e-12, None)), 1)
        b_hat = np.exp(coeffs[0])

        b_hats.append(b_hat)
        b_los.append(b_ci[0])
        b_his.append(b_ci[1])

        if (pi + 1) % 5 == 0:
            print(f"Progress: {pi + 1}/{P}")

    order = np.argsort(plist)
    p_sort = plist[order]
    bh_sort = np.array(b_hats)[order]
    blo_sort = np.array(b_los)[order]
    bhi_sort = np.array(b_his)[order]

    np.savez_compressed(
        "RecoverQAAp.npz",
        plist=p_sort,
        b_hat=bh_sort,
        b_lo=blo_sort,
        b_hi=bhi_sort
    )
    return p_sort, bh_sort, blo_sort, bhi_sort

paths = ["QAAptry0.npz", "QAAptry1.npz", "QAAptry2.npz", "QAAptry3.npz"]
p, bh, lo, hi = plot_b_ci_band_simplified(paths)
