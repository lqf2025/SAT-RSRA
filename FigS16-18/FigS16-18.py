import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import os
from matplotlib.gridspec import GridSpec

# --- 1. Global Style Configuration ---
plt.rcParams.update({
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.linewidth': 1,
    'pdf.fonttype': 42,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.ymargin': 0.02
})

plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Cambria'
mpl.rcParams['mathtext.it'] = 'Computer Modern:italic'
mpl.rcParams['mathtext.bf'] = 'Computer Modern:bold'
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, amssymb, bm} \boldmath"

# --- 2. Colors and Markers ---
g = ["#3C5488", "#C2CBE4", '#FFC4B3', 'o', '^', 'X']  # Right panels
g2 = [
    "#08306B", "#67000D", "#00441B", "#3F007D", "#74C476", "#BCBDDC",
    "#8C2D04", "#E7298A", "#FEC44F", "#4292C6", "#FC9272",
    'o', '^', 'X', '*', 's', 'D', 'v', '<', '>', 'p', 'h'
]

def strround(x, i):
    return f"{x:.{i}f}"

def width(ax, bwith):
    for spine in ax.spines.values():
        spine.set_linewidth(bwith)

def _k_to_tag(k):
    """Stable string tag for k (e.g. 0.30 -> '0.3')."""
    s = f"{float(k):.12g}"
    return s.rstrip("0").rstrip(".")

# ============================================================
# Panel (a): Left Large Plot (Classical Solvers)
# ============================================================
def draw_cdcl_panel(ax, cdcl_file, n_min=70, n_max=125):
    """
    Reads Classical-mix/Recover{k}uniclassical.npz.
    Plots lines for specific solvers: cmini, ccad, cglu, clin, pmini, pcad, pglu, plin.
    """
    if not os.path.exists(cdcl_file):
        print(f"[Warn] Missing Classical file: {cdcl_file}")
        return

    d = np.load(cdcl_file, allow_pickle=True)
    
    # Handle different key names for 'ns' or 'n_list'
    if "ns" in d.files:
        ns_all = np.asarray(d["ns"], dtype=int)
    elif "n_list" in d.files:
        ns_all = np.asarray(d["n_list"], dtype=int)
    else:
        print(f"[Error] No 'ns' or 'n_list' in {cdcl_file}")
        return

    # Apply range mask
    mask = (ns_all >= int(n_min))
    if n_max is not None:
        mask &= (ns_all <= int(n_max))
    ns = ns_all[mask]

    if len(ns) == 0:
        print(f"[Warn] No data points in range [{n_min}, {n_max}] for {cdcl_file}")
        return

    # Solvers to plot
    target_solvers = ["cmini", "ccad", "cglu", "clin", "pmini", "pcad", "pglu", "plin"]
    
    # Mapping for legend names (Optional: customize as needed)
    display_names = {
        "cmini": "Minisat-confl", "ccad": "Cadical-confl",
        "cglu": "Glucose-confl",  "clin": "Lingeling-confl",
        "pmini": "Minisat-prop",  "pcad": "Cadical-prop",
        "pglu": "Glucose-prop",   "plin": "Lingeling-prop"
    }

    handles = []
    
    # Iterate through solvers and plot if data exists
    valid_count = 0
    for solver in target_solvers:
        # Check for necessary keys (mean, CI, fit params)
        key_mean = f"{solver}_mean"
        key_lo   = f"{solver}_ci_lo" # Note: checking your likely key format
        key_hi   = f"{solver}_ci_hi"
        key_a    = f"{solver}_b_hat" # b_hat is the base c
        
        # Fallback for key names if slightly different in new files
        if key_mean not in d.files:
            continue

        # Assign color/marker cyclically
        color = g2[valid_count % len(g2)]
        marker = g2[(valid_count % 11) + 11] # Markers start after colors in g2 list
        valid_count += 1

        # Extract Data
        mean = np.asarray(d[key_mean])[mask]
        
        # Handle CI keys (some files might use _low/_high or _lo/_hi)
        if key_lo in d.files:
            low = np.asarray(d[key_lo])[mask]
            high = np.asarray(d[key_hi])[mask]
        elif f"{solver}_low" in d.files:
            low = np.asarray(d[f"{solver}_low"])[mask]
            high = np.asarray(d[f"{solver}_high"])[mask]
        else:
            low = mean * 0.95 # Fallback dummy
            high = mean * 1.05

        # Get Fit Parameter c (stored as b_hat usually)
        if f"{solver}_b_hat" in d.files:
            b_hat = float(d[f"{solver}_b_hat"])
            a_hat = float(d.get(f"{solver}_a_hat", 1.0)) # Default a=1 if missing
        else:
            b_hat = 0.0
            a_hat = 0.0

        # Plotting
        # 1. CI Band
        ax.fill_between(ns, low, high, color=color, alpha=0.15)
        # 2. Data Points
        ax.plot(ns, mean, marker=marker, ms=6, lw=0, color=color, mfc=color, mew=0)
        # 3. Fit Line
        if b_hat > 0:
            ax.plot(ns, a_hat * (b_hat ** ns), "--", lw=2, color=color)
            label = rf"{display_names.get(solver, solver)}: $c={strround(b_hat,4)}$"
        else:
            label = rf"{display_names.get(solver, solver)}"

        handles.append(
            mlines.Line2D([0], [0], marker=marker, ms=6, mfc=color, mew=0,
                          ls="--", color=color, lw=2, label=label)
        )

    ax.legend(handles=handles, 
          frameon=False, 
          ncol=2, 
          fontsize=10,
          loc="center left",     
          bbox_to_anchor=(0.05, 0.55), 
          columnspacing=0.5)

# ============================================================
# Panel (b): Top Right (QAA)
# ============================================================
def draw_qaa_panel(ax, qaa_file, title_tex, color=g[0], marker=g[3]):
    if not os.path.exists(qaa_file):
        print(f"[Warn] Missing QAA file: {qaa_file}")
        return

    d = np.load(qaa_file, allow_pickle=True)
    
    # Support multiple key naming conventions
    if "ns" in d.files: ns = d["ns"]
    elif "n_list" in d.files: ns = d["n_list"]
    else: return

    # Try to find complexity (1/p) data
    if "y_mean" in d.files:
        mean = d["y_mean"]
        lo = d["y_ci_lo"] if "y_ci_lo" in d.files else mean
        hi = d["y_ci_hi" ] if "y_ci_hi" in d.files else mean
    elif "inv_means" in d.files:
        mean = d["inv_means"]
        lo = d["inv_lows"]
        hi = d["inv_highs"]
    else:
        return

    # Fit params
    if "b_hat" in d.files: b_hat = float(d["b_hat"]); a_hat = float(d["a_hat"])
    elif "B_hat" in d.files: b_hat = float(d["B_hat"]); a_hat = float(d["A_hat"])
    else: b_hat = 0; a_hat = 0

    # Ensure integer type for n and float for data
    ns = np.asarray(ns, dtype=int)
    mean = np.asarray(mean, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    # Plot
    ax.fill_between(ns, lo, hi, color=color, alpha=0.15)
    ax.plot(ns, mean, marker=marker, ms=6, lw=0, color=color, mfc=color, mew=0)
    if b_hat > 0:
        ax.plot(ns, a_hat * (b_hat ** ns), "--", lw=2, color=color)
        label = rf"$c={strround(b_hat,4)}$"
    else:
        label = "Data"

    h = mlines.Line2D([0], [0], marker=marker, ms=6, mfc=color, mew=0,
                      ls="--", color=color, lw=2, label=label)
    ax.legend(handles=[h], frameon=False, fontsize=10, title=title_tex, loc="upper left")

# ============================================================
# Panel (c): Bottom Right (QAOA)
# ============================================================
def draw_qaoa_panel(ax, qaoa_file, title_tex, color=g[1], marker=g[4]):
    if not os.path.exists(qaoa_file):
        print(f"[Warn] Missing QAOA file: {qaoa_file}")
        return

    d = np.load(qaoa_file, allow_pickle=True)
    
    if "ns" in d.files: ns = d["ns"]
    elif "n_list" in d.files: ns = d["n_list"]
    else: return

    # QAOA usually y_mean
    if "y_mean" in d.files:
        mean = d["y_mean"]
        lo = d["y_ci_lo"]
        hi = d["y_ci_hi"]
    else: return

    if "b_hat" in d.files: b_hat = float(d["b_hat"]); a_hat = float(d["a_hat"])
    elif "B_hat" in d.files: b_hat = float(d["B_hat"]); a_hat = float(d["A_hat"])
    else: b_hat = 0; a_hat = 0

    ns = np.asarray(ns, dtype=int)
    
    ax.fill_between(ns, lo, hi, color=color, alpha=0.15)
    ax.plot(ns, mean, marker=marker, ms=6, lw=0, color=color, mfc=color, mew=0)
    if b_hat > 0:
        ax.plot(ns, a_hat * (b_hat ** ns), "--", lw=2, color=color)
        label = rf"$c={strround(b_hat,4)}$"
    else:
        label = "Data"

    h = mlines.Line2D([0], [0], marker=marker, ms=6, mfc=color, mew=0,
                      ls="--", color=color, lw=2, label=label)
    ax.legend(handles=[h], frameon=False, fontsize=10, title=title_tex, loc="upper left")


# ============================================================
# Main Plotting Function
# ============================================================
def plot_mix_three_panel(out_pdf, cdcl_file, qaa_file, qaoa_file, k, TYPE="O"):
    fig = plt.figure(figsize=(11, 6.5))
    gs = GridSpec(2, 3, figure=fig)

    # --- (a) Classical Mix (Left Big) ---
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    draw_cdcl_panel(ax1, cdcl_file, n_min=70, n_max=125)
    ax1.set_yscale("log")
    ax1.set_ylabel("Time complexity", fontsize=12)
    ax1.set_xlabel(r"$n$", fontsize=12)
    ax1.set_title("a", x=-0.06, y=0.996, fontsize=12)
    width(ax1, 1.25)
    # Optional: Customize ticks based on k if needed
    # ax1.set_xticks(...) 

    # --- (b) QAA Mix (Top Right) ---
    ax_b = fig.add_subplot(gs[0, 2])
    draw_qaa_panel(
        ax_b,
        qaa_file=qaa_file,
        title_tex=rf"QAA Mix, $\epsilon={k}$",
        color=g[0], marker=g[3]
    )
    ax_b.set_yscale("log")
    ax_b.set_ylabel("Time complexity", fontsize=10.5)
    ax_b.set_xlabel(r"$n$", fontsize=10.5)
    ax_b.set_title("b", x=-0.06, y=0.99, fontsize=12)
    width(ax_b, 1)

    # --- (c) QAOA Mix (Bottom Right) ---
    ax_c = fig.add_subplot(gs[1, 2])
    draw_qaoa_panel(
        ax_c,
        qaoa_file=qaoa_file,
        title_tex=rf"QAOA Mix, $\epsilon={k}$",
        color=g[1], marker=g[4]
    )
    ax_c.set_yscale("log")
    ax_c.set_ylabel("Time complexity", fontsize=10.5)
    ax_c.set_xlabel(r"$n$", fontsize=10.5)
    ax_c.set_title("c", x=-0.06, y=0.99, fontsize=12)
    width(ax_c, 1)

    # Align Y-limits for QAA/QAOA if desired (visually pleasing)
    ax_b.set_ylim(bottom=0.99)
    ax_c.set_ylim(bottom=0.99)

    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.95, wspace=0.35, hspace=0.25)
    plt.savefig(out_pdf)
    plt.close()
    print(f"[Saved] {out_pdf}")


# ==========================
# Execution Loop
# ==========================
if __name__ == "__main__":
    # Define target k values
    listk = [0.07, 0.3, 0.5]
    TYPE = "M"  # M for Mix

    for i, k in enumerate(listk):
        tag = _k_to_tag(k)
        
        # New File Paths
        f_cls  = f"Classical-FigS16-18a/Recover{tag}uniclassical.npz"
        f_qaa  = f"QAA-FigS16-18b/QAAmixrecover{tag}.npz"
        f_qaoa = f"QAOA-FigS16-18c/RecoverQAOAmix{tag}.npz"

        out_name = f"FigS{i+16}.pdf"
        
        print(f"Processing k={k} -> {out_name}")
        plot_mix_three_panel(
            out_pdf=out_name,
            cdcl_file=f_cls,
            qaa_file=f_qaa,
            qaoa_file=f_qaoa,
            k=k,
            TYPE=TYPE
        )

    print("All plots generated.")