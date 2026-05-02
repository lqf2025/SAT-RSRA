import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# --- 1. Global Style Configuration ---
# ==========================================
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

# ==========================================
# --- 2. Data Processing Functions ---
# ==========================================
k_list_global = [0.55, 0.575, 0.6, 0.626, 0.65, 0.675, 0.7, 0.725, 0.75]
markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*']

def load_rg_algo_data(k_list, key_prefix):
    """Loads random graph algorithm data (Sampling, Grover, 2SAT) for given k values."""
    dataset = {}
    n_list = None
    for k in k_list:
        try:
            d = np.load(f"RG-FigS11acd/RGrecover/recover_RG{k}.npz", allow_pickle=True)
            if n_list is None: n_list = d["n_list"]
            dataset[k] = {
                "ave": d[f"{key_prefix}_ci_ave"],
                "lo": d[f"{key_prefix}_ci_lo"],
                "hi": d[f"{key_prefix}_ci_hi"],
                "a_val": np.mean(d[f"{key_prefix}a_ci"]),
                "b_val": np.mean(d[f"{key_prefix}b_ci"])
            }
        except: pass
    return n_list, dataset

def load_flip_summary():
    """Loads pre-processed Flip algorithm summary data."""
    try:
        d = np.load("Flip-FigS11b/RecoverFilpall.npz", allow_pickle=True)
        return {
            "ns": d["ns"], "k_list": d["k_list"], "y_mean": d["y_mean"],
            "y_ci": d["y_ci"], "a_hat": d["a_hat"], "b_hat": d["b_hat"],
            "fit_n_min": d.get("fit_n_min", 70) or 0,
            "fit_n_max": d.get("fit_n_max", 125) or 9999
        }
    except: return None

# ==========================================
# --- 3. Core Plotting Routine ---
# ==========================================
fig = plt.figure(figsize=(16, 14))
gs = GridSpec(2, 2, figure=fig)
colors = plt.cm.plasma(np.linspace(0, 0.85, len(k_list_global)))

panel_configs = [
    {"type": "RG",   "title": "Sampling-RSRA", "key": "directRSRA"},
    {"type": "Flip", "title": "Flip-RSRA",       "key": None},
    {"type": "RG",   "title": "Grover-RSRA",     "key": "grover"},
    {"type": "RG",   "title": "Sampling-2SAT",    "key": "twoSAT"}
]

axes_list = []

for idx, cfg in enumerate(panel_configs):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    axes_list.append(ax)
    
    # Significantly increase scatter point size
    scatter_size = 130 
    
    if cfg["type"] == "RG":
        n_list, dataset = load_rg_algo_data(k_list_global, cfg["key"])
        if n_list is not None:
            for k_idx, k in enumerate(k_list_global):
                if k not in dataset: continue
                d = dataset[k]
                fit_y = d["a_val"] * (d["b_val"] ** n_list)
                label_tex = rf"$\frac{{m}}{{n}}={k:g}$"
                
                ax.fill_between(n_list, d["lo"], d["hi"], color=colors[k_idx], alpha=0.1, lw=0)
                ax.scatter(n_list, d["ave"], color=colors[k_idx], marker=markers[k_idx], 
                           s=scatter_size, edgecolors='white', lw=1.0, zorder=3, label=label_tex)
                ax.plot(n_list, fit_y, color=colors[k_idx], ls="--", lw=2.0, alpha=0.7)
                
    elif cfg["type"] == "Flip":
        f_data = load_flip_summary()
        if f_data is not None:
            ns_f = f_data["ns"]
            fit_mask = (ns_f >= f_data["fit_n_min"]) & (ns_f <= f_data["fit_n_max"])
            for i, k_val in enumerate(f_data["k_list"]):
                try: c_idx = k_list_global.index(k_val)
                except: c_idx = i % len(colors)
                
                label_tex = rf"$\frac{{m}}{{n}}={k_val:g}$"
                mu, lo, hi = f_data["y_mean"][i], f_data["y_ci"][i,:,0], f_data["y_ci"][i,:,1]
                
                ax.fill_between(ns_f, lo, hi, color=colors[c_idx], alpha=0.1, lw=0)
                ax.scatter(ns_f, mu, color=colors[c_idx], marker=markers[c_idx], 
                           s=scatter_size, edgecolors='white', lw=1.0, zorder=3, label=label_tex)
                y_fit = f_data["a_hat"][i] * (f_data["b_hat"][i] ** ns_f[fit_mask])
                ax.plot(ns_f[fit_mask], y_fit, color=colors[c_idx], ls="--", lw=2.0, alpha=0.7)

    # --- Style Enhancements ---
    # Enlarge legend font; markerscale=1.0 keeps markers same size as in plot
    ax.legend(loc='upper left', fontsize=15.5, ncol=2, columnspacing=0.8, 
              handletextpad=0.2, markerscale=1.0, frameon=False)

    ax.set_yscale("log")
    ax.text(0.95, 0.05, cfg["title"], 
            transform=ax.transAxes, 
            fontsize=20, 
            ha='right', 
            va='bottom',
            zorder=5)
    ax.grid(False) 
    ax.set_ylabel("Time complexity", fontsize=18)
    ax.set_xlabel("$n$", fontsize=18)
    
    # Subplot labels (a, b, c, d)
    ax.text(-0.04, 1.05, f"{chr(97+idx)}", transform=ax.transAxes, 
            fontsize=22, fontweight='bold', va='top', ha='right')
    
    # Tick parameter styling
    ax.tick_params(axis='both', which='major', labelsize=17, length=8, width=1.8)
    ax.tick_params(axis='both', which='minor', labelsize=17, length=8, width=1.8)

plt.subplots_adjust(top=0.96, hspace=0.15, bottom=0.05, left=0.08, right=0.99)

out_pdf = "FigS11.pdf"
plt.savefig(out_pdf)