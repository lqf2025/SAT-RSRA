import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

# --- 1. Global Style and Font Configuration ---
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

# --- 2. Color and Marker Definitions ---
# g used for right-side Panels B/C
g = ["#3C5488", "#C2CBE4", '#FFC4B3', 'o', '^', 'X']

# g2 used for left-side Panel A (colors and markers)
g2 = [
    # --- 11 Colors ---
    # Designed based on pairing requirements: (Dark, Light/Contrast)
    "#08306B",  # 1. Deep Blue (pair with 10)
    "#67000D",  # 2. Deep Red (pair with 11)
    "#00441B",  # 3. Deep Green (pair with 5)
    "#3F007D",  # 4. Deep Purple (pair with 6)
    "#74C476",  # 5. Emerald Green (pair with 3)
    "#BCBDDC",  # 6. Light Purple (pair with 4)
    "#8C2D04",  # 7. Deep Brown (pair with 9)
    "#E7298A",  # 8. Magenta (Unique/High Contrast)
    "#FEC44F",  # 9. Bright Yellow (pair with 7)
    "#4292C6",  # 10. Sky Blue (pair with 1)
    "#FC9272",  # 11. Coral (pair with 2)
    
    # --- 11 Markers ---
    'o', '^', 'X', '*', 's', 'D', 'v', '<', '>', 'p', 'h'
]

# --- 3. Helper Functions ---
def strround(x, i):
    # Formats a float to a string with specified decimal precision.
    return f"{x:.{i}f}"

def _geom_mid(ci_pair):
    # Calculates the geometric mean of a confidence interval pair.
    lo, hi = float(ci_pair[0]), float(ci_pair[1])
    return float(np.sqrt(lo * hi))  

def width(ax, bwith):
    # Sets the linewidth for all spines in the given axes.
    for spine in ax.spines.values():
        spine.set_linewidth(bwith)

# --- 4. Plotting Functions ---

def draw_classical_panel(ax, k):
    # Plots the time complexity comparison for classical algorithms and VQE in Panel (a).
    handles = []
    
    # Full list of 11 lines with corresponding display names
    full_keys = [
    ("cdlx", "Dlx-conflict"), ("cglu", "Glucose-conflict"), ("pmini", "Minisat-propagation"),("plin", "Lingeling-propagation"),
    ("cmini", "Minisat-conflict"), ("clin", "Lingeling-conflict"),("pcad", "Cadical-propagation"), ("VQE_p", "VQE-RSRA"),
    ("ccad", "Cadical-conflict"), ("pdlx", "Dlx-propagation"),("pglu", "Glucose-propagation")
    ]
    
    try:
        path = rf"classical-FigS2-S10a/unirecoverwhole/Classical_k{k}.npz"
        d = np.load(path, allow_pickle=True)
        ns_all = d["ns"]
        mask = (ns_all >= 70) & (ns_all <= 125)
        if (k == 0.9):
            mask = (ns_all >= 20) & (ns_all <= 50)
        ns_plot = ns_all[mask]
        eps = 1e-12
        
        for idx, (key, display_name) in enumerate(full_keys):
            mean_key = f"{key}__mean"
            if mean_key not in d.files: continue
            
            # Color and marker assignment logic
            color = g2[idx % 11]
            marker = g2[(idx % 11) + 11]
            
            # Read raw data
            mu_raw = d[mean_key][mask]
            lo_raw = d[f"{key}__mean_ci_lo"][mask]
            hi_raw = d[f"{key}__mean_ci_hi"][mask]
            
            # --- VQE conversion logic (p -> 1/p) ---
            if key == 'VQE_p':
                # Mean conversion
                mu_plot = 1.0 / np.clip(mu_raw, eps, 1.0)
                # Confidence interval inversion: [1/hi, 1/lo]
                lo_p = 1.0 / np.clip(hi_raw, eps, 1.0)
                hi_p = 1.0 / np.clip(lo_raw, eps, 1.0)
                
                # Parameter conversion: 1/p = (1/a) * (1/b)^n
                a_ci_raw = d[f"{key}__a_ci"]
                b_ci_raw = d[f"{key}__b_ci"]
                # Use the inverse of the geometric mean as point estimate
                a_v = 1.0 / _geom_mid(a_ci_raw)
                b_v = 1.0 / _geom_mid(b_ci_raw)
            else:
                # Read classical algorithms directly
                mu_plot = mu_raw
                lo_p, hi_p = lo_raw, hi_raw
                a_v = _geom_mid(d[f"{key}__a_ci"])
                b_v = _geom_mid(d[f"{key}__b_ci"])

            # --- Plotting ---
            # 1. Fill confidence interval band
            ax.fill_between(ns_plot, lo_p, hi_p, color=color, alpha=0.15)
            # 2. Plot scatter data
            ax.plot(ns_plot, mu_plot, marker=marker, ms=6, lw=0, color=color, mfc=color, mew=0)
            # 3. Plot fitted line (Log space)
            ax.plot(ns_plot, a_v * (b_v**ns_plot), "--", lw=2, color=color)
            
            # --- Generate Legend handles ---
            label = rf"{display_name}: $c={strround(b_v, 4)}$"
            custom_handle = mlines.Line2D([0], [0], marker=marker, ms=6, mfc=color, mew=0, 
                                          ls='--', color=color, lw=2, label=label)
            handles.append(custom_handle)

        # Set axes limits and legend
        ax.set_xlim(69, 126)
        # Dynamically adjust position based on k value
        ypos = 0.7
        if(k == 0.4):
            ypos = 0.78
        if(k == 0.2):
            ypos = 0.35
        if(k == 0.9):
            ypos = 0.2

        bbox = (0.5, ypos)
        ax.legend(handles=handles, frameon=False, ncol=3, fontsize=7.5, 
                  title=rf'Classical and VQE, $m/n={k}$', bbox_to_anchor=bbox, loc='center', columnspacing=0.5)
        
    except Exception as e: 
        print(f"Panel A Error (k={k}): {e}")

def draw_qaa_combined(ax, k, layernum=150):
    # Plots the time complexity for restricted and unrestricted QAA in Panel (b).
    handles = []
    # QAA (Restricted) - g[0] (#3C5488) + g[3] ('o')
    try:
        d = np.load(rf"QAA-FigS2-S10b/QAArecoverwhole/recoverQAA{k}.npz", allow_pickle=True)
        lj = np.where(d["layerlist"] == layernum)[0][0]
        ax.fill_between(d["n_list"], d["inv_ci_lo"][:, lj], d["inv_ci_hi"][:, lj], color=g[0], alpha=0.15)
        ax.plot(d["n_list"], d["inv_mean"][:, lj], marker=g[3], ms=6, lw=0, color=g[0], mfc=g[0], mew=0)
        a, b = float(d["inv_fit_a_hat"][lj]), float(d["inv_fit_b_hat"][lj])
        ax.plot(d["n_list"], a * (b**d["n_list"]), "--", lw=2, color=g[0])
        label = rf"QAA-RSRA: $c={strround(b,4)}$"
        handles.append(mlines.Line2D([0], [0], marker=g[3], ms=6, mfc=g[0], mew=0, ls='--', color=g[0], lw=2, label=label))
    except: pass

    # QAAur (Unrestricted) - g[1] (#C2CBE4) + g[4] ('^')
    try:
        d = np.load(rf"QAAur-FigS2-S10b/QAAurrecover/recoverQAAur{k}.npz", allow_pickle=True)
        ax.fill_between(d["n_list"], d["inv_ci_lo"][:, 0], d["inv_ci_hi"][:, 0], color=g[1], alpha=0.15)
        ax.plot(d["n_list"], d["inv_mean"][:, 0], marker=g[4], ms=6, lw=0, color=g[1], mfc=g[1], mew=0)
        a = _geom_mid([d["inv_fit_a_ci_lo"][0], d["inv_fit_a_ci_hi"][0]])
        b = float(d["inv_fit_b_hat"][0])
        ax.plot(d["n_list"], a * (b**d["n_list"]), "--", lw=2.5, color=g[1])
        label = rf"QAA: $c={strround(b,4)}$"
        handles.append(mlines.Line2D([0], [0], marker=g[4], ms=6, mfc=g[1], mew=0, ls='--', color=g[1], lw=2.5, label=label))
    
    except: pass

    ax.set_ylim(bottom=0.9899)
    if(TYPE == "O"):
        ax.set_ylim(bottom=0.99)
    if(k == 0.2):
        ax.set_ylim(bottom=1.001)
    if(k == 0.4):
        ax.set_ylim(bottom=1)
    loc = 'upper center'
    if(k == 0.9):
        loc = 'upper left'
    ax.legend(handles=handles, frameon=False, fontsize=8, title=rf"{layernum} layer QAA, $m/n={k}$", loc=loc)

def draw_qaoa_combined(ax, k):
    # Plots the time complexity for restricted and unrestricted QAOA in Panel (c).
    handles = []
    # QAOA (Restricted)
    try:
        d = np.load(rf"QAOA-FigS2-S10c/QAOArecover/recoverQAOAs{k}.npz", allow_pickle=True)
        ax.fill_between(d["n_list"], d["y_ci_lo"], d["y_ci_hi"], color="#00A087", alpha=0.15)
        ax.plot(d["n_list"], d["y_mean"], marker='^', ms=6, lw=0, color="#00A087", mfc="#00A087", mew=0)
        a, b = float(d["a_hat"]), float(d["b_hat"])
        ax.plot(d["n_list"], a * (b**d["n_list"]), "--", lw=2, color="#00A087")
        label = rf"QAOA-RSRA: $c={strround(b,4)}$"
        handles.append(mlines.Line2D([0], [0], marker='^', ms=6, mfc="#00A087", mew=0, ls='--', color="#00A087", lw=2, label=label))
    except: pass
    
    # QAOAur (Unrestricted Diamond)
    try:
        d = np.load(rf"QAOAur-FigS2-S10c/QAOAurrecover/recoverQAOAur{k}.npz", allow_pickle=True)
        ax.fill_between(d["n_list"], d["y_ci_lo"], d["y_ci_hi"], color="#843131", alpha=0.15)
        ax.plot(d["n_list"], d["y_mean"], marker='D', ms=5, lw=0, color="#843131", mfc="#843131", mew=0)
        a, b = float(d["a_hat"]), float(d["b_hat"])
        ax.plot(d["n_list"], a * (b**d["n_list"]), "--", lw=2, color="#843131")
        label = rf"QAOA: $c={strround(b,4)}$"
        handles.append(mlines.Line2D([0], [0], marker='D', ms=5, mfc="#843131", mew=0, ls='--', color="#843131", lw=2, label=label))
    except: pass

    ax.set_ylim(bottom=0.99)
    if(TYPE == "O"):
        ax.set_ylim(bottom=0.999)
    if(k==0.9):
        ax.set_ylim(bottom=0.995)

    ax.legend(handles=handles, frameon=False, fontsize=8, title=rf"40 layer QAOA, $m/n={k}$", loc='upper right')
    if(TYPE == "O"):
        ax.legend(handles=handles, frameon=False, fontsize=8, title=rf"40 layer QAOA, $m/n={k}$", loc='upper left')

# --- 5. Main Loop ---
listk = [0.626, 0.55, 0.575, 0.6, 0.65, 0.675, 0.7, 0.725, 0.75]
TYPE = "I"
for i in range(len(listk)):
    k = listk[i]
    fig = plt.figure(figsize=(11, 6.5))
    gs = GridSpec(2, 3, figure=fig)
    
    # Panel (a)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    draw_classical_panel(ax1, k)
    ax1.set_yscale('log'); ax1.set_ylabel('Time complexity', fontsize=12); ax1.set_xlabel(r'$n$', fontsize=12)
    ax1.set_title('a', x=-0.06, y=0.996, fontsize=12); width(ax1, 1.25)
    ax1.set_xticks(range(75, 135, 10))
    if(TYPE == 'O'):
        ax1.set_xticks(range(70, 130, 10))
        ax1.set_xlim(68, 121)
    if(k == 0.9):
        ax1.set_xticks(range(20, 60, 10))
        ax1.set_xlim(18, 51)

    # Panel (b)
    ax_b = fig.add_subplot(gs[0, 2])
    draw_qaa_combined(ax_b, k, 150)
    ax_b.set_yscale('log'); ax_b.set_ylabel('Time complexity', fontsize=10.5); ax_b.set_xlabel(r'$n$', fontsize=10.5)
    
    ax_b.set_title('b',  x=-0.06, y=0.99, fontsize=12); width(ax_b, 1)
    ax_b.set_xticks(range(10, 80, 10))
    ax_b.set_xlim(3, 71)
    if(TYPE == 'O'):
        ax_b.set_xticks(range(10, 70, 10))
        ax_b.set_xlim(3, 61)

    # Panel (c)
    ax_c = fig.add_subplot(gs[1, 2])
    draw_qaoa_combined(ax_c, k)
    ax_c.set_yscale('log'); ax_c.set_ylabel('Time complexity', fontsize=10.5); ax_c.set_xlabel(r'$n$', fontsize=10.5)
    ax_c.set_title('c', x=-0.06, y=0.99, fontsize=12); width(ax_c, 1)
    ax_c.set_xticks(range(15, 60, 15))

    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.06, top=0.97, wspace=0.4, hspace=0.2)
    plt.savefig(rf"FigS{i+2}.pdf")
    plt.close()

listk = [0.2, 0.4, 0.8, 0.9]
TYPE = "O"
for i in range(len(listk)):
    k = listk[i]
    fig = plt.figure(figsize=(11, 6.5))
    gs = GridSpec(2, 3, figure=fig)
    
    # Panel (a)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    draw_classical_panel(ax1, k)
    ax1.set_yscale('log'); ax1.set_ylabel('Time complexity', fontsize=12); ax1.set_xlabel(r'$n$', fontsize=12)
    ax1.set_title('a', x=-0.06, y=0.996, fontsize=12); width(ax1, 1.25)
    ax1.set_xticks(range(75, 135, 10))
    if(TYPE == 'O'):
        ax1.set_xticks(range(70, 130, 10))
        ax1.set_xlim(68, 121)
    if(k == 0.9):
        ax1.set_xticks(range(20, 60, 10))
        ax1.set_xlim(18, 51)

    # Panel (b)
    ax_b = fig.add_subplot(gs[0, 2])
    draw_qaa_combined(ax_b, k, 150)
    ax_b.set_yscale('log'); ax_b.set_ylabel('Time complexity', fontsize=10.5); ax_b.set_xlabel(r'$n$', fontsize=10.5)
    
    ax_b.set_title('b',  x=-0.06, y=0.99, fontsize=12); width(ax_b, 1)
    ax_b.set_xticks(range(10, 80, 10))
    ax_b.set_xlim(3, 71)
    if(TYPE == 'O'):
        ax_b.set_xticks(range(10, 70, 10))
        ax_b.set_xlim(3, 61)

    # Panel (c)
    ax_c = fig.add_subplot(gs[1, 2])
    draw_qaoa_combined(ax_c, k)
    ax_c.set_yscale('log'); ax_c.set_ylabel('Time complexity', fontsize=10.5); ax_c.set_xlabel(r'$n$', fontsize=10.5)
    ax_c.set_title('c', x=-0.06, y=0.99, fontsize=12); width(ax_c, 1)
    ax_c.set_xticks(range(15, 60, 15))

    plt.subplots_adjust(left=0.06, right=0.99, bottom=0.06, top=0.97, wspace=0.4, hspace=0.2)
    if(i==0):
        plt.subplots_adjust(left=0.06, right=0.99, bottom=0.06, top=0.97, wspace=0.45, hspace=0.2)
    plt.savefig(rf"FigS{i+12}.pdf")
    plt.close()

print("Execution Finished.")