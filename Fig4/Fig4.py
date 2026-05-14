import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter


# =============================
# Global matplotlib settings
# =============================

plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Cambria'
mpl.rcParams['mathtext.it'] = 'Computer Modern:italic'
mpl.rcParams['mathtext.bf'] = 'Computer Modern:bold'
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, bm} \boldmath"
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

plt.rcParams.update({
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.linewidth': 1,
    'lines.linewidth': 1.0,
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.figsize': (13, 5.8),
})


# =============================
# Helper functions
# =============================

def bold_g_formatter(x, pos):
    if abs(x) < 1e-12:
        x = 0
    return r'$\mathbf{%g}$' % x


def convertstring(i):
    """Convert i to a 7-bit binary string."""
    return bin(i + 128)[3:]


def convertstring2(i):
    """Convert i to an 8-bit binary string."""
    return bin(i + 256)[3:]


def convertstringlist(array):
    """Convert indices into 7-bit binary labels."""
    return [bin(i + 128)[3:] for i in array]


def convertstringlist2(array):
    """Convert indices into 8-bit binary labels."""
    return [bin(i + 256)[3:] for i in array]


def setup_subplot(ax, title):
    """Apply consistent axis styling and add panel label."""
    ax.spines[:].set_linewidth(1)

    ax.tick_params(
        axis='both',
        which='major',
        labelsize=10,
        width=0.5,
        length=3
    )
    ax.tick_params(axis='both', which='minor', length=0)

    ax.set_ylabel('Frequency', fontsize=18, labelpad=2)

    ax.text(
        -0.01, 1.08, title,
        fontsize=18,
        transform=ax.transAxes,
        ha='right',
        va='top',
        bbox=dict(facecolor='white', edgecolor='none', pad=0)
    )


def plot_panel(ax, x, average, std, dataacc, titlein, label, loc, mov):
    """Plot energy vs iteration with experimental error bars and simulation curve."""
    exp_color = '#6366F1'
    sim_color = '#EF4444'

    ax.errorbar(
        x,
        average,
        yerr=std,
        fmt='o-',
        capsize=8,
        elinewidth=2,
        linewidth=2,
        markersize=11,
        label='Exp.',
        capthick=1.5,
        color=exp_color,
        markerfacecolor=exp_color,
        zorder=3
    )

    ax.plot(
        x,
        dataacc,
        linewidth=2.5,
        linestyle='--',
        marker='s',
        markersize=13,
        label='Sim.',
        color=sim_color,
        markerfacecolor=sim_color,
        zorder=2
    )

    handles, labels = ax.get_legend_handles_labels()

    ax.spines[:].set_linewidth(1)

    ax.tick_params(axis='both', which='major', width=0.6, length=2.5)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.tick_params(axis='y', which='minor', length=0)

    ax.set_xlabel('Iteration', labelpad=2, fontsize=18)
    ax.set_ylabel('Energy', labelpad=2, fontsize=18)

    ax.text(
        -0.01, 1.08, titlein,
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=18
    )

    ax.legend(
        [handles[1], handles[0]],
        [labels[1], labels[0]],
        frameon=False,
        handletextpad=0.4,
        handlelength=2.8,
        fontsize=16,
        loc=loc,
        bbox_to_anchor=mov
    )

    ax.tick_params(
        axis='x',
        which='both',
        bottom=True,
        top=False,
        labelbottom=True,
        labeltop=False,
        labelsize=15
    )

    ax.yaxis.set_major_formatter(FuncFormatter(bold_g_formatter))

    ax.tick_params(
        axis='y',
        which='both',
        left=True,
        right=False,
        labelleft=True,
        labelright=False,
        labelsize=15
    )

    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])

    ax.text(
        0.65,
        0.95,
        label,
        ha='center',
        va='center',
        transform=ax.transAxes,
        fontsize=16
    )


def load_finaldicts(file_list):
    """Load finaldict objects from npz files."""
    return [
        np.load(filename, allow_pickle=True)['finaldict'].item()
        for filename in file_list
    ]


def finaldicts_to_state_matrix(finaldicts, n_states, converter):
    """
    Convert a list of finaldicts into a matrix of shape
    (n_states, n_runs).

    Each row corresponds to one computational basis state.
    Each column corresponds to one independent experimental run.
    """
    values = np.zeros((n_states, len(finaldicts)))

    for run_id, data in enumerate(finaldicts):
        for state_id in range(n_states):
            state_label = str(converter(state_id))
            if state_label in data:
                values[state_id, run_id] = data[state_label]

    return values


def plot_state_box_panel(
    ax,
    file_list,
    n_states,
    converter,
    panel_label,
    method_label,
    highlight_positions=None,   # 需要高亮的 box 序号，例如 [0] 或 [0,2]
    annotation=None,
    annotation_xy=(0.08, 0.82),
    top_k=10,
):
    """
    Plot final-state frequency distributions as Nature-like box plots
    with hollow-circle raw points.

    highlight_positions:
        list of integer positions (0-based among the displayed top_k states)
        to be highlighted in red. All others use a muted gray-blue.
    """
    if highlight_positions is None:
        highlight_positions = []

    # -------------------------
    # Load data
    # -------------------------
    finaldicts = load_finaldicts(file_list)
    values = finaldicts_to_state_matrix(finaldicts, n_states, converter)

    mean_values = values.mean(axis=1)
    top_indices = np.argsort(mean_values)[::-1][:top_k]

    top_values = [values[i, :] for i in top_indices]
    top_labels = [converter(i) for i in top_indices]
    positions = np.arange(1, len(top_values) + 1)

    # -------------------------
    # Minimal color palette
    # -------------------------
    base_color = '#6366F1'       # blue edge, matched to Exp. in a--d
    base_fill  = '#C7D2FE'       # stronger light-blue fill

    hi_color   = '#EF4444'       # red edge, matched to Sim. in a--d
    hi_fill    = '#FECACA'       # stronger light-red fill

    median_color = '#374151'     # neutral dark gray
    edge_colors = []
    fill_colors = []
    for i in range(len(top_values)):
        if i in highlight_positions:
            edge_colors.append(hi_color)
            fill_colors.append(hi_fill)
        else:
            edge_colors.append(base_color)
            fill_colors.append(base_fill)

    # -------------------------
    # Draw boxplot
    # -------------------------
    bp = ax.boxplot(
        top_values,
        positions=positions,
        labels=top_labels,
        patch_artist=True,
        whis=(0, 100),           # whiskers = min/max
        widths=0.48,
        capwidths=0.62,
        showfliers=False,
        showmeans=False,
        medianprops=dict(
            color=median_color,
            linewidth=1.4
        ),
        boxprops=dict(
            linewidth=1.35
        ),
        whiskerprops=dict(
            linewidth=1.35,
            solid_capstyle='butt'
        ),
        capprops=dict(
            linewidth=1.25,
            solid_capstyle='butt'
        )
    )

    # style boxes
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(fill_colors[i])
        patch.set_edgecolor(edge_colors[i])
        patch.set_alpha(0.68)
        patch.set_linewidth(1.35)
        patch.set_zorder(1)

    # style whiskers and caps
    for i in range(len(top_values)):
        c = edge_colors[i]
        for line in bp['whiskers'][2*i:2*i+2]:
            line.set_color(c)
            line.set_linewidth(1.35)
            line.set_zorder(2)
        for line in bp['caps'][2*i:2*i+2]:
            line.set_color(c)
            line.set_linewidth(1.25)
            line.set_zorder(2)

    # -------------------------
    # Overlay hollow-circle raw points
    # -------------------------
    # five runs: small deterministic horizontal spread
    offsets = np.array([-0.09, -0.045, 0.0, 0.045, 0.09])

    for i, vals in enumerate(top_values):
        vals = np.asarray(vals, dtype=float)
        c = edge_colors[i]

        for j in range(vals.size):
            ax.scatter(
                positions[i] + offsets[j],
                vals[j],
                s=28,
                marker='o',
                facecolors='none',     # hollow circle
                edgecolors=c,
                linewidths=1.25,
                alpha=1,
                zorder=4
            )

    # -------------------------
    # Axes styling
    # -------------------------
    setup_subplot(ax, panel_label)

    ax.text(
        0.65, 0.95, method_label,
        fontsize=16,
        transform=ax.transAxes,
        ha='center',
        va='center'
    )

    ax.tick_params(axis='x', which='minor', length=0)
    ax.tick_params(axis='y', which='minor', length=0)

    ax.tick_params(
        axis='x', which='both',
        bottom=True, top=False,
        labelbottom=True, labeltop=False,
        labelsize=12.5
    )
    ax.tick_params(
        axis='y', which='both',
        left=True, right=False,
        labelleft=True, labelright=False,
        labelsize=15
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(top_labels, rotation=68, ha='center')
    ax.margins(y=0.14)

    if annotation is not None:
        ax.text(
            annotation_xy[0],
            annotation_xy[1],
            annotation,
            transform=ax.transAxes,
            fontsize=15,
            color='black'
        )

    return top_labels, top_values


# =============================
# Figure layout
# =============================

fig = plt.figure(figsize=(13, 5.8))

sim_color = '#EF4444'
exp_color = '#6366F1'

gs = GridSpec(
    2,
    4,
    width_ratios=[1, 1, 1, 1],
    height_ratios=[1, 1]
)


# =============================
# Panel a: VQE-RSRA energy
# =============================

n = 5
x = np.array(range(0, 7))

total = np.zeros(7)
total2 = np.zeros(7)

for i in range(1, n + 1):
    data = np.load(
        "VQE-Fig4ae/VQE" + str(i) + ".npz",
        allow_pickle=True
    )['energylist']

    if i <= 3:
        dataapp = np.load(
            "VQE-Fig4ae/VQEappend" + str(i) + ".npz",
            allow_pickle=True
        )['energylist'][1]
        data = np.append(data, dataapp)

    total = total + data
    total2 = total2 + np.square(data)

average = total / n
std = np.sqrt((total2 - np.square(average) * n) / (n - 1))

dataacc = np.load(
    "VQE-Fig4ae/VQE.npz",
    allow_pickle=True
)['energylist']

ax = plt.subplot(gs[0, 0])
plot_panel(
    ax,
    x,
    average,
    std,
    dataacc,
    "a",
    "VQE-RSRA",
    'center right',
    (1.05, 0.5)
)
ax.set_ylim(-0.06, 1.28)


# =============================
# Panel b: VQE energy
# =============================

total = np.zeros(7)
total2 = np.zeros(7)

for i in range(1, n + 1):
    data = np.load(
        "VQEur-Fig4bf/VQEur" + str(i) + ".npz",
        allow_pickle=True
    )['energylist']

    total = total + data
    total2 = total2 + np.square(data)

average = total / n
std = np.sqrt((total2 - np.square(average) * n) / (n - 1))

dataacc = np.load(
    "VQEur-Fig4bf/VQEur.npz",
    allow_pickle=True
)['energylist']

ax = plt.subplot(gs[0, 1])
plot_panel(
    ax,
    x,
    average,
    std,
    dataacc,
    "b",
    "VQE",
    'center right',
    (1.05, 0.5)
)
ax.set_ylim(3.4, 5.1)


# =============================
# Panel c: QAOA-RSRA energy
# =============================

total = np.zeros(7)
total2 = np.zeros(7)

for i in range(1, n + 1):
    data = np.load(
        "QAOA-Fig4cg/QAOA" + str(i) + ".npz",
        allow_pickle=True
    )['energylist']

    total = total + data
    total2 = total2 + np.square(data)

average = total / n
std = np.sqrt((total2 - np.square(average) * n) / (n - 1))

dataacc = np.load(
    "QAOA-Fig4cg/QAOA.npz",
    allow_pickle=True
)['energylist']

ax = plt.subplot(gs[0, 2])
plot_panel(
    ax,
    x,
    average,
    std,
    dataacc,
    "c",
    "QAOA-RSRA",
    'lower left',
    (-0.02, -0.06)
)


# =============================
# Panel d: QAOA energy
# =============================

total = np.zeros(7)
total2 = np.zeros(7)

for i in range(1, n + 1):
    data = np.load(
        "QAOAur-Fig4dh/QAOAur" + str(i) + ".npz",
        allow_pickle=True
    )['energylist']

    total = total + data
    total2 = total2 + np.square(data)

average = total / n
std = np.sqrt((total2 - np.square(average) * n) / (n - 1))

dataacc = np.load(
    "QAOAur-Fig4dh/QAOAur.npz",
    allow_pickle=True
)['energylist']

ax = plt.subplot(gs[0, 3])
plot_panel(
    ax,
    x,
    average,
    std,
    dataacc,
    "d",
    "QAOA",
    'lower left',
    (-0.02, -0.06)
)
ax.set_yticks([4.7, 4.9, 5.1, 5.3])


# =============================
# Panel e: VQE-RSRA final distribution
# =============================

ax = plt.subplot(gs[1, 0])

plot_state_box_panel(
    ax=ax,
    file_list=[
        "VQE-Fig4ae/VQEappend1.npz",
        "VQE-Fig4ae/VQEappend2.npz",
        "VQE-Fig4ae/VQEappend3.npz",
        "VQE-Fig4ae/VQE4.npz",
        "VQE-Fig4ae/VQE5.npz",
    ],
    n_states=128,
    converter=convertstring,
    panel_label='e',
    method_label='VQE-RSRA',
    highlight_positions=[0],
    annotation=r'$p=\frac{4304.0}{9999}=43.04\%$',
    annotation_xy=(0.17, 0.58),
    top_k=10,
)


# =============================
# Panel f: VQE final distribution
# =============================

ax = plt.subplot(gs[1, 1])

plot_state_box_panel(
    ax=ax,
    file_list=[
        "VQEur-Fig4bf/VQEur1.npz",
        "VQEur-Fig4bf/VQEur2.npz",
        "VQEur-Fig4bf/VQEur3.npz",
        "VQEur-Fig4bf/VQEur4.npz",
        "VQEur-Fig4bf/VQEur5.npz",
    ],
    n_states=256,
    converter=convertstring2,
    panel_label='f',
    method_label='VQE',
    highlight_positions=[],
    annotation=None,
    top_k=10,
)


# =============================
# Panel g: QAOA-RSRA final distribution
# =============================

ax = plt.subplot(gs[1, 2])

plot_state_box_panel(
    ax=ax,
    file_list=[
        "QAOA-Fig4cg/QAOA1.npz",
        "QAOA-Fig4cg/QAOA2.npz",
        "QAOA-Fig4cg/QAOA3.npz",
        "QAOA-Fig4cg/QAOA4.npz",
        "QAOA-Fig4cg/QAOA5.npz",
    ],
    n_states=128,
    converter=convertstring,
    panel_label='g',
    method_label='QAOA-RSRA',
    highlight_positions=[0,2],
    annotation=r'$p=\frac{3027.4+610.2}{9999}=36.4\%$',
    annotation_xy=(0.08, 0.58),
    top_k=10,
)


# =============================
# Panel h: QAOA final distribution
# =============================

ax = plt.subplot(gs[1, 3])

plot_state_box_panel(
    ax=ax,
    file_list=[
        "QAOAur-Fig4dh/QAOAur1.npz",
        "QAOAur-Fig4dh/QAOAur2.npz",
        "QAOAur-Fig4dh/QAOAur3.npz",
        "QAOAur-Fig4dh/QAOAur4.npz",
        "QAOAur-Fig4dh/QAOAur5.npz",
    ],
    n_states=256,
    converter=convertstring2,
    panel_label='h',
    method_label='QAOA',
    highlight_positions=[],
    annotation=None,
    top_k=10,
)


# =============================
# Final layout and export
# =============================

plt.subplots_adjust(
    left=0.0,
    right=0.99,
    bottom=0.01,
    top=1.01,
    wspace=0.31,
    hspace=0.2
)

plt.savefig("Fig4.pdf", bbox_inches='tight')