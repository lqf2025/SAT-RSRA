import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Cambria'
mpl.rcParams['mathtext.it'] = 'Computer Modern:italic'
mpl.rcParams['mathtext.bf'] = 'Computer Modern:bold'
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, bm} \boldmath"
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

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
    ax.tick_params(axis='both', which='major', labelsize=10, width=0.5, length=3)
    ax.tick_params(axis='both', which='minor', length=0)
    ax.set_ylabel('Frequency', fontsize=18, labelpad=2)
    ax.text(
        -0.01, 1.08, title, fontsize=18, transform=ax.transAxes,
        ha='right', va='top',
        bbox=dict(facecolor='white', edgecolor='none', pad=0)
    )


def plot_panel(ax, x, average, std, dataacc, titlein, label, loc, mov):
    """Plot energy vs iteration with error bars and simulation curve."""
    exp_color = '#6366F1'
    sim_color = '#EF4444'

    ax.errorbar(
        x, average, std,
        fmt='o-', capsize=8, elinewidth=2, linewidth=2,
        markersize=11, label='Exp.', capthick=1.5, color=exp_color
    )

    ax.plot(
        x, dataacc,
        linewidth=2.5, linestyle='--', marker='s',
        markersize=13, label='Sim.', color=sim_color
    )

    handles, labels = ax.get_legend_handles_labels()

    ax.spines[:].set_linewidth(1)
    ax.tick_params(axis='both', which='major', width=0.6, length=2.5)
    ax.set_xlabel('Iteration', labelpad=2, fontsize=18)
    ax.set_ylabel('Energy', labelpad=2, fontsize=18)

    ax.text(
        -0.01, 1.08, titlein, transform=ax.transAxes,
        ha='right', va='top', fontsize=18
    )

    ax.legend(
        [handles[1], handles[0]], [labels[1], labels[0]],
        frameon=False, handletextpad=0.4, handlelength=2.8,
        fontsize=16, loc=loc, bbox_to_anchor=mov
    )

    ax.tick_params(axis='x', which='minor', length=0)
    ax.tick_params(axis='y', which='minor', length=0)
    ax.tick_params(
        axis='x', which='both',
        bottom=False, top=False,
        labelbottom=True, labeltop=False,
        labelsize=15
    )
    ax.yaxis.set_major_formatter(FuncFormatter(bold_g_formatter))
    ax.tick_params(
        axis='y', which='both',
        left=True, right=False,
        labelleft=True, labelright=False,
        labelsize=15
    )

    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.text(
        0.65, 0.95, label,
        ha='center', va='center',
        transform=ax.transAxes, fontsize=16
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
    box_colors,
    annotation=None,
    annotation_xy=(0.08, 0.82),
    top_k=10,
):
    """
    Plot final-state frequency distributions as pure box plots.

    Each box summarizes five independent experimental runs for one state.
    States are selected according to their mean frequency over the five runs.

    For n=5, whiskers are set to the minimum and maximum values.
    No scatter points, no mean markers, and no outlier markers are shown.
    """
    finaldicts = load_finaldicts(file_list)
    values = finaldicts_to_state_matrix(finaldicts, n_states, converter)

    mean_values = values.mean(axis=1)
    top_indices = np.argsort(mean_values)[::-1][:top_k]

    top_values = [values[i, :] for i in top_indices]
    top_labels = [converter(i) for i in top_indices]
    top_colors = box_colors[:len(top_values)]

    bp = ax.boxplot(
    top_values,
    labels=top_labels,
    patch_artist=True,
    whis=(0, 100),
    widths=0.6,          # 箱体稍微窄一点，不然会显得比上面 errorbar 粗
    capwidths=0.8,       # 更接近上面 capsize=8 的视觉长度
    showfliers=False,
    showmeans=False,
    medianprops=dict(
        color='#4B5563',
        linewidth=1.3
    ),
    boxprops=dict(
        linewidth=1.5
    ),
    whiskerprops=dict(
        linewidth=2.0,    # 对应上面的 elinewidth=2
        solid_capstyle='butt'
    ),
    capprops=dict(
        linewidth=1.5,    # 对应上面的 capthick=1.5
        solid_capstyle='butt'
    )
    )

    for i, patch in enumerate(bp['boxes']):
        c = top_colors[i]
        patch.set_facecolor(c)
        patch.set_edgecolor(c)
        patch.set_alpha(0.82)

    for i, c in enumerate(top_colors):
        for line in bp['whiskers'][2 * i: 2 * i + 2]:
            line.set_color(c)
            line.set_solid_capstyle('butt')
            line.set_linewidth(2)
            line.set_zorder(2)

        for line in bp['caps'][2 * i: 2 * i + 2]:
            line.set_color(c)
            line.set_solid_capstyle('butt')
            line.set_linewidth(1.5)
            line.set_zorder(3)

    setup_subplot(ax, panel_label)

    ax.text(
        0.65, 0.96, method_label,
        fontsize=16,
        transform=ax.transAxes,
        ha='center',
        va='center'
    )

    ax.tick_params(axis='x', which='minor', length=0)
    ax.tick_params(axis='y', which='minor', length=0)
    ax.tick_params(
        axis='x', which='both',
        bottom=False, top=False,
        labelbottom=True, labeltop=False,
        labelsize=12.5
    )
    ax.tick_params(
        axis='y', which='both',
        left=True, right=False,
        labelleft=True, labelright=False,
        labelsize=15
    )

    ax.set_xticklabels(top_labels, rotation=70, ha='center')
    ax.margins(y=0.12)

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


plt.rcParams.update({
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'font.family': 'sans-serif',
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


fig = plt.figure()

# 和上半张图统一配色
sim_color = '#EF4444'   # red
exp_color = '#6366F1'   # blue

plt.subplots_adjust(
    left=0.1,
    right=0.98,
    top=0.92,
    bottom=0.12,
    wspace=0.2,
    hspace=0.15
)

gs = GridSpec(
    2, 4,
    width_ratios=[1, 1, 1, 1],
    height_ratios=[1, 1]
)


# =========================
# Panel a: VQE-RSRA energy
# =========================

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

average = total / 5
std = np.sqrt((total2 - (np.square(average)) * n) / (n - 1))

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


# ===================
# Panel b: VQE energy
# ===================

total = np.zeros(7)
total2 = np.zeros(7)

for i in range(1, n + 1):
    data = np.load(
        "VQEur-Fig4bf/VQEur" + str(i) + ".npz",
        allow_pickle=True
    )['energylist']

    total = total + data
    total2 = total2 + np.square(data)

average = total / 5
std = np.sqrt((total2 - (np.square(average)) * n) / (n - 1))

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


# =========================
# Panel c: QAOA-RSRA energy
# =========================

total = np.zeros(7)
total2 = np.zeros(7)

for i in range(1, n + 1):
    data = np.load(
        "QAOA-Fig4cg/QAOA" + str(i) + ".npz",
        allow_pickle=True
    )['energylist']

    total = total + data
    total2 = total2 + np.square(data)

average = total / 5
std = np.sqrt((total2 - (np.square(average)) * n) / (n - 1))

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


# ====================
# Panel d: QAOA energy
# ====================

total = np.zeros(7)
total2 = np.zeros(7)

for i in range(1, n + 1):
    data = np.load(
        "QAOAur-Fig4dh/QAOAur" + str(i) + ".npz",
        allow_pickle=True
    )['energylist']

    total = total + data
    total2 = total2 + np.square(data)

average = total / 5
std = np.sqrt((total2 - (np.square(average)) * n) / (n - 1))

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


# ======================================
# Panel e: VQE-RSRA final distribution
# ======================================

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
    box_colors=[sim_color, exp_color, exp_color, exp_color, exp_color,
                exp_color, exp_color, exp_color, exp_color, exp_color],
    annotation=r'$p=\frac{4304.0}{9999}=43.04\%$',
    annotation_xy=(0.17, 0.58),
    top_k=10,
)


# ================================
# Panel f: VQE final distribution
# ================================

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
    box_colors=[exp_color, exp_color, exp_color, exp_color, exp_color,
                exp_color, exp_color, exp_color, exp_color, exp_color],
    annotation=None,
    top_k=10,
)


# =======================================
# Panel g: QAOA-RSRA final distribution
# =======================================

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
    box_colors=[sim_color, exp_color, sim_color, exp_color, exp_color,
                exp_color, exp_color, exp_color, exp_color, exp_color],
    annotation=r'$p=\frac{3027.4+610.2}{9999}=36.4\%$',
    annotation_xy=(0.08, 0.58),
    top_k=10,
)


# ================================
# Panel h: QAOA final distribution
# ================================

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
    box_colors=[exp_color, exp_color, exp_color, exp_color, exp_color,
                exp_color, exp_color, exp_color, exp_color, exp_color],
    annotation=None,
    top_k=10,
)


plt.subplots_adjust(
    left=0,
    right=0.99,
    bottom=0.01,
    top=1.01,
    wspace=0.302,
    hspace=0.2
)

plt.savefig("Fig4.pdf", bbox_inches='tight')