import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def mean_sem(datasets, key):
    """Return (mean, SEM) of field `key` across multiple npz runs."""
    vals = np.array([float(np.asarray(d[key])) for d in datasets], dtype=float)
    mean = vals.mean()
    sem = vals.std(ddof=1) / np.sqrt(vals.size)
    return mean, sem


def plot_err(ax, xs, ys, yerrs, label, color, marker):
    """Standardized errorbar call."""
    ax.errorbar(
        xs, ys, yerr=yerrs,
        label=label, color=color, marker=marker,
        capsize=9, elinewidth=2.5, linewidth=2.5,
        capthick=1.5, markersize=13
    )


# -----------------------------
# Matplotlib configuration
# -----------------------------
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'font.family': 'sans-serif',
    'axes.linewidth': 1,
    'lines.linewidth': 1.0,
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.figsize': (8.2, 4),
})

plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Cambria'
mpl.rcParams['mathtext.it'] = 'Computer Modern:italic'
mpl.rcParams['mathtext.bf'] = 'Computer Modern:bold'
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, bm} \boldmath"
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'


# -----------------------------
# Data aggregation
# -----------------------------
colors = ['#EF4444', '#6366F1', "#FBB463"]
markers = ['o', 's', 'p']

start = 5
final = 23
xs = np.arange(start, final, dtype=int)
N = xs.size
runs = 10

sim_s, sim_sE = np.zeros(N), np.zeros(N)
exp_s, exp_sE = np.zeros(N), np.zeros(N)
ran_s, ran_sE = np.zeros(N), np.zeros(N)

sim_E, sim_EE = np.zeros(N), np.zeros(N)
exp_E, exp_EE = np.zeros(N), np.zeros(N)
ran_E, ran_EE = np.zeros(N), np.zeros(N)

for i, n in enumerate(xs):
    datasets = [np.load(f"exptotal/exp{n} {r}.npz") for r in range(runs)]

    sim_s[i], sim_sE[i] = mean_sem(datasets, 'aves')
    sim_E[i], sim_EE[i] = mean_sem(datasets, 'aveE')

    exp_s[i], exp_sE[i] = mean_sem(datasets, 'exps')
    exp_E[i], exp_EE[i] = mean_sem(datasets, 'expE')

    ran_s[i], ran_sE[i] = mean_sem(datasets, 'rans')
    ran_E[i], ran_EE[i] = mean_sem(datasets, 'ranE')


# -----------------------------
# Plotting
# -----------------------------
xticks = np.arange(start, final, 4, dtype=int)

fig = plt.figure()

ax1 = plt.subplot(121)
plot_err(ax1, xs, sim_s, sim_sE, 'Sim.', colors[0], markers[0])
plot_err(ax1, xs, exp_s, exp_sE, 'Exp.', colors[1], markers[1])
plot_err(ax1, xs, ran_s, ran_sE, 'Ran.', colors[2], markers[2])
ax1.set_xlabel(r'$n$')
ax1.set_ylabel('Success probability')
ax1.set_xticks(xticks)
ax1.set_xlim(4.3, final - 0.3)
ax1.set_title('(a)', x=-0.06, y=0.99, fontsize=18)
ax1.legend(
    fontsize=16, frameon=False, handlelength=3, columnspacing=0.5,
    bbox_to_anchor=(0.75, 0.83), loc='center'
)

ax2 = plt.subplot(122)
plot_err(ax2, xs, sim_E, sim_EE, 'Sim.', colors[0], markers[0])
plot_err(ax2, xs, exp_E, exp_EE, 'Exp.', colors[1], markers[1])
plot_err(ax2, xs, ran_E, ran_EE, 'Ran.', colors[2], markers[2])
ax2.set_xlabel(r'$n$')
ax2.set_ylabel('Energy')
ax2.set_xticks(xticks)
ax2.set_xlim(4.3, final - 0.3)
ax2.set_title('(b)', x=-0.06, y=0.99, fontsize=18)
ax2.legend(
    fontsize=16, frameon=False, handlelength=3, columnspacing=0.5,
    bbox_to_anchor=(0.22, 0.83), loc='center'
)

plt.subplots_adjust(
    left=0.09,
    right=0.995,
    top=0.938,
    bottom=0.15
)
plt.savefig('Fig5.pdf')
