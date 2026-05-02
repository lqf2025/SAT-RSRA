import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

def mean_boot(x, ci=0.95, B=2000):
    """Return a percentile-bootstrap confidence interval for the sample mean."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    a = 1.0 - ci
    rng = np.random.default_rng()
    idx = rng.integers(0, n, size=(B, n))
    boot = x[idx].mean(axis=1)
    lo = np.quantile(boot, a / 2.0)
    hi = np.quantile(boot, 1.0 - a / 2.0)
    return float(lo), float(hi)

plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Cambria'
mpl.rcParams['mathtext.it'] = 'Computer Modern:italic'
mpl.rcParams['mathtext.bf'] = 'Computer Modern:bold'
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, bm} \boldmath"
plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 8, 'axes.titlesize': 11,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'axes.linewidth': 1, 'lines.linewidth': 1.0, 'xtick.direction': 'in',
    'ytick.direction': 'in', 'pdf.fonttype': 42,
})

fig = plt.figure(figsize=(6, 4))
colors = ['#1f77b4','#ff7f0e', '#2ca02c','#d62728' , '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

def plotn(m, n, name, label, number):
    """Plot the mean and bootstrap CI of per-instance samples over a selected n-range."""
    number = 8 - number
    npzfile = np.load(name, allow_pickle=True)
    ns = np.asarray(npzfile['ns'], dtype=int)
    vars_all = np.asarray(npzfile['vars_all'], dtype=float)

    mask = (ns >= m) & (ns < n)
    x = ns[mask]
    samples = vars_all[mask, :]

    y = samples.mean(axis=1)

    lows, highs = [], []
    for row in samples:
        lo, hi = mean_boot(row, ci=0.95, B=2000)
        lows.append(lo)
        highs.append(hi)

    lows = np.array(lows)
    highs = np.array(highs)

    yerr = np.vstack([y - lows, highs - y])

    linecolor = mcolors.to_rgba(colors[number], alpha=1.0)
    marker_color = mcolors.to_rgba(colors[number], alpha=0.8)

    plt.plot(
        x, y,
        label=fr'$m/n={str(label)}$',
        linewidth=2.5,
        markevery=4,
        marker=markers[number],
        markersize=9,
        color=linecolor,
        markerfacecolor=marker_color,
        markeredgewidth=0.8,
        markeredgecolor=linecolor
    )

    plt.errorbar(
        x, y, yerr=yerr,
        fmt='none',
        ecolor=linecolor,
        elinewidth=1.2,
        capsize=2.5,
        capthick=1.2,
        alpha=0.65,
        label='_nolegend_'
    )

klist = [0.476, 0.526, 0.576, 0.626, 0.676, 0.726]
for i in range(6):
    plotn(10, 30, "BP/BP" + str(klist[i]) + ".npz", klist[i], i)

plt.xlabel(r'$n$', fontsize=15, labelpad=0)
plt.ylabel('Variance of ' + r'$\frac{\partial\langle H \rangle}{\partial\theta_v}$', fontsize=15, labelpad=2)
plt.xticks([i for i in np.arange(10, 30, 4)], fontsize=15)
plt.xlim(9.5, 29.5)
plt.ylim(-0.01, 4.2)
plt.yticks(fontsize=15)
plt.tick_params(axis='y', which='both', left=True, right=False)
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

legend1 = plt.legend(ncol=3, fontsize=8.5, bbox_to_anchor=(0.454, 0.94), loc='center', frameon=False)
line1 = plt.axhline(1/32, color='#D2D2D2', linewidth=2.5, linestyle='--')
legend2 = plt.legend([line1], [r'y=$\frac{1}{32}$'], loc='center', bbox_to_anchor=(0.124, 0.84), fontsize=8, frameon=False)

fig.add_artist(legend1)
fig.add_artist(legend2)
plt.subplots_adjust(left=0.12, right=0.99, top=0.99, bottom=0.12)
plt.savefig('FigS20.pdf')
