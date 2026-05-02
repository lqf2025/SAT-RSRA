import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['mathtext.fontset'] = 'custom'    # Use a custom mathtext configuration.
#mpl.rcParams['mathtext.rm'] = 'Helvetica'
mpl.rcParams['mathtext.rm'] = 'Cambria'       # Set roman mathtext font.
mpl.rcParams['mathtext.it'] = 'Computer Modern:italic'  # Set italic mathtext font.
mpl.rcParams['mathtext.bf'] = 'Computer Modern:bold'    # Set bold mathtext font.
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, bm} \boldmath"
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

def convertstring(i):
    """Convert i to a 7-bit binary string (offset by 128) for labeling."""
    str = bin(i + 128)[3:]
    return str

def convertstring2(i):
    """Convert i to an 8-bit binary string (offset by 256) for labeling."""
    str = bin(i + 256)[3:]
    return str

def convertstringlist(array):
    """Vectorized helper to convert a list of indices into 7-bit binary labels."""
    label = []
    for i in array:
        str = bin(i + 128)[3:]
        label.append(str)
    return label

def convertstringlist2(array):
    """Vectorized helper to convert a list of indices into 8-bit binary labels."""
    label = []
    for i in array:
        str = bin(i + 256)[3:]
        label.append(str)
    return label

def setup_subplot(ax, title):
    """Apply consistent axis styling and add the panel title label."""
    ax.spines[:].set_linewidth(1)
    ax.tick_params(axis='both', which='major', labelsize=10, width=0.5, length=3)
    ax.tick_params(axis='both', which='minor', length=0)
    #ax.set_xlabel('State', fontsize=9, labelpad=2)
    ax.set_ylabel('Frequency', fontsize=18, labelpad=2)
    ax.text(-0.01, 1.08, title, fontsize=18, transform=ax.transAxes,
            ha='right', va='top', bbox=dict(facecolor='white', edgecolor='none', pad=0))

def plot_panel(ax, x, average, std, dataacc, titlein, label, loc, mov):
    """Plot energy vs iteration with error bars for experiments and a dashed simulation curve."""
    # Data loading logic intentionally left unchanged.
    plt.errorbar(
        x, average, std, fmt='o-', capsize=9, elinewidth=2.5, linewidth=2.5, markersize=11,
        label='Exp.', capthick=1.5, color='#6366F1'
    )
    plt.plot(
        x, dataacc, linewidth=2.5, linestyle='--', marker='s',
        markersize=13, label='Sim.', color='#EF4444'
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.spines[:].set_linewidth(1)
    ax.tick_params(axis='both', which='major', width=0.6, length=2.5)
    ax.set_xlabel('Iteration', labelpad=2, fontsize=18)
    ax.set_ylabel('Energy', labelpad=2, fontsize=18)
    ax.text(-0.01, 1.08, titlein, transform=ax.transAxes,
            ha='right', va='top', fontsize=18)
    plt.legend(
        [handles[1], handles[0]], [labels[1], labels[0]],
        frameon=False, handletextpad=0.4, handlelength=2.8, fontsize=16, loc=loc, bbox_to_anchor=mov
    )
    ax.tick_params(axis='x', which='minor', length=0)
    ax.tick_params(axis='y', which='minor', length=0)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, labeltop=False, labelsize=15)
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False, labelsize=15)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.text(0.65, 0.95, label, ha='center', va='center',
            transform=ax.transAxes, fontsize=16)

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
r, b = '#E40043', '#00A2E7'
plt.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.12,
                    wspace=0.2, hspace=0.15)
gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1])

n = 5
x = np.array(range(0, 7))
total = np.zeros(7)
total2 = np.zeros(7)
for i in range(1, n + 1):
    data = np.load("VQE-Fig4ae/VQE" + str(i) + ".npz", allow_pickle=True)['energylist']
    if (i <= 3):
        dataapp = np.load("VQE-Fig4ae/VQEappend" + str(i) + ".npz", allow_pickle=True)['energylist'][1]
        data = np.append(data, dataapp)
    total = total + data
    total2 = total2 + np.square(data)
average = total / 5
std = np.sqrt((total2 - (np.square(average)) * n) / (n - 1))
dataacc = np.load("VQE-Fig4ae/VQE.npz", allow_pickle=True)['energylist']
plot_panel(plt.subplot(gs[0, 0]), x, average, std, dataacc, "a", "VQE-RSRA", 'center right', (1.05, 0.5))
plt.ylim(-0.05, 1.28)

total = np.zeros(7)
total2 = np.zeros(7)
for i in range(1, n + 1):
    data = np.load("VQEur-Fig4bf/VQEur" + str(i) + ".npz", allow_pickle=True)['energylist']
    total = total + data
    total2 = total2 + np.square(data)
average = total / 5
std = np.sqrt((total2 - (np.square(average)) * n) / (n - 1))
dataacc = np.load("VQEur-Fig4bf/VQEur.npz", allow_pickle=True)['energylist']
plot_panel(plt.subplot(gs[0, 1]), x, average, std, dataacc, "b", "VQE", 'center right', (1.05, 0.5))
plt.ylim(3.4, 5.1)

ax = plt.subplot(gs[1, 0])
ax.text(0.65, 0.95, 'VQE-RSRA', fontsize=16, transform=ax.transAxes, ha='center', va='center')
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=20)
ax.set_ylabel('Frequency', fontsize=20)
#ax.set_xlabel('State', fontsize=20)
setup_subplot(ax, 'e')
ax.tick_params(axis='x', which='minor', length=0)
ax.tick_params(axis='y', which='minor', length=0)
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, labeltop=False, labelsize=12.5)
ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False, labelsize=15)

color = [r, b, b, b, b, b, b, b, b, b]
replacelist = ['#EF4444', '#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1']
data1 = np.load("VQE-Fig4ae/VQEappend1.npz", allow_pickle=True)['finaldict'].item()
data2 = np.load("VQE-Fig4ae/VQEappend2.npz", allow_pickle=True)['finaldict'].item()
data3 = np.load("VQE-Fig4ae/VQEappend3.npz", allow_pickle=True)['finaldict'].item()
data4 = np.load("VQE-Fig4ae/VQE4.npz", allow_pickle=True)['finaldict'].item()
data5 = np.load("VQE-Fig4ae/VQE5.npz", allow_pickle=True)['finaldict'].item()

total1 = np.zeros((128))
total2 = np.zeros((128))
for i in range(128):
    s = str(convertstring(i))
    if s in data1:
        total1[i] = total1[i] + data1[s]
        total2[i] = total2[i] + data1[s] ** 2
    if s in data2:
        total1[i] = total1[i] + data2[s]
        total2[i] = total2[i] + data2[s] ** 2
    if s in data3:
        total1[i] = total1[i] + data3[s]
        total2[i] = total2[i] + data3[s] ** 2
    if s in data4:
        total1[i] = total1[i] + data4[s]
        total2[i] = total2[i] + data4[s] ** 2
    if s in data5:
        total1[i] = total1[i] + data5[s]
        total2[i] = total2[i] + data5[s] ** 2
ave = total1 / 5
var = np.sqrt((total2 - (np.square(ave)) * n) / (n - 1))

a = np.array(range(128))
total = zip(a, ave, var)
totalsort = sorted(total, key=lambda x: x[1], reverse=True)
totalsort = totalsort[0:10]
a1, a2, a3 = zip(*totalsort)
range2 = np.array(range(10))
a1 = convertstringlist(a1)
print(a2[0])

bars = ax.bar(a1, a2, color=color, alpha=0.9)
for bar, error, color2 in zip(bars, a3, replacelist):
    plt.errorbar(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        yerr=error,
        color=color2,
        capsize=5,
        elinewidth=2.5
    )
ax.text(1.35, 3400, 'p' + r'$=\frac{4304.0}{9999}$' + r'$=43.04\%$', fontsize=15, color='black')
plt.xticks(rotation=70)

ax = plt.subplot(gs[1, 1])
ax.text(0.65, 0.95, 'VQE', fontsize=16, transform=ax.transAxes, ha='center', va='center')
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=20)
ax.set_ylabel('Frequency', fontsize=20)
#ax.set_xlabel('State', fontsize=20)
setup_subplot(ax, 'f')
ax.tick_params(axis='x', which='minor', length=0)
ax.tick_params(axis='y', which='minor', length=0)
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, labeltop=False, labelsize=12.5)
ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False, labelsize=15)

color = [b, b, b, b, b, b, b, b, b, b]
replacelist = ['#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1']
data1 = np.load("VQEur-Fig4bf/VQEur1.npz", allow_pickle=True)['finaldict'].item()
data2 = np.load("VQEur-Fig4bf/VQEur2.npz", allow_pickle=True)['finaldict'].item()
data3 = np.load("VQEur-Fig4bf/VQEur3.npz", allow_pickle=True)['finaldict'].item()
data4 = np.load("VQEur-Fig4bf/VQEur4.npz", allow_pickle=True)['finaldict'].item()
data5 = np.load("VQEur-Fig4bf/VQEur5.npz", allow_pickle=True)['finaldict'].item()

total1 = np.zeros((256))
total2 = np.zeros((256))
for i in range(256):
    s = str(convertstring2(i))
    if s in data1:
        total1[i] = total1[i] + data1[s]
        total2[i] = total2[i] + data1[s] ** 2
    if s in data2:
        total1[i] = total1[i] + data2[s]
        total2[i] = total2[i] + data2[s] ** 2
    if s in data3:
        total1[i] = total1[i] + data3[s]
        total2[i] = total2[i] + data3[s] ** 2
    if s in data4:
        total1[i] = total1[i] + data4[s]
        total2[i] = total2[i] + data4[s] ** 2
    if s in data5:
        total1[i] = total1[i] + data5[s]
        total2[i] = total2[i] + data5[s] ** 2
ave = total1 / 5
var = np.sqrt((total2 - (np.square(ave)) * n) / (n - 1))

a = np.array(range(256))
total = zip(a, ave, var)
totalsort = sorted(total, key=lambda x: x[1], reverse=True)
totalsort = totalsort[0:10]
a1, a2, a3 = zip(*totalsort)
range2 = np.array(range(10))
a1 = convertstringlist2(a1)

bars = ax.bar(a1, a2, color=color, alpha=0.9)
for bar, error, color2 in zip(bars, a3, replacelist):
    plt.errorbar(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        yerr=error,
        color=color2,
        capsize=5,
        elinewidth=2.5
    )
plt.xticks(rotation=70)

total = np.zeros(7)
total2 = np.zeros(7)
n = 5
x = np.array(range(0, 7))
for i in range(1, n + 1):
    data = np.load("QAOA-Fig4cg/QAOA" + str(i) + ".npz", allow_pickle=True)['energylist']
    total = total + data
    total2 = total2 + np.square(data)
average = total / 5
std = np.sqrt((total2 - (np.square(average)) * n) / (n - 1))
dataacc = np.load("QAOA-Fig4cg/QAOA.npz", allow_pickle=True)['energylist']
plot_panel(plt.subplot(gs[0, 2]), x, average, std, dataacc, "c", "QAOA-RSRA", 'lower left', (-0.02, -0.06))

total = np.zeros(7)
total2 = np.zeros(7)
for i in range(1, n + 1):
    data = np.load("QAOAur-Fig4dh/QAOAur" + str(i) + ".npz", allow_pickle=True)['energylist']
    total = total + data
    total2 = total2 + np.square(data)
average = total / 5
std = np.sqrt((total2 - (np.square(average)) * n) / (n - 1))
dataacc = np.load("QAOAur-Fig4dh/QAOAur.npz", allow_pickle=True)['energylist']
plot_panel(plt.subplot(gs[0, 3]), x, average, std, dataacc, "d", "QAOA", 'lower left', (-0.02, -0.06))
plt.yticks([4.7, 4.9, 5.1, 5.3])

ax = plt.subplot(gs[1, 3])
ax.text(0.65, 0.95, 'QAOA', fontsize=16, transform=ax.transAxes, ha='center', va='center')
ax.set_ylabel('Frequency', fontsize=15)
#ax.set_xlabel('State', fontsize=20)
setup_subplot(ax, 'h')
ax.tick_params(axis='x', which='minor', length=0)
ax.tick_params(axis='y', which='minor', length=0)
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, labeltop=False, labelsize=12.5)
ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False, labelsize=15)

color = [b, b, b, b, b, b, b, b, b, b]
replacelist = ['#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1']
data1 = np.load("QAOAur-Fig4dh/QAOAur1.npz", allow_pickle=True)['finaldict'].item()
data2 = np.load("QAOAur-Fig4dh/QAOAur2.npz", allow_pickle=True)['finaldict'].item()
data3 = np.load("QAOAur-Fig4dh/QAOAur3.npz", allow_pickle=True)['finaldict'].item()
data4 = np.load("QAOAur-Fig4dh/QAOAur4.npz", allow_pickle=True)['finaldict'].item()
data5 = np.load("QAOAur-Fig4dh/QAOAur5.npz", allow_pickle=True)['finaldict'].item()

total1 = np.zeros((256))
total2 = np.zeros((256))
for i in range(256):
    s = str(convertstring2(i))
    if s in data1:
        total1[i] = total1[i] + data1[s]
        total2[i] = total2[i] + data1[s] ** 2
    if s in data2:
        total1[i] = total1[i] + data2[s]
        total2[i] = total2[i] + data2[s] ** 2
    if s in data3:
        total1[i] = total1[i] + data3[s]
        total2[i] = total2[i] + data3[s] ** 2
    if s in data4:
        total1[i] = total1[i] + data4[s]
        total2[i] = total2[i] + data4[s] ** 2
    if s in data5:
        total1[i] = total1[i] + data5[s]
        total2[i] = total2[i] + data5[s] ** 2
ave = total1 / 5
var = np.sqrt((total2 - (np.square(ave)) * n) / (n - 1))

a = np.array(range(256))
total = zip(a, ave, var)
totalsort = sorted(total, key=lambda x: x[1], reverse=True)
totalsort = totalsort[0:10]
a1, a2, a3 = zip(*totalsort)
range2 = np.array(range(10))
a1 = convertstringlist2(a1)

bars = ax.bar(a1, a2, color=color, alpha=0.9)
for bar, error, color2 in zip(bars, a3, replacelist):
    plt.errorbar(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        yerr=error,
        color=color2,
        capsize=5,
        elinewidth=2.5,
    )
plt.xticks(rotation=70)

ax = plt.subplot(gs[1, 2])
ax.text(0.65, 0.95, 'QAOA-RSRA', fontsize=16, transform=ax.transAxes, ha='center', va='center')
ax.set_ylabel('Frequency', fontsize=15)
#ax.set_xlabel('State', fontsize=20)
setup_subplot(ax, 'g')
ax.tick_params(axis='x', which='minor', length=0)
ax.tick_params(axis='y', which='minor', length=0)
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, labeltop=False, labelsize=12.5)
ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False, labelsize=15)

color = [r, b, r, b, b, b, b, b, b, b]
replacelist = ['#EF4444', '#6366F1', '#EF4444', '#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1', '#6366F1']
data1 = np.load("QAOA-Fig4cg/QAOA1.npz", allow_pickle=True)['finaldict'].item()
data2 = np.load("QAOA-Fig4cg/QAOA2.npz", allow_pickle=True)['finaldict'].item()
data3 = np.load("QAOA-Fig4cg/QAOA3.npz", allow_pickle=True)['finaldict'].item()
data4 = np.load("QAOA-Fig4cg/QAOA4.npz", allow_pickle=True)['finaldict'].item()
data5 = np.load("QAOA-Fig4cg/QAOA5.npz", allow_pickle=True)['finaldict'].item()

total1 = np.zeros((128))
total2 = np.zeros((128))
for i in range(128):
    s = str(convertstring(i))
    if s in data1:
        total1[i] = total1[i] + data1[s]
        total2[i] = total2[i] + data1[s] ** 2
    if s in data2:
        total1[i] = total1[i] + data2[s]
        total2[i] = total2[i] + data2[s] ** 2
    if s in data3:
        total1[i] = total1[i] + data3[s]
        total2[i] = total2[i] + data3[s] ** 2
    if s in data4:
        total1[i] = total1[i] + data4[s]
        total2[i] = total2[i] + data4[s] ** 2
    if s in data5:
        total1[i] = total1[i] + data5[s]
        total2[i] = total2[i] + data5[s] ** 2
ave = total1 / 5
var = np.sqrt((total2 - (np.square(ave)) * n) / (n - 1))

a = np.array(range(128))
total = zip(a, ave, var)
totalsort = sorted(total, key=lambda x: x[1], reverse=True)
totalsort = totalsort[0:10]
a1, a2, a3 = zip(*totalsort)
print(a2[0], a2[2])
range2 = np.array(range(10))
a1 = convertstringlist(a1)

bars = ax.bar(a1, a2, color=color)
for bar, error, color2 in zip(bars, a3, replacelist):
    plt.errorbar(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        yerr=error,
        color=color2,
        capsize=5,
        elinewidth=2.5,
        capthick=1.5,
    )
ax.text(0.75, 2600, ' p' + r'$=\frac{3027.4+610.2}{9999}$' + r'$=36.4\%$', fontsize=15, color='black')
plt.xticks(rotation=70)

plt.subplots_adjust(
    left=0,
    right=0.99,
    bottom=0.01,
    top=1.01,
    wspace=0.3,
    hspace=0.2
)
plt.savefig("Fig4.pdf", bbox_inches='tight')
