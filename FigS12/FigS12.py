import numpy as np
from random import randint
import math,random
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['mathtext.fontset'] = 'custom'    # 设置公式字体为 Computer Modern
#mpl.rcParams['mathtext.rm'] = 'Helvetica'  
mpl.rcParams['mathtext.rm'] = 'Cambria'       # 设置数学公式中的普通文本为 Cambria
mpl.rcParams['mathtext.it'] = 'Computer Modern:italic'  # 设置斜体为 Cambria
mpl.rcParams['mathtext.bf'] = 'Computer Modern:bold'
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, bm} \boldmath"
plt.rcParams['axes.labelweight'] = 'bold'  
plt.rcParams['axes.titleweight'] = 'bold' 
fig=plt.figure(figsize=(6,4))
colors = ['#1f77b4','#ff7f0e', '#2ca02c','#d62728' , '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
plt.rcParams.update({
    'font.size': 10,                  # 主字体大小
    'axes.labelsize': 8,              # 轴标签字号
    'axes.titlesize': 11,              # 轴标题字号
    'xtick.labelsize': 11,             # X轴刻度字号
    'ytick.labelsize': 11,             # Y轴刻度字号
    'legend.fontsize': 11,             # 图例字号
    'font.family': 'sans-serif',      # 使用无衬线字体
    'font.sans-serif': ['Arial'],     # 指定Arial字体
    'axes.linewidth': 1,            # 坐标轴线宽
    'lines.linewidth': 1.0,           # 数据线宽度
    'xtick.major.width': 1,         # X轴主刻度线宽
    'ytick.major.width': 1,         # Y轴主刻度线宽
    'xtick.major.size': 2.5,          # X轴主刻度长度
    'ytick.major.size': 2.5,          # Y轴主刻度长度
    'xtick.direction': 'in',          # 刻度线方向朝内
    'ytick.direction': 'in',          # 刻度线方向朝内               # 输出分辨率
    'pdf.fonttype': 42,               # 确保输出可编辑文本
})
def plotn(m,n,name,label,number):
    number=8-number
    npzfile=np.load(name)
    varn=npzfile['averagvar']
    x=range(m,n)
    y=varn[m:n]
    linecolor=mcolors.to_rgba(colors[number], alpha=1)
    marker_color = mcolors.to_rgba(colors[number], alpha=0.8)
    #plt.plot(x,y,label=r'$\frac{m}{n}=$'+str(label),linewidth=2.5,markevery=5,marker=markers[number],markersize=7,color=linecolor,markerfacecolor=marker_color,markeredgewidth=0)
    plt.plot(x, y,
             label=fr'$m/n={str(label)}$',
             linewidth=2.5,  # 适当减小线宽
             markevery=4,
             marker=markers[number],
             markersize=10,   # 适当减小标记尺寸
             color=linecolor,
             markerfacecolor=marker_color,
             markeredgewidth=0.8,  # 添加标记边框
             markeredgecolor=linecolor)
klist=[0.476,0.526,0.576,0.626,0.676,0.726]
for i in range(6):
    plotn(10,30,"BP/BP"+str(klist[i])+".npz",klist[i],i)
plt.xlabel(r'$n$',fontsize=15,labelpad=0)
plt.ylabel('Variance of '+r'$\frac{\partial\langle H \rangle}{\partial\theta_v}$',fontsize=15,labelpad=2)
plt.xticks([i for i in np.arange(10,30,4)])
#plt.yticks([i for i in np.arange(0.5,3,0.5)])
plt.xticks(fontsize=15)
#plt.ylim(0,2.8)
plt.xlim(9.5,29.5)
plt.yticks(fontsize=15)
plt.tick_params(axis='x', which='minor', length=0)
plt.tick_params(axis='y', which='minor', length=0)
plt.tick_params(axis='y', which='both', left=True, right=False)
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
legend1=plt.legend(ncol=3,fontsize=12.8,bbox_to_anchor=(0.5, 0.665),loc='center',frameon=False,edgecolor='black',fancybox=False,handlelength=1.5,columnspacing=0.5)
line1=plt.axhline(1/32,color='#D2D2D2',linewidth=2.5,label=r'$\frac{1}{32}$',linestyle='--')
legend2 = plt.legend([line1], [r'y=$\frac{1}{32}$'], loc='center', bbox_to_anchor=(0.124, 0.95),fontsize=12.8,handlelength=1.5,frameon=False)
fig.add_artist(legend1)
plt.margins(x=0.05)
fig.add_artist(legend2)
plt.subplots_adjust(left=0.12,   # 图像左边距（0~1之间，越大间距越大）
                    right=0.99,  # 图像右边距（0~1之间，越小间距越大）
                    top=0.99,    # 图像上边距（0~1之间，越小间距越大）
                    bottom=0.1)
plt.savefig('FigS12.pdf')
