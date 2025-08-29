import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
def calculate(label):
    ave=(data0[label]+data1[label]+data2[label]+data3[label]+data4[label]+data5[label]+data6[label]+data7[label]+data8[label]+data9[label])/10
    ss=(data0[label]**2+data1[label]**2+data2[label]**2+data3[label]**2+data4[label]**2+data5[label]**2+data6[label]**2+data7[label]**2+data8[label]**2+data9[label]**2)
    sigma=np.sqrt((ss-10*ave**2)/9)
    error=sigma/np.sqrt(10)
    return ave,error
plt.rcParams.update({
    'font.size': 12,                  # 主字体大小
    'axes.labelsize': 18,              # 轴标签字号
    'axes.titlesize': 18,              # 轴标题字号
    'xtick.labelsize': 18,             # X轴刻度字号
    'ytick.labelsize': 18,             # Y轴刻度字号
    'legend.fontsize': 18,             # 图例字号
    'font.family': 'sans-serif',      # 使用无衬线字体
    'axes.linewidth': 1,            # 坐标轴线宽
    'lines.linewidth': 1.0,           # 数据线宽度
    'xtick.major.width': 1,         # X轴主刻度线宽
    'ytick.major.width': 1,         # Y轴主刻度线宽
    'xtick.major.size': 2.5,          # X轴主刻度长度
    'ytick.major.size': 2.5,          # Y轴主刻度长度
    'xtick.direction': 'in',          # 刻度线方向朝内
    'ytick.direction': 'in',          # 刻度线方向朝内               # 输出分辨率
    'figure.figsize': (8.2, 4),      # 图表尺寸
})
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
colors = ['#EF4444' ,'#6366F1', "#FBB463"]
markers = ['o', 's', 'p']
start=5
final=21
avesrecord=np.zeros((final-start))
aveErecord=np.zeros((final-start))
expsrecord=np.zeros((final-start))
expErecord=np.zeros((final-start))
ransrecord=np.zeros((final-start))
ranErecord=np.zeros((final-start))
avesrecordE=np.zeros((final-start))
aveErecordE=np.zeros((final-start))
expsrecordE=np.zeros((final-start))
expErecordE=np.zeros((final-start))
ransrecordE=np.zeros((final-start))
ranErecordE=np.zeros((final-start))
for n in range(start,final):
    data0=np.load("originaldata/exp"+str(n)+" "+str(0)+".npz")
    data1=np.load("originaldata/exp"+str(n)+" "+str(1)+".npz")
    data2=np.load("originaldata/exp"+str(n)+" "+str(2)+".npz")
    data3=np.load("originaldata/exp"+str(n)+" "+str(3)+".npz")
    data4=np.load("originaldata/exp"+str(n)+" "+str(4)+".npz")
    data5=np.load("originaldata/exp"+str(n)+" "+str(5)+".npz")
    data6=np.load("originaldata/exp"+str(n)+" "+str(6)+".npz")
    data7=np.load("originaldata/exp"+str(n)+" "+str(7)+".npz")
    data8=np.load("originaldata/exp"+str(n)+" "+str(8)+".npz")
    data9=np.load("originaldata/exp"+str(n)+" "+str(9)+".npz")
    avesrecord[n-start],avesrecordE[n-start]=calculate('aves')
    aveErecord[n-start],aveErecordE[n-start]=calculate('aveE')
    expsrecord[n-start],expsrecordE[n-start]=calculate('exps')
    expErecord[n-start],expErecordE[n-start]=calculate('expE')
    ransrecord[n-start],ransrecordE[n-start]=calculate('rans')
    ranErecord[n-start],ranErecordE[n-start]=calculate('ranE')
ax1=plt.subplot(121)
plt.errorbar(range(start,final),avesrecord,avesrecordE,label='Sim.',color=colors[0], capsize=9, elinewidth=2.5,linewidth=2.5,
                capthick=1.5, markersize=13,marker=markers[0])
plt.errorbar(range(start,final),expsrecord,expsrecordE,label='Exp.',color=colors[1], capsize=9, elinewidth=2.5,linewidth=2.5,
                capthick=1.5, markersize=13,marker=markers[1])
plt.errorbar(range(start,final),ransrecord,ransrecordE,label='Ran.',color=colors[2], capsize=9, elinewidth=2.5,linewidth=2.5,
                capthick=1.5, markersize=13,marker=markers[2])
plt.legend(fontsize=16,frameon=False,edgecolor='black',fancybox=False,handlelength=3,columnspacing=0.5,bbox_to_anchor=(0.75,0.83),loc='center')
ax1.set_xlabel(r'$n$')
ax1.set_ylabel('Success possibility')
plt.xticks(np.arange(5,final,3))
plt.xlim(4.3,final-0.3)
plt.title('(a)',x=-0.06,y=0.99,fontsize=18)
ax2=plt.subplot(122)
plt.errorbar(range(start,final),aveErecord,aveErecordE,label='Sim.',color=colors[0], capsize=9, elinewidth=2.5,linewidth=2.5,
                capthick=1.5, markersize=13,marker=markers[0])
plt.errorbar(range(start,final),expErecord,expErecordE,label='Exp.',color=colors[1], capsize=9, elinewidth=2.5,linewidth=2.5,
                capthick=1.5, markersize=13,marker=markers[1])
plt.errorbar(range(start,final),ranErecord,ranErecordE,label='Ran.',color=colors[2], capsize=9, elinewidth=2.5,linewidth=2.5,
                capthick=1.5, markersize=13,marker=markers[2])
plt.xticks(np.arange(5,final,3))
plt.xlim(4.3,final-0.3)
plt.title('(b)',x=-0.06,y=0.99,fontsize=18)

# ax1.text(-0.06, 0.99, '(a)', fontsize=18, transform=ax1.transAxes, 
#             ha='right', va='top', bbox=dict(facecolor='white', edgecolor='none', pad=0))
# ax2.text(-0.06, 0.99, '(b)', fontsize=18, transform=ax2.transAxes, 
#             ha='right', va='top', bbox=dict(facecolor='white', edgecolor='none', pad=0))
ax2.set_xlabel(r'$n$')
ax2.set_ylabel('Energy')
plt.legend(fontsize=16,frameon=False,edgecolor='black',fancybox=False,handlelength=3,columnspacing=0.5,bbox_to_anchor=(0.22,0.83),loc='center')
plt.subplots_adjust(left=0.09,   # 图像左边距（0~1之间，越大间距越大）
                    right=0.995,  # 图像右边距（0~1之间，越小间距越大）
                    top=0.938,    # 图像上边距（0~1之间，越小间距越大）
                    bottom=0.15)
plt.savefig('Fig5.pdf')
