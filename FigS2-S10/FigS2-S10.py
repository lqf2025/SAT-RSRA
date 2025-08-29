import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter, ScalarFormatter
import matplotlib as mpl
plt.rcParams.update({
    'xtick.direction': 'in',          # 刻度线方向朝内
    'ytick.direction': 'in'          # 刻度线方向朝内 
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
def calculateR2(x,data):
    slope, intercept = np.polyfit(x, data, 1)
    datamean=np.mean(data)
    TSS = np.sum((data - datamean) ** 2)
    RSS = np.sum((data - slope*x-intercept) ** 2)
    r2 = 1 - (RSS / TSS)
    return r2
def strround(x,i):
    return f"{x:.{i}f}"
def strchange(s):
    if(float(s)<0):
        return str(s)
    else:
        return '+'+str(s)
def drawclassical(datain,i):
    global handles2
    global labels2
    data=datain.copy()
    x=range(70,121)
    slope, intercept = np.polyfit(x, data, 1)
    datamean=np.mean(data)
    TSS = np.sum((data - datamean) ** 2)
    RSS = np.sum((data - slope*x-intercept) ** 2)
    r2 = 1 - (RSS / TSS)
    plt.plot(x, np.exp(slope * x + intercept),linestyle='--',linewidth=2.5,color=g2[i])
    plt.plot(x,np.exp(data),marker=g2[i+int(len(g2)/2)],linewidth=0,markersize=8,color=g2[i],markerfacecolor=g2[i],markeredgewidth=0,markevery=1)
    plt.tick_params(axis='x', which='minor', length=0)
    #plt.tick_params(axis='y', which='minor', length=0)
    plt.tick_params(axis='both', which='both', right=False, top=False, labelright=False, labeltop=False)
    label=str(methodlabel[i])+': '+str(strround(np.e**intercept,3))+r'$\times{' + str(strround(np.e**slope,3)) + '}^n$'+', '+r'$R^2=$'+strround(r2,3)
    custom_handles = matplotlib.lines.Line2D(
            [0], [0],
            marker=g2[i+int(len(g2)/2)],
            markersize=8,
            markerfacecolor=g2[i],
            markeredgewidth=0,
            linestyle='--',
            color=g2[i],
            linewidth=2,
            label=label
        )
    handles2.append(custom_handles)
    labels2.append(label)
def drawQAA(layernum):
    handles=[]
    labels=[]
    data=np.load('QAA-FigS2-S10b/QAAsinglell'+str(k)+'.npz')['success']
    min=45
    max=65
    max0=max
    i=0
    x=range(min,max)
    slope, intercept = np.polyfit(x, data[min-5:max-5], 1)
    datamean=np.mean(data[min-5:])
    TSS = np.sum((data[min-5:max-5] - datamean) ** 2)
    RSS = np.sum((data[min-5:max-5] - slope*x-intercept) ** 2)
    r2 = 1 - (RSS / TSS)
    x=range(5,max)
    plt.plot(x, np.exp(slope * x + intercept),linestyle='--',linewidth=2,color=g[i])
    plt.plot(x,np.exp(data[:max-5]),marker=g[i+int(len(g)/2)],linewidth=0,markersize=6,color=g[i],markerfacecolor=g[i],markeredgewidth=0,markevery=1)
    label='QAA-RSRA: '+str(strround(np.e**intercept,3))+r'$\times{' + str(strround(np.e**slope,3)) + '}^n$'+', '+r'$R^2=$'+strround(r2,3)
    custom_handles = matplotlib.lines.Line2D(
            [0], [0],
            marker=g[i+int(len(g)/2)],
            markersize=6,
            markerfacecolor=g[i],
            markeredgewidth=0,
            linestyle='--',
            color=g[i],
            linewidth=2,
            label=label
        )
    handles.append(custom_handles)
    labels.append(label)
    data=np.load('QAAur-FigS2-S10b/QAAscaleur'+str(k)+'.npz')['sloperecordur'][:,0]
    i=1
    min=5
    max=15
    x=range(min,max)
    slope, intercept = np.polyfit(x, data[min-5:], 1)
    datamean=np.mean(data[min-5:])
    TSS = np.sum((data[min-5:] - datamean) ** 2)
    RSS = np.sum((data[min-5:] - slope*x-intercept) ** 2)
    r2 = 1 - (RSS / TSS)
    x=range(5,max)
    plt.plot(x, np.exp(slope * x + intercept),linestyle='--',linewidth=2.5,color=g[i])
    plt.plot(x,np.exp(data),marker=g[i+int(len(g)/2)],linewidth=0,markersize=6,color=g[i],markerfacecolor=g[i],markeredgewidth=0,markevery=1)
    label='QAA: '+str(strround(np.e**intercept,3))+r'$\times{' + str(strround(np.e**slope,3)) + '}^n$'+', '+r'$R^2=$'+strround(r2,3)
    custom_handles = matplotlib.lines.Line2D(
            [0], [0],
            marker=g[i+int(len(g)/2)],
            markersize=6,
            markerfacecolor=g[i],
            markeredgewidth=0,
            linestyle='--',
            color=g[i],
            linewidth=2.5,
            label=label
        )
    handles.append(custom_handles)
    labels.append(label)
    plt.xticks(np.arange(5,max0+5,5))
    plt.xticks(fontsize=8.5)
    plt.yscale('log')
    plt.xlabel("n",fontsize=10.5,labelpad=0)
    plt.ylabel('Time complexity',fontsize=10.5)
    plt.tick_params(axis='x', which='minor', length=0)
    plt.tick_params(axis='y', which='both', labelsize=7.5)
    plt.tick_params(axis='both', which='both', right=False, top=False, labelright=False, labeltop=False)
    plt.legend(handles,labels,frameon=False,edgecolor='black',fancybox=False,fontsize=7.5,title_fontsize=9,title=str(layernum)+' layer QAA, m/n='+str(k),bbox_to_anchor=(0.5, 0.85),loc='center',handletextpad=0.4,handlelength=3.2) 
def drawQAOA(layernum):
    handles=[]
    labels=[]
    data=np.load('QAOA-FigS2-S10c/QAOAsingle'+str(k)+'.npz')['success']
    #data=data[:,layernum]
    min=20
    max=40
    i=0
    x=range(min,max)
    slope, intercept = np.polyfit(x, data[min-5:], 1)
    datamean=np.mean(data[min-5:])
    TSS = np.sum((data[min-5:] - datamean) ** 2)
    RSS = np.sum((data[min-5:] - slope*x-intercept) ** 2)
    r2 = 1 - (RSS / TSS)
    x=range(5,max)
    plt.plot(x, np.exp(slope * x + intercept),linestyle='--',linewidth=2,color=g[i])
    plt.plot(x,np.exp(data),marker=g[i+int(len(g)/2)],linewidth=0,markersize=6,color=g[i],markerfacecolor=g[i],markeredgewidth=0,markevery=1)
    label='QAOA-RSRA: '+str(strround(np.e**intercept,3))+r'$\times{' + str(strround(np.e**slope,3)) + '}^n$'+', '+r'$R^2=$'+strround(r2,3)
    custom_handles = matplotlib.lines.Line2D(
            [0], [0],
            marker=g[i+int(len(g)/2)],
            markersize=6,
            markerfacecolor=g[i],
            markeredgewidth=0,
            linestyle='--',
            color=g[i],
            linewidth=2,
            label=label
        )
    handles.append(custom_handles)
    labels.append(label)
    data=np.load('QAOAur-FigS2-S10c/QAOAurscale'+str(k)+'.npz')['slopeurrecord'][:,0]
    i=1
    min=5
    max=12
    x=range(min,max)
    slope, intercept = np.polyfit(x, data[min-5:], 1)
    datamean=np.mean(data[min-5:])
    TSS = np.sum((data[min-5:] - datamean) ** 2)
    RSS = np.sum((data[min-5:] - slope*x-intercept) ** 2)
    r2 = 1 - (RSS / TSS)
    x=range(5,max)
    plt.plot(x, np.exp(slope * x + intercept),linestyle='--',linewidth=2.5,color=g[i])
    plt.plot(x,np.exp(data),marker=g[i+int(len(g)/2)],linewidth=0,markersize=6,color=g[i],markerfacecolor=g[i],markeredgewidth=0,markevery=1)
    label='QAOA: '+str(strround(np.e**intercept,3))+r'$\times{' + str(strround(np.e**slope,3)) + '}^n$'+', '+r'$R^2=$'+strround(r2,3)
    custom_handles = matplotlib.lines.Line2D(
            [0], [0],
            marker=g[i+int(len(g)/2)],
            markersize=6,
            markerfacecolor=g[i],
            markeredgewidth=0,
            linestyle='--',
            color=g[i],
            linewidth=2.5,
            label=label
        )
    handles.append(custom_handles)
    labels.append(label)
    plt.xticks(fontsize=8.5)
    plt.yscale('log')
    plt.xlabel("n",fontsize=10.5,labelpad=0)
    plt.ylabel('Time complexity',fontsize=10.5)
    plt.tick_params(axis='x', which='minor', length=0)
    plt.tick_params(axis='y', which='both', labelsize=7.5)
    plt.tick_params(axis='both', which='both', right=False, top=False, labelright=False, labeltop=False)

    plt.legend(handles,labels,frameon=False,edgecolor='black',fancybox=False,fontsize=7.5,title_fontsize=9,title=str(layernum)+' layer QAOA, m/n='+str(k),bbox_to_anchor=(0.5, 0.85),loc='center',handletextpad=0.4,handlelength=3.2) 
def width(ax,bwith):
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
listk=[0.626,0.55,0.575,0.6,0.65,0.675,0.7,0.725,0.75]
listp=[0.45,0.92,0.35,0.45,0.45,0.5,0.07,0.07,0.07]
for i in range(9):
    fig,ax= plt.subplots()
    ax.axis('off')
    fig.set_size_inches(10, 6)
    plt.rcParams['axes.ymargin'] = 0.02
    gs = GridSpec(2, 3, figure=fig)
    k=listk[i]
    p=listp[i]

    g=["#3C5488", "#C2CBE4",'#FFC4B3','o','^','X']
    g2=["#FBB463", "#80B1D3", "#BDBADB", "#FBC99A", "#8DD1C6", "#F47F72",'o','^','X','*','>','p']
    #g2=['#87CEEB','#C7E6C0','#FFDAB9','#E6D4FF','#FFC4B3','#8B0000','#B0E9E2','#483D8B','o','^','X','*','>','p','P','D']
    ######################################################################## 经典

    ax1 = fig.add_subplot(gs[0:2, 0:2])
    handles2=[]
    labels2=[]
    methodlabel=['dlx','minisat22','Cadical195','Glucose42','Lingeling','VQE-RSRA','QA-DCRA','QA']
    totaldata=np.load('classical-FigS2-S10a/uni'+str(k)+'.npz')
    minidata=totaldata['averagetime']
    caddata=totaldata['averagetime2']
    gludata=totaldata['averagetime3']
    lindata=totaldata['averagetime4']
    dlxdata=totaldata['averagetimedlx']
    VQEdata=totaldata['list1']

    drawclassical(dlxdata,0)
    drawclassical(minidata,1)
    drawclassical(caddata,2)
    drawclassical(gludata,3)
    drawclassical(lindata,4)
    drawclassical(VQEdata,5)
    ############################################################################################################

    plt.xticks(np.arange(70,130,10))
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=8)
    plt.yscale('log')
    plt.xlabel("n",fontsize=12,labelpad=0)
    plt.xlim(69,121)
    place="upper left"
    normal=0.5
    plt.legend(handles2,labels2,frameon=False,edgecolor='black',fancybox=False,ncol=2,title_fontsize=10,columnspacing=0.5,fontsize=9,title='Classical and VQE, m/n='+str(k),bbox_to_anchor=(normal, p),loc='center',handletextpad=0.4,handlelength=3.2)
    if(p==0.07):
        plt.legend(handles2,labels2,frameon=False,edgecolor='black',fancybox=False,ncol=2,title_fontsize=10,columnspacing=0.5,fontsize=7.5,title='Classical and VQE, m/n='+str(k),bbox_to_anchor=(0.59, p),loc='center',handletextpad=0.4,handlelength=3.2)
    plt.ylabel('Time complexity',fontsize=12)
    plt.title('a',x=-0.06,y=0.996,fontsize=12)
    width(ax,1.25)
    # ######################################################################### QAOA
    ax5=fig.add_subplot(gs[1,2])
    drawQAOA(25)
    width(ax5,1)
    plt.title('c',x=-0.06,y=0.99,fontsize=12)

    # ######################################################################### QAA
    ax9=fig.add_subplot(gs[0,2])
    drawQAA(100)
    width(ax9,1)
    plt.title('b',x=-0.06,y=0.99,fontsize=12)
    plt.subplots_adjust(
        left=0.05,    # 左侧边距
        right=0.99,   # 右侧边距
        bottom=0.03,  # 底部边距
        top=0.965,     # 顶部边距
        wspace=0.35,  # 水平子图间距
        hspace=0.16   # 垂直子图间距
    )
    plt.savefig("FigS"+str(i+2)+".pdf")
    #plt.savefig("scalingdrawwide"+str(k)+".pdf")