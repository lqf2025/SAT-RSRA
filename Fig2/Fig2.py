import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.linalg import expm
from matplotlib.gridspec import GridSpec
def fitslope(x, y):
    numerator = sum(xi * yi for xi, yi in zip(x, y))
    denominator = sum(xi ** 2 for xi in x)
    return numerator / denominator
def recover(run):
    datatotal=np.zeros((20,35))
    for n in range(20,40):
        data=-np.log(np.load('pdetermine-Fig2fg/QAOAsingle/'+str(n)+'/QAOAsingle'+str(n)+' '+str(run)+'.npz')['data1'])
        datatotal[n-20,:]=data
    sloperecord=np.zeros(35)
    for i in range(35):
        s1,_=np.polyfit(range(20),datatotal[:,i],1)
        sloperecord[i]=np.e**s1
    return sloperecord
bwith =1
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
fig=plt.figure(figsize=(13.5,6))
gs = plt.GridSpec(2, 4)

#################################################################################
nlist=np.load("reduction-Fig2abh/reduction.npz")['nlist']/5000
klist=np.load("reduction-Fig2abh/reduction.npz")['klist']/5000
nslope=[]
kslope=[]
successlist=np.load("reduction-Fig2abh/reduction.npz")['successlist']
krange=0.55+np.array(range(21))*0.2/20
for i in range(21):
    nslope.append(fitslope(range(10,70),nlist[7:,i]))#n 
    kslope.append(fitslope(range(10,70),klist[7:,i]))#n-k m=0.026 =n-m
maxn=69
x=range(10,70)
marker=['o','P','H','D','v','s']
c=['#8B5CF6', '#06B6D4', '#F59E0B', '#EF4444', '#10B981', '#6366F1']
ax1 =fig.add_subplot(gs[0,0])
ax1.spines[:].set_linewidth(bwith)
ax=plt.gca()
plt.tick_params(axis='x', which='minor', length=0)
plt.tick_params(axis='y', which='minor', length=0)
ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False)
ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False)
plt.xticks(size=12)
plt.yticks(size=12)
plt.title('a',x=-0.06,y=0.99,fontsize=15)
for i in range(6):
    plt.plot(range(3,70),nlist[:,4*i]-klist[:,4*i],label=fr'${round(krange[4*i],2)}$',marker=marker[i],linewidth=2.5,markersize=5,markevery=10,color=c[i])
plt.legend(loc='upper left',labelspacing=0.15,bbox_to_anchor=(-0.01,1.05), fontsize=11,frameon=False,title='$m/n=$',title_fontsize=12.5)
plt.xlabel(r'$n$',fontsize=13.5)
plt.ylabel(r'$k$'+' (rank of '+r'$A$'+')',fontsize=13.5)
plt.xlim(0,maxn+1)
############################################
ax2 =fig.add_subplot(gs[0,1])
ax2.spines[:].set_linewidth(bwith)
plt.yticks(size=12)
plt.plot(krange,np.array(nslope)-np.array(kslope),label='Slope of '+r'$k$',color='#6366F1',marker='o',markersize=8,linewidth =3)
plt.plot(krange,krange,'--',linewidth =3,marker='o',color='#8B5CF6',label=r'$m/n$',markersize=8)
plt.legend(bbox_to_anchor=(0.68,0.165),labelspacing=0.25, fontsize=12.5,frameon=False,loc='center',handlelength=2.3)
plt.tick_params(axis='x', which='minor', length=0)
plt.tick_params(axis='y', which='minor', length=0)
ax2=plt.gca()
ax2.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False)
plt.xlabel(r'$m/n$',fontsize=13.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(np.arange(0.55,0.79,0.05))
plt.tick_params(axis='x', which='minor', length=0)
plt.tick_params(axis='y', which='minor', length=0)
plt.title('b',x=-0.06,y=0.99,fontsize=15)
############################################
ax3 =fig.add_subplot(gs[0,2])
ax3.spines[:].set_linewidth(bwith)
ax = plt.gca()
ax3.text(0.35,0.935, 'VQE-based', ha='center',fontsize=12,transform = ax.transAxes)
npzfile=np.load("VQE-Fig2c/VQE.npz")
fe1=npzfile['fe1']
fe2=npzfile['fe2']
fe3=npzfile['fe3']
fl1=npzfile['fl1']
fl2=npzfile['fl2']
fl3=npzfile['fl3']
se1=npzfile['se1']
se2=npzfile['se2']
se3=npzfile['se3']
sl1=npzfile['sl1']
sl2=npzfile['sl2']
sl3=npzfile['sl3']
plt.plot(fe1,fl1,'--',linewidth =2.5,marker='P',markersize=5,markevery=5,label='Failure example 1',color=c[0])
plt.plot(fe2,fl2,'--',linewidth =2.5,marker='s',markersize=5,markevery=5,label='Failure example 2',color=c[1])
plt.plot(fe3,fl3,'--',linewidth =2.5,marker='^',markersize=5,markevery=5,label='Failure example 3',color=c[2])
plt.plot(se1,sl1,'-',linewidth =2.5,marker='o',markersize=5,markevery=5,label='Success example 1',color=c[3])
plt.plot(se2,sl2,'-',linewidth =2.5,marker='D',markersize=5,markevery=5,label='Success example 2',color=c[4])
plt.plot(se3,sl3,'-',linewidth =2.5,marker='^',markersize=5,markevery=5,label='Success example 3',color=c[5])
plt.title('c',x=-0.06,y=0.99,fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(size=12)
ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False)
plt.tick_params(axis='x', which='minor', length=0)
plt.tick_params(axis='y', which='minor', length=0)
plt.legend(bbox_to_anchor=(1.02,0.62),loc='center right',fontsize=11,handlelength=2.3,frameon=False)
plt.xlabel('Iteration',fontsize=13.5)
plt.ylabel('Energy',fontsize=13.5)
ax3.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False)
#################################################################################
ax4 =fig.add_subplot(gs[1,0])
ax4.spines[:].set_linewidth(bwith)
trange=15
npzfile=np.load("QAAQAOA-Fig2de/QAOAdraw.npz")
x=npzfile['x']
energylist=npzfile['energylist']
poss=npzfile['poss']
plt.tick_params(axis='x', which='minor', length=0)
plt.tick_params(axis='y', which='minor', length=0)
plt.tick_params(top=False)
ax4.tick_params(axis="y", labelcolor="b") 
ax4.plot(x,energylist,'-',marker='o',linewidth =3,markersize=7,color='#6366F1',label='Energy')
legend=plt.legend(bbox_to_anchor=(0.18,0.85),fontsize=12.5,handlelength=2.3,frameon=False,labelspacing=0.25,loc='center left')
ax = plt.gca()
ax4.text(0.35,0.935, 'QAOA-based', ha='center',fontsize=12,transform = ax.transAxes)
ax4.set_xticks([i for i in range(2,trange+2,2)])
plt.xlabel(r'$N_{\mathrm{QAOA}} $',fontsize=13.5)
plt.xticks(fontsize=12)
plt.yticks(size=12)
ax45=ax4.twinx()
plt.tick_params(axis='x', which='minor', length=0)
plt.tick_params(axis='y', which='minor', length=0)
plt.tick_params(top=False)
ax45.tick_params(axis="y", labelcolor="r") 
ax45.plot(x,poss,'-',marker='P',linewidth =3,markersize=7,color='#EF4444',label='Possibility')
legend=plt.legend(bbox_to_anchor=(0.18,0.75),fontsize=12.5,handlelength=2.3,frameon=False,labelspacing=0.25,loc='center left')
plt.xticks(fontsize=12)
plt.yticks(size=12)
plt.title('e',x=-0.06,y=0.99,fontsize=15)
plt.ylim(-0.02,0.5)
#################################################################################
ax5=plt.subplot(gs[0, 3])
ax5.spines[:].set_linewidth(bwith)
ax = plt.gca()
ax5.text(0.35,0.935, 'QAA-based', ha='center',fontsize=12,transform = ax.transAxes)
trange2=152
npzfile=np.load("QAAQAOA-Fig2de/QAAdraw.npz")
x=npzfile['x']
energylist=npzfile['energylist']
possibility=npzfile['possibility']
plt.tick_params(axis='x', which='minor', length=0)
plt.tick_params(axis='y', which='minor', length=0)
plt.tick_params(top=False)
ax5.tick_params(axis="y", labelcolor='#6366F1') 
ax5.plot(x,energylist,'-',linewidth =3,marker='o',markevery=15,markersize=7,label='Energy',color='#6366F1')
ax5.set_xticks([i for i in np.arange(0,trange2,30)+2])
plt.xlabel(r'$N_{\mathrm{QAA}} $',fontsize=13.5)
legend=plt.legend(bbox_to_anchor=(0.38,0.38),fontsize=12.5,handlelength=2.3,frameon=False,labelspacing=0.25,loc='center left')
plt.ylim(-0.32,13.2)
plt.xticks(fontsize=12)
plt.yticks(size=12)
ax55=ax5.twinx()
plt.ylim(-0.05,1.05)
plt.tick_params(axis='x', which='minor', length=0)
plt.tick_params(axis='y', which='minor', length=0)
plt.tick_params(top=False)
ax55.tick_params(axis="y", labelcolor='#EF4444') 
ax55.plot(x,possibility,'-',linewidth =3,marker='P',markevery=15,markersize=7,label='Possibility',color='#EF4444')
legend=plt.legend(bbox_to_anchor=(0.38,0.28),fontsize=12.5,handlelength=2.3,frameon=False,labelspacing=0.25,loc='center left')
plt.xticks(fontsize=12)
plt.yticks(size=12)
plt.title('d',x=-0.06,y=0.99,fontsize=15)
#################################################################################
ax7 =fig.add_subplot(gs[1,1])
ax7.spines[:].set_linewidth(bwith)
confidence_level = 0.95
t_score = stats.t.ppf((1 + confidence_level) / 2, df=4)
colorlist=['#6366F1',"#A5B4FC","#FDBA74"]
linelist=["-","--","-."]

data0=np.load('pdetermine-Fig2fg/QAAscale 0.npz')['scale']-1
data1=np.load('pdetermine-Fig2fg/QAAscale 1.npz')['scale']-1
data2=np.load('pdetermine-Fig2fg/QAAscale 2.npz')['scale']-1
data3=np.load('pdetermine-Fig2fg/QAAscale 3.npz')['scale']-1
data4=np.load('pdetermine-Fig2fg/QAAscale 4.npz')['scale']-1
ave=(data0+data1+data2+data3+data4)/5
print(ave[40],ave[-1])
std=np.sqrt((data0**2+data1**2+data2**2+data3**2+data4**2)-5*ave**2)/np.sqrt(5)*t_score
plt.plot(range(1,150),ave,linewidth=4,label='QAA-RSRA',color=colorlist[0],linestyle=linelist[0])
plt.plot(range(1,150),data0,linewidth=0.9,color=colorlist[2],linestyle='--')
plt.plot(range(1,150),data1,linewidth=0.9,color=colorlist[2],linestyle='--')
plt.plot(range(1,150),data2,linewidth=0.9,color=colorlist[2],linestyle='--')
plt.plot(range(1,150),data3,linewidth=0.9,color=colorlist[2],linestyle='--')
plt.plot(range(1,150),data4,linewidth=0.9,color=colorlist[2],linestyle='--')
plt.fill_between(
    range(1,150), 
    ave - std,  # 下限
    ave + std,  # 上限
    alpha=0.35,
    color=colorlist[1]
) 
plt.xlabel(r'$N_{\mathrm{QAA}}$',fontsize=13.5)
plt.yscale('log')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("Scaling",fontsize=13.5)
plt.legend(frameon=False,fontsize=12.5,bbox_to_anchor=(0.6,0.8),loc='center')
plt.title('f',x=-0.06,y=0.99,fontsize=15)

#################################################################################
ax5=plt.subplot(gs[1, 2])
ax5.spines[:].set_linewidth(bwith)
data0=recover(0)-1
data1=recover(1)-1
data2=recover(2)-1
data3=recover(3)-1
data4=recover(4)-1
ave=(data0+data1+data2+data3+data4)/5
print(ave[25],ave[-1])
std=np.sqrt((data0**2+data1**2+data2**2+data3**2+data4**2)-5*ave**2)/np.sqrt(5)*t_score
plt.plot(range(1,36),ave,linewidth=4,label='QAOA-RSRA',color=colorlist[0],linestyle=linelist[0])
# plt.scatter(range(1,36), data0, marker='+', s=50, c=colorlist[2] , linewidths=2.5)
# plt.scatter(range(1,36), data1, marker='+', s=50, c=colorlist[2] , linewidths=2.5)
# plt.scatter(range(1,36), data2, marker='+', s=50, c=colorlist[2] , linewidths=2.5)
# plt.scatter(range(1,36), data3, marker='+', s=50, c=colorlist[2] , linewidths=2.5)
# plt.scatter(range(1,36), data4, marker='+', s=50, c=colorlist[2] , linewidths=2.5)
plt.plot(range(1,36),data0,linewidth=0.9,color=colorlist[2],linestyle='--')
plt.plot(range(1,36),data1,linewidth=0.9,color=colorlist[2],linestyle='--')
plt.plot(range(1,36),data2,linewidth=0.9,color=colorlist[2],linestyle='--')
plt.plot(range(1,36),data3,linewidth=0.9,color=colorlist[2],linestyle='--')
plt.plot(range(1,36),data4,linewidth=0.9,color=colorlist[2],linestyle='--')
plt.fill_between(
    range(1,36), 
    ave - std,  # 下限
    ave + std,  # 上限
    alpha=0.35,
    color=colorlist)
plt.xlabel(r'$N_{\mathrm{QAOA}}$',fontsize=13.5)
plt.ylabel("Scaling",fontsize=13.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(frameon=False,fontsize=12.5,bbox_to_anchor=(0.6,0.8),loc='center')
plt.title('g',x=-0.06,y=0.99,fontsize=15)
plt.yscale('log')
plt.xlim(0.5,35.5)
#################################################################################
ax8 =fig.add_subplot(gs[1,3])
ax8.spines[:].set_linewidth(bwith)
marker=['o','P','H','D','v','s']
gslope=[]
glist=np.load("reduction-Fig2abh/reduction.npz")['glist']/5000
krange=0.55+np.array(range(21))*0.2/20
for i in range(21):
    gslope.append(fitslope(range(10,70),glist[7:,i]))#n 
plt.plot(krange,gslope,label='Slope of '+r'$g$',color='#EF4444',marker='o',linewidth =3,markersize=7,markevery=1)
#plt.legend(loc='center left',labelspacing=0.1, fontsize=11,frameon=False,bbox_to_anchor=(-0.02,0.56))
plt.xlabel(r'$m/n$',fontsize=13.5)
plt.ylabel('Gradient of '+r'$|G|$',fontsize=13.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('h',x=-0.06,y=0.99,fontsize=15)

plt.subplots_adjust(left=0.04,   # 图像左边距（0~1之间，越大间距越大）
                    right=0.97,  # 图像右边距（0~1之间，越小间距越大）
                    top=0.963,    # 图像上边距（0~1之间，越小间距越大）
                    bottom=0.075,
                    wspace=0.4,
                    hspace=0.21)
#plt.tight_layout()
plt.savefig("Fig2.pdf")