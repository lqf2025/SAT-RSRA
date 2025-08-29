import numpy as np
import matplotlib.pyplot as plt
def printclause(clausein,m):
    clause=clausein.copy()
    for i in range(m):
        string="clause "+str(i)+str(":")+str(clause[i][0])+","+str(clause[i][1])+","+str(clause[i][2])+"  "
        print(string, end=" ")
        if(i%7==6):
            print("") 
    print("") 
def s2(n,o1,o2):
    return (2*o1+o2)^n
def matrix2(th,o1,o2):#两比特作用矩阵
    if(o1==0 and o2==0):
        return np.eye(4) 
    s=np.sin(th/2)**2
    c=np.cos(th/2)**2
    matrix=np.eye(4)*c
    list=[s2(x,o1,o2) for x in range(4)]
    for i in range(4):
        matrix[i][list[i]]=s
    return matrix
def calculateaverage12(p2in,parameter,dim,num1,num2):#num1相当于0，num2相当于1
    p2=p2in.copy()
    r=parameter.copy()
    vec=np.zeros((4),dtype=complex)
    vec[3]=1#初始全1
    for j in range(dim):
        q1=p2[num1-1][j]
        q2=p2[num2-1][j]
        mat=matrix2(r[j],q1,q2)
        vec=mat.dot(vec)
    pr=vec[3]
    return pr
def calculateenergy(p2in,parameter,dim,clauses,m):#计算固定参数下的能量
    Energysum=0
    p2=p2in.copy()
    r=parameter.copy()
    for i in range(m):
        l1=clauses[i][0]
        l2=clauses[i][1]
        #l3=clauses[i][2]
        Energysum=Energysum+calculateaverage12(p2,r,dim,l1,l2)
    return Energysum.real
def mostcommonstate(n,p2in,parameter,dim):
    p2=p2in.copy()
    r=parameter.copy()
    r=r/np.pi
    r=np.rint(r)%2
    s=np.zeros((n),dtype=int)
    s=s+1
    for i in range(dim):
        if(r[i]==1):
            for j in range(n):
                s[j]=s[j]+p2[j][i]
    s=s%2
    s=''.join(str(k)for k in s)
    return s
def calculategradient(p2in,par,dim,clauses,m):
    p2=p2in.copy()
    th=np.pi/2
    gra=np.zeros((dim))
    for i in range(dim):
        tempar1=par.copy()
        tempar1[i]=tempar1[i]+th
        E1=calculateenergy(p2,tempar1,dim,clauses,m)
        tempar2=par.copy()
        tempar2[i]=tempar2[i]-th
        E2=calculateenergy(p2,tempar2,dim,clauses,m)
        gra[i]=(E1-E2)/(2*np.sin(th))
    return gra
def VQEnAdam(clauses,initialparametersin,n,m,lr,maxepoch,p2in,dim,b1,b2,eposilon):
    state=0
    para=initialparametersin.copy()
    p2=p2in.copy()
    energylist=[]
    garafor=np.zeros((dim))#一阶距
    hfor=0#二阶距
    for i in range(maxepoch):
        energy=calculateenergy(p2in,para,dim,clauses,m)
        energylist=np.append(energylist,energy)
        print("NADAM epoch:"+str(i)+"  energy:"+str(energy))
        gradient=calculategradient(p2,para,dim,clauses,m)
        print(np.linalg.norm(gradient))
        if(np.linalg.norm(gradient)<0.1):
            state=1
            x=range(i+1)
            finalenergy=calculateenergy(p2in,para,dim,clauses,m)
            s=mostcommonstate(n,p2,para,dim)        
            return finalenergy,s,para,state,x,energylist
        garafor=garafor*b1+gradient*(1-b1)
        hfor=b2*hfor+(1-b2)*np.linalg.norm(gradient)*np.linalg.norm(gradient)
        vhat=hfor/(1-b2**(i+1))
        mhat=garafor/(1-b1**(i+1))
        mov=-lr*1/np.sqrt(eposilon+vhat)*(b1*mhat+(1-b1)/(1-b1**(i+1))*gradient)
        print(np.linalg.norm(mov))
        para=para+mov
    finalenergy=calculateenergy(p2in,para,dim,clauses,m)
    s=mostcommonstate(n,p2,para,dim)        
    return finalenergy,s,para,state,x,energylist
def verify(clauses,m,s):#验证是不是解
    e=0
    for i in range(m):
        l1=clauses[i][0]
        l2=clauses[i][1]
        l3=clauses[i][2]
        if(int(s[l1-1])+int(s[l2-1])+int(s[l3-1])!=1):
            e=e+1
    if(e==0):
        print("verified")
    else:
        print("not verified")
def readnpz(n,m):
    npzfile=np.load("PQC"+str(n)+","+str(m)+".npz")
    return npzfile['clauses'],npzfile['p2']
n=150
m=94
#adam算法相关参数
learningadam=1
b1=0.9
b2=0.999
maxadamepoch=150
clauses,p2=readnpz(n,m)
dimention=np.shape(p2)[1]#维数
counter2=0#成功的
for counter in range(400):
    state=0
    print("try "+str(counter)+" start")
    initialparameters=np.random.rand(dimention)*4*np.pi
    paramove=initialparameters.copy()
    groundE,groundS,paramove,state,epochnum,elist=VQEnAdam(clauses,paramove,n,m,learningadam,maxadamepoch,p2,dimention,b1,b2,1e-8)
    if(groundE<0.95):
        counter2=counter2+1
print(counter2)
np.savez("VQEcount4.npz",counter2=counter2)
