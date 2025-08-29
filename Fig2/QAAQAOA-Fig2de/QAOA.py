import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import expm
from parfor import parfor
import random
def calculatesquantumstate(p2in,s):#s可以是数组或字符串，计算s对应的量子态,返回数组
    p2=p2in.copy()
    n=np.shape(p2)[0]#n
    col=np.shape(p2)[1]#列数
    state=np.zeros((n),dtype=int)
    for i in range(col):
        if(int(s[i])==1):
            state=state+p2[:,i]
    state=(state+1)%2
    return state
def calculateenergy(clausesin,statein):#计算对于一个量子态的能量
    clauses=clausesin.copy()
    state=statein.copy()
    m=np.shape(clausesin)[0]#条例数
    energy=0
    for i in range(m):
        l1=clauses[i][0]
        l2=clauses[i][1]
        energy=energy+state[l1-1]*state[l2-1]
    return energy
def convertbin(x,n):
    o_bin = bin(x)[2:] 
    out_bin = o_bin.rjust(n,'0')
    table=np.zeros((n),dtype='int')
    for i in range(n):
        table[i]=int(out_bin[i])
    return table
def convert210(s2in):
    s2=s2in.copy()
    length=len(s2)
    ten=0
    for i in range(length):
        ten=ten+s2[i]*2**(length-1-i)
    return ten
def calculateenergymatrix(dim,clausesin,p2in):#H1矩阵
    p2=p2in.copy()
    matrix=np.zeros((2**dim))
    clauses=clausesin.copy()
    @parfor(range(2**dim), disable=True)
    def single(i):
        s=convertbin(i,dim)
        state=calculatesquantumstate(p2,s)
        energy=calculateenergy(clauses,state)
        return energy
    return np.array(single)
def calculate(v1,mat):#计算v1^{dagger}*mat*v1
    v=np.dot(np.array(mat),np.array(v1))
    E=np.inner(np.array(v1).conjugate(),v)
    return E
def calculatecorrectmatrix(dim,clausesin,p2in):#H2矩阵
    p2=p2in.copy()
    matrix=np.zeros((2**dim))
    clauses=clausesin.copy()
    @parfor(range(2**dim), disable=True)
    def single(i):
        s=convertbin(i,dim)
        state=calculatesquantumstate(p2,s)
        energy=calculateenergy(clauses,state)
        if(energy<0.5):
            return 1
        else:
            return 0
    return np.array(single)
def generatecase(n,k):#生成一个有解的问题
    m=int(np.floor(n*k))
    if(random.random()<n*k-m):
        m=m+1
    while(1):
        clauses,p2,matrix,c=generator.generaterandom(n,m)
        dimention=np.shape(p2)[1]#维数
        print(dimention)
        m2=calculatecorrectmatrix(dimention,clauses,p2)
        if(np.max(m2)>0.5):
            break
    return m,clauses,p2,matrix,c
def initial(t):###########################
    #可以加一个参数
    sum=0
    vec=np.zeros((2*t))
    for i in range(1,t+1):
        s=i/(t+1)
        sum=sum+np.exp(-5*s*(1-s))
        vec[i+t-1]=sum
    vec=vec/sum
    for i in range(t):
        vec[i]=1-vec[i+t]
    return vec
def QAAposs(parain):
    para=parain.copy()
    t=int(len(para)/2)
    initialstate=np.ones((2**dimention))/np.sqrt(2**dimention)
    for i in range(t):
        #print(i)
        g=para[t+i]
        b=para[i]
        r=np.exp(1j*g*diag)
        initialstate=np.multiply(r,initialstate)
        trans=np.array([[0,1],[1,0]])
        matrixsingle=np.cos(b)*np.eye(2)-np.sin(b)*trans*1j
        tensor = initialstate.reshape([2] * dimention)
        for k in range(dimention):
            # 将第k个维度移动到第一个位置
            tensor = np.moveaxis(tensor, k, 0)
            # 重塑为 (2, ...) 以便矩阵乘法
            reshaped = tensor.reshape(2, -1)
            # 应用矩阵乘法到当前量子位
            reshaped = matrixsingle @ reshaped
            # 恢复形状并移回原维度顺序
            tensor = reshaped.reshape(tensor.shape)
            tensor = np.moveaxis(tensor, 0, k)
        initialstate=tensor.reshape(-1)
    return sum((initialstate*initialstate.conjugate())*diag2).real,sum((initialstate*initialstate.conjugate())*diag).real
def callback(xk):
    global totalenergy
    e=f1(xk)
    print(e)
def f1(parain):
    para=parain.copy()
    t=int(len(para)/2)
    initialstate=np.ones((2**dimention))/np.sqrt(2**dimention)
    for i in range(t):
        #print(i)
        g=para[t+i]
        b=para[i]
        r=np.exp(1j*g*diag)
        initialstate=np.multiply(r,initialstate)
        trans=np.array([[0,1],[1,0]])
        matrixsingle=np.cos(b)*np.eye(2)-np.sin(b)*trans*1j
        tensor = initialstate.reshape([2] * dimention)
        for k in range(dimention):
            # 将第k个维度移动到第一个位置
            tensor = np.moveaxis(tensor, k, 0)
            # 重塑为 (2, ...) 以便矩阵乘法
            reshaped = tensor.reshape(2, -1)
            # 应用矩阵乘法到当前量子位
            reshaped = matrixsingle @ reshaped
            # 恢复形状并移回原维度顺序
            tensor = reshaped.reshape(tensor.shape)
            tensor = np.moveaxis(tensor, 0, k)
        initialstate=tensor.reshape(-1)
    return sum((initialstate*initialstate.conjugate())*diag).real
if __name__ == '__main__':
    clauses=np.load("PQC100,63.npz")['clauses']
    m=np.load("PQC100,63.npz")['m']
    p2=np.load("PQC100,63.npz")['p2']
    diag=calculateenergymatrix(np.shape(p2)[1],clauses,p2)
    diag2=calculatecorrectmatrix(np.shape(p2)[1],clauses,p2)
    dimention=np.shape(p2)[1]#维数
    energylist=[]
    posslist=[]
    trange=1
    x=[]
    possibility=[]
    energylist=[]
    for t in (np.array(range(trange))+1):
        print("layer number:"+str(t))
        initialparameters=initial(t)*0.5
        boundst=[(-5,5)]*(2*t)
        res=minimize(fun=f1, method='BFGS',x0=initialparameters, callback=callback)
        x=np.append(x,t)
        poss,E=QAAposs(res.x)
        print("final energy:"+str(E))
        print("final poss:"+str(poss))
        energylist=np.append(energylist,E)
        possibility=np.append(possibility,poss)
    np.savez("QAOAdraw.npz", x=x,energylist=energylist,possibility=possibility)
        