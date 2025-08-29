import numpy as np
import generator
import random
from parfor import parfor
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
    for i in range(2**dim):
        s=convertbin(i,dim)
        state=calculatesquantumstate(p2,s)
        energy=calculateenergy(clauses,state)
        matrix[i]=energy
    return matrix
def calculate(v1,mat):#计算v1^{dagger}*mat*v1
    v=np.dot(np.array(mat),np.array(v1))
    E=np.inner(np.array(v1).conjugate(),v)
    return E
def calculatecorrectmatrix(dim,clausesin,p2in):#H2矩阵
    p2=p2in.copy()
    matrix=np.zeros((2**dim))
    clauses=clausesin.copy()
    for i in range(2**dim):
        s=convertbin(i,dim)
        state=calculatesquantumstate(p2,s)
        energy=calculateenergy(clauses,state)
        if(energy<0.5):
            matrix[i]=1
    return matrix
def calculateenergymatrixur(n,clausesin):#H1矩阵 UR
    clauses=clausesin.copy()
    matrix=np.zeros((2**n,2**n))
    m=np.shape(clauses)[0]
    for i in range(2**n):
        s=convertbin(i,n)
        energy=0
        for j in range(m):
            l1=clauses[j][0]
            l2=clauses[j][1]
            l3=clauses[j][2]
            energy=energy+(s[l1-1]+s[l2-1]+s[l3-1]-1)*(s[l1-1]+s[l2-1]+s[l3-1]-1)
        matrix[i][i]=energy
    return matrix
def calculatecorrectmatrixur(n,clausesin):#H1矩阵 UR
    clauses=clausesin.copy()
    matrix=np.zeros((2**n,2**n))
    m=np.shape(clauses)[0]
    for i in range(2**n):
        s=convertbin(i,n)
        energy=0
        for j in range(m):
            l1=clauses[j][0]
            l2=clauses[j][1]
            l3=clauses[j][2]
            energy=energy+(s[l1-1]+s[l2-1]+s[l3-1]-1)*(s[l1-1]+s[l2-1]+s[l3-1]-1)
        if(energy<0.5):
            matrix[i][i]=1
    return matrix
def generatecase(n,k):#生成一个有解的问题
    m=int(np.floor(n*k))
    if(random.random()<n*k-m):
        m=m+1
    while(1):
        clauses,p2,matrix,c=generator.generaterandom(n,m)
        dimention=np.shape(p2)[1]#维数
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
def QAOApossur(clausesin,n,t):
    clauses=clausesin.copy()
    para=initial(t)*0.25
    m1=calculateenergymatrixur(n,clauses)
    m2=calculatecorrectmatrixur(n,clauses)
    diag2=np.diag(m2)
    initialstate=np.ones((2**n))/np.sqrt(2**n)
    diag=np.diag(m1)
    for i in range(t):
        g=para[t+i]
        b=para[i]
        r=np.exp(1j*g*diag)
        initialstate=r*initialstate
        trans=np.array([[0,1],[1,0]])
        matrixsingle=np.cos(b)*np.eye(2)-np.sin(b)*trans*1j
        tensor = initialstate.reshape([2] * n)
        for k in range(n):
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
    return sum((initialstate*initialstate.conjugate())*diag2).real
def slopeur(plist,k):
    plen=len(plist)
    sloperecordur=np.zeros((10,plen))
    for n in range(5,15):
        print(n)
        @parfor(range(2000), disable=True)
        def single(i):
            m,clauses,p2,matrix,c=generatecase(n,k)
            return clauses,p2,matrix,c
            
        for pos in range(plen): 
            @parfor(range(2000), disable=True)
            def single2(i):
                return QAOApossur(single[i][0],n,plist[pos])
            sloperecordur[n-5][pos]=-np.log(sum(single2)/2000)
    np.savez('QAAscaleur'+str(k)+'.npz',sloperecordur=sloperecordur)
if __name__ == '__main__':

    slopeur(range(100,101),0.7)
    slopeur(range(100,101),0.725)
    slopeur(range(100,101),0.75)
    slopeur(range(100,101),0.55)
    slopeur(range(100,101),0.575)
    slopeur(range(100,101),0.6)
    slopeur(range(100,101),0.626)
    slopeur(range(100,101),0.65)
    slopeur(range(100,101),0.675)