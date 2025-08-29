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
def QAAposs(clausesin,p2in,t):
    clauses=clausesin.copy()
    p2=p2in.copy()
    para=initial(t)*0.5
    dimention=np.shape(p2)[1]#维数
    diag=calculateenergymatrix(dimention,clauses,p2)
    diag2=calculatecorrectmatrix(dimention,clauses,p2)
    initialstate=np.ones((2**dimention))/np.sqrt(2**dimention)
    for i in range(t):
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
    return sum((initialstate*initialstate.conjugate())*diag2).real
def QAOApossur(clausesin,n,t):
    clauses=clausesin.copy()
    para=initial(t)*0.5
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
def generateenergymatrix(n,matrixin,c):#h1qs
    matrix=matrixin.copy()
    energymatrix=np.zeros((2**n))
    for i in range(2**n):
        s=convertbin(i,n)
        energymatrix[i]=calculate(s,matrix)
    return energymatrix
def generateenergymatrix2(n,matrixin,c):#h2qs
    matrix=matrixin.copy()
    correctmatrix=np.zeros((2**n))
    for i in range(2**n):
        s=convertbin(i,n)
        if(calculate(s,matrix)+c==0):
            correctmatrix[i]=1
    return correctmatrix
def QAApossqs(matrixin,c,t):
    para=initial(t)*0.25
    matrix=matrixin.copy()
    size=np.shape(matrix)[0]
    diag1=generateenergymatrix(size,matrix,c)
    diag2=generateenergymatrix2(size,matrix,c)
    initialstate=np.ones((2**size))/np.sqrt(2**size)
    for i in range(t):
        g=para[t+i]
        b=para[i]
        r=np.exp(1j*g*diag1)
        initialstate=np.multiply(r,initialstate)
        trans=np.array([[0,1],[1,0]])
        matrixsingle=np.cos(b)*np.eye(2)-np.sin(b)*trans*1j
        tensor = initialstate.reshape([2] * size)
        for k in range(size):
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
def data(layerlist,k):
    final=60
    success=np.zeros((final-5,50))
    for n in range(5,final):
        @parfor(range(2000), disable=True)
        def single(i):
            m,clauses,p2,matrix,c=generatecase(n,k)
            return clauses,p2,matrix,c
            
        for layer in layerlist:
            @parfor(range(2000), disable=True)
            def single2(i):
                return QAAposs(single[i][0],single[i][1],layer)
            single2=np.array(single2)
            success[n-5][layer]=-np.log(sum(single2)/2000)
            print(n,success[n-5][layer])
    np.savez('QAAsinglell'+str(k)+'.npz',success=success)
if __name__ == '__main__':
    scale=[]
    scale2=[]
    # data([40],0.7)
    # data([40],0.725)
    # data([40],0.75)
    # data([40],0.55)
    data([40],0.626)
    # data([40],0.65)
    # data([40],0.675)
    # data([40],0.6)
    # data([40],0.575)


    

        

