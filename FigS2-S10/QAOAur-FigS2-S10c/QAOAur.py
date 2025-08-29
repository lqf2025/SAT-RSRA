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
    matrix=np.zeros((2**n))
    m=np.shape(clauses)[0]
    for i in range(2**n):
        s=convertbin(i,n)
        energy=0
        for j in range(m):
            l1=clauses[j][0]
            l2=clauses[j][1]
            l3=clauses[j][2]
            energy=energy+(s[l1-1]+s[l2-1]+s[l3-1]-1)*(s[l1-1]+s[l2-1]+s[l3-1]-1)
        matrix[i]=energy
    return matrix
def calculatecorrectmatrixur(n,clausesin):#H1矩阵 UR
    clauses=clausesin.copy()
    matrix=np.zeros((2**n))
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
            matrix[i]=1
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
def QAOApossur(n,t,parain,diag,diag2):
    para=parain.copy()
    initialstate=np.ones((2**n))/np.sqrt(2**n)
    for i in range(t):
        g=para[t+i]
        b=para[i]
        r=np.exp(1j*g*diag)
        initialstate=np.multiply(r,initialstate)
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
def averageQAOAursuccess(single,parain,num,n):#返回参数下成功率
    para=parain.copy()
    t=int(len(para)/2)
    @parfor(range(num), disable=True) ####################
    def singlecase(i):
        return QAOApossur(n,t,para,single[i][0],single[i][1])
    return sum(singlecase)/num
def gradientur(single,layer,par,n):
    list=[]
    for i in range(2*layer):
        temp1=par.copy()
        temp1[i]=temp1[i]+0.001
        list.append(temp1)
    @parfor(range(2*layer), disable=True)
    def singlecase(i):
        return averageQAOAursuccess(single,list[i],train,n)
    return singlecase
def slope(start,final,k):
    plist=range(start,final+1)
    lr1=0.1
    b1=0.9
    b2=0.999
    eposilon=1e-8
    plen=len(plist)
    slopeurrecord=np.zeros((5,plen))
    for n in range(5,10):
        print(n)
        @parfor(range(train), disable=True)
        def single0(i):
            m,clauses,p2,matrix,c=generatecase(n,k)
            return calculateenergymatrixur(n,clauses),calculatecorrectmatrixur(n,clauses)
        @parfor(range(test), disable=True)
        def single1(i):
            m,clauses,p2,matrix,c=generatecase(n,k)
            return calculateenergymatrixur(n,clauses),calculatecorrectmatrixur(n,clauses)
        for pos in range(plen): 
            print(pos)
            layer=plist[pos]
            guess=initial(layer)*0.5
            garafor=np.zeros((2*layer))#一阶距
            hfor=0#二阶距
            for _ in range(epoch):
                print(_)
                E0=averageQAOAursuccess(single0,guess,train,n)
                gradient=(np.array(gradientur(single0,layer,guess,n)-E0))/0.001
                gradient=-gradient #改为求最大值
                garafor=garafor*b1+gradient*(1-b1)
                hfor=b2*hfor+(1-b2)*np.linalg.norm(gradient)*np.linalg.norm(gradient)
                vhat=hfor/(1-b2**(_+1))
                mhat=garafor/(1-b1**(_+1))
                mov=-lr1*1/np.sqrt(eposilon+vhat)*(b1*mhat+(1-b1)/(1-b1**(_+1))*gradient)
                guess=guess+mov
            finalQAOA=averageQAOAursuccess(single1,guess,test,n)
            slopeurrecord[n-5][pos]=-np.log(finalQAOA)
    np.savez('QAOAurscale'+str(k)+'.npz',slopeurrecord=slopeurrecord)
if __name__ == '__main__':
    epoch=30
    train=500
    test=2000
    slope(25,25,0.675)