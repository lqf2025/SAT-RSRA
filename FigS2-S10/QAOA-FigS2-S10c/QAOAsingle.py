import numpy as np
import generator
import random
from parfor import parfor
def initial(t):#两边快，中间慢的退火
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
def QAOAposs(dimention,t,parain,diag,diag2):
    para=parain.copy()
    trans=np.array([[0,1],[1,0]])
    initialstate=np.ones((2**dimention))/np.sqrt(2**dimention)
    for i in range(t):
        g=para[t+i]
        b=para[i]
        r=np.exp(1j*g*diag)
        initialstate=r*initialstate
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
def generateenergymatrix(s,matrixin,c):#h1qs
    matrix=matrixin.copy()
    energymatrix=np.zeros((2**s))
    for i in range(2**s):
        istr=convertbin(i,s)
        energymatrix[i]=calculate(istr,matrix)
    return energymatrix
def generateenergymatrix2(s,matrixin,c):#h2qs
    matrix=matrixin.copy()
    correctmatrix=np.zeros((2**s))
    for i in range(2**s):
        istr=convertbin(i,s)
        print(calculate(istr,matrix)+c)
        if(calculate(istr,matrix)+c<0.5):
            correctmatrix[i]=1
    return correctmatrix
def QAOApossqs(size,t,parain,diag,diag2):
    para=parain.copy()
    initialstate=np.ones((2**size))/np.sqrt(2**size)
    trans=np.array([[0,1],[1,0]])
    for i in range(t):
        g=para[t+i]
        b=para[i]
        r=np.exp(1j*g*diag)
        initialstate=r*initialstate
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
def averageQAOAqssuccess(single,parain,num):#返回参数下成功率
    para=parain.copy()
    t=int(len(para)/2)
    @parfor(range(num), disable=True) ####################
    def singlecase(i):
        return QAOApossqs(np.shape(single[i][2])[0],t,para,single[i][6],single[i][7])
    return sum(singlecase)/num
def averageQAOAsuccess(single,parain,num):#返回参数下成功率
    para=parain.copy()
    t=int(len(para)/2)
    @parfor(range(num), disable=True) ####################
    def singlecase(i):
        return QAOAposs(np.shape(single[i][1])[1],t,para,single[i][4],single[i][5])
    return sum(singlecase)/num
def gradient2(single,layer,par):
    list=[]
    for i in range(2*layer):
        temp1=par.copy()
        temp1[i]=temp1[i]+0.001
        list.append(temp1)
    @parfor(range(2*layer), disable=True)
    def singlecase(i):
        return averageQAOAsuccess(single,list[i],train)
    return singlecase

def gradientqs(single,layer,par):
    list=[]
    for i in range(2*layer):
        temp1=par.copy()
        temp1[i]=temp1[i]+0.001
        list.append(temp1)
    @parfor(range(2*layer), disable=True)
    def singlecase(i):
        return averageQAOAqssuccess(single,list[i],train)
    return singlecase
def data(layer,k):
    lr1=0.1
    lr2=0.05
    b1=0.9
    b2=0.999
    eposilon=1e-8
    success=np.zeros((35))
    successqs=np.zeros((35))
    for n in range(5,40):
        print(n)
        @parfor(range(train), disable=True)
        def single0(i):
            m,clauses,p2,matrix,c=generatecase(n,k)
            return clauses,p2,matrix,c,calculateenergymatrix(np.shape(p2)[1],clauses,p2),calculatecorrectmatrix(np.shape(p2)[1],clauses,p2),generateenergymatrix(np.shape(matrix)[0],matrix,c),generateenergymatrix2(np.shape(matrix)[0],matrix,c)
        @parfor(range(test), disable=True)
        def single1(i):
            m,clauses,p2,matrix,c=generatecase(n,k)
            return clauses,p2,matrix,c,calculateenergymatrix(np.shape(p2)[1],clauses,p2),calculatecorrectmatrix(np.shape(p2)[1],clauses,p2),generateenergymatrix(np.shape(matrix)[0],matrix,c),generateenergymatrix2(np.shape(matrix)[0],matrix,c)
        guess=initial(layer)*0.5
        guess2=initial(layer)*0.25
        garafor=np.zeros((2*layer))#一阶距
        hfor=0#二阶距
        garafor2=np.zeros((2*layer))#一阶距
        hfor2=0#二阶距
        for _ in range(epoch):
            print(_)
            E0=averageQAOAsuccess(single0,guess,train)
            gradient=(np.array(gradient2(single0,layer,guess)-E0))/0.001
            gradient=-gradient #改为求最大值
            garafor=garafor*b1+gradient*(1-b1)
            hfor=b2*hfor+(1-b2)*np.linalg.norm(gradient)*np.linalg.norm(gradient)
            vhat=hfor/(1-b2**(_+1))
            mhat=garafor/(1-b1**(_+1))
            mov=-lr1*1/np.sqrt(eposilon+vhat)*(b1*mhat+(1-b1)/(1-b1**(_+1))*gradient)
            guess=guess+mov
            #print(guess)
            #print(np.linalg.norm(mov),np.linalg.norm(gradient))
            E0=averageQAOAqssuccess(single0,guess2,train)
            gradient=np.zeros((2*layer))
            gradient=(np.array(gradientqs(single0,layer,guess2))-E0)/0.001
            gradient=-gradient
            garafor2=garafor2*b1+gradient*(1-b1)
            hfor2=b2*hfor2+(1-b2)*np.linalg.norm(gradient)*np.linalg.norm(gradient)
            vhat=hfor2/(1-b2**(_+1))
            mhat=garafor2/(1-b1**(_+1))
            mov=-lr2*1/np.sqrt(eposilon+vhat)*(b1*mhat+(1-b1)/(1-b1**(_+1))*gradient)
            guess2=guess2+mov
        finalQAOA=averageQAOAsuccess(single1,guess,test)
        finalQAOAqs=averageQAOAqssuccess(single1,guess2,test)
        success[n-5]=-np.log(finalQAOA)
        successqs[n-5]=-np.log(finalQAOAqs)
    np.savez('QAOAsingle'+str(k)+'.npz',success=success,successqs=successqs)

if __name__ == '__main__':
    epoch=30
    train=500
    test=2000
    data(25,0.675)
    
