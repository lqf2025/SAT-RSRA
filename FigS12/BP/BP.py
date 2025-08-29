import numpy as np
import generator
import random
from parfor import parfor
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
        Energysum=Energysum+calculateaverage12(p2,r,dim,l1,l2)
    return Energysum.real
def generatecase(n,k):#生成一个有解的问题
    m=int(np.floor(n*k))
    if(random.random()<n*k-m):
        m=m+1
    clauses,p2,matrix,c=generator.generaterandom(n,m)
    return clauses,np.shape(p2)[1],p2
def calculatevariance(n,k):
    while(1):
        clause,dimention,p2=generatecase(n,k)
        if(dimention>=1):
            break
    partial=random.randint(0,dimention-1)
    th=np.pi/2
    sum1=0
    sum2=0
    varcopy=100
    for i in range(varcopy):
        parameters=np.random.rand(dimention)*4*np.pi
        tempar1=parameters.copy()
        tempar1[partial]=tempar1[partial]+th
        E1=calculateenergy(p2,tempar1,dimention,clause,np.shape(clause)[0])
        tempar2=parameters.copy()
        tempar2[partial]=tempar2[partial]-th
        E2=calculateenergy(p2,tempar2,dimention,clause,np.shape(clause)[0])
        gra=(E1-E2)/(2*np.sin(th))
        sum1=sum1+gra
        sum2=sum2+gra*gra
    return sum2/varcopy-(sum1/varcopy)**2
def savenpz(k):
    randomcopy=1000
    averagvar=np.zeros((100))
    for n in range(10,30):
        sumvar=0
        for i in range(randomcopy):
            var=calculatevariance(n,k)
            sumvar=sumvar+var
        averagvar[n]=sumvar/randomcopy
        print(n,averagvar[n])
    np.savez("BP"+str(k)+".npz",averagvar=averagvar)
savenpz(0.626)
