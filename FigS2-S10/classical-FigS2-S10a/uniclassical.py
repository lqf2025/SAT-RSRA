import numpy as np
import generator
import random
import dlx
from pysat.solvers import Solver
from parfor import parfor
def generatecase(n,k):#生成一个有解的问题
    m=int(np.floor(n*k))
    if(random.random()<n*k-m):
        m=m+1
    while(1):
        clauses,p2,matrix,c=generator.generaterandom(n,m)
        s=forms(clauses)
        if(s.solve()==True):
            s2=forms2(clauses)
            s2.solve()
            s3=forms3(clauses)
            s3.solve()
            s4=forms4(clauses)
            s4.solve()
            break
    return int(np.shape(p2)[0]),m,clauses,p2,matrix,c,s.accum_stats()['conflicts'],s2.accum_stats()['conflicts'],s3.accum_stats()['conflicts'],s4.accum_stats()['conflicts']
def forms(clausesin):
    clauses=clausesin.copy()
    s = Solver(name='minisat22')
    m=np.shape(clauses)[0]
    for i in range(m):
        a,b,c=int(clauses[i][0]), int(clauses[i][1]),int(clauses[i][2])
        s.add_clause([a,b,c])
        s.add_clause([-a, -b,-c])
        s.add_clause([a, -b,-c])
        s.add_clause([-a, b,-c])
        s.add_clause([-a,-b,c])
    return s
def forms2(clausesin):
    clauses=clausesin.copy()
    s = Solver(name='Cadical195')
    m=np.shape(clauses)[0]
    for i in range(m):
        a,b,c=int(clauses[i][0]), int(clauses[i][1]),int(clauses[i][2])
        s.add_clause([a,b,c])
        s.add_clause([-a, -b,-c])
        s.add_clause([a, -b,-c])
        s.add_clause([-a, b,-c])
        s.add_clause([-a,-b,c])
    return s
def forms3(clausesin):
    clauses=clausesin.copy()
    s = Solver(name='Glucose42')
    m=np.shape(clauses)[0]
    for i in range(m):
        a,b,c=int(clauses[i][0]), int(clauses[i][1]),int(clauses[i][2])
        s.add_clause([a,b,c])
        s.add_clause([-a, -b,-c])
        s.add_clause([a, -b,-c])
        s.add_clause([-a, b,-c])
        s.add_clause([-a,-b,c])
    return s
def forms4(clausesin):
    clauses=clausesin.copy()
    s = Solver(name='Lingeling')
    m=np.shape(clauses)[0]
    for i in range(m):
        a,b,c=int(clauses[i][0]), int(clauses[i][1]),int(clauses[i][2])
        s.add_clause([a,b,c])
        s.add_clause([-a, -b,-c])
        s.add_clause([a, -b,-c])
        s.add_clause([-a, b,-c])
        s.add_clause([-a,-b,c])
    return s
#########################################
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
def QUBOmatrixorigin(clausesin,n,m):
    clause=clausesin.copy()
    QUBOm=np.zeros((n,n))
    QUBOC=0
    for i in range(m):
        a1=clause[i][0]-1
        a2=clause[i][1]-1
        a3=clause[i][2]-1
        if(a1>a2):
            a1,a2=a2,a1
        if(a2>a3):
            a2,a3=a3,a2
        if(a1>a2):
            a1,a2=a2,a1
        QUBOm[a1][a1]=QUBOm[a1][a1]-1
        QUBOm[a2][a2]=QUBOm[a2][a2]-1
        QUBOm[a3][a3]=QUBOm[a3][a3]-1
        QUBOm[a1][a2]=QUBOm[a1][a2]+2
        QUBOm[a1][a3]=QUBOm[a1][a3]+2
        QUBOm[a2][a3]=QUBOm[a2][a3]+2
        QUBOC=QUBOC+1
    #print(QUBO)
    return QUBOm,QUBOC
def calculateenergy(p2in,parameter,dim,clauses,m):#计算固定参数下的能量
    Energysum=0
    p2=p2in.copy()
    r=parameter.copy()
    for i in range(m):
        l1=clauses[i][0]
        l2=clauses[i][1]
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
###################################################
def VQEnAdam(clauses,initialparametersin,m,lr,maxepoch,p2in,dim,b1,b2,eposilon):
    para=initialparametersin.copy()
    p2=p2in.copy()
    energylist=[]
    garafor=np.zeros((dim))#一阶距
    hfor=0#二阶距
    for i in range(maxepoch):
        energy=calculateenergy(p2in,para,dim,clauses,m)
        energylist=np.append(energylist,energy)
        #print("NADAM epoch:"+str(i)+"  energy:"+str(energy))
        gradient=calculategradient(p2,para,dim,clauses,m)
        #print(np.linalg.norm(gradient))
        if(np.linalg.norm(gradient)<0.01):
            #x=range(i+1)
            finalenergy=calculateenergy(p2in,para,dim,clauses,m)        
            return finalenergy
        garafor=garafor*b1+gradient*(1-b1)
        hfor=b2*hfor+(1-b2)*np.linalg.norm(gradient)*np.linalg.norm(gradient)
        vhat=hfor/(1-b2**(i+1))
        mhat=garafor/(1-b1**(i+1))
        mov=-lr*1/np.sqrt(eposilon+vhat)*(b1*mhat+(1-b1)/(1-b1**(i+1))*gradient)
        #print(np.linalg.norm(mov))
        para=para+mov
    finalenergy=calculateenergy(p2in,para,dim,clauses,m)      
    return finalenergy
def convertexactcover(clausesin,n,m):#转换为exact cover
    clauses=clausesin.copy()
    convertedmatrix=np.zeros((n,m))
    for i in range(m):
        l1=clauses[i][0]-1
        l2=clauses[i][1]-1
        l3=clauses[i][2]-1
        convertedmatrix[l1][i]=1
        convertedmatrix[l2][i]=1
        convertedmatrix[l3][i]=1
    return convertedmatrix
""" def sample(Qin,c,time):
    Q=Qin.copy()
    N=np.shape(Q)[1]
    d1= {(i, j): Q[i][j]+Q[j][i] for i in range(N) for j in range(i+1, N)}
    d2= {(i, i): Q[i][i] for i in range(N)}
    d1.update(d2)
    sampler = oj.SQASampler()
    response = sampler.sample_qubo(d1,num_reads=1,num_sweeps=time,gamma=int(np.max(np.abs(Q))))
    energy=response.energies[0]+c
    print(energy)
    return energy """
def singlesize(n,k):
    @parfor(range(copy)) ####################
    def singlecase(i):
        learningadam=1
        b1=0.9
        b2=0.999
        maxadamepoch=150
        VQEsuccess,status1,status2=0,0,0
        nr,m,clauses,p2,matrix,c,timemini,timecad,timeglu,timelin=generatecase(n,k)
        mdlx=convertexactcover(clauses,nr,m)
        Q0,C0=QUBOmatrixorigin(clauses,nr,m)
        status,timedlx=dlx.solve(mdlx)
        """ EO=sample(matrix,c,50*n)
        if(EO<0.5):
            status1=1
        EO2 = sample(Q0,C0,50*n)
        if(EO2<0.5):
            status2=1 """
        if(i<copy2): ###########################
            dimention=np.shape(p2)[1]#维数
            initialparameters=np.random.rand(dimention)*4*np.pi
            paramove=initialparameters.copy()
            groundE=VQEnAdam(clauses,paramove,m,learningadam,maxadamepoch,p2,dimention,b1,b2,1e-8)
            if(groundE<0.95):
                VQEsuccess=1
        return timemini,timecad,timeglu,timelin,timedlx,VQEsuccess#,status1,status2
    return singlecase
def uni(ratio):
    averagetime=np.array([])
    averagetime2=np.array([])
    averagetime3=np.array([])
    averagetime4=np.array([])
    averagetimedlx=np.array([])
    list1=[]
    for n in range(start,end+1):
        l=np.array(singlesize(n,ratio))
        #totaltime,totaltime2,totaltime3,totaltime4,totaltimedlx,counter,counterQA,counterQA0=np.sum(l[ :,0]),np.sum(l[:,1]),np.sum(l[:,2]),np.sum(l[ :,3]),np.sum(l[:,4]),np.sum(l[:,5]),np.sum(l[:,6]),np.sum(l[:,7])
        totaltime,totaltime2,totaltime3,totaltime4,totaltimedlx,counter=np.sum(l[ :,0]),np.sum(l[:,1]),np.sum(l[:,2]),np.sum(l[ :,3]),np.sum(l[:,4]),np.sum(l[:,5])
        print(n,totaltime,totaltime2,totaltime3,totaltime4,totaltimedlx,counter)
        list1.append(np.log(copy2/(counter+small)))
        averagetime=np.append(averagetime,np.log(totaltime/copy+small))
        averagetime2=np.append(averagetime2,np.log(totaltime2/copy+small))
        averagetime3=np.append(averagetime3,np.log(totaltime3/copy+small))
        averagetime4=np.append(averagetime4,np.log(totaltime4/copy+small))
        averagetimedlx=np.append(averagetimedlx,np.log(totaltimedlx/copy+small))
    np.savez('uni'+str(ratio)+'.npz',averagetime=averagetime,averagetime2=averagetime2,averagetime3=averagetime3,averagetime4=averagetime4,averagetimedlx=averagetimedlx,list1=list1) ###############################
if __name__ == '__main__':
    copy=5000
    copy2=1000
    small=1e-10
    start=10
    end=120
    uni(0.626)



