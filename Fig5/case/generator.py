import numpy as np
from random import randint
import matplotlib.pyplot as plt
def generateclause(n,m):#序号范围[1,n]要求n>=3
    c=np.zeros((m,3),dtype=int)
    for i in range(m):
        a1=randint(1,n)
        while(1):
            a2=randint(1,n)
            if(a1!=a2):
                break
        while(1):
            a3=randint(1,n)
            if(a3!=a1 and a3!=a2):
                break
        c[i][0]=a1
        c[i][1]=a2
        c[i][2]=a3
    return c
def printclause(clausein,m):
    clause=clausein.copy()
    for i in range(m):
        string="clause "+str(i)+str(":")+str(clause[i][0])+","+str(clause[i][1])+","+str(clause[i][2])+"  "
        print(string, end=" ")
        if(i%7==6):
            print("") 
    print("") 
def calculatetimes(clausesetin,statein,n):
    state=statein.copy()
    clauseset=clausesetin.copy()
    numbercount=np.zeros((n+1),dtype=int)
    s=np.shape(clauseset)
    clausenumber=s[0]
    for i in range(clausenumber):
        if(state[i]==1):
            continue
        for j in range(0,3):
            literal=clauseset[i][j]
            numbercount[literal]=numbercount[literal]+1
    return numbercount
def judgestate(x,y,z):
    if(min([x,y,z])!=0):
        return 0
    if(max([x,y,z])==0):
        return 0
    return 1
def strictjudge(x,y,z):
    if(judgestate(x,y,z)==0):
        return 0
    mid=x+y+z-min([x,y,z])-max([x,y,z])
    if(mid==0):
        return 0
    return 1
def seekclause(clausesetin,statevectorin,n,clausestatein):#ai,aj,ak  j在决定区,k在非决定区，i随意,clausestate未使用
    #返回 条例位置，控制1(可能加入前决定区)，控制2，受控
    clauseset=clausesetin.copy()
    statevector=statevectorin.copy()
    clausestate=clausestatein.copy()
    count=calculatetimes(clauseset,clausestate,n)#更新度
    s=np.shape(clauseset)
    clausenumber=s[0]
    for i in range(clausenumber):
        if(clausestate[i]==1):
            continue
        else:
            a1,a2,a3=clauseset[i][0],clauseset[i][1],clauseset[i][2]
            s1,s2,s3=statevector[a1],statevector[a2],statevector[a3]
            if(strictjudge(s1,s2,s3)==1):
                ts=np.array([s1,s2,s3])
                h=np.where(ts==0)[0][0]
                return i,clauseset[i][(h+1)%3],clauseset[i][(h+2)%3],clauseset[i][h]
    for i in range(clausenumber):
        if(clausestate[i]==1):
            continue
        else:
            a1,a2,a3=clauseset[i][0],clauseset[i][1],clauseset[i][2]
            s1,s2,s3=statevector[a1],statevector[a2],statevector[a3]
            if(judgestate(s1,s2,s3)==0):
                continue
            ts=np.array([s1,s2,s3])
            h=np.where(ts>0)[0][0]
            if(count[clauseset[i][(h+1)%3]]>count[clauseset[i][(h+2)%3]]):
                return i,clauseset[i][(h+1)%3],clauseset[i][h],clauseset[i][(h+2)%3]
            else:
                return i,clauseset[i][(h+2)%3],clauseset[i][h],clauseset[i][(h+1)%3]
    return -1,-1,-1,-1         
def calculateoccupiedclause(clausesetin,statein):
    clauseset=clausesetin.copy()
    state=statein.copy()
    s=np.shape(clauseset)
    clausenumber=s[0]
    occupied=[]
    for i in range(clausenumber):
        if(state[clausesetin[i][0]]==2 or state[clausesetin[i][1]]==2 or state[clausesetin[i][2]]==2):
            occupied=np.append(occupied,i)
    return occupied
def subreduction(clausesetin,n,statein):#state 1前决定区,2后决定区,0待定,-1已经被删除
    clauseset=clausesetin.copy()
    state=statein.copy()
    s=np.shape(clauseset)
    clausenumber=s[0]
    clausestate=np.zeros((clausenumber),dtype=int)#0未使用,1已使用
    count=calculatetimes(clauseset,clausestate,n)
    initialclause=randint(0,clausenumber-1)
    clausestate[initialclause]=1
    if(count[clauseset[initialclause][0]]<count[clauseset[initialclause][2]]):
        clauseset[initialclause][0],clauseset[initialclause][2]=clauseset[initialclause][2],clauseset[initialclause][0]
    if(count[clauseset[initialclause][1]]<count[clauseset[initialclause][2]]):
        clauseset[initialclause][1],clauseset[initialclause][2]=clauseset[initialclause][2],clauseset[initialclause][1]
    #print("find clause "+str(initialclause)+":"+str(clauseset[initialclause][0])+","+str(clauseset[initialclause][1])+","+str(clauseset[initialclause][2]))
    state[clauseset[initialclause][2]]=2
    state[clauseset[initialclause][1]]=1
    state[clauseset[initialclause][0]]=1
    control=np.array([clauseset[initialclause][0],clauseset[initialclause][1],clauseset[initialclause][2]])
    decidedpar=np.array([clauseset[initialclause][2]])
    while(1):
        cs,f1s,f2s,ds=seekclause(clauseset,state,n,clausestate)
        if(cs==-1):
            #print("subreduction ended")
            break
        #print("find clause "+str(cs)+":"+str(f1s)+","+str(f2s)+","+str(ds))
        cl=np.array([f1s,f2s,ds])
        control=np.c_[control,cl]
        if(state[f1s]>0):#只需要加入后决定区
            state[ds]=2
        if(state[f1s]==0):#前后都要
            state[ds]=2
            state[f1s]=1
        clausestate[cs]=1
        decidedpar=np.append(decidedpar,ds)
    occupied=np.array(calculateoccupiedclause(clauseset,state))
    occupied=occupied.astype("int")
    subreducedclauseset=np.delete(clauseset,occupied,axis=0)
    formerpar=np.where(state==1)[0]
    #print(str(formerpar)+"|"+str(decidedpar))
    for i in range(n+1):
        if(state[i]==2):
            state[i]=-1
        if(state[i]==1):
            state[i]=0
    return subreducedclauseset,state,control       
def reduction(clausesetin,n):#clauseset为一个m*3尺寸，每一行是one in three的一个条例
    clauseset=clausesetin.copy()
    state=np.zeros((n+1),dtype=int)
    controlchain = np.zeros((3,0))
    while(1):
        clauseset,state,reducedcontrol=subreduction(clauseset,n,state)#返回最后一段控制
        controlchain=np.c_[reducedcontrol,controlchain]
        s=np.shape(clauseset)
        clausenumber=s[0]
        if(clausenumber==0):
            break
        #printclause(clauseset,clausenumber)
    left=np.where(state==0)[0]
    left=left[1:]
    #print("left parameternum:"+str(len(left)))
    #print("namely"+str(left))
    return np.array(left),controlchain
def formrepresent(leftin,chainin,n):
    chain=chainin.copy()
    left=leftin.copy()
    usedclausenum=np.shape(chain)[1]
    l=len(left)
    represenrtable=np.zeros((l+1,n),dtype=int)#第一行为常数
    I=np.zeros((l+1),dtype=int)#常数列
    I[0]=1
    for i in range(l):#初始赋值
        represenrtable[i+1][left[i]-1]=1
    for j in range(usedclausenum):
        c1=int(chain[0][j])
        c2=int(chain[1][j])
        d=int(chain[2][j])
        represenrtable[:,d-1]=I-represenrtable[:,c1-1]-represenrtable[:,c2-1]
    return represenrtable
def calculate(v1,mat):#计算v1T*mat*v2
    v=np.dot(np.array(mat),np.array(v1))
    E=np.inner(np.array(v1),v)
    return E
def formanswer(sin,chainin,leftnumin,n):
    s=sin.copy()
    chain=chainin.copy()
    leftnum=leftnumin.copy()
    answer=np.zeros((n),dtype=int)
    for i in range(len(leftnum)):
        answer[leftnum[i]-1]=s[i]
    usedclausenum=np.shape(chain)[1]
    for j in range(usedclausenum):
        c1=int(chain[0][j])
        c2=int(chain[1][j])
        d=int(chain[2][j])
        answer[d-1]=1-answer[c1-1]-answer[c2-1]
    return answer
def calculatefficentclause(l,tablein,clausesin):#计算剩余条例形式
    table=tablein.copy()
    clauses=clausesin.copy()
    I=np.zeros((l+1),dtype=int)#常数列
    I[0]=1
    eclausetable=np.zeros((l+1,0),dtype=int)
    clausesize=np.shape(clauses)[0]
    for i in range(clausesize): #总共size:n+m 前n变量，后m语句
        a1=clauses[i][0]
        a2=clauses[i][1]
        a3=clauses[i][2]
        new=table[:,a1-1]+table[:,a2-1]+table[:,a3-1]-I
        new=new%2
        if(np.count_nonzero(new)!=0):
            eclausetable=np.c_[eclausetable,new]
    return eclausetable
def calculateshifttime(tin,cnum,listin):#计算出现次数的奇偶性
    t=tin.copy()
    list=listin.copy()
    time=0
    for i in list:
        if(t[i][cnum]==1):
            time=time+1
    return time%2
def shiftchainpart1(eclausetablein,l):
    eclausetable=eclausetablein.copy()
    col=0
    eclausesize=np.shape(eclausetable)[1]
    if(eclausesize==0):
        return np.eye(l)
    list=np.array([]).astype('int')
    for i in reversed(range(1,l+1)):
        for j in range(col,eclausesize):
            if(eclausetable[i][j]==1):
                if(col!=j):
                    eclausetable[:,[j,col]]=eclausetable[:,[col,j]]#将标记态挪到第col列
                for k in range(col+1,eclausesize):
                    if(eclausetable[i][k]==1):
                        eclausetable[:,k]=(eclausetable[:,k]+eclausetable[:,col])%2
                list=np.append(list,i-1)
                col=col+1
    #print(list)
    for c in range(0,eclausesize):
        if(np.count_nonzero(eclausetable[:,c])==0):
            break
        if(c==eclausesize-1):
            c=c+1
    eclausetablenew=eclausetable[1:,:c]
    #print(eclausetablenew)
    shiftchainp1=np.zeros((l,l-c),dtype=int)
    pl=0
    for num in range(0,l):
        if(num not in list):
            shiftchainp1[num][pl]=1
            shiftnum=[num]
            for i in reversed(range(c)):
                if(eclausetablenew[num][i]==1 and calculateshifttime(eclausetablenew,i,shiftnum)==1):
                    shiftchainp1[list[i]][pl]=1
                    shiftnum=np.append(shiftnum,list[i])
            pl=pl+1
    return shiftchainp1
def enlargep2(l,n,p1in,leftnumin,chainin):
    p1=p1in.copy()
    s1=np.shape(p1)[1]
    leftnum=leftnumin.copy()
    chain=chainin.copy()
    p2=np.zeros((n,s1),dtype=int)
    for i in range(l):
        p2[leftnum[i]-1,:]=p1[i,:]
    usedclausenum=np.shape(chain)[1]
    for j in range(usedclausenum):
        c1=int(chain[0][j])
        c2=int(chain[1][j])
        d=int(chain[2][j])
        p2[d-1,:]=(p2[c1-1,:]+p2[c2-1,:])%2
    return p2
def verify(m,clausesin,p2in):
    clauses=clausesin.copy()
    p2=p2in.copy()
    s=0
    for i in range(m):
        l1=clauses[i][0]
        l2=clauses[i][1]
        l3=clauses[i][2]
        s=s+np.sum((p2[l1-1,:]+p2[l2-1,:]+p2[l3-1,:])%2)
    if(s==0):
        print("verified")
    else:
        print("not verified")
def formrepresent(leftin,chainin,n):
    chain=chainin.copy()
    left=leftin.copy()
    usedclausenum=np.shape(chain)[1]
    l=len(left)
    represenrtable=np.zeros((l+1,n),dtype=int)#第一行为常数
    I=np.zeros((l+1),dtype=int)#常数列
    I[0]=1
    for i in range(l):#初始赋值
        represenrtable[i+1][left[i]-1]=1
    for j in range(usedclausenum):
        c1=int(chain[0][j])
        c2=int(chain[1][j])
        d=int(chain[2][j])
        represenrtable[:,d-1]=I-represenrtable[:,c1-1]-represenrtable[:,c2-1]
    return represenrtable
def calculate(v1,mat):#计算v1T*mat*v2
    v=np.dot(np.array(mat),np.array(v1))
    E=np.inner(np.array(v1),v)
    return E
def formmatrix(l,tablein,clausesin):
    table=tablein.copy()
    clauses=clausesin.copy()
    I=np.zeros((l+1),dtype=int)#常数列
    I[0]=1
    QUBOmatrix=np.zeros((l,l),dtype=int)
    clausesize=np.shape(clauses)[0]
    parsize=np.shape(table)[1]
    for i in range(clausesize): #总共size:n+m 前n变量，后m语句
        a1=clauses[i][0]
        a2=clauses[i][1]
        a3=clauses[i][2]
        new=table[:,a1-1]+table[:,a2-1]+table[:,a3-1]-I
        table=np.c_[table,new]
    #print(table)
    #所有完全平方
    const=np.inner(table[0,:],table[0,:])
    for i in range(l):
        QUBOmatrix[i][i]=np.inner(table[i+1,:],table[i+1,:])
        QUBOmatrix[i][i]=QUBOmatrix[i][i]+2*np.inner(table[i+1,:],table[0,:])
        for j in range(i+1,l):
                QUBOmatrix[i][j]=2*np.inner(table[i+1,:],table[j+1,:])
    #减去前n列
    const=const-sum(table[0,0:parsize])
    for i in range(l):
        QUBOmatrix[i][i]=QUBOmatrix[i][i]-sum(table[i+1,0:parsize])
    return QUBOmatrix,const

def generaterandom(n,m):   
    clauses=generateclause(n,m)
    #printclause(clauses,m)
    n=int(len(np.unique(clauses)))
    for i in range(n):
        replace_dict = dict(zip(np.unique(clauses), range(1,n+1)))
    result=clauses.copy()
    for old, new in replace_dict.items():
        result[clauses == old] = new
    clauses=result
    leftnum,chain=reduction(clauses,n)
    #print(leftnum)
    #print(chain)
    table=formrepresent(leftnum,chain,n)
    #print(table)
    l=len(leftnum)
    efficentleftclause=calculatefficentclause(l,table,clauses)
    #print(efficentleftclause)
    p1=shiftchainpart1(efficentleftclause,l)
    #print(p1)
    p2=enlargep2(l,n,p1,leftnum,chain)
    #print(np.shape(p2))
    #verify(m,clauses,p2)
    table=formrepresent(leftnum,chain,n)
    #print(table)
    matrix,c=formmatrix(len(leftnum),table,clauses)
    return clauses,p2,matrix,c
    #np.savez("PQC"+str(n)+","+str(m)+".npz", clauses=clauses,leftnum=leftnum,chain=chain,p2=p2)
    #np.savez("QUBO"+str(n)+","+str(m)+".npz", clauses=clauses,leftnum=leftnum,chain=chain,matrix=matrix,c=c)