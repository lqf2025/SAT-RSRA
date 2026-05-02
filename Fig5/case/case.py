import generator
import random
import numpy as np
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
        if(np.sum(m2)>0.5 and np.sum(m2)<1.5):
            break
    return m,clauses,p2,matrix,c
def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
def replace_elements(matrix, elements_to_zero):
    target_set = set(elements_to_zero)
    return [
        [0 if elem in target_set else elem for elem in row]
        for row in matrix
    ]
def greedy(clausesin):
    clauses=clausesin.copy()
    n=np.max(clauses)
    #print(n)
    listdelete=[]
    while(np.max(clauses)!=0):
        flattened = clauses.ravel()
        freq = np.bincount(flattened, minlength=n+1)[1:n+1]
        min_val = min(freq[freq != 0])
        min_indices = [i for i, val in enumerate(freq) if val == min_val]
        p=random.choice(min_indices)+1
        rows_with_p = np.any(clauses == p, axis=1)
        elements = clauses[rows_with_p, :]
        list=elements.flatten()
        list=remove_duplicates(list)
        #print(p,list)
        clauses=np.array(replace_elements(clauses,list))
        listdelete.append(p)
    return n-len(listdelete),listdelete
for n in range(5,21):
    for run in range(0,10):
            m,clauses,p2,matrix,c=generatecase(n,0.626)
            size,list=greedy(clauses)
            print(np.max(clauses),np.shape(matrix)[0],np.shape(p2)[1],size)
            np.savez("case/experimentcase"+str(n)+" "+str(run)+".npz",m=m,clauses=clauses,p2=p2,matrix=matrix,c=c,size=size,list=list)