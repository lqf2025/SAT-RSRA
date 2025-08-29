import numpy as np
from random import randint
import math,random
import generator
from pysat.solvers import Solver
from parfor import parfor
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
    print(n)
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
        print(p,list)
        clauses=np.array(replace_elements(clauses,list))
        listdelete.append(p)
    return n-len(listdelete)
    

def generatecase(n,k):#生成一个有解的问题
    m=int(np.floor(n*k))
    if(random.random()<n*k-m):
        m=m+1
    clauses,p2,matrix,c=generator.generaterandom(n,m)

    return len(np.unique(clauses)),np.shape(p2)[1],np.shape(matrix)[0],greedy(clauses)
def forms(clausesin):
    clauses=clausesin.copy()
    s = Solver(name='minisat22')
    m=np.shape(clauses)[0]
    for i in range(m):
        a,b,c=int(clauses[i][0]), int(clauses[i][1]),int(clauses[i][2])
        s.add_clause([a,b,c])
        s.add_clause([-a, -b,-c])
        s.add_clause([a, b,-c])
        s.add_clause([-a, b,c])
        s.add_clause([a,-b,c])
    return s
def count(n,k):#生成一个有解的问题
    @parfor(range(5000), disable=True)
    def single2(i):
        m=int(np.floor(n*k))
        if(random.random()<n*k-m):
            m=m+1
        clauses=generator.generateclause(n,m)
        s=forms(clauses)
        if(s.solve()==True):
            return 1
        return 0
    p=sum(np.array(single2))/5000     
    return p

if __name__ == '__main__':
    krange=0.55+np.array(range(21))*0.2/20
    nlist=np.zeros((67,21))##3-70
    klist=np.zeros((67,21))
    slist=np.zeros((67,21))
    glist=np.zeros((67,21))
    successlist=np.zeros((21))
    for pos in range(21):
        k=krange[pos]
        successlist[pos]=count(1000,k)
        for n in range(3,70):
            print(k,n)
            @parfor(range(5000), disable=True)
            def single(i):
                a1,a2,a3,a4=generatecase(n,k)
                return a1,a2,a3,a4
            single=np.array(single)
            nlist[n-3][pos]=sum(single[:,0])
            klist[n-3][pos]=sum(single[:,1])
            slist[n-3][pos]=sum(single[:,2])
            glist[n-3][pos]=sum(single[:,3])
    np.savez("reduction.npz",nlist=nlist,klist=klist,slist=slist,glist=glist,successlist=successlist)




