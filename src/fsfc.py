from __future__ import division
from math import log
import math  
from random import shuffle
import numpy as np

class FSFC:

    def __init__(self, training_features):
        self.training_features = training_features
        self.n= training_features.shape[1]

    def convert_cont_to_discrete(self):
        for i in range(0,self.n):
            mean= np.mean(self.training_features[:,i])
            var= np.var(self.training_features[:,i])
            bins= np.array([mean-1.5*var, mean-var, mean-0.5*var, mean, mean+0.5*var, mean+var, mean+1.5*var])
            inds= np.digitize(self.training_features[:,i],bins)
            self.training_features[:,i]= inds
                              
    def entropy(self, i):
        f = self.training_features[:,i]

        unique,counts= np.unique(f, return_counts=True)
        l=len(f)

        d= dict(zip(unique,counts))

        probs=[]
        for v in d.values():
            probs.append(v/l)

        H= 0 
        for p in probs:
            if p != 0:
                H = H + p * math.log(p, 2)
            else:
                H= H + 0
        H = H*-1
        return H

    def gain(self, i, j):

        total = 0
        f_i = self.training_features[:,i]
        f_j = self.training_features[:,j]

        unique_i,counts_i= np.unique(f_i, return_counts=True)
        unique_j,counts_j= np.unique(f_j, return_counts=True)

        di=dict(zip(unique_i,counts_i))
        dj=dict(zip(unique_j,counts_j))

        vals_i=list(di.keys())
        vals_j=list(dj.keys())

        #dij is a dictionary containing count of each y value with respect to every x value
        dij=dict.fromkeys(vals_i)
        for key in dij.keys():
            dij[key]=[]
            for x in range(0,len(vals_j)):
                dij[key].append(0)

        for x in range(0,len(f_j)):
            for y in range(0,len(vals_j)):
                if f_j[x]==vals_j[y]:
                    for key in dij.keys():
                        if f_i[x]==key:
                            dij[key][y]= dij[key][y] + 1

        for key in dij.keys():
            prod=1
            pre= di[key]/len(f_i)
            sm=0
            for y in dij[key]:
                p=y/di[key]
                if p!=0:
                    sm = sm + p*math.log(p,2)
                else:
                    sm = sm + 0
            sm=sm*-1
            prod= pre*sm
            total= total + prod
                
        IG = self.entropy(i) - total
        return IG

    def SU(self, i, j):
        su = 2*self.gain(i,j)/(self.entropy(i)+self.entropy(j))
        return su

    def kNNDensity(self, sym_unc, i, k): 
        num=0

        for x in sym_unc[i]:
            num=num+x
            
        redundancy_degree = num/k
        
        return redundancy_degree


    def kNN(self, i ,k, sym_unc):
        neighbours=[]

        indexes= list(range(0,len(sym_unc[i])))
        d=dict.fromkeys(indexes)
        for key in d.keys():
            d[key]=sym_unc[i][key]

        su=list(d.values())
        sorted_su=sorted(su, key=float, reverse=True)

        for x in sorted_su:
            for key in d.keys():
                if d[key]==x:
                    neighbours.append(key)

        cnt=0
        knn=[]
        for n in neighbours:
            knn.append(n)
            cnt=cnt+1
            if cnt==k:
                break
        
        return knn

    def AR(self, i_f, C, sym_unc):
        sm=0
        for feature_index in C:
            sm= sm+ sym_unc[i_f][feature_index]
            
        avg_redundancy = sm/len(C)
        return avg_redundancy


    def fsfc(self):
        k=4
        n= self.training_features.shape[1]
        self.convert_cont_to_discrete()
        Fs=[]
        Fc=[]
        m=0

        #Calculate symmetric uncertainty of every pair of features
        sym_unc=[]

        for i in range(0,n):
            l=[]
            for j in range(0,n):
                if i==j:
                    l.append(0)
                else:
                    l.append(self.SU(i,j))
            sym_unc.append(l)

        dknn=[]

        for i in range(0,n):
            dknn.append(self.kNNDensity(sym_unc,i,k))

        F=list(range(0,n))

        dknn_map=dict.fromkeys(F)

        for key in dknn_map.keys():
            dknn_map[key]=dknn[key]

        vals = list(dknn_map.values())
        new_vals= sorted(vals, key=float, reverse=True)
        new_keys=[]

        for x in new_vals:
            for key in dknn_map.keys():
                if dknn_map[key]==x:
                    new_keys.append(key)
                    
        #contains the keys of the sorted features
        sorted_features= new_keys
        Fs = sorted_features
        fs0 = Fs[0]
        Fc.append(fs0)
        #m=1
        m=m+1

        maxSU=0
        for fs in Fs:
            i= fs
            for fc in Fc:
                j= fc
                if j not in self.kNN(i,k,sym_unc) and i not in self.kNN(j,k,sym_unc):
                    Fc.append(fs)
                    m=m+1
                    
                    maxSU=max(maxSU, sym_unc[i][j])
        
        C=[]
        
        for p in range(0,m-1):
            i=Fc[p]
            l=[]
            l.append(i)
            C.append(l)

        print(C)
        
        for fi in Fs:
            if fi not in Fc:
                i=fi
                pos=0
                for fcj in Fc:
                    cj=fcj
                    if(sym_unc[i][cj]>j):
                        j=sym_unc[i][cj]
                        pos=cj
                        
                if(sym_unc[i][pos]> maxSU):
                    C[pos].append(i)
                else:
                    Fc.append(i)
                    m=m+1
                    l=[]
                    l.append(i)
                    C.append(l)

        S=[]            
        for Cj in C:
            ar=0
            pos=0
            for i_f in Cj:
                if self.AR(i_f,Cj,sym_unc)>ar:
                    ar=self.AR(i_f,Cj,sym_unc)
                    pos=i_f
            S.append(pos)
            
        return S
            
