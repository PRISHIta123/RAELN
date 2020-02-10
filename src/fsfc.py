from __future__ import division
from math import log
import math  
from random import shuffle 

def entropy(pi):
    total = 0
    for p in pi:
        p = p / sum(pi)
        if p != 0:
            total += p * log(p, 2)
        else:
            total += 0
    total *= -1
    return total


def gain(d, a):
    total = 0
    for v in a:
        total += sum(v) / sum(d) * entropy(v)

    gain = entropy(d) - total
    return gain

def SU(X,Y):
    su = 2*gain(X,Y)/(entropy(X)+entropy(Y))
    return su

def CalculateNeighborsClass(neighbors, k): 
    count = {} 
  
    for i in range(k): 
        if neighbors[i][1] not in count: 
  
            # The class at the ith index is 
            # not in the count dict.  
            # Initialize it to 1. 
            count[neighbors[i][1]] = 1
        else: 
  
            # Found another item of class  
            # c[i]. Increment its counter. 
            count[neighbors[i][1]] += 1
  
    return count 
  
def FindMax(Dict): 
  
    # Find max in dictionary, return  
    # max value and max index 
    maximum = -1
    classification = '' 
  
    for key in Dict.keys(): 
          
        if Dict[key] > maximum: 
            maximum = Dict[key] 
            classification = key 
  
    return (classification, maximum)

def UpdateNeighbors(neighbors, item, distance, k, ): 
    if len(neighbors) < k: 
  
        # List is not full, add  
        # new item and sort 
        neighbors.append([distance, item['Class']]) 
        neighbors = sorted(neighbors) 
    else: 
  
        # List is full Check if new  
        # item should be entered 
        if neighbors[-1][0] > distance: 
  
            # If yes, replace the  
            # last element with new item 
            neighbors[-1] = [distance, item['Class']] 
            neighbors = sorted(neighbors) 
  
    return neighbors

def kNNDensity(nItem, k, Items): 
    if(k > len(Items)): 
          
        # k is larger than list 
        # length, abort 
        return "k larger than list length"; 
      
    # Hold nearest neighbors. 
    # First item is distance,  
    # second class 
    neighbors = []; 
  
    for item in Items: 
        
        # Find Euclidean Distance 
        distance = EuclideanDistance(nItem, item); 
  
        # Update neighbors, either adding 
        # the current item in neighbors  
        # or not. 
        neighbors = UpdateNeighbors(neighbors, item, distance, k); 
  
    # Count the number of each 
    # class in neighbors 
    count = CalculateNeighborsClass(neighbors, k); 
  
    # Find the max in count, aka the 
    # class with the most appearances. 
    sum=0;
    k = FindMax(count);
    for i in neighbors:
        sum+=symmetric_uncertainity(item,i);
    redundancy_degree = sum/k;
    
    return redundancy_degree



def kNN(nItem, k, Items):
    if(k > len(Items)): 
          
        # k is larger than list 
        # length, abort 
        return "k larger than list length"; 
      
    # Hold nearest neighbors. 
    # First item is distance,  
    # second class 
    neighbors = []; 
  
    for item in Items: 
        
        # Find Euclidean Distance 
        distance = EuclideanDistance(nItem, item); 
  
        # Update neighbors, either adding 
        # the current item in neighbors  
        # or not. 
        neighbors = UpdateNeighbors(neighbors, item, distance, k); 
    return neighbors

def AR(f,C):
    sum=0;
    for fi in C:
        sum=sum+SU(f,fi);
    avg_redundancy = sum/len(C);
    return avg_redundancy


def fsfc(Dict): 
    sorted_d = dict( sorted(Dict.items(), key=operator.itemgetter(1),reverse=True));
    Fs = list(sorted_d);
    fs0 = Fs[0];
    Fc=[];
    Fc.append(fs0);
    m=1;
    maxSU=0;
    for fs in Fs:
        for fc in Fc:
            if fc in kNN(fs) and fs in kNN(fc):
                Fc.append(fs);
                m=m+1;
                maxSU=max(maxSU, SU(fs,fc));
    
    C=[]
    label: step
    for p in range(0,m-1):
        C.append(Fc[p]);
    
    
    for fi in Fs:
        if fi not in Fc:
            j=0;
            for fcj in Fc:
                if(SU(fi,fcj)>j):
                    j=SU(fi,fcj);
                    x=list2.index(fcj);
            fcj=Fc[x];
            if(SU(fi,fcj)>maxSU):
                C.append(fi);
            else:
                Fc.append(fi);
                m=m+1;
                goto step;
    S=[]            
    for Cj in C:
        ar=0;
        for f in Cj:
            if(AR(f,Cj),ar):
                ar=AR(f,Cj);
                S.append(f);
        
