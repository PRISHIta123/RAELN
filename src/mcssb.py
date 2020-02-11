import numpy as np
import pandas as pd
import math
import tensorflow as tf
import matplotlib.pyplot as plt

class MCSSB:

    def __init(self, training_data, labels, test_data, tlabels, num_samples, num_labeled):
        self.training_data = training_data
        self.labels = labels
        self.ohe_labels= np.zeros((self.labels.size, self.labels.max()+1))
        self.num_classes= 10
        self.test_data= test_data
        self.tlabels= tlabels
        self.num_samples= num_samples
        self.num_labeled = num_labeled
        self.S= []
        self.Z_ul= []
        self.Z_l= []
        self.b = []
        self.tau= []
        self.phi= []

    def one_hot_encode(self):
        self.ohe_labels[np.arange(self.labels.size),self.labels] = 1

    
    def set_S(self):

        for i in range(0,self.num_samples):
            l=[]
            for j in range(0,self.num_samples):
                if i==j:
                    l.append(0)
                #calculate cosine similarity between x_i and x_j
                else:
                    a,b= self.training_data[i],self.training_data[j]
                    dot = np.dot(a,b)
                    norma = np.linalg.norm(a)
                    normb = np.linalg.norm(b)
                    cos = dot / (norma * normb)
                    l.append(cos)    
            self.S.append(l)

    def set_b(self, pred_y):

        for i in range(num_labeled, num_samples):
            den=0
            for x in range(0,self.num_classes):
                den= den+ exp(pred_y[i][x])

            b_i=[]
            for x in range(0,self.num_classes):
                b_i.append(exp(pred_y[i][x])//den)

            self.b.append(b_i)

    def set_tau(self):

        for i in range(self.num_labeled, self.num_samples):
            l1=[]
            for j in range(self.num_labeled, self.num_samples):
                sm=0
                l=[]
                for k in range(0,self.num_classes):
                    prod= b[i][k]*b[j][k]
                    sm=sm +prod
                for k in range(0,self.num_classes):
                    x= (b[i][k]*b[j][k])//sm
                    l.append(x)
                l1.append(l)
            self.tau.append(l1)
                

    def set_phi(self):

        for i in range(0, self.num_labeled):
            l1=[]
            for j in range(self.num_labeled, self.num_samples):
                l=[]
                for k in range(0,self.num_classes):
                    con= np.dot(b[i][k],self.ohe_labels[j][k])
                    sm=0
                    for k1 in range(0,self.num_classes):
                        sm=sm+ (1//b[i][k1])
                    x= con*sm
                    l.append(x)
                l1.append(l)
            self.phi.append(l1)
                    
        

    def set_Z_ij_ul(self, i, j):

        Z_i_j = np.dot(b[i],b[j])

        return Z_i_j

    def set_Z_ul(self,pred_y):
        
        for i in range(self.num_labeled, self.num_samples):
            l=[]
            for j in range(self.num_labeled, self.num_samples):
                z= self.set_Z_ij_ul(i,j)
                l.append(z)
            self.Z_u.append(l)


    #inconsistency of unlabeled samples
    def F_u(self):

        fu=0

        for i in range(self.num_labeled, self.num_samples):
            for j in range(self.num_labeled, self.num_samples):
                fu= fu + (self.S[i][j]//self.Z_u[i][j])

        return fu
        

    def set_Z_ij_l_ul(self, i, j):

        Z_i_j = np.dot(self.ohe_labels[i],b[j])

        return Z_i_j


    def set_Z_l(self,pred_y):
        
        for i in range(0, self.num_labeled):
            l=[]
            for j in range(self.num_labeled, self.num_samples):
                z= self.set_Z_ij_l(i,j)
                l.append(z)
            self.Z_l.append(l)
            

    #inconsistency of unlabeled samples with labeled samples
    def F_l(self):

        fl=0

        for i in range(0, self.num_labeled):
            for j in range(self.num_labeled, self.num_samples):
                fl= fl + (self.S[i][j]//self.Z_l[i][j])

        return fl

    def obj_func(self, fu, fl):

        return fu + 10000*fl

    def training(self):

        training_data_ul= self.training_data[self.num_labeled:self.num_samples]

        l=len(training_data_ul)
        training_data_ul= np.random.shuffle(training_data_ul)
        s= random.randrange(1,l+1)

        sample_data= training_data_ul[0:s]
        
        T= 1000
        alpha=1

        for iter in range(0,T):
            if alpha<=0:
                break
            else:
                for i in range(0,s):
                    for k in range(0,self.num_classes):
                        alpha_i_k= 0
                        beta_i_k= 0
                        
                        for j in range(self.num_labeled, self.num_samples):
                            alpha_i_k= alpha_i_k +((self.S[i][j]*(self.b[i][k]- self.tau[i][j][k]))//self.Z_u[i][j])

                        for j in range(0,self.num_labeled):
                            beta_i_k= beta_i_k +(self.S[i][j]*self.phi[i][j][k])

                        beta_i_k = 0.5*beta_i_k
            
    
    


    
            

        
            
                

        
        
        

    

    
        

        
        
