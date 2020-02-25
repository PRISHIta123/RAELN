import numpy as np
import pandas as pd
import math
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


class MCSSB:

    def __init__(self, training_data, labels, test_data, tlabels, num_samples, num_labeled, num_classes):
        self.training_data = training_data
        self.labels = labels
        self.num_classes = num_classes
        self.ohe_labels= np.zeros((len(self.labels), num_classes))
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

        for i in range(self.num_labeled, self.num_samples):
            den=0
            for x in range(0,self.num_classes):
                den= den+ exp(pred_y[i][x])

            b_i=[]
            for x in range(0,self.num_classes):
                b_i.append(exp(pred_y[i][x])//den)

            self.b.append(b_i)

        print(self.b[0])
            

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

        for i in range(self.num_labeled, self.num_samples):
            l1=[]
            for j in range(0,self.num_labeled):
                l=[]
                for k in range(0,self.num_classes):
                    con= self.b[i][k]*self.ohe_labels[j][k]
                    sm=0
                    for k1 in range(0,self.num_classes):
                        sm=sm+ (1//self.b[i][k1])
                    x= con*sm
                    x= x- (self.ohe_labels[i][k]//self.b[i][k])
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
        training_data_l= self.training_data[0:self.num_labeled]

        T= 50
        alpha=1
        H=[]
        h_actual=[]

        for iter in range(0,T):
            w=[]
            #H=y_pred
            y_pred=[]
            for i in range(self.num_labeled, self.num_samples):
                alpha_i=[]
                beta_i=[]
                w1=[]
                for k in range(0,self.num_classes):
                    alpha_i.append(0)
                    beta_i.append(0)

                for k in range(0,self.num_classes):
                        
                    for j in range(self.num_labeled, self.num_samples):
                        alpha_i[k]= alpha_i[k] +((self.S[i][j]*(self.b[i][k]- self.tau[i][j][k]))//self.Z_u[i][j])

                    for j in range(0,self.num_labeled):
                        beta_i[k]= beta_i[k] +(self.S[i][j]*self.phi[i][j][k])

                    beta_i[k] = 0.5*beta_i[k]
                    w1.append(alpha_i[k] + 1000*beta_i[k])

                y_pred.append(w1.index(max(w1)))
                w.append(max(w1))
                    
            s=max(30000,self.num_samples//5)

            y_pred=np.array(H)

            indexes= random.choices(np.arange(self.num_labeled,self.num_samples),w,s)
            sample_data_ul= [self.training_data[i] for i in indexes]
            y_pred = [y_pred[i] for i in indexes]
            sample_data_ul= np.array(sample_data_ul)
            y_pred= np.array(y_pred)

            acc_training_data= np.vstack((training_data_l, sample_data_ul))
            acc_labels= np.vstack((self.labels,y_pred))

            classifier = RandomForestClassifier(criterion='entropy')
            classifier.fit(acc_training_data, acc_labels)

            test_data= training_data_ul

            h=[]
            for i in range(self.num_labeled, self.num_samples):
                x = classifier.predict(test_data[i])
                l=[]
                for k in range(0,self.num_classes):
                    if k==x:
                        l.append(1)
                    else:
                        l.append(0)
                h.append(l)

            A_u=0
            A_l=0
            B_u=0
            B_l=0

            #Calculate A_u
            for i in range(self.num_labeled,self.num_samples):
                for j in range(self.num_labeled,self.num_samples):
                    pre=(self.S[i][j]//self.Z_u[i][j])
                    sm=0
                    for k in range(0,self.num_classes):
                        sm= sm + h[i][k]*self.b[i][k]
                    prod=pre*sm
                    A_u= A_u+prod

            #Calculate A_l
            for i in range(0,self.num_labeled):
                for j in range(self.num_labeled,self.num_samples):
                    pre=self.S[i][j]
                    sm=0
                    for k in range(0,self.num_classes):
                        for k1 in range(0,self.num_classes):
                            sm= sm + ((self.ohe_labels[i][k]*self.b[j][k1]*h[j][k1])//self.b[j][k])
                    prod=pre*sm
                    A_l=A_l+prod

            A_l=A_l*0.5

            #Calculate B_u
            for i in range(self.num_labeled,self.num_samples):
                for j in range(self.num_labeled,self.num_samples):
                    pre=(self.S[i][j]//self.Z_u[i][j])
                    sm=0
                    for k in range(0,self.num_classes):
                        sm= sm + h[i][k]*self.tau[i][j][k]
                    prod=pre*sm
                    B_u= B_u+prod

            #Calculate B_l
            for i in range(0,self.num_labeled):
                for j in range(self.num_labeled,self.num_samples):
                    pre=self.S[i][j]
                    sm=0
                    for k in range(0,self.num_classes):
                        sm= sm + ((self.ohe_labels[i][k]*self.h[j][k])//self.b[j][k])
                    prod=pre*sm
                    B_l=B_l+prod

            B_l=B_l*0.5

            num= B_u + 1000*B_l
            den= A_u + 1000*A_l

            alpha= 0.25*math.log(num//den)
            print(alpha)
            if alpha<=0:
                break
            else:
                for i in range(self.num_labeled, self.num_samples):
                    for k in range(0,self.num_classes):
                        if iter==0:
                            H[i][k]=y_pred[i][k]
                        H[i][k]= H[i][k]+ alpha*h[i][k]
                h_actual=h

        cnt=0
        cntt=0
        for i in range(self.num_labeled, self.num_samples):
            for k in range(0,self.num_classes):
                if H[i][k]==1 and H[i][k]==h[i][k]:
                    cnt=cnt + 1

            cntt=cntt+1

        print("Accuracy: ",((cnt/cntt)*100),"%")

            

            

            

        

            
                

                

                
                

                

                
                    
                    

                    
            
    
    


    
            

        
            
                

        
        
        

    

    
        

        
        
