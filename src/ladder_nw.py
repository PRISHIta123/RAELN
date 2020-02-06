import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from tensorflow.contrib.layers import fully_connected

class Ladder:

    def __init__(self, training_features, training_labels, lr, actf1, actf2, layer_sizes, num_labeled, num_samples):

        self.training_features = training_features
        self.training_labels = training_labels
        self.lr = lr
        self.actf1 = actf1
        self.actf2 = actf2
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.num_labeled = num_labeled
        self.num_samples = num_samples

    def batch_norm(self, vals):
        mean, var = tf.nn.moments(vals, axes=[0])
        return (vals - mean) / tf.sqrt(var + tf.constant(1e-10))

    def calc_var(z,n):
        mz, vz = tf.nn.moments(z,axes=[0])
        mn, vn = tf.nn.moments(n,axes=[0])
        var = tf.math.divide(vz,tf.math.add(vz,vn))
        return var

    def denoising(self, z_corr, u, var):
        z_calc=0
        z_calc= var*z_corr + (1-var)*u
        return z_calc

    def encoder(self, x, w, beta, gamma, noise_std):
        z=[]
        h=[]
        n= tf.random_normal(tf.shape(x)) * noise_std
        z[0] = x + n
        h[0] = x + n

        for l in range(1,self.L + 1):
            mul= tf.matmul(h[l-1],w[l-1])
            z[l]= self.batch_norm(mul) + tf.random_normal(tf.shape(mul)) * noise_std

            if l==self.L:
                h[l]= self.actf2(tf.math.multiply(gamma[l],tf.math.add(z[l],beta[l])))
            else:
                h[l]= self.actf1(tf.math.multiply(gamma[l],tf.math.add(z[l],beta[l])))

        prob_y_x = h[self.L]
        return z, n, prob_y_x

    def decoder(self, h_corr, v, z_corr, var):
        u=[]
        l=L
        while l>=0:
            u.append(0)
            l=l-1

        l=L
        while l>=0:
            if l==L:
                u[L]=self.batch_norm(h_corr[L])
            else:
                z_calc[l]= self.denoising(u[l],z_corr[l],var)
                mul= tf.matmul(z_calc[l],v[l])
                u[l]= self.batch_norm(mul)
            l=l-1

        return u

    def next_batch(self, num, data):
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]

        return np.asarray(data_shuffle)

    def training(self):

        X=tf.placeholder(tf.float32,shape=[None,self.layer_sizes[0])

        initializer=tf.variance_scaling_initializer()

        W=[]
        V=[]
        beta=[]
        gamma=[]
                         
        for l in range(self.L):
                         w=tf.Variable(tf.random_normal(self.layer_sizes[l],self.layer_sizes[l+1]),dtype=tf.float32)
                         W.append(w)
                         v=tf.Variable(tf.random_normal(self.layer_sizes[l+1],self.layer_sizes[l]),dtype=tf.float32)
                         V.append(v)


        for l in range(self.L):
                         beta.append(tf.Variable(0.0 * tf.ones([self.layer_sizes[l+1]]))
                         gamma.append(tf.Variable(1.0 * tf.ones([self.layer_sizes[l+1]]))

        noise_std=0.3

        z_corr, n, p_enc_corr= self.encoder(X,W,beta,gamma,noise_std)
        z, n_0, p_enc = self.encoder(X,W,beta,gamma,0.0)
        var= self.calc_var(z,n)
        u= self.decoder(p_enc_corr,V,z_corr,var)
        
        
                                

        
                                    
                                
        
        
        

    
