import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from tensorflow.contrib.layers import fully_connected

class Ladder:

    def __init__(self, training_features, training_labels, lr, actf1, actf2, num_layers, num_labeled, num_samples):

        self.training_features = training_features
        self.training_labels = training_labels
        self.lr = lr
        self.actf1 = actf1
        self.actf2 = actf2
        self.num_layers = num_layers
        self.num_labeled = num_labeled
        self.num_samples = num_samples

    def join(labeled, unlabeled):
        return tf.concat([labeled, unlabeled],0)

    def split(inputs, batch_size):
        labeled = tf.slice(inputs, [0, 0], [batch_size, -1])
        unlabeled = tf.slice(inputs, [batch_size, 0], [-1, -1])
        return labeled, unlabeled

    def gen_weights(shape):
        return tf.Variable(tf.random_normal(shape) / math.sqrt(shape[0])

    def gen_BNP(inits, size):
        return tf.Variable(inits * tf.ones([size])

    def batch_normalization(self, batch):

        mean, var = tf.nn.moments(batch, axes=[0])
        return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

    def encoder(self, inputs, noise_std, batch_size):

        h = inputs + tf.random_normal(tf.shape(inputs))*noise_std
        var=['z','m','v','h']

        #to store the pre-activation, mean, variance, activation for each layer
        #Variables for labeled and unlabeled examples
        labeled = dict.fromkeys(var,None)
        unlabeled = dict.fromkeys(var,None)

        for x in var:
            labeled[x] = {}
            unlabeled[x] = {}

        labeled['z'][0]= tf.slice(h, [0, 0], [batch_size, -1])
        unlabeled['z'][0]= tf.slice(h, [batch_size, 0], [-1, -1])

        for l in range(1, L+1):
            labeled['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h)
            z_pre = tf.matmul(h, weights['W'][l-1])  # pre-activation
            z_pre_l, z_pre_u = split_lu(z_pre)  # split labeled and unlabeled examples

        m, v = tf.nn.moments(z_pre_u, axes=[0])

        return h, labeled, unlabeled

    def decoder(self,):


    def training(self, noise_std):
        inputs = tf.placeholder(tf.float32,shape=[None,self.num_layers[0])
        outputs = tf.placeholder(tf.float32,shape=[None,1])

        batch_size=10000
        
        y_c, corr_vals_l, corr_vals_ul = encoder(inputs, noise_std, batch_size)
        y, vals_l, vals_ul = encoder(inputs,0.0)

        num_epoch=1000
        num_iter = (self.num_samples/batch_size) * num_epoch

        noise_std=0.3

        shapes = zip(num_layers[:-1], num_layers[1:])

        L= len(num_layers)- 1

        hp=['W','V','beta','gamma']

        hyper_param= dict.fromkeys(hp)

        for x in hp:
            hyper_param[hp]=[]

        for s in shapes:
            hyper_param['W'].append(self.gen_weights(s))
            hyper_param['V'].append(self.gen_weights(s[::-1])

        for l in range(L):
            hyper_param['beta'].append(self.gen_BNP(0.0,num_layers[l+1]))
            hyper_param['gamma'].append(self.gen_BNP(1.0,num_layers[l+1]))

        #Not sure what to do with this
        #denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]

        
                                

        
                                    
                                
        
        
        

    
