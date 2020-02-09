import numpy as np
import pandas as pd
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from tensorflow.contrib.layers import fully_connected

class Ladder:

    def __init__(self, training_data, labels, testing_data, t_labels, lr, actf1, actf2, layer_sizes, num_labeled, num_samples, batch_size):

        self.training_data= training_data
        self.labels = labels
        self.testing_data= testing_data
        self.t_labels = t_labels
        self.lr = lr
        self.actf1 = actf1
        self.actf2 = actf2
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.num_samples = num_samples
        self.batch_size= batch_size
        self.denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10]
        self.num_batches = len(training_data)//self.batch_size
    
    def batch_norm(self, vals):
        mean, var = tf.nn.moments(vals, axes=[0])
        return (vals - mean) / tf.sqrt(var + tf.constant(1e-10))

    def encoder(self, x, w, beta, gamma, noise_std):
        z=[]
        h=[]
        n= tf.random_normal(tf.shape(x)) * noise_std
        noise=[]
        
        z.append(x + n)
        h.append(x + n)
        noise.append(n)

        for l in range(1,self.L + 1):
            mul= tf.matmul(h[l-1],w[l-1])
            n1= tf.random_normal(tf.shape(mul)) * noise_std
            noise.append(n1)
            z.append(self.batch_norm(mul) + n1)

            if l==self.L:
                h.append(self.actf2(tf.multiply(tf.add(z[l],beta[l-1]),gamma[l-1])))
            else:
                h.append(self.actf1(tf.multiply(tf.add(z[l],beta[l-1]),gamma[l-1])))

        prob_y_x = h[self.L]
        return z, noise, prob_y_x
    
    def next_batch(self, num, data, labels):
        #num= batch size

        idy = np.arange(0 , len(labels))
        np.random.shuffle(idy)

        idy = idy[:num]
        idx = idy
        
        data_shuffle_y = [labels[i] for i in idy]

        data_shuffle_x = [data[i] for i in idx]

        return np.asarray(data_shuffle_x),np.asarray(data_shuffle_y)


    def training(self):
        X=tf.placeholder(tf.float32,shape=[None,self.layer_sizes[0]])
        Y=tf.placeholder(tf.float32)

        training = tf.placeholder(tf.bool)

        initializer=tf.variance_scaling_initializer()

        W=[]
        V=[]
        beta=[]
        gamma=[]
   
        for l in range(self.L):
                         w=tf.Variable(tf.random_normal((self.layer_sizes[l],self.layer_sizes[l+1]), seed=0))/ math.sqrt(self.layer_sizes[l])
                         W.append(w)
                         v=tf.Variable(tf.random_normal((self.layer_sizes[l+1],self.layer_sizes[l]), seed=0))/ math.sqrt(self.layer_sizes[l+1])
                         V.append(v)


        for l in range(self.L):
                         beta.append(tf.Variable(0.0 * tf.ones([self.layer_sizes[l+1]])))
                         gamma.append(tf.Variable(1.0 * tf.ones([self.layer_sizes[l+1]])))

        noise_std=0.3

        z_corr, noise, p_enc_corr= self.encoder(X,W,beta,gamma,noise_std)
        z, n_0, p_enc = self.encoder(X,W,beta,gamma,0.0)

        #total supervised cost
        l_cost = -tf.reduce_mean(tf.reduce_sum(Y*(tf.log(p_enc_corr +tf.constant(1e-10))), 1))

        loss= l_cost 

        pred_cost = -tf.reduce_mean(tf.reduce_sum(Y*(tf.log(p_enc +tf.constant(1e-10))), 1)) #correct prediction cost

        correct_prediction = tf.equal(tf.argmax(p_enc, 1), tf.argmax(Y, 1))  # no of correct predictions
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)

        optimizer=tf.train.AdamOptimizer(self.lr)
        train=optimizer.minimize(loss)
        init=tf.global_variables_initializer()
        num_epoch=1000

        x_vals=[]
        y_vals=[]

        with tf.Session() as sess:
            sess.run(init)
            
            for epoch in range(num_epoch):
                
                #learning rate decay
                self.lr = self.lr * (0.96 **(epoch//25))
                for iteration in range(self.num_batches):
                    X_batch, Y_batch =self.next_batch(self.batch_size,self.training_data,self.labels)
                    sess.run(train,feed_dict={X:X_batch, Y:Y_batch, training: True})
                train_loss=loss.eval(feed_dict={X:X_batch, Y:Y_batch, training: True})
                print("epoch {} loss {}".format(epoch,train_loss))
                x_vals.append(epoch)
                y_vals.append(train_loss[0])

            #print ("Final Accuracy: ", sess.run(accuracy, feed_dict={X:self.testing_data, Y:self.t_labels}), "%")

        plt.rcParams['figure.figsize']=(20,20)
        plt.plot(x_vals,y_vals)
        plt.xlabel("Epochs")
        plt.ylabel("Training loss")
        plt.show()

        

        
        



        
        
        
        
        
                                

        
                                    
                                
        
        
        

    
