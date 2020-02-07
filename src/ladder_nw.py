import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from tensorflow.contrib.layers import fully_connected

class Ladder:

    def __init__(self, training_data, labels, lr, actf1, actf2, layer_sizes, num_labeled, num_samples, batch_size):

        self.training_data= training_data
        self.labels = labels
        self.lr = lr
        self.actf1 = actf1
        self.actf2 = actf2
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.num_labeled = num_labeled
        self.num_samples = num_samples
        self.batch_size= batch_size
        self.denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]

    def unlabeled(self, x):
        ul = tf.slice(x,[self.batch_size,0],[-1,-1])
        return ul

    def labeled(self, x):
        l = tf.slice(x,[0,0],[self.batch_size,-1])
        return l
    
    def batch_norm(self, vals):
        mean, var = tf.nn.moments(vals, axes=[0])
        return (vals - mean) / tf.sqrt(var + tf.constant(1e-10))

    def calc_var(z,n):
        mz, vz = tf.nn.moments(z,axes=[0])
        mn, vn = tf.nn.moments(n,axes=[0])
        var = tf.math.divide(vz,tf.math.add(vz,vn))
        return var

    def denoising(self, u, z_corr, var):
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

    def decoder(self, h_corr, v, z_corr, var, z, n):
        u=[]
        #denoising cost
        d_cost =[]
        l=L
        while l>=0:
            u.append(0)
            l=l-1

        l=L
        while l>=0:
            if l==L:
                u[L]=self.batch_norm(self.unlabeled(h_corr[L]))
            else:
                mul= tf.matmul(self.unlabeled(z_calc[l+1]),v[l])
                u[l]= self.batch_norm(mul)
                var = self.calc_var(self.unlabeled(z[l]),self.unlabeled(n[l]))
                z_calc[l]= self.denoising(u[l],self.unlabeled(z_corr[l]),var)
                z_calc_bn[l]= tf.batch_norm(z_calc[l])
                d_cost.append((tf.reduce_sum(tf.square(z_calc_bn[l] - self.unlabeled(z[l])),1)//self.layer_sizes[l])* self.denoising_cost[l])
            l=l-1

        return u,d_cost

    def next_batch(self, num, data, labels):
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle_x = [data[i] for i in idx]

        idy = np.arange(0 , len(labels))
        np.random.shuffle(idy)
        idy = idy[:num]
        data_shuffle_y = [data[i] for i in idy]

        return np.asarray(data_shuffle_x),np.asarray(data_shuffle_y)

    def training(self):
        X=tf.placeholder(tf.float32,shape=[None,self.layer_sizes[0]])
        labeled= tf.slice(x, [0, 0], [self.batch_size, -1])
        unlabeled= tf.slice(x, [self.batch_size, 0], [-1, -1])
        
        y=tf.placeholder(tf.float32)

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
                         beta.append(tf.Variable(0.0 * tf.ones([self.layer_sizes[l+1]])))
                         gamma.append(tf.Variable(1.0 * tf.ones([self.layer_sizes[l+1]])))

        noise_std=0.3

        z_corr, n, p_enc_corr= self.encoder(X,W,beta,gamma,noise_std)
        z, n_0, p_enc = self.encoder(X,W,beta,gamma,0.0)

        #get denoising cost of each layer
        d_cost, u= self.decoder(p_enc_corr,V,z_corr,var,z,n)

        #total unsupervised cost
        u_cost= tf.add_n(d_cost)

        y_N=labeled(p_enc_corr)

        #total supervised cost
        cost = -tf.reduce_mean(tf.reduce_sum(y*tf.log(y_N), 1))

        loss= cost + u_cost

        pred_cost = -tf.reduce_mean(tf.reduce_sum(y*tf.log(p_enc), 1)) #correct prediction cost

        correct_prediction = tf.equal(tf.argmax(p_enc, 1), tf.argmax(y, 1))  # no of correct predictions
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)

        optimizer=tf.train.AdamOptimizer(self.lr)
        train=optimizer.minimize(loss)
        init=tf.global_variables_initializer()
        num_epoch=5000

        x_vals=[]
        y_vals=[]

        with tf.Session() as sess:
            sess.run(init)
            
            for epoch in range(num_epoch):
                num_batches=len(self.training_data)//batch_size
                
                #learning rate decay
                self.lr = self.lr * (0.7 **(epoch//25))
                for iteration in range(num_batches):
                    X_batch, y_batch =self.next_batch(self.batch_size,self.training_data,self.labels)
                    sess.run(train,feed_dict={X:X_batch, Y:Y_batch, training: True})
                train_loss=loss.eval(feed_dict={X:X_batch, Y:Y_batch, training: True})
                print("epoch {} loss {}".format(epoch,train_loss))
                x_vals.append(epoch)
                y_vals.append(train_loss)

            print ("Final Accuracy: ", sess.run(accuracy, feed_dict={X:self.training_data, Y:self.labels, training: False}), "%")

        plt.rcParams['figure.figsize']=(20,20)
        plt.plot(x_vals,y_vals)
        plt.xlabel("Epochs")
        plt.ylabel("Training loss")
        plt.show()

        

        
        



        
        
        
        
        
                                

        
                                    
                                
        
        
        

    
