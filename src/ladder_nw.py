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
        self.num_labeled = num_labeled
        self.num_samples = num_samples
        self.batch_size= batch_size
        self.denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10]
        self.num_batches = len(training_data)//self.batch_size
        self.num_l_per_batch = self.num_labeled//self.num_batches

    def unlabeled(self, x):
        ul = tf.slice(x,[1714,0],[-1,-1])
        return ul

    def labeled(self, x):
        l = tf.slice(x,[0,0],[1714,-1])
        return l
    
    def batch_norm(self, vals):
        mean, var = tf.nn.moments(vals, axes=[0])
        return (vals - mean) / tf.sqrt(var + tf.constant(1e-10))

    def calc_var(self,z,n):
        mz, vz = tf.nn.moments(z,axes=[0])
        mn, vn = tf.nn.moments(n,axes=[0])
        var = tf.divide(vz,tf.add(vz,vn))
        return var

    def denoising(self, u, z_corr, var):
        z_calc=0
        z_calc= var*z_corr + (1-var)*u
        return z_calc

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

    def decoder(self, h_corr, v, z_corr, z, noise):
        u=[]
        z_calc=[]
        z_calc_bn=[]
        #denoising cost
        d_cost =[]
        l=self.L
        while l>=0:
            u.append(0)
            z_calc.append(0)
            z_calc_bn.append(0)
            l=l-1

        l=self.L
        while l>=0:
            if l==self.L:
                h_corr_ul=self.unlabeled(h_corr)
                u[l]=self.batch_norm(h_corr_ul)
            else:
                mul= tf.matmul(z_calc[l+1],v[l])
                u[l]= self.batch_norm(mul)
            var = self.calc_var(self.unlabeled(z[l]),self.unlabeled(noise[l]))
            z_calc[l]= self.denoising(u[l],self.unlabeled(z_corr[l]),var)
            z_calc_bn[l]= self.batch_norm(z_calc[l])
            d_cost.append((tf.reduce_sum(tf.square(z_calc_bn[l] - self.unlabeled(z[l])),1)//self.layer_sizes[l])* self.denoising_cost[l])
            l=l-1

        return u,d_cost

    def next_batch(self, num, data, labels):
        #num= batch size
        #num_l_per_batch= number of labels per batch
        #n_ex_pc= number of labels per class
        
        num_classes= 10
        n_ex_pc = self.num_l_per_batch//num_classes
        
        idy = np.arange(0 , len(labels))
        np.random.shuffle(idy)
        idy = idy.tolist()
        y=[]
        for i in range(num_classes):
            y.append(0)

        indexes=[]

        for i in idy:
            for j in range(num_classes):
                if labels[i]==j and y[j]< n_ex_pc:
                    indexes.append(i)
                    y[j]=y[j]+1
                    break
                
        indexes= np.array(indexes)
        data_shuffle_y = [labels[i] for i in indexes]

        data_labeled = [data[i] for i in indexes]

        index_all = np.arange(0, len(data))
        np.random.shuffle(index_all)

        idx = []

        for i in index_all:
            if i not in indexes:
                idx.append(i)


        num_unlabeled = num - len(data_shuffle_y)
        idx = idx[:num_unlabeled]
        data_unlabeled = [data[i] for i in idx]

        data_shuffle_x= []
        for x in data_labeled:
            data_shuffle_x.append(x)

        for x in data_unlabeled:
            data_shuffle_x.append(x)

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

        #get denoising cost of each layer
        u, d_cost= self.decoder(p_enc_corr,V,z_corr,z,noise)

        #total unsupervised cost
        ul_cost= tf.add_n(d_cost)
        print(type(ul_cost))

        p_enc_corr_l=self.labeled(p_enc_corr)

        #total supervised cost
        l_cost = -tf.reduce_mean(tf.reduce_sum(Y*(tf.log(p_enc_corr_l+tf.constant(1e-10))), 1))
        print(type(l_cost))

        loss= l_cost + ul_cost

        pred_cost = -tf.reduce_mean(tf.reduce_sum(Y*(tf.log(p_enc+tf.constant(1e-10))), 1)) #correct prediction cost

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
                self.lr = self.lr * (0.95 **(epoch//25))
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

        

        
        



        
        
        
        
        
                                

        
                                    
                                
        
        
        

    
