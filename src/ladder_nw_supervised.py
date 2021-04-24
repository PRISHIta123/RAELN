import numpy as np
import pandas as pd
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from tensorflow.contrib.layers import fully_connected

class Ladder:

    def __init__(self, training_data, labels, testing_data, t_labels, lr, actf1, actf2, layer_sizes, num_samples, num_classes, batch_size):

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
        self.num_classes = num_classes
        self.batch_size= batch_size
        self.denoising_cost = [1000.0, 10.0, 0.1, 0.1]
        self.num_batches = len(training_data)//self.batch_size

    def convert_to_one_hot(self):
        L=[]
        for i in range(0,len(self.labels)):
            l=[]
            for j in range(0,self.num_classes):
                if self.labels[i]==j:
                    l.append(1)
                else:
                    l.append(0)
            L.append(l)
        L=np.array(L)
        self.labels=L

        L=[]
        for i in range(0,len(self.t_labels)):
            l=[]
            for j in range(0,self.num_classes):
                if self.t_labels[i]==j:
                    l.append(1)
                else:
                    l.append(0)
            L.append(l)
        L=np.array(L)
        self.t_labels=L
    
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
    
    def next_batch(self):
        #num= batch size

        idy = np.arange(0 , len(self.labels))
        np.random.shuffle(idy)

        idy = idy[:self.batch_size]
        idx = idy
        
        data_shuffle_y = [self.labels[i] for i in idy]

        data_shuffle_x = [self.training_data[i] for i in idx]

        return np.asarray(data_shuffle_x),np.asarray(data_shuffle_y)


    def training(self):
        X=tf.placeholder(tf.float32,shape=[None,self.layer_sizes[0]])
        Y=tf.placeholder(tf.float32)

        self.convert_to_one_hot()

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

        noise_std=0.1

        z_corr, noise, p_enc_corr= self.encoder(X,W,beta,gamma,noise_std)
        z, n_0, p_enc = self.encoder(X,W,beta,gamma,0.0)

        #total supervised cost
        l_cost = -tf.reduce_mean(tf.reduce_sum(Y*(tf.log(p_enc_corr +tf.constant(1e-10))), 1))

        loss= l_cost 

        pred_cost = -tf.reduce_mean(tf.reduce_sum(Y*(tf.log(p_enc +tf.constant(1e-10))), 1)) #correct prediction cost

        correct_prediction = tf.equal(tf.argmax(p_enc, -1), tf.argmax(Y, -1))  # no of correct predictions
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)

        #For Per class accuracy
        pred_func= tf.argmax(p_enc,-1)
        #For Per class accuracy

        optimizer=tf.train.AdamOptimizer(self.lr)
        train=optimizer.minimize(loss)
        init=tf.global_variables_initializer()
        num_epoch=250

        with tf.device('/gpu:0'):
            with tf.Session() as sess:
                sess.run(init)
                
                for epoch in range(num_epoch):
                    
                    #learning rate decay
                    self.lr = self.lr * (0.96 **(epoch//50))
                    for iteration in range(self.num_batches):
                        X_batch, Y_batch =self.next_batch()
                        sess.run(train,feed_dict={X:X_batch, Y:Y_batch, training: True})
                    train_loss=loss.eval(feed_dict={X:X_batch, Y:Y_batch, training: True})
                    print("epoch {}".format(epoch))

                preds=sess.run(pred_func, feed_dict={X:self.training_data})
                preds1=sess.run(pred_func, feed_dict={X:self.testing_data})

                #Per Class Train
                cnt=[]
                cnt_pred=[]
                for i in range(0,self.num_classes):
                    cnt.append(0)
                    cnt_pred.append(0)

                lbls=[np.where(r==1)[0][0] for r in self.labels]
                    
                for label in lbls:
                    for i in range(0,self.num_classes):
                        if label==i:
                            cnt[i]=cnt[i]+1
                            break
                        
                for i in range(0,len(self.labels)):
                    if preds[i]==lbls[i]:
                        for j in range(0,self.num_classes):
                            if j==lbls[i]:
                                cnt_pred[j]=cnt_pred[j]+1
                                break
                per_class_acc=[]
                for i in range(0,self.num_classes):
                    per_class_acc.append(cnt_pred[i]/(cnt[i]+1e-6))
                    
                #Per Class Test
                cnt1=[]
                cnt_pred1=[]
                for i in range(0,self.num_classes):
                    cnt1.append(0)
                    cnt_pred1.append(0)

                lbls1=[np.where(r==1)[0][0] for r in self.t_labels]
                    
                for label in lbls1:
                    for i in range(0,self.num_classes):
                        if label==i:
                            cnt1[i]=cnt1[i]+1
                            break
                        
                for i in range(0,len(self.t_labels)):
                    if preds1[i]==lbls1[i]:
                        for j in range(0,self.num_classes):
                            if j==lbls1[i]:
                                cnt_pred1[j]=cnt_pred1[j]+1
                                break
                per_class_acc1=[]
                for i in range(0,self.num_classes):
                    per_class_acc1.append(cnt_pred1[i]/(cnt1[i]+1e-6))

                print("Per Class Accuracy (Training):" ,per_class_acc)
                print("Per Class Accuracy (Testing):",per_class_acc1)
                print ("Training Accuracy: ", sess.run(accuracy, feed_dict={X:self.training_data, Y:self.labels, training: False}), "%")
                print ("Testing Accuracy: ", sess.run(accuracy, feed_dict={X:self.testing_data, Y:self.t_labels, training:False}), "%")

                return preds1
        
        



        
        
        
        
        
                                

        
                                    
                                
        
        
        

    
