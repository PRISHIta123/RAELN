import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from tensorflow.contrib.layers import fully_connected

class Autoencoder_L2:
    
    def __init__(self, training_data, lr, actf, num_inputs, num_hid, num_output, training_df):
        self.training_data = training_data
        self.training_df = training_df
        self.lr = lr
        self.actf = actf
        self.num_inputs = num_inputs
        self.num_hid = num_hid
        self.num_output = num_output
        self.var_imp=[]
    
        
    def next_batch(self, num, data):
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]

        return np.asarray(data_shuffle)
    
    
    
    def variable_importance(self, l):
    
        for row in l:
            total_weight=0
            for a in row:
                total_weight+=a
            self.var_imp.append(total_weight)
    
    
    def training(self):

        X=tf.placeholder(tf.float32,shape=[None,self.num_inputs])

        initializer=tf.variance_scaling_initializer()

        w1=tf.Variable(initializer([self.num_inputs,self.num_hid[0]]),dtype=tf.float32)
        w2=tf.Variable(initializer([self.num_hid[0],self.num_hid[1]]),dtype=tf.float32)
        w3=tf.Variable(initializer([self.num_hid[1],self.num_hid[2]]),dtype=tf.float32)
        w4=tf.Variable(initializer([self.num_hid[2],self.num_output]),dtype=tf.float32)

        b1=tf.Variable(tf.zeros(self.num_hid[0]))
        b2=tf.Variable(tf.zeros(self.num_hid[1]))
        b3=tf.Variable(tf.zeros(self.num_hid[2]))
        b4=tf.Variable(tf.zeros(self.num_output))

        hid_layer1=self.actf(tf.matmul(X,w1)+b1)
        hid_layer2=self.actf(tf.matmul(hid_layer1,w2)+b2)
        hid_layer3=self.actf(tf.matmul(hid_layer2,w3)+b3)
        output_layer=self.actf(tf.matmul(hid_layer3,w4)+b4)

        
        all_weights=[w1]+[w2]+[w3]+[w4]

        l2_regularizer=tf.contrib.layers.l2_regularizer(scale=0.005,scope=None)
        regularization_penalty=tf.contrib.layers.apply_regularization(l2_regularizer,all_weights)

        loss=tf.reduce_mean(tf.square(output_layer-X))
        
        optimizer=tf.train.AdamOptimizer(self.lr)
        train=optimizer.minimize(loss + regularization_penalty)
        init=tf.global_variables_initializer()
        num_epoch=500
        batch_size=10000
        
        x=[]
        y=[]
        l=[]
        
        with tf.Session() as sess:
            sess.run(init)
            vars=tf.trainable_variables()
            vars_vals=sess.run(vars)
            
            for epoch in range(num_epoch):
                num_batches=len(self.training_data)//batch_size
                
                #learning rate decay
                self.lr = self.lr * (0.7 **(epoch//25))
                for iteration in range(num_batches):
                    X_batch=self.next_batch(batch_size,self.training_data)
                    sess.run(train,feed_dict={X:X_batch})
                train_loss=loss.eval(feed_dict={X:X_batch})
                print("epoch {} loss {}".format(epoch,train_loss))
                x.append(epoch)
                y.append(train_loss)
                
        for var, val in zip(vars, vars_vals):
            if var.get_shape()==(41, 20):
                l=val
                
        plt.rcParams['figure.figsize']=(20,20)
        plt.plot(x,y)
        plt.xlabel("Epochs")
        plt.ylabel("Training loss")
        plt.show()
        
        self.variable_importance(l)
        bf= self.select_features()

        return bf
        
        
    def select_features(self):
        features=list(range(0,self.training_data.shape[1]))
        feature_names=list(self.training_df.columns)

        plt.close()
        plt.rcParams['figure.figsize']=(20,20)
        x_vals=np.arange(len(features))
        plt.bar(x_vals,self.var_imp,align='center',alpha=1)
        plt.xticks(x_vals,features)
        plt.xlabel("Feature Indexes")
        plt.ylabel("Feature Importance Values")
        plt.show()
        
        best_features=sorted(range(len(self.var_imp)), key=lambda i: self.var_imp[i], reverse=True)[:10]
        print("Top 10 features:")
        for i in best_features:
            print(feature_names[i])

        print("Variance", np.var(self.var_imp))

        return best_features

        
