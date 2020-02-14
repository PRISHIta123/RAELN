import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from tensorflow.contrib.layers import fully_connected

import preprocessing as prep
import autoencoder as ae
import autoencoder_l1 as ae_l1
import autoencoder_l2 as ae_l2
import autoencoder_dropout as ae_dropout
import fsfc as fs
import ladder_nw as ladder
import ladder_nw_supervised as ladder_sp
import mcssb as mcssb

training_df=pd.read_csv("C://Users//PrishitaRay//Desktop//Malware_Classification_using_ML//data//UNSW_datasets//UNSW_NB15_training-set.csv")
testing_df=pd.read_csv("C://Users//PrishitaRay//Desktop//Malware_Classification_using_ML//data//UNSW_datasets//UNSW_NB15_testing-set.csv")

training_df= training_df.dropna()
testing_df= testing_df.dropna()

training_df= prep.integer_encode(training_df)
testing_df= prep.integer_encode(testing_df)

training_data=np.array(training_df)
testing_data=np.array(testing_df)

training_labels=training_data[:,43]
training_features=training_data[:,0:43]

testing_labels=testing_data[:,43]
testing_features=testing_data[:,0:43]

#Perform normalization on dataset
training_features = prep.normalize_dataset(training_features)
testing_features = prep.normalize_dataset(testing_features)

num_inputs=43
num_hid=[22,10,22]
num_output=43

lr=0.01
actf=tf.nn.elu

#bf1=ae.Autoencoder(training_features, lr, actf, num_inputs, num_hid, num_output, training_df).training()
#print(bf1)
#bf2=ae_l1.Autoencoder_L1(training_features, lr, actf, num_inputs, num_hid, num_output, training_df).training()
#print(bf2)
#bf3=ae_l2.Autoencoder_L2(training_features, lr, actf, num_inputs, num_hid, num_output, training_df).training()
#print(bf3)
#bf4=ae_dropout.Autoencoder_Dropout(training_features, lr, actf, num_inputs, num_hid, num_output, training_df).training()
#print(bf4)

bf=fs.FSFC(training_features,training_df).fsfc()
print(bf)

sf=bf3

sf.append(43)

#choosing top 10 feature values from datasets to train classifier
training_df = training_df.ix[:,sf]
testing_df = testing_df.ix[:,sf]

training_data = np.array(training_df)
testing_data = np.array(testing_df)

np.random.shuffle(training_data)

data = training_data[:,0:10]
labels = training_data[0:5000,10]
labels_sp = training_data[:,10]
test_data= testing_data[:,0:10]
tlabels= testing_data[:,10]


data= prep.normalize_dataset(data)
test_data= prep.normalize_dataset(test_data)

learning_rate= 0.001
actf1=tf.nn.elu
actf2=tf.nn.softmax
layer_sizes=[10, 8, 4, 10]
num_labeled=5000
num_samples=len(data)
num_classes=10
batch_size=50000

#ladder.Ladder(data,labels,test_data,tlabels,learning_rate,actf1,actf2,layer_sizes,num_labeled,num_samples,num_classes,batch_size).training()
ladder_sp.Ladder(data,labels_sp,test_data,tlabels,learning_rate,actf1,actf2,layer_sizes,num_samples,num_classes,batch_size).training()
