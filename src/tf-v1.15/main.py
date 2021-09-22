import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import LabelEncoder

from tensorflow.contrib.layers import fully_connected

import preprocessing as prep
import autoencoder as ae
import fsfc as fs
import ladder_nw as ladder
import ladder_nw_supervised as ladder_sp
import random_forest as rf
import naive_bayes as nb

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--dataset', action='store', type=str, required=True)
my_parser.add_argument('--feature_selector', action='store', type=str, required=True)
my_parser.add_argument('--classifier',action='store', type=str, required=True)

args = my_parser.parse_args()

print("\n\n########Multi-Class Classification of Network Intrusion Malwares with Reduced Label Dependency########\n")

print("\nModel Details:\n")

v=vars(args)

print("Dataset: ",v["dataset"])
print("Feature Selection Algorithm: ",v["feature_selector"])
print("Classification Algorithm: ",v["classifier"])

dataset=v["dataset"]
fs_algo=v["feature_selector"]
class_algo=v["classifier"]

if dataset=="UNSWNB15":

	training_df=pd.read_csv("../../data//UNSW_datasets//UNSW_NB15_training-set.csv")
	testing_df=pd.read_csv("../../data//UNSW_datasets//UNSW_NB15_testing-set.csv")

	training_df= training_df.dropna()
	testing_df= testing_df.dropna()

	orig= np.array(testing_df)
	orig_labels= orig[0:50000,43]

	training_df= prep.integer_encode(training_df)
	testing_df= prep.integer_encode(testing_df)

	training_data=np.array(training_df)
	testing_data=np.array(testing_df)

	training_labels=training_data[0:150000,43]
	training_features=training_data[0:150000,0:43]

	testing_labels=testing_data[0:50000,43]
	testing_features=testing_data[0:50000,0:43]

	#Perform normalization on dataset
	training_features = prep.normalize_dataset(training_features)
	testing_features = prep.normalize_dataset(testing_features)

	num_inputs=43
	num_hid=[38,30,38]
	num_output=43

	lr=0.01
	actf=tf.nn.elu

	print("\nPerforming Feature Selection... \n")
	#Selecting features from chosen dataset	
	sf=[]

	if fs_algo=="AE" or fs_algo=="AEL1" or fs_algo=="AEL2" or fs_algo=="AEDropout":
		sf= ae.Autoencoder(fs_algo,training_features, lr, actf, num_inputs, num_hid, num_output, training_df).training()

	elif fs_algo=="FSFC":
		sf= fs.FSFC(training_features, training_df).fsfc()

	elif fs_algo=="None":
		sf= list(range(43))

	sf.append(43)
	#l=len(sf)

	#choosing top l feature values from datasets to train classifier
	training_df = training_df.iloc[:,sf]
	testing_df = testing_df.iloc[:,sf]

	training_data = np.array(training_df)
	testing_data = np.array(testing_df)

	np.random.shuffle(training_data)

	if fs_algo=="None":

		data = training_data[0:150000,0:43]
		labels = training_data[0:5000,43]
		labels_sp = training_data[0:150000,43]
		test_data= testing_data[0:50000,0:43]
		tlabels= testing_data[0:50000,43]
		layer_sizes=[43, 30, 20, 10]

	else:

		data = training_data[0:150000,0:30]
		labels = training_data[0:5000,30]
		labels_sp = training_data[0:150000,30]
		test_data= testing_data[0:50000,0:30]
		tlabels= testing_data[0:50000,30]
		layer_sizes=[30, 25, 15, 10]

	data= prep.normalize_dataset(data)
	test_data= prep.normalize_dataset(test_data)

	learning_rate= 0.005
	actf1=tf.nn.elu
	actf2=tf.nn.softmax
	num_labeled=5000
	num_samples=len(data)
	num_classes=10
	batch_size=1000

	print("\nTraining Classifier and classifying Malware Samples ... \n")

	#Training and Testing accuracy of the different models on the UNSW dataset
	if class_algo=="RandomForest":
		preds=rf.RF(data,labels_sp,test_data,tlabels).training()

	elif class_algo=="NaiveBayes":
		preds=nb.NB(data,labels_sp,test_data,tlabels).training()

	elif class_algo=="LadderNW":
		preds=ladder.Ladder(data,labels,test_data,tlabels,learning_rate,actf1,actf2,layer_sizes,num_labeled,num_samples,num_classes,batch_size).training()

	elif class_algo=="LadderNWsup":
		preds=ladder_sp.Ladder(data,labels_sp,test_data,tlabels,learning_rate,actf1,actf2,layer_sizes,num_samples,num_classes,batch_size).training()

	mapping=dict.fromkeys(list(range(num_classes)))

	for i in range(0,len(testing_labels)):
		for clss in mapping.keys():
			if int(testing_labels[i])==clss:
				mapping[clss]= orig_labels[i]

	pred_labels=[]

	for i in range(0,len(preds)):
		pred_labels.append(mapping[int(preds[i])])

	with open('../../output/predictions.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["SN", "Actual Class", "Predicted Class"])

		for i in range(0,len(preds)):
			writer.writerow([i+1, orig_labels[i], pred_labels[i]])

elif dataset=="NSLKDD":

	training_df=pd.read_csv("../../data//NSL_KDD_datasets//NSL_KDDTrain+.csv",header=None)
	testing_df=pd.read_csv("../../data//NSL_KDD_datasets//NSL_KDDTest+.csv",header=None)

	training_df= training_df.dropna()
	testing_df= testing_df.dropna()

	orig= np.array(testing_df)
	orig_labels= orig[0:40000,41]

	training_df= prep.integer_encode(training_df)
	testing_df= prep.integer_encode(testing_df)

	training_data=np.array(training_df)
	testing_data=np.array(testing_df)

	training_labels=training_data[0:120000,41]
	training_features=training_data[0:120000,0:41]

	testing_labels=testing_data[0:40000,41]
	testing_features=testing_data[0:40000,0:41]

	#Perform normalization on dataset
	training_features = prep.normalize_dataset(training_features)
	testing_features = prep.normalize_dataset(testing_features)

	num_inputs=41
	num_hid=[35,30,35]
	num_output=41

	lr=0.001
	actf=tf.nn.elu

	print("\nPerforming Feature Selection... \n")
	#Selecting features from chosen dataset	
	sf=[]

	if fs_algo=="AE" or fs_algo=="AEL1" or fs_algo=="AEL2" or fs_algo=="AEDropout":
		sf= ae.Autoencoder(fs_algo,training_features, lr, actf, num_inputs, num_hid, num_output, training_df).training()

	elif fs_algo=="FSFC":
		sf= fs.FSFC(training_features, training_df).fsfc()

	elif fs_algo=="None":
		sf= list(range(41))

	sf.append(41)
	#l=len(sf)

	#choosing top l feature values from datasets to train classifier
	training_df = training_df.iloc[:,sf]
	testing_df = testing_df.iloc[:,sf]

	training_data = np.array(training_df)
	testing_data = np.array(testing_df)

	np.random.shuffle(training_data)

	if fs_algo=="None":

		data = training_data[0:120000,0:41]
		labels = training_data[0:1000,41]
		labels_sp = training_data[0:120000,41]
		test_data= testing_data[0:40000,0:41]
		tlabels= testing_data[0:40000,41]
		layer_sizes=[41, 35, 15, 5]

	else:

		data = training_data[0:120000,0:30]
		labels = training_data[0:1000,30]
		labels_sp = training_data[0:120000,30]
		test_data= testing_data[0:40000,0:30]
		tlabels= testing_data[0:40000,30]
		layer_sizes=[30, 25, 10, 5]

	data= prep.normalize_dataset(data)
	test_data= prep.normalize_dataset(test_data)

	learning_rate= 0.005
	actf1=tf.nn.elu
	actf2=tf.nn.softmax
	num_labeled=1000
	num_samples=len(data)
	num_classes=5
	batch_size=1000

	print("\nTraining Classifier and classifying Malware Samples ... \n")

	#Training and Testing accuracy of the different models on the UNSW dataset
	if class_algo=="RandomForest":
		preds=rf.RF(data,labels_sp,test_data,tlabels).training()

	elif class_algo=="NaiveBayes":
		preds=nb.NB(data,labels_sp,test_data,tlabels).training()

	elif class_algo=="LadderNW":
		preds=ladder.Ladder(data,labels,test_data,tlabels,learning_rate,actf1,actf2,layer_sizes,num_labeled,num_samples,num_classes,batch_size).training()

	elif class_algo=="LadderNWsup":
		preds=ladder_sp.Ladder(data,labels_sp,test_data,tlabels,learning_rate,actf1,actf2,layer_sizes,num_samples,num_classes,batch_size).training()

	mapping=dict.fromkeys(list(range(num_classes)))

	for i in range(0,len(testing_labels)):
		for clss in mapping.keys():
			if int(testing_labels[i])==clss:
				mapping[clss]= orig_labels[i]

	pred_labels=[]

	for i in range(0,len(preds)):
		pred_labels.append(mapping[int(preds[i])])

	with open('../../output/predictions.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["SN", "Actual Class", "Predicted Class"])

		for i in range(0,len(preds)):
			writer.writerow([i+1, orig_labels[i], pred_labels[i]])



