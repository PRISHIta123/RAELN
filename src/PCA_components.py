import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import preprocessing as prep
import pandas as pd
import matplotlib.pyplot as plt

print("\n#######PCA based estimation of Optimal Number of Features#######\n")

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--dataset', action='store', type=str, required=True)

args = my_parser.parse_args()
v=vars(args)
dataset= v["dataset"]

if dataset=="UNSWNB15":
	training_df=pd.read_csv("..//data//UNSW_datasets//UNSW_NB15_training-set.csv")

elif dataset=="NSLKDD":
	training_df=pd.read_csv("..//data//NSL_KDD_datasets//NSL_KDDTrain+.csv",header=None)

training_df= training_df.dropna()

training_df= prep.integer_encode(training_df)

training_data=np.array(training_df)

training_data= prep.normalize_dataset(training_data)

scaler = MinMaxScaler(feature_range=[0, 1])
data_rescaled = scaler.fit_transform(training_data)

#Fitting the PCA algorithm with our Data
pca = PCA().fit(data_rescaled)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Features')
plt.ylabel('Variance (%)') #for each component
plt.title('PCA Estimation of Optimal No. of Features with Explained Variance')
plt.show()
