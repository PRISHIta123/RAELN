import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import preprocessing as prep
import pandas as pd
import matplotlib.pyplot as plt

training_df=pd.read_csv("C://Users//Desktop//Malware_Classification_using_ML//data//UNSW_datasets//UNSW_NB15_training-set.csv")

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
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()
