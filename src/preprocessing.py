import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def integer_encode(df):
    
    col_list=df.select_dtypes(include='object')
    
    le=LabelEncoder()
    
    for x in col_list:
        df[x]=le.fit_transform(df[x])

    return df
        

def normalize_dataset(features):
    
    num_rows,num_cols=features.shape
    
    for i in range(0,num_cols):
        mean=np.mean(features[:,i])
        std=np.std(features[:,i])
        
        for j in range(0,num_rows):
            features[j][i]=(features[j][i]-mean)/std

    return features
