import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class Autoencoder(nn.Module):
    
    def __init__(self, variant, training_data, lr, actf, num_inputs, num_hid, num_output, training_df):
        super(Autoencoder,self).__init__()
        self.training_data = training_data
        self.training_df = training_df
        self.variant= variant
        self.actf = actf
        self.num_inputs = num_inputs
        self.num_hid = num_hid
        self.num_output = num_output
        self.var_imp=[]
        self.layer1=nn.Linear(self.num_inputs,self.num_hid[0])
        self.layer2=nn.Linear(self.num_hid[0],self.num_hid[1])
        if self.variant=="AEDropout":
            self.dp= nn.Dropout(0.8)
        self.layer3=nn.Linear(self.num_hid[1],self.num_hid[2])
        self.layer4=nn.Linear(self.num_hid[2],self.num_output)

    def forward(self,x):
        
        x= self.layer1(x)
        x= self.actf(x)
        x= self.layer2(x)
        if self.variant=="AEDropout":
            x= self.dp(x)
        x= self.actf(x)
        x= self.layer3(x)
        x= self.actf(x)
        x= self.layer4(x)
        op= self.actf(x)

        return op

def training(fs_algo,training_features, lr, actf, num_inputs, num_hid, num_output, training_df):

    batch_size=10000
    
    torch.manual_seed(0)
    device = torch.device("cpu")

    model= Autoencoder(fs_algo,training_features, lr, actf, num_inputs, num_hid, num_output, training_df)

    optimizer= optim.Adam(model.parameters(),lr = lr)

    if fs_algo=="AEL2":
        optimizer= optim.Adam(model.parameters(),lr = lr, weight_decay= 0.01)

    total_loss=0

    num_epoch=250

    x=[]
    y=[]
    l=[]

    l1_lambda = 0.001
    l1_norm = sum(p.sum()
                  for p in model.parameters())

    loss_fn= nn.MSELoss()

    train_dataloader= torch.utils.data.DataLoader(training_features, batch_size=batch_size, shuffle= True)

    for epoch in range(1, num_epoch):

        model.train()

        total_loss=0

        for sample in train_dataloader:

            X= sample.to(device)

            optimizer.zero_grad()
            
            if fs_algo=="AE" or fs_algo=="AEDropout" or fs_algo=="AEL2":
                loss= loss_fn(model(X.float()),X.float())
        
            elif fs_algo=="AEL1":
                loss= loss_fn(model(X.float()),X.float())+ l1_lambda*l1_norm

            loss.backward()
            optimizer.step()

            total_loss+=loss.item()

        print("epoch {} loss {}".format(epoch,total_loss/len(train_dataloader)))
        lr = lr * (0.7 **(epoch//25))
            
        x.append(epoch)
        y.append(total_loss/len(train_dataloader))
    
    #To plot variable importances
    for name, param in model.named_parameters():
        if param.requires_grad:
            if list(param.shape)==[43,38] or list(param.shape)==[41,35]:
                l=param
                l=l.tolist()
            
    plt.rcParams['figure.figsize']=(20,20)
    plt.plot(x,y)
    plt.xlabel("Epochs")
    plt.ylabel("Autoencoder Training loss")
    plt.show()
    
    var_imp=variable_importance(l)
    bf= select_features(training_features, training_df,var_imp)

    return bf

def variable_importance(l):

    var_imp=[]

    for row in l:
        total_weight=0
        for a in row:
            total_weight+=a
        var_imp.append(total_weight)

    return var_imp
        
def select_features(training_data,training_df,var_imp):
    features=list(range(0,training_data.shape[1]))
    feature_names=list(training_df.columns)

    plt.close()
    plt.rcParams['figure.figsize']=(20,20)
    x_vals=np.arange(len(features))
    print(x_vals.shape)
    print(len(var_imp))
    plt.bar(x_vals,var_imp,align='center',alpha=1)
    plt.xticks(x_vals,features)
    plt.xlabel("Feature Indices")
    plt.ylabel("Feature Importance Values")
    plt.show()
    
    best_features=sorted(range(len(var_imp)), key=lambda i: var_imp[i], reverse=True)[:30]
    print("\nTop 30 selected features:")
    for i in best_features:
        print(feature_names[i])

    print("\nVariance", np.var(var_imp))

    return best_features


        
