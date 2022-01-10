import numpy as np
import pandas as pd
import math
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import matplotlib.pyplot as plt


class Encoder(torch.nn.Module):

    def __init__(self, actf1, actf2, layer_sizes):

        super(Encoder, self).__init__()
        self.actf1 = actf1
        self.actf2 = actf2
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.actf1 = actf1
        self.actf2 = actf2
        self.W = []
        self.bn_normalize = []
        self.beta = []
        self.gamma =[]
        self.layer1 = torch.nn.Linear(self.layer_sizes[0],self.layer_sizes[1], bias=False)
        self.layer2 = torch.nn.Linear(self.layer_sizes[1],self.layer_sizes[2], bias=False)
        self.layer3 = torch.nn.Linear(self.layer_sizes[2],self.layer_sizes[3], bias=False)
        self.layers = [self.layer1, self.layer2, self.layer3]

        for l in range(self.L):
            self.layers[l].weight.data = torch.randn(self.layers[l].weight.data.size()) / np.sqrt(layer_sizes[l])
            self.layers[l].weight.data = self.layers[l].weight.data.double()
            bn_normalize = torch.nn.BatchNorm1d(self.layer_sizes[l], affine=False)
            self.bn_normalize.append(bn_normalize.double())
            bn_beta = Parameter(torch.FloatTensor(1, self.layer_sizes[l+1]))
            self.beta.append(bn_beta)
            self.beta[-1].data.zero_()
            bn_gamma = Parameter(torch.FloatTensor(1, self.layer_sizes[l+1]))
            self.gamma.append(bn_gamma)
            self.gamma[-1].data = torch.ones(self.gamma[-1].size())


    def forward(self, x, noise_std=0.0):
        z=[]
        h=[]
        noise=[]
        n = np.random.normal(loc=0.0, scale= noise_std, size=x.shape)
        z.append(x + n)
        h.append(x + n)
        n = Variable(torch.FloatTensor(n))
        noise.append(n)

        for l in range(1,self.L):

            mul = self.layers[l-1](h[l-1])
            mul = self.bn_normalize[l](mul.double())
            n1 =  np.random.normal(loc=0.0, scale= noise_std, size=mul.shape)
            n1 = Variable(torch.FloatTensor(n1))
            noise.append(n1)
            z.append(mul + n1)

            if l==self.L:
                h.append(self.actf2((z[l]+self.beta[l-1])*self.gamma[l-1]))
            else:
                h.append(self.actf1((z[l]+self.beta[l-1])*self.gamma[l-1]))

        prob_y_x = h[self.L - 1]

        return z, noise, prob_y_x


class Decoder(torch.nn.Module):

    def __init__(self, layer_sizes):
        super(Decoder, self).__init__()
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes)
        self.denoising_cost = [0.1, 0.1, 10, 1000]
        self.bn_normalize = []
        self.layer1 = torch.nn.Linear(self.layer_sizes[1],self.layer_sizes[0], bias=False)
        self.layer2 = torch.nn.Linear(self.layer_sizes[2],self.layer_sizes[1], bias=False)
        self.layer3 = torch.nn.Linear(self.layer_sizes[3],self.layer_sizes[2], bias=False)
        self.layers = [self.layer1, self.layer2, self.layer3]
        for l in range(0,self.L-1):
            self.layers[l].weight.data = torch.randn(self.layers[l].weight.data.size()) / np.sqrt(layer_sizes[l])
            self.layers[l].weight.data = self.layers[l].weight.data.double()
            bn_normalize = torch.nn.BatchNorm1d(self.layer_sizes[l], affine=False)
            self.bn_normalize.append(bn_normalize.double())

    def calc_var(self,z,n):
        mz, vz = torch.mean(z), torch.var(z)
        mn, vn = torch.mean(n), torch.var(n)
        var = vz / (vz+vn)
        return var

    def denoising(self, u, z_corr, var):
        z_calc=0
        z_corr=torch.sigmoid(z_corr)
        u=torch.sigmoid(u)
        z_calc= var*z_corr + (1-var)*u
        return z_calc

    def forward(self, h_corr, z_corr, z, noise, loss_unsupervised):

        u=[]
        z_calc=[]
        z_calc_bn=[]
        #denoising cost
        d_cost =[]
        l= self.L-1
        while l>=0:
            u.append(0)
            z_calc.append(0)
            z_calc_bn.append(0)
            l=l-1

        l= self.L-2
        while l>=0:
            if l==self.L-2:
                u[l]= h_corr
            else:
                mul= self.layers[l](z_calc[l])
                u[l]= self.bn_normalize[l](mul.double())

            var = self.calc_var(z[l-1],noise[l-1])

            z_calc[l-1]= self.denoising(u[l],z_corr[l],var)
            z_calc_bn[l-1]= self.bn_normalize[l](z_calc[l-1].double())
            d_cost.append(self.denoising_cost[l] * loss_unsupervised.forward(z_calc_bn[l-1], z[l]))
            l=l-1

        return u,d_cost


class Ladder(torch.nn.Module):

    def __init__(self, actf1, actf2, layer_sizes):

        super(Ladder, self).__init__()
        self.encoder = Encoder(actf1, actf2, layer_sizes)
        self.decoder = Decoder(layer_sizes)

    def forward_encoders_clean(self, data):
        return self.encoder.forward(data)

    def forward_encoders_noise(self, data, noise_std):
        return self.encoder.forward(data, noise_std)

    def forward_decoders(self, h_corr, z_corr, z, noise, loss_unsupervised):
        return self.decoder.forward(h_corr, z_corr, z, noise, loss_unsupervised)


def unlabeled(x):
    ul = x[100:-1,0:-1]
    return ul


def labeled(x):
    l = x[0:100,0:-1]
    return l


def convert_to_one_hot(labels, t_labels, num_classes):

    L=[]
    for i in range(0,len(labels)):
        l=[]
        for j in range(0,num_classes):
            if labels[i]==j:
                l.append(1)
            else:
                l.append(0)
        L.append(l)
    L=np.array(L)
    labels=L

    L=[]
    for i in range(0,len(t_labels)):
        l=[]
        for j in range(0,num_classes):
            if t_labels[i]==j:
                l.append(1)
            else:
                l.append(0)
        L.append(l)
    L=np.array(L)
    t_labels=L

    return labels, t_labels


def get_data(training_data, labels, num_labeled, num_samples, batch_size):
        
    idy = np.arange(0, num_labeled)
    np.random.shuffle(idy)
    idy=idy[:100]
    data_labeled = [training_data[i] for i in idy]
    data_shuffle_y = [labels[i] for i in idy]

    idx = np.arange(num_labeled, num_samples)
    np.random.shuffle(idx)
    num_ul_per_batch= batch_size - len(idy)
    idx=idx[:num_ul_per_batch]
    data_unlabeled = [training_data[i] for i in idx]

    data_shuffle_x= []
    for x in data_labeled:
        data_shuffle_x.append(x)

    data_shuffle_x_ul=[]
    for x in data_unlabeled:
        data_shuffle_x_ul.append(x)

    tensor_x = torch.FloatTensor(data_shuffle_x).double() 
    tensor_y = torch.LongTensor(data_shuffle_y)
    tensor_x_ul = torch.FloatTensor(data_shuffle_x_ul).double()

    dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)

    dataset_unlabeled = torch.utils.data.TensorDataset(tensor_x_ul)
    
    train_dataloader_labelled = DataLoader(dataset, batch_size=batch_size)
    train_dataloader_unlabelled = DataLoader(dataset_unlabeled, batch_size=batch_size)

    return train_dataloader_labelled, train_dataloader_unlabelled


def get_test_data(test_data, tlabels, num_samples):

    tensor_x = torch.FloatTensor(test_data).double()
    tensor_y = torch.FloatTensor(tlabels)

    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)

    test_dataloader = DataLoader(dataset, batch_size= num_samples)
    
    return test_dataloader


def training(training_data, labels, testing_data, t_labels, lr, actf1, actf2, layer_sizes, num_labeled, num_samples, num_classes, batch_size=1000):
    
    num_epoch = 150
    noise_std = 0.1
    seed = torch.manual_seed(0)
    num_batches = len(training_data)//batch_size
    
    device = torch.device("cpu")

    ladder = Ladder(actf1, actf2, layer_sizes)

    testing_dataset = TensorDataset(torch.FloatTensor(testing_data), torch.LongTensor(t_labels))
    testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True)

    optimizer = Adam(params= ladder.parameters(), lr=lr)
    loss_supervised = torch.nn.CrossEntropyLoss()
    loss_unsupervised = torch.nn.MSELoss()


    for epoch in range(num_epoch):

        lr = lr * (0.96 **(epoch//50))

        optimizer = Adam(params= ladder.parameters(), lr=lr)

        ladder.train()

        for batch in range(num_batches):

            train_dataloader_labelled, train_dataloader_unlabelled = get_data(training_data, labels, num_labeled, num_samples, batch_size)

            optimizer.zero_grad()

            X_label,Y = next(iter(train_dataloader_labelled))

            z_corr, noise, p_enc_corr = ladder.forward_encoders_noise(X_label, noise_std)
            z, n_0, p_enc_label = ladder.forward_encoders_clean(X_label)

            # pass through decoders
            u, d_cost_l = ladder.forward_decoders(p_enc_corr,z_corr,z,noise,loss_unsupervised)

            # calculate costs
            l_cost = loss_supervised.forward(p_enc_corr, Y)

            X = next(iter(train_dataloader_unlabelled))

            z_corr, noise, p_enc_corr = ladder.forward_encoders_noise(X[0], noise_std)
            z, n_0, p_enc = ladder.forward_encoders_clean(X[0])

            # pass through decoders
            u, d_cost_ul = ladder.forward_decoders(p_enc_corr,z_corr,z,noise,loss_unsupervised)

            ul_cost = sum(d_cost_l)+sum(d_cost_ul)

            # backprop
            loss = l_cost + ul_cost
            loss.backward()
            optimizer.step()

        print("epoch {} ".format(epoch))

    p_enc_label = p_enc_label.data.numpy()
    preds = np.argmax(p_enc_label, axis=1)
    target_train = Y.data.numpy()
    correct_train = np.sum(target_train == preds)
    total_train = target_train.shape[0]

    print("Training Accuracy:", correct_train/total_train)

    test_dataloader = get_test_data(testing_data, t_labels, num_samples)

    test_X, test_Y = next(iter(test_dataloader))

    _, _, output = ladder.forward_encoders_clean(test_X)
   
    output = output.data.numpy()
    preds1 = np.argmax(output, axis=1)
    target_test = test_Y.data.numpy()
    correct_test = np.sum(target_test == preds1)
    total_test = target_test.shape[0]
  
    print("Testing Accuracy:", correct_test/total_test)

    #Per Class Train
    cnt=[]
    cnt_pred=[]
    for i in range(0,num_classes):
        cnt.append(0)
        cnt_pred.append(0)
        
    for label in target_train:
        for i in range(0,num_classes):
            if label==i:
                cnt[i]=cnt[i]+1
                break
            
    for i in range(0,len(target_train)):
        if preds[i]==target_train[i]:
            for j in range(0,num_classes):
                if j==target_train[i]:
                    cnt_pred[j]=cnt_pred[j]+1
                    break

    per_class_acc=[]
    for i in range(0,num_classes):
        per_class_acc.append(cnt_pred[i]/(cnt[i]+1e-6))
        
    #Per Class Test
    cnt1=[]
    cnt_pred1=[]
    for i in range(0,num_classes):
        cnt1.append(0)
        cnt_pred1.append(0)
        
    for label in target_test:
        for i in range(0,num_classes):
            if label==i:
                cnt1[i]=cnt1[i]+1
                break
            
    for i in range(0,len(target_test)):
        if preds1[i]==target_test[i]:
            for j in range(0,num_classes):
                if j==target_test[i]:
                    cnt_pred1[j]=cnt_pred1[j]+1
                    break
    per_class_acc1=[]
    for i in range(0,num_classes):
        per_class_acc1.append(cnt_pred1[i]/(cnt1[i]+1e-6))

    print("Per Class Accuracy (Training):" ,per_class_acc)
    print("Per Class Accuracy (Testing):",per_class_acc1)

    return preds1
