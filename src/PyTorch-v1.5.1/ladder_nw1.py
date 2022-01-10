import numpy as np
import pandas as pd
import math
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class Encoder(torch.nn.Module):

    def __init__(self, x, actf1, actf2, noise_std):

        super(Encoder, self).__init__()


    def forward(self, x, noise_std=0.0):
        z=[]
        h=[]
        n= torch.randn(list(x.shape)) * noise_std
        noise=[]
        
        z.append(x + n)
        h.append(x + n)
        noise.append(n)

        for l in range(1,self.L + 1):
            mul= torch.matmul(h[l-1],self.w[l-1])
            n1= torch.randn(list(mul.shape)) * noise_std
            noise.append(n1)
            z.append(self.batch_norm(mul) + n1)

            if l==self.L:
                h.append(self.actf2((z[l]+self.beta[l-1])*self.gamma[l-1]))
            else:
                h.append(self.actf1((z[l]+self.beta[l-1])*self.gamma[l-1]))

        prob_y_x = h[self.L]

        return z, noise, prob_y_x


class Decoder(torch.nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()


    def forward(self, h_corr, z_corr, z, noise):

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
                mul= torch.matmul(z_calc[l+1],self.v[l])
                u[l]= self.batch_norm(mul)
            var = self.calc_var(self.unlabeled(z[l]),self.unlabeled(noise[l]))
            z_calc[l]= self.denoising(u[l],self.unlabeled(z_corr[l]),var)
            z_calc_bn[l]= self.batch_norm(z_calc[l])
            d_cost.append((torch.sum(torch.square(z_calc_bn[l] - self.unlabeled(z[l])),1)//self.layer_sizes[l])* self.denoising_cost[l])
            l=l-1

        return u,d_cost


class Ladder(torch.nn.Module):

    def __init__(self, layer_sizes, actf1, actf2, noise_std):

        super(Ladder, self).__init__()
        self.encoder = Encoder(x, actf1, actf2)
        self.decoder = Decoder(h_corr, z_corr, z, noise)

    def forward_encoders_clean(self, data):
        return self.encoder.forward(data, 0.0)

    def forward_encoders_noise(self, data, noise_std):
        return self.encoder.forward(data, noise_std)

    def forward_decoders(self, h_corr, z_corr, z, noise):
        return self.decoder.forward(h_corr, z_corr, z, noise)


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


def next_batch(training_data, labels, num_labeled, num_samples, batch_size):
        
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

    tensor_x = torch.FloatTensor(data_shuffle_x) 
    tensor_y = torch.LongTensor(data_shuffle_y)
    
    tensor_x_ul = torch.FloatTensor(data_shuffle_x_ul)

    dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    dataset_unlabeled = torch.utils.data.TensorDataset(tensor_x_ul)
    
    dataloader = torch.utils.data.DataLoader(ConcatDataset(
             dataset,
             dataset_unlabeled
         ))

    return dataloader


def training(training_data, labels, testing_data, t_labels, lr, actf1, actf2, layer_sizes, num_labeled, num_samples, num_classes, batch_size):
    
    num_epoch = 250
    noise_std = 0.1
    seed = torch.manual_seed(0)
    
    device = torch.device("cpu")

    np.random.seed(seed)
    torch.manual_seed(seed)

    ladder = Ladder(layer_sizes, actf1, actf2, noise_std)

    labels, t_labels = convert_to_one_hot(labels, t_labels, num_classes)

    training_dataloader = get_data(training_data, labels, num_labeled, num_samples, batch_size)

    testing_dataset = TensorDataset(torch.FloatTensor(testing_data), torch.LongTensor(t_labels))
    testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    optimizer = Adam(ladder.parameters(), lr=lr)
    loss_supervised = torch.nn.CrossEntropyLoss()
    loss_unsupervised = torch.nn.MSELoss()


    for epoch in range(num_epoch):

        ladder.train()

        '''
        labelled_data = Variable(batch_train_labelled_images, requires_grad=False)
        labelled_target = Variable(batch_train_labelled_labels, requires_grad=False)
        unlabelled_data = Variable(unlabelled_images)
        '''

        for sample in training_dataloader:
            
            X= sample[0].to(device)
            Y= sample[1].to(device)

            optimizer.zero_grad()

            z_corr, noise, p_enc_corr = ladder.forward_encoders_noise(X, noise_std)

            z, n_0, p_enc = ladder.forward_encoders_clean(X)

            # pass through decoders
            u, d_cost = ladder.forward_decoders(p_enc_corr,z_corr,z,noise)

            ul_cost = torch.sum(d_cost)

            # calculate costs
            l_cost = loss_supervised.forward(noise[0:100], Y)

            # backprop
            loss = l_cost + ul_cost
            loss.backward()
            optimizer.step()


        print("epoch {} ".format(epoch))

'''

def evaluate_performance(ladder, valid_loader, e, agg_cost_scaled, agg_supervised_cost_scaled,
                         agg_unsupervised_cost_scaled, args):
    correct = 0.
    total = 0.
    for batch_idx, (data, target) in enumerate(valid_loader):
        data, target = Variable(data), Variable(target)
        output = ladder.forward_encoders_clean(data)
        # TODO: Do away with the below hack for GPU tensors.
        if args.cuda:
            output = output.cpu()
            target = target.cpu()
        output = output.data.numpy()
        preds = np.argmax(output, axis=1)
        target = target.data.numpy()
        correct += np.sum(target == preds)
        total += target.shape[0]

    print("Epoch:", e + 1, "\t",
          "Total Cost:", "{:.4f}".format(agg_cost_scaled), "\t",
          "Supervised Cost:", "{:.4f}".format(agg_supervised_cost_scaled), "\t",
          "Unsupervised Cost:", "{:.4f}".format(agg_unsupervised_cost_scaled), "\t",
          "Validation Accuracy:", correct / total)