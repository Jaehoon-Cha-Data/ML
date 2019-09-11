# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 21:14:12 2019

@author: jaehooncha

@email: Jaehoon.Cha@xjtlu.edu.cn

Multi-Layer Perceptron pytorch
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

### data load ###
features = pd.read_pickle('../datasets/four_features.pickle')
features = np.array(features)
features = scale(features)

train = features[:-365,:]
test = features[-365:,:]


### seperate features and target ###
train_x = np.array(train[:,:3])
train_y = np.array(train[:,3:])

test_x = np.array(test[:,:3])
test_y = np.array(test[:,3:])  

n_samples = train_x.shape[0]

### hyper parameter ###
n_input = train_x.shape[1]
n_hidden_node_1 = 64
n_hidden_node_2 = 32
n_output = 1
Lr = 0.01
n_epochs = 100
s_batch = 100

### before training ###
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


### MLP ###
class mlp_network(nn.Module):
    def __init__(self, n_input, n_hidden_node_1, n_hidden_node_2, n_output):
        super(mlp_network, self).__init__()
        self.n_input = n_input
        self.n_hidden_node_1 = n_hidden_node_1
        self.n_hidden_node_2 = n_hidden_node_2
        self.n_output = n_output
        
        self.layer1 = nn.Linear(self.n_input, self.n_hidden_node_1)
        self.layer2 = nn.Linear(self.n_hidden_node_1, self.n_hidden_node_2)
        self.output = nn.Linear(self.n_hidden_node_2, self.n_output)
        
    def forward(self, x):
        in_size = x.size(0)
        x = x.view(in_size, -1)
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = self.output(x)
        return x


### initialize ###
mlp = mlp_network(n_input, n_hidden_node_1, n_hidden_node_2, n_output)
mlp.cuda()                                ### gpu mode ###
torch.backends.cudnn.benchmark = True     ### gpu mode ###

### loss ###
loss_func = nn.MSELoss()


### optimizer ###
optimizer = torch.optim.Adam(mlp.parameters(), lr = Lr)


def mlp_train(n_epochs = 100):
    n_batch = int(n_samples / s_batch)
    for epoch in range(n_epochs):
        mlp.train()    
        avg_loss = 0.
        for i in range(n_batch):
            batch_idx = np.random.choice(range(n_samples), s_batch)
            batch_x, batch_y = Variable(torch.Tensor(train_x[batch_idx])).cuda(), Variable(torch.Tensor(train_y[batch_idx])).cuda() ### gpu mode ###
#            batch_x, batch_y = Variable(torch.Tensor(train_x[batch_idx])), Variable(torch.Tensor(train_y[batch_idx]))
            optimizer.zero_grad()
            output = mlp(batch_x)
            batch_loss = loss_func(output, batch_y)
            batch_loss.backward()
            optimizer.step()
            avg_loss += batch_loss/ n_samples * s_batch
        print('Epoch: {:4d} loss: {:.9f}'.format(epoch, avg_loss.data))


def mlp_predict(X):
    mlp.eval()
    with torch.no_grad():            #for inference mode
        X_ = Variable(torch.Tensor(X)).cuda() 
    #    X_ = Variable(torch.Tensor(X), volatile = True)
        mlp_prediction = mlp(X_)
        mlp_prediction = mlp_prediction.cpu().numpy().reshape(-1)
    return mlp_prediction

    
### implement ###
mlp_train()
train_predict_y = mlp_predict(train_x)
test_predict_y = mlp_predict(test_x)


### root mean squared error ###
train_rmse = np.sqrt(np.mean((train_predict_y - train_y.reshape(-1))**2))
test_rmse = np.sqrt(np.mean((test_predict_y - test_y.reshape(-1))**2))
print('train RMSE is %.4f' %(train_rmse))
print('test RMSE is %.4f' %(test_rmse))


### font size ###
plt.rcParams.update({'font.size': 15})


### draw outputs ###
plt.figure(figsize=(15,7))
plt.plot(test_y, label = 'true', c = 'r', marker = '_')
plt.plot(test_predict_y, label = 'prediction', c = 'k')
plt.title('Multi-Layer Perceptron Pytorch')
plt.xlabel('X', size = 20)
plt.ylabel('Y', size = 20)
plt.legend(loc = 1)


