# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:30:18 2019

@author: jaehooncha

@email: Jaehoon.Cha@xjtlu.edu.cn

Gaussian process regression
using the Radial Basis Function (RBF) kernel for prediction
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

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


### GPR ###
def gpr_train(X, Y):
    dy = 0.1 * (0.5 + 1.0 * np.random.random(Y.shape[0]))
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    clf = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2, n_restarts_optimizer=10)
    #kernel=kernel, n_restarts_optimizer=9
    clf.fit(X, Y)
    return clf

def gpr_prediction(gpr_model, X):
    gpr_regression = gpr_model
    gpr_prediction, gpr_sigma = gpr_regression.predict(X, return_std=True)
    gpr_prediction = gpr_prediction.reshape(-1)
    return gpr_prediction, gpr_sigma


### implement ###
gpr_model = gpr_train(train_x, train_y)
train_predict_y, train_sigma = gpr_prediction(gpr_model, train_x)
test_predict_y, test_sigma = gpr_prediction(gpr_model, test_x)


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
plt.fill(np.concatenate([np.array(range(len(test_x))), np.array(range(len(test_x)))[::-1]]),
         np.concatenate([test_predict_y - 1.9600 * test_sigma,
                        (test_predict_y + 1.9600 * test_sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.title('Gaussian process regression')
plt.xlabel('X', size = 20)
plt.ylabel('Y', size = 20)
plt.legend(loc = 1)
