# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 23:00:20 2019

@author: jaehooncha

@email: Jaehoon.Cha@xjtlu.edu.cn

Linear regression
"""
import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

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


### Linear regression ###
def linear_regression_train(X, Y):
    clf = LinearRegression()
    clf.fit(X, Y)
    return clf

def linear_regression_prediction(lr_model, X):
    linear_regression = lr_model
    linear_prediction = linear_regression.predict(X)
    linear_prediction = linear_prediction.reshape(-1)
    return linear_prediction


### implement ###
linear_regression_model = linear_regression_train(train_x, train_y)
train_predict_y = linear_regression_prediction(linear_regression_model, train_x)
test_predict_y = linear_regression_prediction(linear_regression_model, test_x)


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
plt.title('Linear regression')
plt.xlabel('X', size = 20)
plt.ylabel('Y', size = 20)
plt.legend(loc = 1)
