# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 23:15:24 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com

Support Vector Machine 
using the Radial Basis Function (RBF) kernel for prediction
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVR
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


### SVM ###
def svm_train(X, Y):
    clf = SVR(kernel= 'rbf', gamma= 0.01)
    clf.fit(X, Y)
    return clf

def svm_prediction(svm_model, X):
    svm_regression = svm_model
    svm_prediction = svm_regression.predict(X)
    svm_prediction = svm_prediction.reshape(-1)
    return svm_prediction


### implement ###
svm_model = svm_train(train_x, train_y)
train_predict_y = svm_prediction(svm_model, train_x)
test_predict_y = svm_prediction(svm_model, test_x)


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
plt.title('Support Vector Machine')
plt.xlabel('X', size = 20)
plt.ylabel('Y', size = 20)
plt.legend(loc = 1)
