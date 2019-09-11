# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:33:00 2019

@author: jaehooncha

@email: Jaehoon.Cha@xjtlu.edu.cn

Multi-Layer Perceptron keras in tensorflow
"""
import tensorflow as tf
import numpy as np 
import pandas as pd
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


### MLP ###
def mlp_train(X, Y, h1 = 64, h2 = 32, n_epochs = 100, s_batch = 100, Lr = 0.01):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape = (X.shape[1:])))
    model.add(tf.keras.layers.Dense(h1, activation = tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(h2, activation = tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.Adam(lr=Lr)
    
    model.compile(optimizer = optimizer,
                 loss = 'mse',
                 metrics =['mae', 'mse'])

    hist = model.fit(X, Y, epochs = n_epochs, batch_size = s_batch)   
    return hist

def mlp_predict(mlp_model, X):
    mlp = mlp_model
    mlp_prediction = mlp.predict(X)
    mlp_prediction = mlp_prediction.reshape(-1)
    return mlp_prediction


### implement ###
mlp_model = mlp_train(train_x, train_y, n_epochs = 100, s_batch = 100).model
train_predict_y = mlp_predict(mlp_model, train_x)
test_predict_y = mlp_predict(mlp_model, test_x)


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
plt.title('Multi-Layer Perceptron Keras')
plt.xlabel('X', size = 20)
plt.ylabel('Y', size = 20)
plt.legend(loc = 1)

