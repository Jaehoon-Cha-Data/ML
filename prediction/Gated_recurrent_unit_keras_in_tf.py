# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 17:12:30 2019

@author: jaehooncha
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GRU, BatchNormalization
from tensorflow.keras.layers import TimeDistributed

### data load ###
features = pd.read_pickle('../datasets/four_features.pickle')
features = np.array(features)
features = scale(features)


time_seq = 10
n_features = features.shape[1]-1

def make_sequence(features, s_seq):
    features_seq = []
    m, n = features.shape
    for i in range(m - s_seq + 1):
        features_seq.append([np.array(features[i:i+s_seq, :-1]), features[i+s_seq-1, -1]])
    return features_seq


features_seq = make_sequence(features, time_seq)

train = features_seq[:-365]
test = features_seq[-365:]


### seperate features and target ###
train_x = [item[0] for item in train]
train_x = np.concatenate(train_x)
train_x = train_x.reshape(-1,time_seq, n_features)
train_y = [item[1] for item in train]
train_y = np.array(train_y)

test_x = [item[0] for item in test]
test_x = np.concatenate(test_x)
test_x = test_x.reshape(-1,time_seq, n_features)
test_y = [item[1] for item in test]
test_y = np.array(test_y)


### LSTM ###
def rnn_train(X, Y, h1 = 128, h2 = 32, n_epochs = 100, s_batch = 100, Lr = 0.001):
    model = tf.keras.models.Sequential()
    model.add(GRU(h1, input_shape = (X.shape[1:]), return_sequences = True))
    model.add(GRU(h2))
    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(lr=Lr)
    
    model.compile(optimizer = optimizer,
                 loss = 'mse',
                 metrics =['mae', 'mse'])

    hist = model.fit(X, Y, epochs = n_epochs, batch_size = s_batch)   
    return hist


def rnn_predict(rnn_model, X):
    rnn = rnn_model
    rnn_prediction = rnn.predict(X)
    rnn_prediction = rnn_prediction.reshape(-1)
    return rnn_prediction
    
    
### implement ###
rnn_model = rnn_train(train_x, train_y, h1 = 64, h2 = 32, n_epochs = 100, s_batch = 100).model
train_predict_y = rnn_predict(rnn_model, train_x)
test_predict_y = rnn_predict(rnn_model, test_x)


### root mean squared error ###
train_rmse = np.sqrt(np.mean((train_predict_y - train_y)**2))
test_rmse = np.sqrt(np.mean((test_predict_y - test_y)**2))
print('train RMSE is %.4f' %(train_rmse))
print('test RMSE is %.4f' %(test_rmse))


### font size ###
plt.rcParams.update({'font.size': 15})


### draw outputs ###
plt.figure(figsize=(15,7))
plt.plot(test_y, label = 'true', c = 'r', marker = '_')
plt.plot(test_predict_y, label = 'prediction', c = 'k')
plt.title('Gated Recurrent Unit Keras')
plt.xlabel('X', size = 20)
plt.ylabel('Y', size = 20)
plt.legend(loc = 1)




