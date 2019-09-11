# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:14:54 2019

@author: jaehooncha

@email: Jaehoon.Cha@xjtlu.edu.cn

Multi-Layer Perceptron tensorflow

tf.InteractiveSession._active_session_count
tf.reset_default_graph()
"""
import tensorflow as tf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import gc; gc.collect()
tf.reset_default_graph()

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
X_ = tf.placeholder('float', [None, n_input])
Y_ = tf.placeholder('float')


def mlp_network(X_, n_hidden_node_1, n_hidden_node_2, n_output):
    xavier_init = tf.contrib.layers.xavier_initializer
    
    m, n_input = X_.shape
    
    W = {
        'h1': tf.get_variable('h1',  shape = [n_input, n_hidden_node_1], initializer= xavier_init()),
        'h2': tf.get_variable('h2',  shape = [n_hidden_node_1, n_hidden_node_2], initializer= xavier_init()),
        'hout': tf.get_variable('hout',  shape = [n_hidden_node_2, n_output], initializer= xavier_init())
        }
    B = {
        'b1': tf.get_variable('b1',  shape = [n_hidden_node_1], initializer= tf.constant_initializer(0.)),
        'b2': tf.get_variable('b2',  shape = [n_hidden_node_2], initializer= tf.constant_initializer(0.)),
        'bout': tf.get_variable('bout',  shape = [n_output], initializer= tf.constant_initializer(0.))
         }
    
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X_, W['h1']), B['b1']), name="hidden_layer1")
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, W['h2']), B['b2']), name="hidden_layer2")
    output = tf.add(tf.matmul(layer_2, W['hout']), B['bout'], name="output_layer")
    return output




### prediction ###
pred = mlp_network(X_, n_hidden_node_1, n_hidden_node_2, n_output)


### loss ###
loss = tf.losses.mean_squared_error(labels = Y_, predictions = pred)


### optimizer ###
optimizer = tf.train.AdamOptimizer(learning_rate=Lr).minimize(loss)


### initialize ###
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)


### train ###
def mlp_train(n_epochs = n_epochs, n_samples = n_samples, s_batch = s_batch):
    n_batch = int(n_samples / s_batch)
    for epoch in range(n_epochs):
        avg_loss = 0.
        for i in range(n_batch):
            batch_idx = np.random.choice(range(n_samples), s_batch)
            batch_x = train_x[batch_idx]
            batch_y = train_y[batch_idx]
            
            # Fit  batch
            _, batch_loss = sess.run((optimizer, loss), feed_dict = {X_: batch_x, Y_: batch_y})
            
            # Compute average loss
            avg_loss += batch_loss/ n_samples * s_batch
        print("Epoch:", '%04d' % (epoch), "loss = ", "{:.9f}".format(avg_loss))


def mlp_predict(X):
    mlp_prediction = sess.run(pred, feed_dict={X_:X})
    mlp_prediction = mlp_prediction.reshape(-1)
    return mlp_prediction


### implement ###
mlp_train()
train_predict_y = mlp_predict(train_x)
test_predict_y = mlp_predict(test_x)
sess.close()

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
plt.title('Multi-Layer Perceptron Tensorflow')
plt.xlabel('X', size = 20)
plt.ylabel('Y', size = 20)
plt.legend(loc = 1)

