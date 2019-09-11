# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:44:54 2019

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


time_seq = 10
n_features = features.shape[1]-1

def make_sequence(features, s_seq):
    features_seq = []
    m, n = features.shape
    for i in range(m - s_seq + 1):
#        features_seq.append([np.array(features[i:i+s_seq, :-1]), features[i+s_seq-1, -1]])
        features_seq.append([np.array(features[i:i+s_seq, :-1]), features[i:i+s_seq, -1]])
    return features_seq


features_seq = make_sequence(features, time_seq)

train = features_seq[:-365]
test = features_seq[-365:]


### seperate features and target ###
train_x = [item[0] for item in train]
train_x = np.concatenate(train_x)
train_x = train_x.reshape(-1,time_seq, n_features)
train_y = [item[1] for item in train]
train_y = np.array(train_y).reshape(-1,time_seq, 1)

test_x = [item[0] for item in test]
test_x = np.concatenate(test_x)
test_x = test_x.reshape(-1,time_seq, n_features)
test_y = [item[1] for item in test]
test_y = np.array(test_y).reshape(-1,time_seq, 1)


n_samples = train_x.shape[0]


### hyper parameter ###
n_input_ = train_x.shape[-1]
n_hidden_node = 64
#n_hidden_node_2 = 32
n_output = 1
Lr = 0.01
n_epochs = 100
s_batch = 100


### before training ###
X_ = tf.placeholder('float', [None, time_seq, n_input_])
Y_ = tf.placeholder('float', [None, time_seq])


def lstm_network(X_, n_hidden_node, n_output):
    xavier_init = tf.contrib.layers.xavier_initializer
    
    m, n_seq, n_input = X_.shape
    
    W = {
        'wa': tf.get_variable('wa',  shape = [n_input, n_hidden_node], initializer= xavier_init()),
        'wi': tf.get_variable('wi',  shape = [n_input, n_hidden_node], initializer= xavier_init()),
        'wf': tf.get_variable('wf',  shape = [n_input, n_hidden_node], initializer= xavier_init()),
        'wo': tf.get_variable('wo',  shape = [n_input, n_hidden_node], initializer= xavier_init()),
        'hout': tf.get_variable('hout',  shape = [n_hidden_node, n_output], initializer= xavier_init())
        }
    U = {
        'ua': tf.get_variable('ua',  shape = [n_hidden_node, n_hidden_node], initializer= xavier_init()),
        'ui': tf.get_variable('ui',  shape = [n_hidden_node, n_hidden_node], initializer= xavier_init()),
        'uf': tf.get_variable('uf',  shape = [n_hidden_node, n_hidden_node], initializer= xavier_init()),
        'uo': tf.get_variable('uo',  shape = [n_hidden_node, n_hidden_node], initializer= xavier_init())
        }    
    B = {
        'ba': tf.get_variable('ba',  shape = [n_hidden_node], initializer= tf.constant_initializer(0.)),
        'bi': tf.get_variable('bi',  shape = [n_hidden_node], initializer= tf.constant_initializer(0.)),
        'bf': tf.get_variable('bf',  shape = [n_hidden_node], initializer= tf.constant_initializer(0.)),
        'bo': tf.get_variable('bo',  shape = [n_hidden_node], initializer= tf.constant_initializer(0.)),
        'bout': tf.get_variable('bout',  shape = [n_output], initializer= tf.constant_initializer(0.))
        }
    
    h_t = tf.zeros_like(tf.matmul(X_[:, 0, :], W['wa']), name = 'h_t')
    c_t = tf.zeros_like(tf.matmul(X_[:, 0, :], W['wa']), name = 'c_t')

    h_seq = []
    for t in range(n_seq):
        x_t = X_[:, t, :]
        a_t = tf.nn.tanh(tf.add(tf.add(tf.matmul(x_t, W['wa']), tf.matmul(h_t, U['ua'])), B['ba']), name="activation_gate")
        i_t = tf.nn.sigmoid(tf.add(tf.add(tf.matmul(x_t, W['wi']), tf.matmul(h_t, U['ui'])), B['bi']), name="input_gate")
        f_t = tf.nn.sigmoid(tf.add(tf.add(tf.matmul(x_t, W['wf']), tf.matmul(h_t, U['uf'])), B['bf']), name="forget_gate")
        o_t = tf.nn.sigmoid(tf.add(tf.add(tf.matmul(x_t, W['wo']), tf.matmul(h_t, U['uo'])), B['bo']), name="forget_gate")
        c_t = tf.add(tf.multiply(a_t, i_t), tf.multiply(f_t, c_t))
        h_t = tf.multiply(tf.nn.tanh(c_t), o_t)
        h_seq.append(h_t)
    o_seq = tf.add(tf.matmul(h_seq[0], W['hout']), B['bout'], name="output_layer")
    for t in range(1, n_seq):
        output = tf.add(tf.matmul(h_seq[t], W['hout']), B['bout'], name="output_layer")
        o_seq = tf.concat((o_seq, output), axis = 1)
    return o_seq
        

### prediction ###
pred = lstm_network(X_, n_hidden_node, n_output)


#### loss ###
loss = tf.losses.mean_squared_error(labels = Y_, predictions = pred)
#
#
#### optimizer ###
optimizer = tf.train.AdamOptimizer(learning_rate=Lr).minimize(loss)
#
#
#### initialize ###
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)


### train ###
def lstm_train(n_epochs = n_epochs, n_samples = n_samples, s_batch = s_batch):
    n_batch = int(n_samples / s_batch)
    for epoch in range(n_epochs):
        avg_loss = 0.
        for i in range(n_batch):
            batch_idx = np.random.choice(range(n_samples), s_batch)
            batch_x = train_x[batch_idx]
            batch_y = train_y[batch_idx]
            
            # Fit  batch
            _, batch_loss = sess.run((optimizer, loss), feed_dict = {X_: batch_x, Y_: batch_y.reshape(-1, time_seq)})
            
            # Compute average loss
            avg_loss += batch_loss/ n_samples * s_batch
        print("Epoch:", '%04d' % (epoch), "loss = ", "{:.9f}".format(avg_loss))
        
#
#
def lstm_predict(X):
    lstm_predict = sess.run(pred, feed_dict={X_:X})
    lstm_predict = lstm_predict
    return lstm_predict
#
#
### implement ###
lstm_train()
train_predict_y = lstm_predict(train_x)
test_predict_y = lstm_predict(test_x)
sess.close()

### root mean squared error ###
train_rmse = np.sqrt(np.mean((train_predict_y - train_y.reshape(-1, time_seq))**2))
test_rmse = np.sqrt(np.mean((test_predict_y - test_y.reshape(-1, time_seq))**2))
print('train RMSE is %.4f' %(train_rmse))
print('test RMSE is %.4f' %(test_rmse))


### font size ###
plt.rcParams.update({'font.size': 15})


### draw outputs ###
plt.figure(figsize=(15,7))
plt.plot(test_y[:,-1], label = 'true', c = 'r', marker = '_')
plt.plot(test_predict_y[:,-1], label = 'prediction', c = 'k')
plt.title('Long Short Term Memory Tensorflow')
plt.xlabel('X', size = 20)
plt.ylabel('Y', size = 20)
plt.legend(loc = 1)


