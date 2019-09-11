# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 20:41:21 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com

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


n_samples = train_x.shape[0]


### hyper parameter ###
n_input_ = train_x.shape[-1]
n_hidden_node_1 = 64
n_hidden_node_2 = 32
n_output = 1
Lr = 0.01
n_epochs = 100
s_batch = 100


### before training ###
X_ = tf.placeholder('float', [None, time_seq, n_input_])
Y_ = tf.placeholder('float')


def lstm_network(X_, n_hidden_node_1, n_hidden_node2, n_output):
    xavier_init = tf.contrib.layers.xavier_initializer
    
    m, n_seq, n_input = X_.shape
    
    W = {
        'wa1': tf.get_variable('wa1',  shape = [n_input, n_hidden_node_1], initializer= xavier_init()),
        'wi1': tf.get_variable('wi1',  shape = [n_input, n_hidden_node_1], initializer= xavier_init()),
        'wf1': tf.get_variable('wf1',  shape = [n_input, n_hidden_node_1], initializer= xavier_init()),
        'wo1': tf.get_variable('wo1',  shape = [n_input, n_hidden_node_1], initializer= xavier_init()),
        'wa2': tf.get_variable('wa2',  shape = [n_hidden_node_1, n_hidden_node_2], initializer= xavier_init()),
        'wi2': tf.get_variable('wi2',  shape = [n_hidden_node_1, n_hidden_node_2], initializer= xavier_init()),
        'wf2': tf.get_variable('wf2',  shape = [n_hidden_node_1, n_hidden_node_2], initializer= xavier_init()),
        'wo2': tf.get_variable('wo2',  shape = [n_hidden_node_1, n_hidden_node_2], initializer= xavier_init()),
        'hout': tf.get_variable('hout',  shape = [n_hidden_node_2, n_output], initializer= xavier_init())
        }
    U = {
        'ua1': tf.get_variable('ua1',  shape = [n_hidden_node_1, n_hidden_node_1], initializer= xavier_init()),
        'ui1': tf.get_variable('ui1',  shape = [n_hidden_node_1, n_hidden_node_1], initializer= xavier_init()),
        'uf1': tf.get_variable('uf1',  shape = [n_hidden_node_1, n_hidden_node_1], initializer= xavier_init()),
        'uo1': tf.get_variable('uo1',  shape = [n_hidden_node_1, n_hidden_node_1], initializer= xavier_init()),
        'ua2': tf.get_variable('ua2',  shape = [n_hidden_node_2, n_hidden_node_2], initializer= xavier_init()),
        'ui2': tf.get_variable('ui2',  shape = [n_hidden_node_2, n_hidden_node_2], initializer= xavier_init()),
        'uf2': tf.get_variable('uf2',  shape = [n_hidden_node_2, n_hidden_node_2], initializer= xavier_init()),
        'uo2': tf.get_variable('uo2',  shape = [n_hidden_node2, n_hidden_node_2], initializer= xavier_init())        
        }    
    B = {
        'ba1': tf.get_variable('ba1',  shape = [n_hidden_node_1], initializer= tf.constant_initializer(0.)),
        'bi1': tf.get_variable('bi1',  shape = [n_hidden_node_1], initializer= tf.constant_initializer(0.)),
        'bf1': tf.get_variable('bf1',  shape = [n_hidden_node_1], initializer= tf.constant_initializer(0.)),
        'bo1': tf.get_variable('bo1',  shape = [n_hidden_node_1], initializer= tf.constant_initializer(0.)),
        'ba2': tf.get_variable('ba2',  shape = [n_hidden_node_2], initializer= tf.constant_initializer(0.)),
        'bi2': tf.get_variable('bi2',  shape = [n_hidden_node_2], initializer= tf.constant_initializer(0.)),
        'bf2': tf.get_variable('bf2',  shape = [n_hidden_node_2], initializer= tf.constant_initializer(0.)),
        'bo2': tf.get_variable('bo2',  shape = [n_hidden_node_2], initializer= tf.constant_initializer(0.)),        
        'bout': tf.get_variable('bout',  shape = [n_output], initializer= tf.constant_initializer(0.))
        }
    
    h_t1 = tf.zeros_like(tf.matmul(X_[:, 0, :], W['wa1']), name = 'h_t1')
    c_t1 = tf.zeros_like(tf.matmul(X_[:, 0, :], W['wa1']), name = 'c_t1')
    h_t2 = tf.zeros_like(tf.matmul(h_t1, W['wa2']), name = 'h_t2')
    c_t2 = tf.zeros_like(tf.matmul(h_t1, W['wa2']), name = 'c_t2')
    
    h_seq1 = []
    for t in range(n_seq):
        x_t = X_[:, t, :]
        a_t = tf.nn.tanh(tf.add(tf.add(tf.matmul(x_t, W['wa1']), tf.matmul(h_t1, U['ua1'])), B['ba1']), name="activation_gate")
        i_t = tf.nn.sigmoid(tf.add(tf.add(tf.matmul(x_t, W['wi1']), tf.matmul(h_t1, U['ui1'])), B['bi1']), name="input_gate")
        f_t = tf.nn.sigmoid(tf.add(tf.add(tf.matmul(x_t, W['wf1']), tf.matmul(h_t1, U['uf1'])), B['bf1']), name="forget_gate")
        o_t = tf.nn.sigmoid(tf.add(tf.add(tf.matmul(x_t, W['wo1']), tf.matmul(h_t1, U['uo1'])), B['bo1']), name="forget_gate")
        c_t1 = tf.add(tf.multiply(a_t, i_t), tf.multiply(f_t, c_t1))
        h_t1 = tf.multiply(tf.nn.tanh(c_t1), o_t)
        h_seq1.append(h_t1)
    h_seq2 = []    
    for t in range(n_seq):
        x_t2 = h_seq1[t]
        a_t2 = tf.nn.tanh(tf.add(tf.add(tf.matmul(x_t2, W['wa2']), tf.matmul(h_t2, U['ua2'])), B['ba2']), name="activation_gate")
        i_t2 = tf.nn.sigmoid(tf.add(tf.add(tf.matmul(x_t2, W['wi2']), tf.matmul(h_t2, U['ui2'])), B['bi2']), name="input_gate")
        f_t2 = tf.nn.sigmoid(tf.add(tf.add(tf.matmul(x_t2, W['wf2']), tf.matmul(h_t2, U['uf2'])), B['bf2']), name="forget_gate")
        o_t2 = tf.nn.sigmoid(tf.add(tf.add(tf.matmul(x_t2, W['wo2']), tf.matmul(h_t2, U['uo2'])), B['bo2']), name="forget_gate")
        c_t2 = tf.add(tf.multiply(a_t2, i_t2), tf.multiply(f_t2, c_t2))
        h_t2 = tf.multiply(tf.nn.tanh(c_t2), o_t2)
        h_seq2.append(h_t2)
    output = tf.add(tf.matmul(h_seq2[-1], W['hout']), B['bout'], name="output_layer")
    return output
        

### prediction ###
pred = lstm_network(X_, n_hidden_node_1, n_hidden_node_2, n_output)


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
            _, batch_loss = sess.run((optimizer, loss), feed_dict = {X_: batch_x, Y_: batch_y.reshape(-1,1)})
            
            # Compute average loss
            avg_loss += batch_loss/ n_samples * s_batch
        print("Epoch:", '%04d' % (epoch), "loss = ", "{:.9f}".format(avg_loss))
        
#
#
def lstm_predict(X):
    lstm_prediction = sess.run(pred, feed_dict={X_:X})
    lstm_prediction = lstm_prediction.reshape(-1)
    return lstm_prediction
#
#
### implement ###
lstm_train()
train_predict_y = lstm_predict(train_x)
test_predict_y = lstm_predict(test_x)
sess.close()

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
plt.title('Long Short Term Memory Tensorflow')
plt.xlabel('X', size = 20)
plt.ylabel('Y', size = 20)
plt.legend(loc = 1)


