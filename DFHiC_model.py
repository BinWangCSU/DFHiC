import os, sys
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

def DFHiC(input_matrix, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  
    # g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("DFHiC", reuse=reuse) as vs:
        x = InputLayer(input_matrix, name='in')
        
        ################## DFHiC ##########################
        n_0 = Conv2d(x, 32, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n32s1/c1/0')
        n_1 = Conv2d(n_0, 32, (3, 3), (1, 1), dilation_rate=(2, 2), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n32s1/c1/1')
        n_2 = Conv2d(n_1, 32, (3, 3), (1, 1), dilation_rate=(2, 2), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n32s1/c1/2')
        n_3 = ElementwiseLayer([n_0, n_2], tf.add, name='add')
        n_4 = Conv2d(n_3, 64, (3, 3), (1, 1), dilation_rate=(2, 2), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/3')
        n_5 = Conv2d(n_4, 64, (3, 3), (1, 1), dilation_rate=(2, 2), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/4')
        n_6 = ElementwiseLayer([n_4, n_5], tf.add, name='add')
        n_7 = Conv2d(n_6, 128, (3, 3), (1, 1), dilation_rate=(2, 2), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c1/5')
        n_8 = Conv2d(n_7, 128, (3, 3), (1, 1), dilation_rate=(2, 2), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c1/6')
        n_9 = ElementwiseLayer([n_7, n_8], tf.add, name='add')
        n_10 = Conv2d(n_9, 256, (3, 3), (1, 1), dilation_rate=(2, 2), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c1/7')
        n_11 = Conv2d(n_10, 256, (3, 3), (1, 1), dilation_rate=(2, 2), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c1/8')
        n = Conv2d(n_11, 1, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n1s1/c/m')
        n = ElementwiseLayer([n, x], tf.add, name='add')
        ############################################################
        return n 

def DFHiC_predict(data, input_matrix, net, model_name, batch=64):
    print(model_name)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=model_name, network=net)
    out = np.zeros(data.shape)
    for i in range(out.shape[0]//batch):
        out[batch*i:batch*(i+1)] = sess.run(net.outputs, {input_matrix: data[batch*i:batch*(i+1)]})
    out[batch*(i+1):] = sess.run(net.outputs, {input_matrix: data[batch*(i+1):]})
    return out
