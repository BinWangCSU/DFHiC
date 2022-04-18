# coding: utf-8
import os, time, pickle, random, sys, math
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import matplotlib.pyplot as plt
import hickle as hkl
from skimage.measure import compare_mse
from skimage.measure import compare_ssim

#GPU setting and Global parameters
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
#checkpoint = "checkpoint"
checkpoint = sys.argv[2]
graph_dir = sys.argv[3]
block_size = sys.argv[4]
tl.global_flag['mode']='DFHiC'
tl.files.exists_or_mkdir(checkpoint)
tl.files.exists_or_mkdir(graph_dir)
batch_size = int(sys.argv[5])  #128
lr_init = 1e-4

beta1 = 0.9
#n_epoch_init = 100
n_epoch_init = 1
n_epoch = 500
lr_decay = 0.1
decay_every = int(n_epoch / 2)
ni = int(np.sqrt(batch_size))


def calculate_psnr(mat1,mat2):
    data_range=np.max(mat1)-np.min(mat1)
    err=compare_mse(mat1,mat2)
    return 10 * np.log10((data_range ** 2) / err)

def calculate_ssim(mat1,mat2):
    data_range=np.max(mat1)-np.min(mat1)
    return compare_ssim(mat1,mat2,data_range=data_range)

train_data=np.load("preprocess/data/GM12878/train_data_raw_ratio16.npz")
lr_mats_full=train_data['train_lr']
hr_mats_full=train_data['train_hr']

lr_mats_train = lr_mats_full[:int(0.95*len(lr_mats_full))]
hr_mats_train = hr_mats_full[:int(0.95*len(hr_mats_full))]

lr_mats_valid = lr_mats_full[int(0.95*len(lr_mats_full)):]
hr_mats_valid = hr_mats_full[int(0.95*len(hr_mats_full)):]


# zssr
def DFHiC(t_matrix, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    # g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("DFHiC", reuse=reuse) as vs:
        x = InputLayer(t_matrix, name='in')

        ################## multi_dialted_cnn ##########################
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
        return n

t_matrix = tf.placeholder('float32', [None, block_size, block_size, 1], name='input_hic_matrix')
t_target_matrix = tf.placeholder('float32', [None, block_size, block_size, 1], name='t_target_hic_matrix')

net = DFHiC(t_matrix, is_train=True, reuse=False)
net_test = DFHiC(t_matrix, is_train=False, reuse=True)

l1_loss = tl.cost.absolute_difference_error(net.outputs, t_target_matrix, is_mean=True)
g_vars = tl.layers.get_variables_with_name('DFHiC', True, True)

with tf.variable_scope('learning_rate'):
    lr_v = tf.Variable(lr_init, trainable=False)

g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(l1_loss, var_list=g_vars)

#summary variables
merged_summary = tf.summary.scalar("l1_loss", l1_loss)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
tl.layers.initialize_global_variables(sess)

#record variables for TensorBoard visualization
summary_writer=tf.summary.FileWriter('%s'%graph_dir,graph=tf.get_default_graph())

wait=0
patience=20
best_mse_val = np.inf
best_epoch=0
for epoch in range(0, n_epoch + 1):
    ## update learning rate
    if epoch != 0 and (epoch % decay_every == 0):
        #new_lr_decay = lr_decay**(epoch // decay_every)
        new_lr_decay=1
        sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
        log = " ** new learning rate: %f (for DFHiC)" % (lr_init * new_lr_decay)
        print(log)
    elif epoch == 0:
        sess.run(tf.assign(lr_v, lr_init))
        log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for DFHiC)" % (lr_init, decay_every, lr_decay)
        print(log)

    epoch_time = time.time()
    total_loss = 0

    for idx in range(0, len(hr_mats_train)-batch_size, batch_size):
        b_mats_input = lr_mats_train[idx:idx + batch_size]
        b_mats_target = hr_mats_train[idx:idx + batch_size]
        errM, _ = sess.run([l1_loss, g_optim], {t_matrix: b_mats_input, t_target_matrix: b_mats_target})
    print("Epoch [%2d/%2d] time: %4.4fs, mse: %.6f" %
              (epoch, n_epoch, time.time() - epoch_time, errM))
    #validation
    hr_mats_pre = np.zeros(hr_mats_valid.shape)
    for i in range(hr_mats_pre.shape[0]//batch_size):
        hr_mats_pre[batch_size*i:batch_size*(i+1)] = sess.run(net_test.outputs, {t_matrix: lr_mats_valid[batch_size*i:batch_size*(i+1)]})
    hr_mats_pre[batch_size*(i+1):] = sess.run(net_test.outputs, {t_matrix: lr_mats_valid[batch_size*(i+1):]})
    mse_val=np.median(list(map(compare_mse,hr_mats_pre[:,:,:,0],hr_mats_valid[:,:,:,0])))
    if mse_val < best_mse_val:
        wait=0
        best_mse_val = mse_val
        #save the model with minimal MSE in validation samples
        tl.files.save_npz(net.all_params, name=checkpoint + '/{}_best.npz'.format(tl.global_flag['mode']), sess=sess)
        best_epoch=epoch
        # np.savetxt(checkpoint + 'best_epoch.txt',np.array(best_epoch))
    else:
        wait+=1
        if wait >= patience:
            print("Early stopping! The validation median mse is %.6f\n"%best_mse_val)
            #sys.exit() 

    log = "[*] Epoch: [%2d/%2d] time: %4.4fs, valid_mse:%.8f\n" % (epoch, n_epoch, time.time() - epoch_time,mse_val)
    print(log)
    #record variables for TensorBoard visualization
    summary=sess.run(merged_summary,{t_matrix: b_mats_input, t_target_matrix: b_mats_target})
    summary_writer.add_summary(summary, epoch)
    
print("epoch")
print(best_epoch)