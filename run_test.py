import os, sys
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from skimage.measure import compare_mse
from skimage.measure import compare_ssim
from DFHiC_model import DFHiC,DFHiC_predict

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

def calculate_psnr(mat1,mat2):
    data_range=np.max(mat1)-np.min(mat1)
    err=compare_mse(mat1,mat2)
    return 0 if err==0 else 10 * np.log10((data_range ** 2) / err)

def calculate_ssim(mat1,mat2):
    return compare_ssim(mat1,mat2)


input_matrix = tf.placeholder('float32', [None, None, None, 1], name='matrix_input')
net = DFHiC(input_matrix, is_train=False, reuse=False)   

test_data=np.load("preprocess/data/GM12878/test_data_raw_ratio16.npz")
lr_mats_test=test_data['test_lr']
hr_mats_test=test_data['test_hr']

model_path="/home/liukun/code/zssr/Pretrained_weights/DFHiC_model.npz"
sr_matrix = DFHiC_predict(lr_mats_test, input_matrix, net, model_path)

print(sr_matrix.shape)

np.save('DFHiC_predicted_result.npy',sr_matrix)
    
mse_score=list(map(compare_mse,hr_mats_test[:,:,:,0],sr_matrix[:,:,:,0]))
psnr_score=list(map(calculate_psnr,hr_mats_test[:,:,:,0],sr_matrix[:,:,:,0]))
ssim_score=list(map(calculate_ssim,hr_mats_test[:,:,:,0],sr_matrix[:,:,:,0]))
print("##################################")
print('MSE score:%.5f'%np.median(mse_score))
print('PSNR score:%.5f'%np.median(psnr_score))
print('SSIM score:%.5f'%np.median(ssim_score))
print("##################################")
