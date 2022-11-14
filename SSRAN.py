# -*- coding: utf-8 -*-
"""
@author: Wenjing Chen
"""

from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import os
import scipy.io
import time
import pandas as pd
from random import shuffle
import random
import scipy.ndimage
import math
from local_band_attention import ECA

# -------------------------------------------------------parameter setting-------------------------------------------------------------
data_augment = False
patch_size = 4            # input size of network
epoch = 200               # epoch size
batch_size = 100          # batch size
num_epoch_deacy = 50      # deacy for learnin_rate_base
learning_rate_base = 0.01 # learning rate
learning_rate_decay = 0.1 # learning rate decay


# -------------------------------------------------------load data----------------------------------------------------------------------
print('Loading Data and Generate Training/Testing Samples.....')
# Load data
DATA_PATH = os.path.join(os.getcwd(), "Data")


# Pavia University database
HR_MSI = scipy.io.loadmat(os.path.join(DATA_PATH, 'pavia_cnn.mat'))['HR_MSI']
HR_HSI = scipy.io.loadmat(os.path.join(DATA_PATH, 'pavia_cnn.mat'))['HR_HSI']


# -------------------------------------------------------Height, Width, Band ----------------------------------------------------------
Height_HR, Width_HR, Band_MSI = HR_MSI.shape[0], HR_MSI.shape[1], HR_MSI.shape[2]
Band_HSI = HR_HSI.shape[2]
print('The Shape of HR-MSI:',Height_HR, Width_HR, Band_MSI)
print('The Shape of HR-HSI:',Height_HR, Width_HR, Band_HSI)
HR_MSI = HR_MSI.astype(float)
HR_HSI = HR_HSI.astype(float)


# ------------------------------------------------------Divide training set and testing set
Training_MSI = HR_MSI[:, 0:100, :]         # 120 for DC, 100 for Pavia, 36 for Paris 
Training_HSI = HR_HSI[:, 0:100, :]         # 120 for DC, 100 for Pavia, 36 for Paris 
Testing_MSI = HR_MSI[:, 100:Width_HR, :]   # 120 for DC, 100 for Pavia, 36 for Paris 
Testing_HSI = HR_HSI[:, 100:Width_HR, :]   # 120 for DC, 100 for Pavia, 36 for Paris 
print('The Shape of HR-MSI:',Training_MSI.shape, Testing_MSI.shape)
print('The Shape of HR-HSI:',Training_HSI.shape, Testing_HSI.shape)


# ------------------------------------------------------define: crop patches ---------------------------------------------------------
def Crop_Patch(data_for_crop,height_index,width_index,size):
    """ function to extract patches from the orignal data """
    transpose_array = np.transpose(data_for_crop, (2, 0, 1))  # why transpose?
    height_slice = slice(height_index, height_index + size)
    width_slice = slice(width_index, width_index + size)
    patch = transpose_array[:, height_slice, width_slice]
    return np.array(patch)   
  

# ------------------------------------------crop Training patches for training, with overlap, stride: 1-------------------------------------
Train_HSI_Patches, Train_MSI_Patches = [], []
for j in range(0, Training_MSI.shape[1]-patch_size+1, 1):
    for i in range(0, Training_MSI.shape[0]-patch_size+1, 1):
        curr_hsi = Crop_Patch(Training_HSI, i, j, patch_size)
        Train_HSI_Patches.append(curr_hsi)
        curr_msi = Crop_Patch(Training_MSI, i, j, patch_size)
        Train_MSI_Patches.append(curr_msi)
Train_HSI_Patches = np.array(Train_HSI_Patches)
Train_MSI_Patches = np.array(Train_MSI_Patches)

# ------------------------------------------crop HR patches for testing, with overlap, stride: patch_size-------------------------------------
Test_MSI_Patches, Test_HSI_Patches = [], []
for j in range(0, Testing_MSI.shape[1]-patch_size+1, patch_size):
    for i in range(0, Testing_MSI.shape[0]-patch_size+1, patch_size):
        curr_hsi = Crop_Patch(Testing_HSI, i, j, patch_size)
        Test_HSI_Patches.append(curr_hsi)
        curr_msi = Crop_Patch(Testing_MSI, i, j, patch_size)
        Test_MSI_Patches.append(curr_msi)
Test_HSI_Patches = np.array(Test_HSI_Patches)
Test_MSI_Patches = np.array(Test_MSI_Patches)

print('The shape of Inputs Train_MSI:',Train_MSI_Patches.shape)
print('The shape of Outputs Train_HSI:',Train_HSI_Patches.shape)
print('The shape of Inputs Test_MSI:',Test_MSI_Patches.shape)
print('The shape of Outputs Test_HSI:',Test_HSI_Patches.shape)

# ------------------------------------------caculate the number of iterations --------------------------------------------------
num_train = Train_HSI_Patches.shape[0]       # number of training samples     
training_iters = np.ceil(num_train*1.0/batch_size*epoch).astype(int)  
print("The Total Iterations is: " + np.str(training_iters))

# ------------------------------------------data preprocess---------------------------------------------------------------------
Train_HSI_Patches = np.transpose(Train_HSI_Patches,(0,2,3,1))
Train_MSI_Patches = np.transpose(Train_MSI_Patches,(0,2,3,1))
Test_HSI_Patches = np.transpose(Test_HSI_Patches,(0,2,3,1))
Test_MSI_Patches = np.transpose(Test_MSI_Patches,(0,2,3,1))


# ------------------------------------------define the network ----------------------------------------------------------------

x = tf.placeholder("float", [None, patch_size, patch_size, Band_MSI])
y = tf.placeholder("float", [None, patch_size, patch_size, Band_HSI])
is_training = tf.placeholder(tf.bool)

NNN = 64  # kernel number
kernel_size = 5 # number of local spectral bands for attention 
def conv_net(x):
    batch_norm_params = {
      'decay': 0.95,
      'epsilon': 1e-5,
      'scale': True,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'is_training': is_training,
      'fused': None,
    }

    with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      normalizer_fn=None):
        tf.set_random_seed(0)                   # the seed can be removed
        # to extract intial feature
        Spe1 = slim.conv2d(x, NNN, 1, padding='SAME', scope='Spe1')

        # ------------------residual spectral-spatial blocks 1------------------  
        # 1. 3x3 spatial branch
        Spe2_3 = slim.conv2d(Spe1, NNN, 3, padding='SAME', scope='Sep2_3')
        Spe3_3 = slim.conv2d(Spe2_3, NNN, 3, padding='SAME', scope='Spe3_3')
        # 2. 1x1 spectral branch
        Spe2_1 = slim.conv2d(Spe1, NNN, 1, padding='SAME', scope='Sep2_1')
        Spe3_1 = slim.conv2d(Spe2_1, NNN, 1, padding='SAME', scope='Spe3_1')
        # 3. aggregate spatial and spectral features 
        #    concat spectral and spatial features, then a 1x1 convolution is employed to generate spectral-spatial features 
        Spe3_concat = tf.concat([Spe3_1,Spe3_3],3)
        Spe3 = slim.conv2d(Spe3_concat, NNN, 1, padding='SAME', scope='Sep3_concat')
        # 4. attention for each feature channel
        Spe3 = ECA(Spe3, k_size = kernel_size, name = 'ECA1')
        # 5. skip-connection 
        Spe3 = tf.nn.relu(tf.add(Spe3,Spe1)) 

        # ------------------residual spectral-spatial blocks 2------------------ 
        # 1. 3x3 spatial branch    
        Spe4_3 = slim.conv2d(Spe3, NNN, 3, padding='SAME', scope='Sep4_3')
        Spe5_3 = slim.conv2d(Spe4_3, NNN, 3, padding='SAME', scope='Spe5_3')
        # 2. 1x1 spectral branch
        Spe4_1 = slim.conv2d(Spe3, NNN, 1, padding='SAME', scope='Sep4_1')
        Spe5_1 = slim.conv2d(Spe4_1, NNN, 1, padding='SAME', scope='Spe5_1')
        # 3. aggregate spatial and spectral features 
        #    concat spectral and spatial features, then a 1x1 convolution is employed to generate spectral-spatial features 
        Spe5_concat = tf.concat([Spe5_3,Spe5_1],3)
        Spe5 = slim.conv2d(Spe5_concat, NNN, 1, padding='SAME', scope='Sep5_concat')
        # 4. attention for each feature channel
        Spe5 = ECA(Spe5, k_size = kernel_size, name = 'ECA2')
        # 5. skip-connection 
        Spe5 = tf.nn.relu(tf.add(Spe5,Spe3)) 

        # ------------------residual spectral-spatial blocks 3------------------ 
        # 1. 3x3 spatial branch    
        Spe6_3 = slim.conv2d(Spe5, NNN, 3, padding='SAME', scope='Sep6_3')
        Spe7_3 = slim.conv2d(Spe6_3, NNN, 3, padding='SAME', scope='Spe7_3')
        # 2. 1x1 spectral branch
        Spe6_1 = slim.conv2d(Spe5, NNN, 1, padding='SAME', scope='Sep6_1')
        Spe7_1 = slim.conv2d(Spe6_1, NNN, 1, padding='SAME', scope='Spe7_1')
        # 3. aggregate spatial and spectral features 
        #    concat spectral and spatial features, then a 1x1 convolution is employed to generate spectral-spatial features 
        Spe7_concat = tf.concat([Spe7_3,Spe7_1],3)
        Spe7 = slim.conv2d(Spe7_concat, NNN, 1, padding='SAME', scope='Sep7_concat')
        # 4. attention for each feature channel
        Spe7 = ECA(Spe7, k_size = kernel_size, name = 'ECA3')
        # 5. skip-connection 
        Spe7 = tf.nn.relu(tf.add(Spe7,Spe5))  

        # Out
        Output_HSI = slim.conv2d(Spe7, Band_HSI, 1, padding='SAME', activation_fn=None)  
        Output_HSI = ECA(Output_HSI,k_size = kernel_size, name = 'ECA_output')
    return Output_HSI

# Construct model
pred = conv_net(x)

# ------------------------------------------------------Define loss and optimizer------------------------------------
# L2 loss
L2_loss = tf.reduce_mean(tf.square(pred - y))
