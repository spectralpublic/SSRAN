from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim

""" Implementation of ECA """
def ECA(net, k_size = 5, name = 'ECA'):
  print('k_size:',k_size)
  batchsize, height, width, in_channels = net.get_shape().as_list()
  # GAP
  # net_GAP = slim.avg_pool2d(net, height, stride = height, padding='VALID')
  net_GAP = tf.reduce_mean(net, axis=[1,2], keep_dims=True)

  # 1-D ECA
  net_GAP_for_1D = tf.reshape(net_GAP, [tf.shape(net_GAP)[0], in_channels, 1]) 
  atten = tf.layers.conv1d(net_GAP_for_1D, filters=1, kernel_size=k_size, padding='same', activation = tf.nn.sigmoid, use_bias=False, name=name)
  #print('att.', atten.shape)
  atten_3D = tf.reshape(atten, [tf.shape(net_GAP)[0], 1, 1, in_channels])

  scale = net * atten_3D

  return scale

