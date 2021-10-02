# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 10:11:43 2021

@author: sense
"""

import tensorflow as tf
import numpy as np

class ResBlock(tf.keras.layers.Layer):
    def __init__(self,kernel_size,dil_rate,dim):
        super(ResBlock, self).__init__()
        self.dilated_conv = tf.keras.layers.Conv1D(dim, kernel_size, dilation_rate=dil_rate, padding='same')
        self.conv1d = tf.keras.layers.Conv1D(dim, 1, padding='same')
        
        
    def call(self,tensor):
        x = self.dilated_conv(tensor)
        x = tf.nn.tanh(x)
        
        y = self.dilated_conv(tensor)
        y = tf.nn.sigmoid(y)
        
        out = x * y
        
        out = self.conv1d(out)
        
        return out + tensor, out
        

class WaveNetModel(tf.keras.Model):
    def __init__(self,vocab_size,res_kernel_size,dil_rate,res_dim):
        super(WaveNetModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(res_dim, 1, padding='same')
        self.conv2 = tf.keras.layers.Conv1D(res_dim, 1, padding='same')
        self.out_conv = tf.keras.layers.Conv1D(vocab_size, 1, padding='same')
        
        self.resblock1 = ResBlock(res_kernel_size,dil_rate[0],res_dim)
        self.resblock2 = ResBlock(res_kernel_size,dil_rate[1],res_dim)
        self.resblock3 = ResBlock(res_kernel_size,dil_rate[2],res_dim)
        self.resblock4 = ResBlock(res_kernel_size,dil_rate[3],res_dim)
        self.resblock5 = ResBlock(res_kernel_size,dil_rate[4],res_dim)
        
    def call(self,z):
        #print(z.shape)
        #print(z.dtype)
        z = self.conv1(z)
        z = tf.nn.tanh(z)
        
        skip = 0
        for i in range(3):
            z, s = self.resblock1(z)
            skip += s
            z, s = self.resblock2(z)
            skip += s
            z, s = self.resblock3(z)
            skip += s
            z, s = self.resblock4(z)
            skip += s
            z, s = self.resblock5(z)
            skip += s
            
        skip = self.conv2(skip)
        skip = self.out_conv(skip)
        
        return skip
        
        
        