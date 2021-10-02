# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 17:46:55 2021

@author: sense
"""

import numpy as np
import pandas as pd
import glob
import csv
import librosa
# import scikits.audiolab
import data_preproc
import os
from scipy.io import wavfile
import tensorflow as tf
from model import *
import time

# data path
_data_path = "asset/data/"

# index to byte mapping
index2byte = ['<EMP>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
              'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# byte to index mapping
byte2index = {}
for i, ch in enumerate(index2byte):
    byte2index[ch] = i

# vocabulary size
voca_size = len(index2byte)

label, mfcc_file = [], []
with open('valid.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        if len(row)>0:
            # mfcc file
            mfcc_file.append('mfcc/' + row[0] + '.npy')
            # label info ( convert to string object for variable-length support )
            label.append(np.asarray(row[1:], dtype=np.int).tostring())
            

def _augment_speech(mfcc):

    # random frequency shift ( == speed perturbation effect on MFCC )
    r = np.random.randint(-2, 2)

    # shifting mfcc
    mfcc = np.roll(mfcc, r, axis=0)

    # zero padding
    if r > 0:
        mfcc[:r, :] = 0
    elif r < 0:
        mfcc[r:, :] = 0

    return mfcc

def _load_mfcc(mfcc_files,labels):
    mfcc_transf = []
    label_transf = []
    
    for mfcc_file,label in zip(mfcc_files,labels):

        # label, wave_file
        #mfcc_file,label = src_list
    
        # decode string to integer
        label = np.fromstring(label, np.int)
    
        # load mfcc
        mfcc = np.load(mfcc_file, allow_pickle=False)
    
        # speed perturbation augmenting
        mfcc = _augment_speech(mfcc)
        
        mfcc_transf.append(mfcc)
        label_transf.append(label)

    return mfcc_transf,label_transf#mfcc,label

 
mfcc_transf,label_transf = _load_mfcc(mfcc_file,label)

mfcc_padded = []

for arr in mfcc_transf:
    mfcc_padded.append(tf.keras.preprocessing.sequence.pad_sequences(arr,padding='post',maxlen=1406))

mfcc_padded = np.array(mfcc_padded)

label_padded = tf.keras.preprocessing.sequence.pad_sequences(label_transf,padding='post',maxlen=516)


dataset = tf.data.Dataset.from_tensor_slices((mfcc_padded, label_padded))

# Use map to load the numpy files in parallel
# dataset = dataset.map(lambda item1, item2: tf.numpy_function(_load_mfcc, [item1, item2], [tf.float32, tf.int32]),num_parallel_calls=tf.data.experimental.AUTOTUNE)
# #mfcc_test, label_test = _load_mfcc(mfcc_file[0],label[0])
# dataset = dataset.padded_batch(16,padded_shapes=)
dataset = dataset.batch(16).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

i = 0
max_el1 = 0
min_el1 = 516
for ele1,ele2 in dataset:
    # if ele1.shape[0]>max_el1:
    #     max_el1 = ele1.shape[0]
        
    # if ele1.shape[0]<min_el1:
    #     min_el1 = ele1.shape[0]
    
  if i != 1:
    print(ele1.shape)
    print(ele2.shape)
    i += 1
  else:
    break

rate_list = [1,2,4,8,16]
model = WaveNetModel(voca_size, 7,rate_list, 128)

i = 0
for ele1,ele2 in dataset:
    # if ele1.shape[0]>max_el1:
    #     max_el1 = ele1.shape[0]
        
    # if ele1.shape[0]<min_el1:
    #     min_el1 = ele1.shape[0]
    
  if i != 1:
    # input_shape = (4, 10, 128)
    # x = tf.random.normal(input_shape)
    # print(x.dtype)
    ele1 = tf.cast(ele1, dtype=tf.float32)
    op = model(ele1)
    #loss = 
    # print(op.shape)
    # print(ele2.shape)
    
    pred_length = ele1.shape[1]
    label_length = ele2.shape[1]
    
    pred_length = pred_length * tf.ones(shape=16,dtype=tf.int64)
    label_length = label_length * tf.ones(shape=16,dtype=tf.int64)
    print(ele2.shape)
    print(op.shape)
    # print(pred_length.dtype)
    # print(label_length.dtype)
    loss = tf.nn.ctc_loss(ele2, op, label_length, pred_length, logits_time_major = False)
    i+=1
  else:
    break

optimizer = optimizer = tf.keras.optimizers.Adam()
epochs = 5
num_steps = (mfcc_padded.shape[0])//16

def loss_function(real, pred, real_label_length,pred_label_length):
   loss_ = tf.nn.ctc_loss(real,pred,real_label_length,pred_label_length,logits_time_major = False)
   return tf.reduce_mean(loss_)

for i in range(epochs):
    start = time.time()
    total_loss = 0
    
    for (batch, (mfcc_feat,target)) in enumerate(dataset):
        mfcc_feat = tf.cast(mfcc_feat, dtype=tf.float32)
        pred_length = mfcc_feat.shape[1] * tf.ones(shape=mfcc_feat.shape[0],dtype=tf.int64)
        label_length = target.shape[1] * tf.ones(shape=target.shape[0],dtype=tf.int64)
        with tf.GradientTape() as tape:
            output = model(mfcc_feat)
            loss = loss_function(target,output,label_length,pred_length)
        
        total_loss += loss
        trainable_variables = model.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        
        if batch % 40 == 0:
         print ('Epoch {} Batch {} Loss {:.4f}'.format(i + 1, batch, loss))
        
    
    print(f'Epoch {i+1} Loss {total_loss/num_steps:.6f}')
    print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')
        
    
    