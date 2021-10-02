# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 12:40:24 2021

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
from transf_model import Encoder,Decoder, Transformer,create_look_ahead_mask,create_padding_mask 
import time


# data path
_data_path = "asset/data/"

# index to byte mapping
index2byte = ['<EMP>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
              'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','<start>','<end>']

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

label_padded_preproc = []
for l in label_transf:
    lab = np.append(l,29)
    label_padded_preproc.append(np.insert(lab,0,28))
    

mfcc_padded = []

for arr in mfcc_transf:
    mfcc_padded.append(tf.keras.preprocessing.sequence.pad_sequences(arr,padding='post',maxlen=1406))

mfcc_padded = np.array(mfcc_padded)

label_padded = tf.keras.preprocessing.sequence.pad_sequences(label_padded_preproc,padding='post',maxlen=516)


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

'''Transformer stuff'''
num_layers = 3
d_model = 512
num_heads = 8
dff = 2048
pe_dim = 1000
maximum_position_encoding = 1000

def create_masks_decoder(tar):
    # Encoder padding mask

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return look_ahead_mask


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
   mask = tf.math.logical_not(tf.math.equal(real, 0))
   loss_ = loss_object(real, pred)
   mask = tf.cast(mask, dtype=loss_.dtype)
   loss_ *= mask
   return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


encoder = Encoder(1,512,8,512,100)
decoder = Decoder(1,512,8,512,voca_size,1000)

transformer = Transformer(num_layers,d_model,num_heads,dff,pe_dim,voca_size,max_pos_encoding=maximum_position_encoding)

i = 0

for ele1,ele2 in dataset:
    # if ele1.shape[0]>max_el1:
    #     max_el1 = ele1.shape[0]
        
    # if ele1.shape[0]<min_el1:
    #     min_el1 = ele1.shape[0]
    
  if i != 1:
    # print(ele1.shape)
    # print(ele2.shape)
    # print('Testing element:', ele2[3,5])
    
    # op = encoder(ele1,training=False)
    
    # op2,_ = decoder(ele2,op)
    # print(op.shape)
    # print(op2.shape)
    
    op,_ = transformer(ele1,ele2,False)
    
    print(op.shape)
    
    loss = loss_function(ele2,op)
    print(loss)
    i += 1
  else:
    break

max_ele = 0
for label in label_padded:
    #print(label)
    for ele in label:
        if ele>max_ele:
            max_ele = ele
            
            
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
   def __init__(self, d_model, warmup_steps=4000):
      super(CustomSchedule, self).__init__()
      self.d_model = d_model
      self.d_model = tf.cast(self.d_model, tf.float32)
      self.warmup_steps = warmup_steps

   def __call__(self, step):
      arg1 = tf.math.rsqrt(step)
      arg2 = step * (self.warmup_steps ** -1.5)
      return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)
train_loss = tf.keras.metrics.Mean(name='train_loss')
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                    epsilon=1e-9)

@tf.function
def train_step(img_tensor, tar):
   tar_inp = tar[:, :-1]
   tar_real = tar[:, 1:]
   dec_mask = create_masks_decoder(tar_inp)
   with tf.GradientTape() as tape:
      predictions, _ = transformer(img_tensor, tar_inp,True, look_ahead_mask = dec_mask)
      loss = loss_function(tar_real, predictions)

   gradients = tape.gradient(loss, transformer.trainable_variables)   
   optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
   train_loss(loss)

for epoch in range(20):
   start = time.time()
   train_loss.reset_states()
   for (batch, (img_tensor, tar)) in enumerate(dataset):
      train_step(img_tensor, tar)
      if batch % 50 == 0:
         print ('Epoch {} Batch {} Loss {:.4f}'.format(
         epoch + 1, batch, train_loss.result()))

   print ('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                               train_loss.result()))
   print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
   
   
'''Evaluation'''   
def evaluate(audio_file):
    mfcc = np.load(audio_file, allow_pickle=False)
    mfcc = _augment_speech(mfcc)
    padded_mfcc = tf.keras.preprocessing.sequence.pad_sequences(mfcc,padding='post',maxlen=1406)
    
    padded_mfcc = tf.expand_dims(padded_mfcc, axis=0)
    
    decoder_input = [28]
    decoder_input = tf.expand_dims(decoder_input,axis = 0)
    result = []
    
    for i in range(100):
        dec_mask = create_masks_decoder(decoder_input)
        predictions,_ = transformer(padded_mfcc,decoder_input,False,look_ahead_mask = dec_mask)
        predictions = predictions[: ,-1:, :]
        
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if predicted_id == 29:
            return result,tf.squeeze(decoder_input, axis=0)
    
        result.append(int(predicted_id))
        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)
    #print(predicted_id)
    return result,decoder_input#tf.squeeze(decoder_input, axis=0)

res,dec_input_final = evaluate(mfcc_file[0])
text_op = ''

for char in res:
    text_op += index2byte[char]
    
print("Text output: ", text_op)



 
