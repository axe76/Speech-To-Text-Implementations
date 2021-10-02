# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 15:51:41 2021

@author: sense
"""

import string

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

def str2index(str_):

    # clean white space
    str_ = ' '.join(str_.split())
    # remove punctuation and make lower case
    str_ = str_.translate(str.maketrans('','',string.punctuation)).lower()#str_.translate(None, string.punctuation).lower()

    res = []
    for ch in str_:
        try:
            res.append(byte2index[ch])
        except KeyError:
            # drop OOV
            pass
    return res