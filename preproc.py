# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 17:45:46 2021

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

def process_libri(csv_file, category):

    parent_path = 'LibriSpeech/' + category + '/'
    labels, wave_files = [], []

    # create csv writer
    writer = csv.writer(csv_file, delimiter=',')

    # read directory list by speaker
    speaker_list = glob.glob(parent_path + '*')
    for spk in speaker_list:

        # read directory list by chapter
        chapter_list = glob.glob(spk + '/*/')
        for chap in chapter_list:

            # read label text file list
            txt_list = glob.glob(chap + '/*.txt')
            for txt in txt_list:
                with open(txt, 'rt') as f:
                    records = f.readlines()
                    for record in records:
                        # parsing record
                        field = record.split('-')  # split by '-'
                        speaker = field[0]
                        chapter = field[1]
                        field = field[2].split()  # split field[2] by ' '
                        utterance = field[0]  # first column is utterance id

                        # wave file name
                        wave_file = parent_path + '%s/%s/%s-%s-%s.flac' % \
                                                  (speaker, chapter, speaker, chapter, utterance)
                        wave_files.append(wave_file)

                        # label index
                        labels.append(data_preproc.str2index(' '.join(field[1:])))  # last column is text label

    # save results
    for i, (wave_file, label) in enumerate(zip(wave_files, labels)):
        fn = wave_file.split('/')[-1]
        target_filename = 'mfcc/' + fn + '.npy'
        if os.path.exists( target_filename ):
            continue
        # print info
        print("LibriSpeech corpus preprocessing (%d / %d) - '%s']" % (i, len(wave_files), wave_file))

        # load flac file
        #wave, sr, _ = scikits.audiolab.flacread(wave_file)
        wave, sr = librosa.load(wave_file)#wavfile.read(wave_file)

        # get mfcc feature
        mfcc = librosa.feature.mfcc(wave, sr=16000)

        # save result ( exclude small mfcc data to prevent ctc loss )
        if len(label) < mfcc.shape[1]:
            # filename

            # save meta info
            writer.writerow([fn] + label)

            # save mfcc
            np.save(target_filename, mfcc, allow_pickle=False)
            
csv_f = open('valid.csv', 'w')
process_libri(csv_f, 'dev-clean')
csv_f.close()