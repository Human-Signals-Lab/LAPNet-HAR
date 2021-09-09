#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 16:31:56 2018

@author: Rebecca Adaimi

PAMAP2 Dataset Sliding window + Train/Test Split 
Participants 5 and 6 are used for testing 
"""

import numpy as np
import pandas as pd
import os
import math as m
import matplotlib.pyplot as plt 
from scipy import stats
import scipy.fftpack 
import copy
import scipy as sp
import scipy.signal
from collections import Counter
import _pickle as cp

SAMPLING_FREQ = 100  # 100Hz

SLIDING_WINDOW_LENGTH = int(5.12 * SAMPLING_FREQ)
#SLIDING_WINDOW_LENGTH = int(1.*SAMPLING_FREQ)

SLIDING_WINDOW_STEP = int(1*SAMPLING_FREQ)
#SLIDING_WINDOW_STEP = int(SLIDING_WINDOW_LENGTH/2)

def __rearrange(a,y, window, overlap):
    l, f = a.shape
    shape = (int( (l-overlap)/(window-overlap) ), window, f)
    stride = (a.itemsize*f*(window-overlap), a.itemsize*f, a.itemsize)
    X = np.lib.stride_tricks.as_strided(a, shape=shape, strides=stride)

    l,f = y.shape
    shape = (int( (l-overlap)/(window-overlap) ), window, f)
    stride = (y.itemsize*f*(window-overlap), y.itemsize*f, y.itemsize)
    Y = np.lib.stride_tricks.as_strided(y, shape=shape, strides=stride)
    Y = Y.max(axis=1)

    return X, Y.flatten()

def normalize(x):
    """Normalizes all sensor channels by mean substraction,
    dividing by the standard deviation and by 2.
    :param x: numpy integer matrix
        Sensor data
    :return:
        Normalized sensor data
    """
    x = np.array(x, dtype=np.float32)
    m = np.mean(x, axis=0)
    x -= m
    std = np.std(x, axis=0)
    std += 0.000001
    x /= (std * 2)  # 2 is for having smaller values
    return x
        
if __name__ == "__main__":   

    path = './PAMAP2_Dataset/Protocol/' 
    participants = os.listdir(path)

    test_participants = [105, 106]  ## taking participants 5 and 6 as test data 
    
    train_data = []
    test_data = []
    test_labels = []
    train_labels = []


    for p in sorted(participants):
        print (str(p))
        full_path = os.sep.join((path,p))
        data = pd.DataFrame(np.genfromtxt(full_path, delimiter=' '))            
        print(np.shape(data))
        data.iloc[:,2] = data.iloc[:,2].interpolate()
        
        transient = np.where(data.iloc[:,1] == 0)[0]
        data = np.delete(np.array(data), transient,axis=0)
        
        data = data[~np.isnan(data).any(axis=1)]

        label = data[:,1].astype(int)

        data = data[:, 2:]
        print(np.shape(data))

        data = normalize(data)
        stacked_data, label_segments = __rearrange(data, label.reshape((-1,1)), SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
        print(np.shape(stacked_data))
        if int(p.split('.')[0][-3:]) in test_participants:
            test_data.extend(stacked_data)
            test_labels.extend(label_segments)

        else:
            train_data.extend(stacked_data)
            train_labels.extend(label_segments)

    assert len(test_data) == len(test_labels)
    assert len(train_data) == len(train_labels)

    print("Train Data: {}".format(np.shape(train_data)))
    print("Test Data: {}".format(np.shape(test_data)))

    obj = [(np.array(train_data), np.array(train_labels)), (np.array(test_data), np.array(test_labels))]
    target_filename = './PAMAP2_Dataset/PAMAP2_Train_Test_{}_{}_normalized.data'.format(SLIDING_WINDOW_LENGTH,SLIDING_WINDOW_STEP)
    f = open(target_filename, 'wb')
    cp.dump(obj, f)
    f.close()
     
                
                
                
        
        
    
    

