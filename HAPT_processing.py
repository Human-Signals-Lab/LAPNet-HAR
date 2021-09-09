# !/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 10 2021

@author: Rebecca Adaimi

HAPT dataset loading and preprocessing
Participants 29 and 30 used as test data 
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
import sys
import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import glob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SAMPLING_FREQ = 50 # Hz
#SLIDING_WINDOW_LENGTH = int(49)
SLIDING_WINDOW_LENGTH = int(2.56*SAMPLING_FREQ)

#SLIDING_WINDOW_STEP = int(1*SAMPLING_FREQ)
SLIDING_WINDOW_STEP = int(SLIDING_WINDOW_LENGTH/2)

def standardize(mat):
    """ standardize each sensor data columnwise"""
    for i in range(mat.shape[1]):
        mean = np.mean(mat[:, [i]])
        std = np.std(mat[:, [i]])
        mat[:, [i]] -= mean
        mat[:, [i]] /= std

    return mat


def __rearrange(a,y, window, overlap):
    l, f = a.shape
    shape = (int( (l-overlap)/(window-overlap) ), window, f)
    stride = (a.itemsize*f*(window-overlap), a.itemsize*f, a.itemsize)
    X = np.lib.stride_tricks.as_strided(a, shape=shape, strides=stride)
    #import pdb; pdb.set_trace()

    l,f = y.shape
    shape = (int( (l-overlap)/(window-overlap) ), window, f)
    stride = (y.itemsize*f*(window-overlap), y.itemsize*f, y.itemsize)
    Y = np.lib.stride_tricks.as_strided(y, shape=shape, strides=stride)
    Y = Y.max(axis=1)

    return X, Y.flatten()
# def normalize(x):
#     """Normalizes all sensor channels by mean substraction,
#     dividing by the standard deviation and by 2.
#     :param x: numpy integer matrix
#         Sensor data
#     :return:
#         Normalized sensor data
#     """
#     x = np.array(x, dtype=np.float32)
#     m = np.mean(x, axis=0)
#     x -= m
#     std = np.std(x, axis=0)
#     std += 0.000001
#     x /= (std * 2)  # 2 is for having smaller values
#     return x

def normalize(data):
    """ l2 normalization can be used"""

    y = data[:, 0].reshape(-1, 1)
    X = np.delete(data, 0, axis=1)
    transformer = Normalizer(norm='l2', copy=True).fit(X)
    X = transformer.transform(X)

    return np.concatenate((y, X), 1)


def normalize_df(data):
    """ l2 normalization can be used"""

    #y = data[:, 0].reshape(-1, 1)
    #X = np.delete(data, 0, axis=1)
    transformer = Normalizer(norm='l2', copy=True).fit(data)
    data = transformer.transform(data)

    return data

def min_max_scaler(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    return data

def read_dir(DIR):

    folder1=sorted(os.listdir(DIR))
    #import pdb; pdb.set_trace()

    labels = np.genfromtxt(os.path.join(DIR,folder1[-1]), delimiter=' ')
    accel_files = folder1[:int(len(folder1[:-1])/2)]
    gyro_files = folder1[int(len(folder1[:-1])/2):-1]

    train_d = []
    test_d = []
    for a_file,g_file in zip(accel_files,gyro_files):
        #import pdb; pdb.set_trace()
        a_ff = os.path.join(DIR, a_file)
        g_ff = os.path.join(DIR, g_file)
        a_df = np.genfromtxt(a_ff, delimiter=' ')
        g_df = np.genfromtxt(g_ff, delimiter=' ')
        ss = a_file.split('.')[0].split('_')
        exp, user = int(ss[1][-2:]), int(ss[2][-2:])

        indices = labels[labels[:,0]==exp]
        indices = indices[indices[:,1]==user]
        for ii in range(len(indices)):
            a_sub = a_df[int(indices[ii][-2]):int(indices[ii][-1]),:]
            g_sub = g_df[int(indices[ii][-2]):int(indices[ii][-1]),:]
            if user == 29 or user == 30:
                test_d.extend(np.append(np.append(a_sub,g_sub,axis=1),np.array([indices[ii][-3]]*len(a_sub))[:,None],axis=1))
            else:
                train_d.extend(np.append(np.append(a_sub,g_sub,axis=1),np.array([indices[ii][-3]]*len(a_sub))[:,None],axis=1))

    train_x = np.array(train_d)[:,:-1]
    test_x = np.array(test_d)[:,:-1]
    train_y = np.array(train_d)[:,-1]
    test_y = np.array(test_d)[:,-1]

    #x_train, y_train, x_test, y_test = down_sample(x_train, y_train, x_test, y_test, True)
    print(np.unique(train_y),np.unique(test_y))
    train_x = normalize(train_x)
    test_x = normalize(test_x)
    train_x, train_y = __rearrange(train_x, train_y.astype(int).reshape((-1,1)), SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    test_x, test_y = __rearrange(test_x, test_y.astype(int).reshape((-1,1)), SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)

    #import pdb; pdb.set_trace()

    return train_x, train_y, test_x, test_y
    
if __name__ == "__main__": 

    path = './HAPT_data/RawData'

    # activity = []
    subject = []
    # age = []
    act_num = []
    sensor_readings = []

    ## Corrupt datapoint:
    # act_num[258] = '11' 
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    
    train_data, train_labels, test_data, test_labels = read_dir(path)

    assert len(test_data) == len(test_labels)
    assert len(train_data) == len(train_labels)

    print("Train Data: {}".format(np.shape(train_data)))
    print("Test Data: {}".format(np.shape(test_data)))

    obj = [(np.array(train_data), np.array(train_labels)), (np.array(test_data), np.array(test_labels))]
    target_filename = './HAPT_data/HAPT_Train_Test_{}_{}.data'.format(SLIDING_WINDOW_LENGTH,SLIDING_WINDOW_STEP)
    f = open(target_filename, 'wb')
    cp.dump(obj, f)
    f.close()