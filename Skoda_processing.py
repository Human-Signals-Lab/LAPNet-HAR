# !/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 10 2021

@author: Rebecca Adaimi

Skoda dataset loading and preprocessing

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

SAMPLING_FREQ = 98 # Hz
#SLIDING_WINDOW_LENGTH = int(49)
SLIDING_WINDOW_LENGTH = int(1.*SAMPLING_FREQ)

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
    import pdb; pdb.set_trace()

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


def label_count_from_zero(all_data):
    """ start all labels from 0 to total number of activities"""

    labels = {32: 'null class', 48: 'write on notepad', 49: 'open hood', 50: 'close hood',
              51: 'check gaps on the front door', 52: 'open left front door',
              53: 'close left front door', 54: 'close both left door', 55: 'check trunk gaps',
              56: 'open and close trunk', 57: 'check steering wheel'}

    a = np.unique(all_data[:, 0])

    for i in range(len(a)):
        all_data[:, 0][all_data[:, 0] == a[i]] = i
        print(i, labels[a[i]])
    return all_data

def split(data):
    """ get 80% train, 10% test and 10% validation data from each activity """

    y = data[:, 0]  # .reshape(-1, 1)
    X = np.delete(data, 0, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

    return X_train, y_train, X_test, y_test

def get_train_val_test(data):
    # removing sensor ids
    #import pdb; pdb.set_trace() 
    #print(np.shape(data))
    # for i in range(1, 60, 6):
    #     #print(i ,data[:,i])
    #     data = np.delete(data, i, 1)
    no_of_sensor = 10
    columns2drop = [1 + s * 7 for s in range(no_of_sensor)] + [5 + s * 7 for s in range(no_of_sensor)] + [6 + s * 7 for s in range(no_of_sensor)] + [7 + s * 7 for s in range(no_of_sensor)]
    data = np.delete(data, columns2drop, 1)
    data = data[data[:, 0] != 32]  # remove null class activity

    data = label_count_from_zero(data)
    data = normalize(data)

    activity_id = np.unique(data[:, 0])
    number_of_activity = len(activity_id)

    for i in range(number_of_activity):

        data_for_a_single_activity = data[np.where(data[:, 0] == activity_id[i])]
        trainx, trainy, testx, testy = split(data_for_a_single_activity)

        if i == 0:
            x_train, y_train, x_test, y_test = trainx, trainy, testx, testy

        else:
            x_train = np.concatenate((x_train, trainx))
            y_train = np.concatenate((y_train, trainy))

            x_test = np.concatenate((x_test, testx))
            y_test = np.concatenate((y_test, testy))

    return x_train, y_train, x_test, y_test

def down_sample(x_train, y_train, x_test, y_test, verbose=False):
    print('Before Downsampling: ')
    print("x_train shape = ", x_train.shape)
    print("y_train shape =", y_train.shape)
    print("x_test shape =", x_test.shape)
    print("y_test shape =", y_test.shape)    

    x_train = x_train[::3, :]
    y_train = y_train[::3]
    x_test = x_test[::3, :]
    y_test = y_test[::3]
    if verbose:
        print("x_train shape(downsampled) = ", x_train.shape)
        print("y_train shape(downsampled) =", y_train.shape)
        print("x_test shape(downsampled) =", x_test.shape)
        print("y_test shape(downsampled) =", y_test.shape)
    return x_train, y_train, x_test, y_test

def read_dir(DIR):

    right_path = os.path.join(DIR, 'right_classall_clean.mat')
    #left_path = os.path.join(DIR, 'left_classall_clean.mat')

    data_dict = scipy.io.loadmat(right_path, squeeze_me=True)
    #left_data = scipy.io.loadmat(left_path)['left_classall_clean']

    all_data = data_dict[list(data_dict.keys())[3]]  

    x_train, y_train, x_test, y_test = get_train_val_test(all_data)
    #x_train, y_train, x_test, y_test = down_sample(x_train, y_train, x_test, y_test, True)
    print(np.unique(y_train))
    train_x, train_y = __rearrange(x_train, y_train.astype(int).reshape((-1,1)), SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    test_x, test_y = __rearrange(x_test, y_test.astype(int).reshape((-1,1)), SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    #import pdb; pdb.set_trace()

    return train_x, train_y, test_x, test_y
    
if __name__ == "__main__": 

    path = './Skoda_data/'

    # activity = []
    subject = []
    # age = []
    act_num = []
    sensor_readings = []

    ## Corrupt datapoint:
    # act_num[258] = '11' 
    
    train_data, train_labels, test_data, test_labels = read_dir(path)

    assert len(test_data) == len(test_labels)
    assert len(train_data) == len(train_labels)

    print("Train Data: {}".format(np.shape(train_data)))
    print("Test Data: {}".format(np.shape(test_data)))

    print(np.unique(train_labels))

    obj = [(np.array(train_data), np.array(train_labels)), (np.array(test_data), np.array(test_labels))]
    target_filename = './Skoda_data/Skoda_Train_Test_{}_{}.data'.format(SLIDING_WINDOW_LENGTH,SLIDING_WINDOW_STEP)
    f = open(target_filename, 'wb')
    cp.dump(obj, f)
    f.close()