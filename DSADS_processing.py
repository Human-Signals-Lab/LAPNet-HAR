# !/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 10 2021

@author: Rebecca Adaimi

DSADS dataset loading and preprocessing
Participants 7 and 8 are taken as test data
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

SAMPLING_FREQ = 25 # Hz

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

    path = './DSADS_data'
    activities = sorted(os.listdir(path))

    print(activities)


    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    for a in activities:
        activity_path = os.sep.join((path,a))
        participants = sorted(os.listdir(activity_path))

        print(participants)
        test_participants = participants[-2:]
        train_participants = participants[:-2]
        for p in train_participants:
            train_data_sub = []
            full_path = os.sep.join((activity_path, p))

            segments = sorted(os.listdir(full_path))
            for seg in segments:
                segment_path = os.sep.join((full_path, seg))
                print(segment_path)
                data = pd.DataFrame(np.genfromtxt(segment_path, delimiter=','))
                data = data[~np.isnan(data).any(axis=1)]   
                train_data_sub.extend(np.reshape(np.array(data),(1,np.shape(data)[0], np.shape(data)[1])))
                train_labels.extend([int(a[-2:])])
            #train_data_sub = normalize(train_data_sub)
            train_data.extend(train_data_sub)

        for p in test_participants:
            test_data_sub = []
            full_path = os.sep.join((activity_path, p))

            segments = sorted(os.listdir(full_path))
            for seg in segments:
                segment_path = os.sep.join((full_path, seg))
                print(segment_path)
                data = pd.DataFrame(np.genfromtxt(segment_path, delimiter=','))
                data = data[~np.isnan(data).any(axis=1)]   
                test_data_sub.extend(np.reshape(np.array(data),(1,np.shape(data)[0], np.shape(data)[1])))
                test_labels.extend([int(a[-2:])])  
            #test_data_sub = normalize(test_data_sub)
            test_data.extend(test_data_sub)

    assert len(test_data) == len(test_labels)
    assert len(train_data) == len(train_labels)

    print("Train Data: {}".format(np.shape(train_data)))
    print("Test Data: {}".format(np.shape(test_data)))

    obj = [(np.array(train_data), np.array(train_labels)), (np.array(test_data), np.array(test_labels))]
    target_filename = './DSADS_Train_Test_normalized.data'
    f = open(target_filename, 'wb')
    cp.dump(obj, f)
    f.close()
     
















