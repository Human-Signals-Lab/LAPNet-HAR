
import time
import numpy as np
#import tensorflow as tf
import sys
import pickle
import os
import copy

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score
from sklearn import utils
import matplotlib.pyplot as plt
import itertools
import csv
from sklearn.decomposition import PCA
from inc_pca import IncPCA
import pickle 

from sklearn import metrics
from enum import Enum
import librosa.display
import sys
from scipy import stats 
import datetime
from scipy.fftpack import dct
import _pickle as cPickle
import copy
import os
from collections import Counter
import torch
from torch.autograd import Variable
import random
from subprocess import call
import json
import torch

def rearrange(a,y, window, overlap):
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

def plotCNNStatistics(statistics_path):

    statistics_dict = cPickle.load(open(statistics_path, 'rb'))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    lines = []
 
    bal_alpha = 0.3
    test_alpha = 1.0
    bal_map = np.array([statistics['Trainloss'].cpu().data.numpy() for statistics in statistics_dict['Trainloss']])    # (N, classes_num)
    test_map = np.array([statistics['Testloss'] for statistics in statistics_dict['Testloss']])    # (N, classes_num)
    test_f1 = np.array([statistics['test_f1'] for statistics in statistics_dict['test_f1']])    # (N, classes_num)

    basetrain_map = np.array([statistics['BaseTrainloss'].cpu().data.numpy() for statistics in statistics_dict['BaseTrainloss']])
    basetrain_f1 = np.array([statistics['BaseTrain_f1'] for statistics in statistics_dict['BaseTrain_f1']])
    
    newClasses_test_map = np.array([statistics['Testloss_NewClasses'].cpu().data.numpy() for statistics in statistics_dict['Testloss_NewClasses']])
    newClasses_test_f1 = np.array([statistics['newClasses_test_f1'] for statistics in statistics_dict['newClasses_test_f1']])

    line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
    line, = ax.plot(test_map, color='r', alpha=test_alpha)

    lines.append(line)
     
    ax.grid(color='b', linestyle='solid', linewidth=0.3)
    plt.legend(labels=['Training Loss','Testing Loss'], loc=2)

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    line, = ax.plot(test_f1, color='r', alpha=test_alpha)
    ax.yaxis.set_ticks(np.arange(0, 1.01, 0.05))
    ax.yaxis.set_ticklabels(np.around(np.arange(0, 1.01, 0.05), decimals=2))        
    plt.ylabel('Test Average Fscore')

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    line, = ax.plot(basetrain_map, color='r', alpha=test_alpha)   
    plt.ylabel('Base Train Loss')

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    line, = ax.plot(basetrain_f1, color='r', alpha=test_alpha)
    ax.yaxis.set_ticks(np.arange(0, 1.01, 0.05))
    ax.yaxis.set_ticklabels(np.around(np.arange(0, 1.01, 0.05), decimals=2))        
    plt.ylabel('Base train Average Fscore')

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    line, = ax.plot(newClasses_test_map, color='r', alpha=test_alpha)   
    plt.ylabel('New Classes Test Loss')

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    line, = ax.plot(newClasses_test_f1, color='r', alpha=test_alpha)
    ax.yaxis.set_ticks(np.arange(0, 1.01, 0.05))
    ax.yaxis.set_ticklabels(np.around(np.arange(0, 1.01, 0.05), decimals=2))        
    plt.ylabel('New Classes Test Average Fscore')

def plot_confusion_matrix(cm, class_list,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks, class_list, rotation=45)
    plt.yticks(tick_marks, class_list)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.clim(0.,1.)
    plt.ylabel('Ground True Activities')
    plt.xlabel('Predicted Activities')

def extract_sample(n_classes, n_support, n_query, inputs, labels, seed, shuffle=False):

    support = []
    y_support = []
    query = []
    y_query = []
    np.random.seed(seed)
    #print(Counter(labels.data.cpu().numpy()))
    K = np.random.choice(np.unique(labels), n_classes, replace=False)
    #print(K)
    for cls in K:
        datax_cls = copy.deepcopy(inputs[labels == cls])
        perm = utils.shuffle(datax_cls.data.cpu().numpy())
        #print(perm)
        #perm = np.random.permutation(datax_cls)
        #print(np.shape(perm))
        # if len(perm) < n_support:
        #     change = n_support - len(perm)
        support_cls = copy.deepcopy(perm[:n_support])
        #print(np.shape(support_cls))
        support.extend(support_cls)
        #print(support)
        y_support.extend([cls]*len(support_cls))
        query_cls = copy.deepcopy(perm[n_support:])
        query.extend(query_cls)
        y_query.extend([cls]*len(query_cls))

        #print(np.shape(support_cls), np.shape(query_cls),np.shape(perm))

    if len(y_query) < 1:
        y_query = copy.deepcopy(y_support)
        query = copy.deepcopy(support)
    elif len(np.unique(y_query)) < len(np.unique(y_support)):
        size = int(np.mean(list(Counter(y_query).values())))
        for cls in np.setdiff1d(list(np.unique(y_support)), list(np.unique(y_query))):
            datax_cls = np.where(y_support == cls)[0]
            #print(size, datax_cls)
            idx = np.random.choice(datax_cls,min(len(datax_cls),size),replace=False)
           # print(idx)
            y_query.extend(list(np.array(y_support)[idx]))
            query.extend(list(np.array(support)[idx]))

    support = np.array(support)
    query = np.array(query)
    y_support = np.array(y_support)
    y_query = np.array(y_query)

    if shuffle:
        support, y_support = utils.shuffle(support,y_support,random_state=seed)
        query, y_query = utils.shuffle(query,y_query,random_state=seed)

    support = torch.from_numpy(support).float()
    query = torch.from_numpy(query).float()
    return support, y_support, query, y_query


def load_dataset(filename):

    f = open(filename, 'rb')
    data = cPickle.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test


def extract_sample(n_classes, n_support, n_query, inputs, labels, seed, shuffle=False):

    support = []
    y_support = []
    query = []
    y_query = []
    np.random.seed(seed)
    #print(Counter(labels.data.cpu().numpy()))
    K = np.random.choice(np.unique(labels), n_classes, replace=False)
    #print(K)
    for cls in K:
        datax_cls = copy.deepcopy(inputs[labels == cls])
        perm = utils.shuffle(datax_cls.data.cpu().numpy())
        #print(perm)
        #perm = np.random.permutation(datax_cls)
        #print(np.shape(perm))
        # if len(perm) < n_support:
        #     change = n_support - len(perm)
        support_cls = copy.deepcopy(perm[:n_support])
        #print(np.shape(support_cls))
        support.extend(support_cls)
        #print(support)
        y_support.extend([cls]*len(support_cls))
        query_cls = copy.deepcopy(perm[n_support:])
        query.extend(query_cls)
        y_query.extend([cls]*len(query_cls))

        #print(np.shape(support_cls), np.shape(query_cls),np.shape(perm))

    if len(y_query) < 1:
        y_query = copy.deepcopy(y_support)
        query = copy.deepcopy(support)
    elif len(np.unique(y_query)) < len(np.unique(y_support)):
        size = int(np.mean(list(Counter(y_query).values())))
        for cls in np.setdiff1d(list(np.unique(y_support)), list(np.unique(y_query))):
            datax_cls = np.where(y_support == cls)[0]
            #print(size, datax_cls)
            idx = np.random.choice(datax_cls,min(len(datax_cls),size),replace=False)
           # print(idx)
            y_query.extend(list(np.array(y_support)[idx]))
            query.extend(list(np.array(support)[idx]))

    support = np.array(support)
    query = np.array(query)
    y_support = np.array(y_support)
    y_query = np.array(y_query)

    if shuffle:
        support, y_support = utils.shuffle(support,y_support,random_state=seed)
        query, y_query = utils.shuffle(query,y_query,random_state=seed)

    support = torch.from_numpy(support).float()
    query = torch.from_numpy(query).float()
    return support, y_support, query, y_query

def order_classes(inputs, labels,seed):

    data_x = []
    data_y = []
    np.random.seed(seed)
    #print(Counter(labels.data.cpu().numpy()))
    n_classes = len(np.unique(labels))
    K = np.random.choice(np.unique(labels), n_classes, replace=False)
    #print(np.shape(inputs))
    change = 0
    for cls in K:
        datax_cls = inputs[labels == cls]
        perm = np.random.permutation(datax_cls)

        data_x.extend(perm)
        data_y.extend([cls]*len(datax_cls))

    data_x = np.array(data_x)
    data_y = np.array(data_y)


    data_x = torch.from_numpy(data_x).float()
    data_y = torch.from_numpy(data_y).float()
    return data_x, data_y


def modify_new_logits(p, pre_p, old_classes, beta=.5):

    """
    Adapted from https://arxiv.org/pdf/2003.13191.pdf
    p : output logits of new classifier
    pre_p : old classifier output logits

    """
    beta = beta # from paper
    #new_p = torch.index_select(p, 1 , old_classes) * beta + torch.index_select(pre_p, 1, old_classes) * (1 - beta)

    for c in old_classes:
        p[:,c] = p[:,c] * beta + pre_p[:,c] * (1 - beta) 

    return p
    #return new_p

def MultiClassCrossEntropy(logits, labels, T, device):
    """
    Source: https://github.com/ngailapdi/LWF/blob/baa07ee322d4b2f93a28eba092ad37379f565aca/model.py#L16
    :param logits: output logits of the model
    :param labels: ground truth labels
    :param T: temperature scaler
    :return: the loss value wrapped in torch.autograd.Variable
    """
    #print(type(logits), logits.requires_grad)
    #print(labels, type(labels), labels.requires_grad)
    #labels = Variable(labels.data, requires_grad=False).to(device)
    outputs = torch.log_softmax(-logits / T, dim=1)  # compute the log of softmax values
    labels = torch.softmax(-labels / T, dim=1)

    #print('outputs: ', outputs)
    #print('labels: ', labels.shape)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    #print('OUT: ', outputs, outputs.requires_grad)
    return outputs #Variable(outputs.data, requires_grad=True).to(device)


def calculate_forgetting(f1_scores_t, max_f1_scores_up2t):
    #forgetting_scores = {seq_id: {} for seq_id in f1_scores_t.keys()}
    F_k = []
    #import pdb; pdb.set_trace()
    for class_id, f1 in f1_scores_t.items():
        if max_f1_scores_up2t[class_id] == 0.:
            continue
        F_k.append(1.0 - f1 / max_f1_scores_up2t[class_id]) # consider only those tasks that the model remembers
    #import pdb; pdb.set_trace()
    forgetting_score = np.mean(F_k)

    return forgetting_score


# def calculate_forgetting(sequence_to_previous_task_scores, sequence_to_max_task_scores):
#     forgetting_scores = {seq_id: {} for seq_id in sequence_to_previous_task_scores.keys()}
#     for seq_id, prev_task_to_scores_dict in sequence_to_previous_task_scores.items():
#         for task_id, prev_task_scores in prev_task_to_scores_dict.items():
#             # F_k = [sequence_to_max_task_scores[seq_id][prev_task_id] - score for prev_task_id, score in
#             #        prev_task_scores.items() if sequence_to_max_task_scores[seq_id][prev_task_id] > 0]
#             F_k = [1.0 - score / sequence_to_max_task_scores[seq_id][prev_task_id]  for prev_task_id, score in
#                    prev_task_scores.items() if sequence_to_max_task_scores[seq_id][prev_task_id] > 0] # consider only those tasks that the model remembers
#             forgetting_scores[seq_id][task_id] = mean(F_k)
#     df = pd.DataFrame.from_dict(forgetting_scores).T
#     df.columns = ['Task ' + str(i+1) for i in range(len(forgetting_scores[0]))]
#     return df