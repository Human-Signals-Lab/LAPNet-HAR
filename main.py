import time
import numpy as np
import tensorflow as tf
import sys
import pickle
import os
import copy

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score
import matplotlib.pyplot as plt
import itertools
import csv

from sklearn import metrics
from enum import Enum
import librosa.display
import sys
from scipy import stats 
import datetime
from scipy.fftpack import dct
import _pickle as cp
import copy
import os
from collections import Counter

import random
random.seed(1)
from imblearn.over_sampling import SMOTE
from subprocess import call
from models import *

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import shutil


def smote_oversample(X, y):
    smote = SMOTE()

    xx = X.reshape((-1, 6))
    yy = [y]*np.shape(X)[1]
    yy = np.array(yy).flatten()
    print(np.shape(xx),np.shape(yy))
    X_smote, y_smote = smote.fit_resample(xx, yy)
    return X_smote.reshape((-1, np.shape(X)[1], np.shape(X)[2])), y_smote[::np.shape(X)[1]]

def random_oversample(X, y):

    count= Counter(y)

    majority = count.most_common(1)[0]
    #print(majority)
    for k in count:
        if k != majority[0]:
            num_samples = majority[1] - count[k]
            #print(k, num_samples)
            #print(len(X[y == k]))
            sampled = random.choices(range(len(X[y == k])), k=num_samples)
            subset = X[y==k]
            #print(np.shape(subset))
            X = np.vstack((X,subset[sampled]))
            #print(np.shape(y))
            y = np.append(y,k*np.ones(len(sampled)))
            #print(np.shape(X), np.shape(y))
    return X, y

def plotCNNStatistics(statistics_path):

    statistics_dict = cPickle.load(open(statistics_path, 'rb'))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    lines = []
 
    #print(statistics_dict)
    bal_alpha = 0.3
    test_alpha = 1.0
    bal_map = np.array([statistics['Trainloss'].cpu().data.numpy() for statistics in statistics_dict['Trainloss']])    # (N, classes_num)
    test_map = np.array([statistics['Testloss'] for statistics in statistics_dict['Testloss']])    # (N, classes_num)
    test_f1 = np.array([statistics['test_f1'] for statistics in statistics_dict['test_f1']])    # (N, classes_num)

    # val_map = np.array([statistics['val_f1'] for statistics in statistics_dict['val_f1']])

    #print(bal_map)
    #print(test_map)
    line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
    line, = ax.plot(test_map, color='r', alpha=test_alpha)
    # line, = ax.plot(val_map, color='g', alpha=test_alpha)

    lines.append(line)


    ax.set_ylim(0, 1.)
    #ax.set_xlim(0, len(iterations))
    #ax.xaxis.set_ticks(np.arange(0, len(iterations), 25))
    #ax.xaxis.set_ticklabels(np.arange(0, max_plot_iteration, 50000))
    ax.yaxis.set_ticks(np.arange(0, 1.01, 0.05))
    ax.yaxis.set_ticklabels(np.around(np.arange(0, 1.01, 0.05), decimals=2))        
    ax.grid(color='b', linestyle='solid', linewidth=0.3)
    plt.legend(labels=['Training Loss','Testing Loss'], loc=2)

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    line, = ax.plot(test_f1, color='r', alpha=test_alpha)
    ax.set_ylim(0,1.)
    ax.yaxis.set_ticks(np.arange(0, 1.01, 0.05))
    ax.yaxis.set_ticklabels(np.around(np.arange(0, 1.01, 0.05), decimals=2))        
    plt.ylabel('Test Average Fscore')


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



print("Downloading opportunity dataset...")
if not os.path.exists("OpportunityUCIDataset.zip"):
    call(
        'wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip"',
        shell=True
    )
    print("Downloading done.\n")
else:
    print("Dataset already downloaded. Did not download twice.\n")


print("Extracting...")
if not os.path.exists("oppChallenge_gestures.data"):
    from preprocess_Oppdata import generate_data
    generate_data("OpportunityUCIDataset.zip", "oppChallenge_gestures.data", "gestures")
    print("Extracting successfully done to oppChallenge_gestures.data.")
else:
    print("Dataset already extracted. Did not extract twice.\n")



#--------------------------------------------
# Dataset-specific constants and functions
#--------------------------------------------

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113
NB_SENSOR_CHANNELS_WITH_FILTERING = 149

# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 18

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH =24


# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = int(SLIDING_WINDOW_LENGTH/2)

# Batch Size
BATCH_SIZE = 100
BATCH_SIZE_VAL = 100


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

print("Loading data...")
X_train, y_train, X_test, y_test = load_dataset('oppChallenge_gestures.data')
print(np.shape(y_train))
assert (NB_SENSOR_CHANNELS_WITH_FILTERING == X_train.shape[1] or NB_SENSOR_CHANNELS == X_train.shape[1])



X_train, y_train_segments = __rearrange(X_train, y_train.reshape((-1,1)), SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
X_test, y_test_segments = __rearrange(X_test, y_test.reshape((-1,1)), SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
#sys.exit()

# Data is reshaped
X_train = X_train.reshape((-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS)) # for input to Conv1D
X_test = X_test.reshape((-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS)) # for input to Conv1D

print(" ..after sliding and reshaping, train data: inputs {0}, targets {1}".format(X_train.shape, y_train_segments.shape))
print(" ..after sliding and reshaping, test data : inputs {0}, targets {1}".format(X_test.shape, y_test_segments.shape))

print(np.shape(X_train))
print(Counter(y_train_segments))

# # remove null class

# X_train = X_train[y_train_segments != 0]
# X_test = X_test[y_test_segments != 0]

# y_train_segments = y_train_segments[y_train_segments != 0]
# y_train_segments = y_train_segments -1
# y_test_segments = y_test_segments[y_test_segments != 0]
# y_test_segments = y_test_segments - 1

print(Counter(y_train_segments))

classes = np.unique(y_test_segments)


y_train = tf.keras.utils.to_categorical(y_train_segments, num_classes=NUM_CLASSES, dtype='int32')
y_test = tf.keras.utils.to_categorical(y_test_segments, num_classes=NUM_CLASSES, dtype='int32')

#model = InceptionNN(NUM_CLASSES)
model = DeepConvLSTM()

if torch.cuda.is_available():
    model.cuda()


# Statistics
statistics_path = './statistics/DeepConvLSTM_Opportunity.pkl'

if not os.path.exists(os.path.dirname(statistics_path)):
    os.makedirs(os.path.dirname(statistics_path))
statistics_container = StatisticsContainer(statistics_path)


x_train_tensor = torch.from_numpy(np.array(X_train)).float()
y_train_tensor = torch.from_numpy(np.array(y_train)).float()
x_test_tensor = torch.from_numpy(np.array(X_test)).float()
y_test_tensor = torch.from_numpy(np.array(y_test)).float()


x_val_tensor = torch.from_numpy(np.array(X_test)).float()
y_val_tensor = torch.from_numpy(np.array(y_test)).float()

train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)
val_data = TensorDataset(x_val_tensor, y_val_tensor)

print(np.shape(x_train_tensor))
train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                    batch_size=BATCH_SIZE,
                    num_workers=1, pin_memory=True, shuffle = True,drop_last=True)

val_loader = torch.utils.data.DataLoader(dataset=val_data, 
                        batch_size=BATCH_SIZE_VAL,
                        num_workers=1, pin_memory=True, shuffle = True,drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                        batch_size=BATCH_SIZE_VAL,
                        num_workers=1, pin_memory=True, shuffle = True,drop_last=True)

optimizer = optim.Adam(model.parameters(), lr=1e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
#optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
#optimizer = optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=0.9)

#criterion = nn.CrossEntropyLoss()

num_epochs = 100
### Training Loop ########
iteration = 0
for epoch in range(num_epochs):

    h = model.init_hidden(BATCH_SIZE) 
    model.train()       

    #running_loss = 0.0
    for d in train_loader:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = d
        inputs = inputs.cuda()
        labels = labels.cuda()
        h = tuple([each.data for each in h])

        # zero the parameter gradients
        optimizer.zero_grad()



        #print(np.shape(labels))
        yhat, h = model(inputs,h,BATCH_SIZE)

        #clipwise_output = model(inputs,inputs.shape[0])
        #print("....",np.shape(clipwise_output))
        #clipwise_output = outputs['clipwise_output']
        loss = F.binary_cross_entropy(yhat, labels)
        loss.backward()
        optimizer.step()


    print('[Epoch %d]' % (epoch + 1))
    print('Train loss: {}'.format(loss))
    eval_output = []
    true_output = []
    test_output = []
    true_test_output = []
    val_h = model.init_hidden(BATCH_SIZE_VAL)
    #print(np.shape(val_h[0]))        

    model.eval()

    with torch.no_grad():

        for val_input, y_val in test_loader:


            val_input = torch.from_numpy(np.array(val_input)).float()
            val_input = val_input.cuda()
            y_val = y_val.cuda()
            val_h = tuple([each.data for each in val_h])
            #print(np.shape(val_h[0]), np.shape(val_input))        


            yhat, val_h = model(val_input, val_h, BATCH_SIZE_VAL)
            #clipwise_output = model(val_input,val_input.shape[0])
            #clipwise_output = yhat['clipwise_output']
            test_loss = F.binary_cross_entropy(yhat, y_val)

            test_output.append(yhat.data.cpu().numpy())
            true_test_output.append(y_val.data.cpu().numpy())

        test_oo = np.argmax(np.vstack(test_output), axis = 1)
        true_test_oo = np.argmax(np.vstack(true_test_output), axis = 1)

        accuracy = metrics.accuracy_score(true_test_oo, test_oo)
        precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_test_oo, test_oo, average='weighted')
        try:
            auc_test = metrics.roc_auc_score(np.vstack(true_test_output), np.vstack(test_output), average="weighted")
        except ValueError:
            auc_test = None
        print('Test loss: {}'.format(test_loss))
        print('TEST average_precision: {}'.format(precision))
        print('TEST average f1: {}'.format(fscore))
        print('TEST average recall: {}'.format(recall))
        print('TEST auc: {}'.format(accuracy))

        trainLoss = {'Trainloss': loss}
        statistics_container.append(iteration, trainLoss, data_type='Trainloss')
        testLoss = {'Testloss': test_loss}
        statistics_container.append(iteration, testLoss, data_type='Testloss')
        test_f1 = {'test_f1':fscore}
        statistics_container.append(iteration, test_f1, data_type='test_f1')

        statistics_container.dump()
        
print('Finished Training')


print('Inference...............')
eval_output = []
true_output = []
val_h = model.init_hidden(BATCH_SIZE_VAL)
true_output = []
#embedding_output = np.empty((0,60928))
#embedding_output = np.empty((0,2048))
model.eval()
with torch.no_grad():
    for val_input, y_val in test_loader:

        val_h = tuple([each.data for each in val_h])


        val_input = torch.from_numpy(np.array(val_input)).float()
        val_input = val_input.cuda()
        y_val = y_val.cuda()
    

        yhat, val_h = model(val_input, val_h, BATCH_SIZE_VAL)
        #embeddings = yhat['embedding']
        #embedding_output = np.vstack((embedding_output,embeddings.cpu().numpy()))
        #clipwise_output = model(val_input, val_input.shape[0])
        # yhat = model(x_val)
        #clipwise_output = yhat['clipwise_output']
        eval_output.extend(yhat.data.cpu().numpy().tolist())
        true_output.extend(y_val.data.cpu().numpy().tolist())
        #print(clipwise_output.data.cpu().numpy())

    eval_oo = np.argmax(np.vstack(eval_output), axis = 1)
    true_oo = np.argmax(np.vstack(true_output), axis = 1)

    # average_precision = metrics.average_precision_score(np.vstack(true_output), np.vstack(eval_output),average="weighted")
    precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_oo, eval_oo, average='weighted')
    accuracy = metrics.accuracy_score(true_oo, eval_oo)
    print('Inference precision: {}'.format(precision))
    print('Inference f1: {}'.format(fscore))
    print('Inference recall: {}'.format(recall))
    print('Inference accuracy: {}'.format(accuracy))

print(np.shape(np.argsort(true_output)[:,::-1][:,0]))
true_y = []
pred_y = []
for i in range(len(np.argsort(true_output)[:,::-1][:,0])):
    true_y.append(np.argsort(true_output)[[i],::-1][0][0])
    pred_y.append(np.argsort(eval_output)[[i],::-1][0][0])
#true_y = cat[np.argsort(true_output)[:,::-1][:,0]]
#pred_y = cat[np.argsort(eval_output)[:,::-1][:,0]]
C = confusion_matrix(true_y, pred_y)
plt.figure(figsize=(10,10))
plot_confusion_matrix(C, class_list=list(classes), normalize=True, title='Predicted Results')

plotCNNStatistics(statistics_path)
shutil.rmtree('./statistics')
plt.show()