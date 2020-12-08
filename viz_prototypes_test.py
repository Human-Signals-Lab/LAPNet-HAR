"""
Testing class prototypes in offline setting
- use pretrained Opp_model saved 
- extract embeddings from DeepConvLSTM 
- extract class prototypes and visualize


"""

import time
import numpy as np
import tensorflow as tf
import sys
import pickle
import os
import copy

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import itertools
import csv
import seaborn as sns
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

NO_NULL = True
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

# # remove null class
if NO_NULL:
    X_train = X_train[y_train_segments != 0]
    X_test = X_test[y_test_segments != 0]

    y_train_segments = y_train_segments[y_train_segments != 0]
    y_train_segments = y_train_segments -1
    y_test_segments = y_test_segments[y_test_segments != 0]
    y_test_segments = y_test_segments - 1


print(" ..after sliding and reshaping, train data: inputs {0}, targets {1}".format(X_train.shape, y_train_segments.shape))
print(" ..after sliding and reshaping, test data : inputs {0}, targets {1}".format(X_test.shape, y_test_segments.shape))

X = np.append(X_train,X_test, axis=0)
y = np.append(y_train_segments,y_test_segments, axis=0)

print(np.shape(X),np.shape(y))



print(Counter(y))

classes = np.unique(y)


y = tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES, dtype='int32')

#model = InceptionNN(NUM_CLASSES)
model = DeepConvLSTM()
if NO_NULL:
    path = 'Opp_model/DeepConvLSTM_Opportunity_noNull.pth'
else:
    path = 'Opp_model/DeepConvLSTM_Opportunity.pth'

checkpoint = torch.load(path, map_location='cuda')
model.load_state_dict(checkpoint)

if torch.cuda.is_available():
    model.cuda()

x_tensor = torch.from_numpy(np.array(X)).float()
y_tensor = torch.from_numpy(np.array(y)).float()



data = TensorDataset(x_tensor, y_tensor)

print(np.shape(x_tensor))
data_loader = torch.utils.data.DataLoader(dataset=data, 
                    batch_size=BATCH_SIZE,
                    num_workers=1, pin_memory=True, shuffle = False,drop_last=True)


tr_h = model.init_hidden(BATCH_SIZE)
true_output = []
#embedding_output = np.empty((0,60928))
#embedding_output = np.empty((0,2048))
model.eval()
embeddings_list = dict()
embeddings_list['embeddings'] = []
embeddings_list['labels'] = []
with torch.no_grad():

    for tr_input, y_tr in data_loader:
        tr_h = tuple(each.data for each in tr_h)

        tr_input = torch.from_numpy(np.array(tr_input)).float()
        tr_input = tr_input.cuda()
        y_tr = y_tr.cuda()

        yhat, tr_h, embeddings = model(tr_input, tr_h, BATCH_SIZE)
        embeddings_list['embeddings'].extend(embeddings.data.cpu().numpy())
        embeddings_list['labels'].extend(y_tr.data.cpu().numpy())

def extract_prototypes(X,y):

    classes = np.sort(np.unique(y))
    prototypes = np.empty((len(classes),np.shape(X)[1]))
    prot_std = np.empty((len(classes),np.shape(X)[1]))
    #print(classes)
    for c in classes:
        p_mean = np.mean(np.array(X)[y==c], axis = 0)
        p_std = np.std(np.array(X)[y==c], axis = 0)
        prototypes[c] = p_mean
        prot_std[c] = p_std

    return prototypes,prot_std

prototypes,prot_std = extract_prototypes(embeddings_list['embeddings'],np.argmax(embeddings_list['labels'],axis=1))

## visualization

pca = PCA(n_components=2)
prototypes_pca = pca.fit_transform(prototypes)
prot_std_pca = pca.transform(prot_std)
fig = plt.figure(figsize=(8, 3))
NUM_COLORS = len(prototypes)

cm = plt.get_cmap('gist_rainbow')
#plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
kk = np.arange(NUM_COLORS)
markers = ['.',  'x', 'h','1']
#markers = ['.']
np.random.shuffle(kk)
colors = [cm((1.*i)/NUM_COLORS) for i in kk]
#print(Counter(colors))
#sys.exit()
#colors = ['#4EACC5', '#FF9C34', '#4E9A06','#F544FE'] #,'#FF4040']
labels = ['Null','OpenDoor1', 'OpenDoor2','CloseDoor1','CloseDoor2','OpenFridge','CloseFridge','OpenDishwasher','CloseDishwasher','OpenDrawer1','CloseDrawer1','OpenDrawer2','CloseDrawer2','OpenDrawer3','CloseDrawer3','CleanTable','DrinkFromCup','ToogleSwitch']
if NO_NULL:
    labels = labels[1:]
yy = np.argmax(embeddings_list['labels'],axis=1)

## visualize embeddings
for k, col in zip(range(len(prototypes)), colors):

    s = pca.transform(np.array(embeddings_list['embeddings'])[yy==k])

    plt.plot(s[:,0], s[:,1], 'o',
            markerfacecolor=col, markeredgecolor=col,
             marker=markers[k%len(markers)],markersize=7)

    #add label
    plt.annotate(labels[k], (prototypes_pca[k,0], prototypes_pca[k,1]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=10, weight='bold',rotation=45,
                 color='k') 

## vizualize prototypes extracted from all data
fig = plt.figure(figsize=(8, 3))
markers = ['.']

for k, col in zip(range(len(prototypes)), colors):
    # plt.plot(s[:,0], s[:,1], 'o',
    #         markerfacecolor=col, markeredgecolor=col,
    #          marker=markers[k%len(markers)],markersize=7)

    plt.plot(prototypes_pca[k,0],prototypes_pca[k,1], 'o',
            markerfacecolor=col, markeredgecolor=col, 
            marker=markers[k%len(markers)],markersize=20) 

    #add label
    plt.annotate(labels[k], (prototypes_pca[k,0], prototypes_pca[k,1]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=10, weight='bold',rotation=45,
                 color='k')   

## extract prototypes from train only and infer labels on test data using euclidean distance 

prototypes,prot_std = extract_prototypes(embeddings_list['embeddings'][:len(X_train)],np.argmax(embeddings_list['labels'],axis=1)[:len(X_train)])

pca = PCA(n_components=2)
prototypes_pca = pca.fit_transform(prototypes)
#prot_std_pca = pca.transform(prot_std)
test = embeddings_list['embeddings'][len(X_train):]
def compute_euclidean(a,b):
    a2 = tf.cast(tf.reduce_sum(tf.square(a),[-1],keepdims=True),dtype=tf.float32)
    ab = tf.cast(tf.matmul(a,b, transpose_b=True), dtype=tf.float32)
    b2 = tf.cast(tf.repeat(tf.reduce_sum(tf.square(b),[-1],keepdims=True), len(a),axis=0), dtype=tf.float32)
    #print(np.shape(a),np.shape(b),np.shape(a2),np.shape(ab),np.shape(b2))

    return a2 - 2*ab + b2
logits = np.empty((len(test), 0))
for c in range(len(prototypes)):
    logits = np.hstack((logits,-compute_euclidean(test,prototypes[c].reshape((1,-1)))))
pred = tf.nn.softmax(logits)   
y_pred = np.argmax(pred, axis = 1)
true_y = np.argmax(embeddings_list['labels'][len(X_train):], axis=1).reshape((-1,1))

test_embd_pca = pca.transform(test)
fig = plt.figure(figsize=(8, 3))
markers = ['.',  'x', 'h','1']

for k, col in zip(range(len(prototypes)), colors):
    #print(k, np.shape(test_embd_pca[y_pred==k,0]))
    plt.plot(test_embd_pca[y_pred==k,0], test_embd_pca[y_pred==k,1], 'o',
            markerfacecolor=col, markeredgecolor=col,
             marker=markers[k%len(markers)],markersize=7)

    plt.plot(prototypes_pca[k,0],prototypes_pca[k,1], 'o',
            markerfacecolor=col, markeredgecolor=col, 
            marker=markers[k%len(markers)],markersize=7) 

    #add label
    plt.annotate(labels[k], (prototypes_pca[k,0], prototypes_pca[k,1]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=10, weight='bold',rotation=45,
                 color='k') 
# y_pred = y_pred[(true_y != 0).flatten()]
# true_y = true_y[true_y != 0]
# true_y = true_y - 1
precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_y, y_pred.reshape((-1,1)), average='weighted')

print('Inference Results: F1-score: {}, Recall: {}, Precision: {}'.format(fscore,recall,precision))
C = confusion_matrix(true_y, y_pred.reshape((-1,1)))
plt.figure(figsize=(10,10))
plot_confusion_matrix(C, class_list=labels, normalize=True, title='Predicted Results')

plt.show()