## streaming data + updating prototypes

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
from sklearn.decomposition import PCA

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
from data_handler import *
import shutil
from prototype_memory import *


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

# remove null class

X_train = X_train[y_train_segments != 0]
X_test = X_test[y_test_segments != 0]

y_train_segments = y_train_segments[y_train_segments != 0]
y_train_segments = y_train_segments -1
y_test_segments = y_test_segments[y_test_segments != 0]
y_test_segments = y_test_segments - 1

print(Counter(y_train_segments))

classes = np.unique(y_test_segments)

## streaming data
baseClassesNb = 5
percentage = .2# 20%
dataHandler = DataHandler(nb_baseClasses=baseClassesNb, seed=0, train={'data':X_train,'label':y_train_segments}, ClassPercentage=percentage)
dataHandler.streaming_data()
baseData = dataHandler.getBaseData()
baseClasses = np.unique(baseData['label'])


mapping = {}
for i in np.arange(baseClassesNb):
    mapping[baseClasses[i]] = i
for x in range(len(baseData['label'])):
    baseData['label'][x] = mapping[baseData['label'][x]]


## select base classes in test data 
X_test_select = []
y_test_select = []
for c in np.arange(baseClassesNb):
    d,l = X_test[y_test_segments == c,:], y_test_segments[y_test_segments == c]
    X_test_select.extend(d)
    y_test_select.extend(l)

y_train = tf.keras.utils.to_categorical(baseData['label'], num_classes=baseClassesNb, dtype='int32')
y_test = tf.keras.utils.to_categorical(y_test_select, num_classes=baseClassesNb, dtype='int32')

#model = InceptionNN(NUM_CLASSES)
model = DeepConvLSTM(n_classes=len(np.unique(baseData['label'])))

if torch.cuda.is_available():
    model.cuda()


## pretrain base model
x_train_tensor = torch.from_numpy(np.array(baseData['data'])).float()
y_train_tensor = torch.from_numpy(np.array(y_train)).float()
train_data = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                    batch_size=BATCH_SIZE,
                    num_workers=1, pin_memory=True, shuffle = True,drop_last=True)

optimizer = optim.Adam(model.parameters(), lr=1e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)


num_epochs = 30
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
        yhat, h, _  = model(inputs,h,BATCH_SIZE)

        #clipwise_output = model(inputs,inputs.shape[0])
        #print("....",np.shape(clipwise_output))
        #clipwise_output = outputs['clipwise_output']
        loss = F.binary_cross_entropy(yhat, labels)
        loss.backward()
        optimizer.step()


    print('[Epoch %d]' % (epoch + 1))
    print('Train loss: {}'.format(loss))


x_test_tensor = torch.from_numpy(np.array(X_test_select)).float()
y_test_tensor = torch.from_numpy(np.array(y_test)).float()
test_data = TensorDataset(x_test_tensor, y_test_tensor)

test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                        batch_size=BATCH_SIZE_VAL,
                        num_workers=1, pin_memory=True, shuffle = True,drop_last=True)

## extract training embeddings
tr_h = model.init_hidden(BATCH_SIZE) 

model.eval()
embeddings_list = dict()
embeddings_list['embeddings'] = []
embeddings_list['labels'] = []
with torch.no_grad():

    for tr_input, y_tr in train_loader:
        tr_h = tuple(each.data for each in tr_h)

        tr_input = torch.from_numpy(np.array(tr_input)).float()
        tr_input = tr_input.cuda()
        y_tr = y_tr.cuda()

        yhat, tr_h, embeddings = model(tr_input, tr_h, BATCH_SIZE)
        embeddings_list['embeddings'].extend(embeddings.data.cpu().numpy())
        embeddings_list['labels'].extend(y_tr.data.cpu().numpy())

# ## extract class prototypes
# def extract_prototypes(X,y):

#     classes = np.sort(np.unique(y))
#     prototypes = dict()
#     prot_std = dict()
#     for c in classes:
#         prototypes[c] = []
#         prot_std[c] = []
#     # prototypes = np.empty((len(classes),np.shape(X)[1]))
#     # prot_std = np.empty((len(classes),np.shape(X)[1]))
#     #print(classes)
#     for c in classes:
#         p_mean = np.mean(np.array(X)[y==c], axis = 0)
#         p_std = np.std(np.array(X)[y==c], axis = 0)
#         prototypes[c].append(p_mean)
#         prot_std[c].append(p_std)

#     return prototypes,prot_std

# Statistics
statistics_path = './statistics/OnlineProtoNet_DeepConvLSTM_Opportunity.pkl'

if not os.path.exists(os.path.dirname(statistics_path)):
    os.makedirs(os.path.dirname(statistics_path))
statistics_container = StatisticsContainer(statistics_path)


labels = np.argmax(embeddings_list['labels'],axis=1)
for i in range(len(labels)):
    labels[i] = list(mapping.keys())[labels[i]]

prot_mem = PrototypeMemory()
prot_mem.initialize_prototypes(torch.from_numpy(np.array(embeddings_list['embeddings'])).float(),torch.from_numpy(np.array(labels)).float())

#prototypes,prot_std = extract_prototypes(embeddings_list['embeddings'],labels)
cm = plt.get_cmap('gist_rainbow')
NUM_COLORS = len(classes)

colors = [cm((1.*i)/NUM_COLORS) for i in np.arange(NUM_COLORS)]
## visualization

pca = PCA(n_components=2)
pca.fit(embeddings_list['embeddings'])

prototypes_pca = pca.transform(list(prot_mem.prototypes.values()))

fig, ax = plt.subplots(figsize=(10,10))
# ax.set_xlim(-6,6)
# ax.set_ylim(-6,6)
xdata, ydata = [], []
ln, = plt.plot([],[],'ro')

xdata.extend(prototypes_pca[:,0])
ydata.extend(prototypes_pca[:,1])
LABELS = ['OpenDoor1', 'OpenDoor2','CloseDoor1','CloseDoor2','OpenFridge','CloseFridge','OpenDishwasher','CloseDishwasher','OpenDrawer1','CloseDrawer1','OpenDrawer2','CloseDrawer2','OpenDrawer3','CloseDrawer3','CleanTable','DrinkFromCup','ToogleSwitch']
annotations= set()

def plt_dynamic(x,y,labels,ax,colors,markers=['.',  'x', 'h','1']):
    #print(x,y,labels,x[labels==6],y[labels==6])
    for k, col in zip(np.unique(labels),colors):
        #print(k,x,y,labels)
        xx,yy = x[labels == k], y[labels == k]
        ax.plot(xx,yy, 'o',
        markerfacecolor=col, markeredgecolor=col, 
        marker=markers[k%len(markers)],markersize=20) 

        #add label
        if annotate and LABELS[k] not in annotations:
            annotations.add(LABELS[k])
            ax.annotate(LABELS[k], (xx, yy),
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=10, weight='bold',rotation=45,
                     color='k') 

    fig.canvas.draw()

ytrue = np.array(list(prot_mem.prototypes.keys()), dtype=np.int32)
annotate = True
print(xdata, ydata, ytrue)
plt_dynamic(np.array(xdata), np.array(ydata), ytrue, ax, colors)
annotate = True
plt.show(block=False)
#sys.exit()
model.eval()
val_h = model.init_hidden(1)
counter = 1
xdata, ydata, ytrue = [], [], []
embeddings=[]
ll=[]

def compute_euclidean(query, proto):
    x = query.unsqueeze(1).expand(query.size(0),proto.size(0),query.size(1))
    y = proto.unsqueeze(0).expand(query.size(0),proto.size(0),query.size(1))

    return torch.pow(x-y,2).sum(2)

print("Started Streaming Data ...")
iteration = 0
while not dataHandler.endOfStream():
    #print(counter)
    d, l = dataHandler.getNextData()
    with torch.no_grad():
        d = torch.from_numpy(np.array(d)).float()
        d = d.cuda()
        #l = l.cuda()
        val_h = tuple([each.data for each in val_h])
        #print(np.shape(val_h[0]), np.shape(val_input))        

        yhat, val_h, embedding = model(d, val_h, 1)
        embeddings.append(embedding.data.cpu().numpy().flatten())
        #print(np.shape(embeddings))
        ll.append(l)

        #upd_prot = pca.transform(prot_mem.prototypes[l][-1].reshape((-1,1))).flatten()

        if counter % 20 == 0:
            prot_mem.update_prototypes(embeddings,np.array(ll))
            ## extract training embeddings
            tr_h = model.init_hidden(BATCH_SIZE) 

            model.eval()
            embeddings_list = dict()
            embeddings_list['embeddings'] = []
            embeddings_list['labels'] = []

            with torch.no_grad():

                for tr_input, y_tr in test_loader:
                    tr_h = tuple(each.data for each in tr_h)

                    tr_input = torch.from_numpy(np.array(tr_input)).float()
                    tr_input = tr_input.cuda()
                    y_tr = y_tr.cuda()

                    yhat, tr_h, embedding = model(tr_input, tr_h, BATCH_SIZE)
                    embeddings_list['embeddings'].extend(embedding.data.cpu().numpy())
                    embeddings_list['labels'].extend(y_tr.data.cpu().numpy())

            logits = compute_euclidean(torch.from_numpy(np.array(embeddings_list['embeddings'])).float(),torch.from_numpy(np.array(list(prot_mem.prototypes.values()))).float())
            pred = F.softmax(-logits, dim=1)   
            test_loss = F.binary_cross_entropy(pred, torch.from_numpy(np.array(embeddings_list['labels'])))

            test_oo = np.argmax(pred, axis = 1)
            true_test_oo = np.argmax(embeddings_list['labels'], axis=1).reshape((-1,1))

            accuracy = metrics.accuracy_score(true_test_oo, test_oo)
            precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_test_oo, test_oo, average='weighted')
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

            #print(prot_mem.prototypes)
            xdata.extend(list(pca.transform(list(prot_mem.prototypes.values()))[:,0]))
            ydata.extend(list(pca.transform(list(prot_mem.prototypes.values()))[:,1]))
            ytrue.extend(np.array(list(prot_mem.prototypes.keys())).astype(int))
            #print(k, xdata[-1],ydata[-1])

            plt_dynamic(np.array(xdata).T, np.array(ydata).T, ytrue, ax, colors)  
            iteration += 1     
    counter += 1
#plt.show()
plotCNNStatistics(statistics_path)
shutil.rmtree('./statistics')
plt.show()
