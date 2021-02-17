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
from inc_pca import IncPCA

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
from proto_net import *


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


    #ax.set_ylim(0, 1.)
    #ax.set_xlim(0, len(iterations))
    #ax.xaxis.set_ticks(np.arange(0, len(iterations), 25))
    #ax.xaxis.set_ticklabels(np.arange(0, max_plot_iteration, 50000))
    #ax.yaxis.set_ticks(np.arange(0, 1.01, 0.05))
    #ax.yaxis.set_ticklabels(np.around(np.arange(0, 1.01, 0.05), decimals=2))        
    ax.grid(color='b', linestyle='solid', linewidth=0.3)
    plt.legend(labels=['Training Loss','Testing Loss'], loc=2)

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    line, = ax.plot(test_f1, color='r', alpha=test_alpha)
    #ax.set_ylim(0,1.)
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


def extract_sample(n_classes, n_support, n_query, inputs, labels, seed, shuffle=False):

    support = []
    y_support = []
    query = []
    y_query = []
    np.random.seed(seed)
    #print(Counter(labels.data.cpu().numpy()))
    K = np.random.choice(np.unique(labels), n_classes, replace=False)
    #print(K)
    change = 0
    for cls in K:
        datax_cls = inputs[labels == cls]
        perm = np.random.permutation(datax_cls)
        #print(np.shape(perm))
        if len(perm) < n_support:
            change += n_support - len(perm)
        support_cls = perm[:n_support]
        #print(np.shape(support_cls))
        support.extend(support_cls)
        y_support.extend([cls]*n_support)
        query_cls = perm[n_support:]
        query.extend(query_cls)
        y_query.extend([cls]*n_query)
        #sample.append(sample_cls)

    if change > 0:
        support.extend(query[-change:])
        query = query[:-change]
    #print(np.shape(support), type(support))
    #print(np.shape(support), type(support))
    support = np.array(support)
    query = np.array(query)
    y_support = np.array(y_support)
    y_query = np.array(y_query)

    if shuffle:
        shuffler = np.random.permutation(len(support))
        support = support[shuffler]
        y_support = y_support[shuffler]

        shuffler = np.random.permutation(len(query))
        query = query[shuffler]
        y_query = y_query[shuffler]

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
percentage = .05  #.2 #.05 # 20%
dataHandler = DataHandler(nb_baseClasses=baseClassesNb, seed=0, train={'data':X_train,'label':y_train_segments}, ClassPercentage=percentage)
dataHandler.streaming_data()
baseData = dataHandler.getBaseData()
baseClasses = np.unique(baseData['label'])


mapping = {}
for i in np.arange(baseClassesNb):
    mapping[baseClasses[i]] = i
for x in range(len(baseData['label'])):
    baseData['label'][x] = mapping[baseData['label'][x]]
print(mapping)
## select base classes in test data 
X_test_select = []
y_test_select = []
for c in baseClasses:
    d,l = X_test[y_test_segments == c,:], y_test_segments[y_test_segments == c]
    X_test_select.extend(d)
    y_test_select.extend(l)

for x in range(len(y_test_select)):
    y_test_select[x] = mapping[y_test_select[x]]

y_train = tf.keras.utils.to_categorical(baseData['label'], num_classes=baseClassesNb, dtype='int32')
y_test = tf.keras.utils.to_categorical(y_test_select, num_classes=baseClassesNb, dtype='int32')

#model = InceptionNN(NUM_CLASSES)
extractor = DeepConvLSTM(n_classes=len(np.unique(baseData['label'])))
model = ProtoNet(extractor,128,baseClassesNb)

if torch.cuda.is_available():
    model.cuda()

# Statistics
statistics_path = './statistics/OnlineProtoNet_DeepConvLSTM_Opportunity.pkl'

if not os.path.exists(os.path.dirname(statistics_path)):
    os.makedirs(os.path.dirname(statistics_path))
statistics_container = StatisticsContainer(statistics_path)



## pretrain base model
x_train_tensor = torch.from_numpy(np.array(baseData['data'])).float()
y_train_tensor = torch.from_numpy(np.array(y_train)).float()
x_test_tensor = torch.from_numpy(np.array(X_test_select)).float()
y_test_tensor = torch.from_numpy(np.array(y_test)).float()
train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                    batch_size=BATCH_SIZE,
                    num_workers=1, pin_memory=True, shuffle = True,drop_last=True)


test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                        batch_size=BATCH_SIZE_VAL,
                        num_workers=1, pin_memory=True, shuffle = True,drop_last=True)

optimizer = optim.Adam(model.parameters(), lr=1e-4,betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
#scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5,last_epoch=-1)
## offline base training 
n_epochs = 50
n_support = int((1.*BATCH_SIZE/2.)/float(baseClassesNb)) ## 10-shot
iteration = 0
for epoch in range(n_epochs):
    h = model.extractor.init_hidden(n_support*baseClassesNb) 
    model.train()   

    running_loss = 0.0
    n_steps = 0
    for d in train_loader:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = d
        #print(labels)
        #print(labels)
        x_support, y_support, x_query, y_query = extract_sample(baseClassesNb, n_support, n_support, inputs, np.argmax(labels, axis = 1), seed = iteration,shuffle=False)
        x_support = x_support.cuda()
        #print(y_support)
        y_support = tf.keras.utils.to_categorical(y_support, num_classes=baseClassesNb, dtype='int32')
        #print(y_support)
        #sys.exit()
        y_support = torch.from_numpy(y_support).float().cuda()
        x_query = x_query.cuda()
        y_query = tf.keras.utils.to_categorical(y_query, num_classes=baseClassesNb, dtype='int32')
        y_query = torch.from_numpy(y_query).float().cuda()
        h = tuple([each.data for each in h])

        #print(np.shape(x_support), np.shape(x_query))
        # zero the parameter gradients
        optimizer.zero_grad()

        #print(np.shape(labels))
        log_p,h = model.forward_offline(x_support,y_support,x_query,h)
        #print(log_p, y_query)
        #clipwise_output = model(inputs,inputs.shape[0])
        #print("....",np.shape(clipwise_output))
        #clipwise_output = outputs['clipwise_output']
        loss = F.binary_cross_entropy(log_p, y_query)
        running_loss += loss
        loss.backward()
        optimizer.step()
        n_steps += 1

    print('[Epoch %d]' % (epoch + 1))
    epoch_train_loss = running_loss / n_steps
    print('Train loss: {}'.format(epoch_train_loss))


    eval_output = []
    true_output = []
    test_output = []
    true_test_output = []
    #h = model.extractor.init_hidden(n_support*baseClassesNb)
    #val_h = model.extractor.init_hidden(n_support*baseClassesNb)
    val_h = model.extractor.init_hidden(BATCH_SIZE)

    #print(np.shape(val_h[0]))        

    model.eval()

    with torch.no_grad():
        print('TESTING !!')
        running_test_loss = 0.0
        n_steps = 0
        for d in test_loader:

            inputs, labels = d
            #print(labels)
            # x_support, y_support, x_query, y_query = extract_sample(baseClassesNb, n_support, n_support, inputs, np.argmax(labels, axis = 1), seed = iteration,shuffle=False)
            # x_support = x_support.cuda()
            # y_support = tf.keras.utils.to_categorical(y_support, num_classes=baseClassesNb, dtype='int32')
            # y_support = torch.from_numpy(y_support).float().cuda()
            # x_query = x_query.cuda()
            # y_query = tf.keras.utils.to_categorical(y_query, num_classes=baseClassesNb, dtype='int32')
            # y_query = torch.from_numpy(y_query).float().cuda()
            # h = tuple([each.data for each in h])

            # #print(np.shape(x_support), np.shape(x_query))
            # # zero the parameter gradients

            # #print(np.shape(labels))
            # log_p,h = model.forward_offline(x_support,y_support,x_query,h)
            # #print(log_p, y_query)
            # #clipwise_output = model(inputs,inputs.shape[0])
            # #print("....",np.shape(clipwise_output))
            # #clipwise_output = outputs['clipwise_output']
            # test_loss = F.binary_cross_entropy(log_p, y_query)

            # test_output.append(log_p.data.cpu().numpy())
            # true_test_output.append(y_query.data.cpu().numpy())

####### Testing without selecting random support ########################################################################
            inputs , labels = order_classes(inputs,np.argmax(labels, axis = 1),iteration)
            labels = tf.keras.utils.to_categorical(labels,num_classes=baseClassesNb,dtype='int32')
            labels = torch.from_numpy(labels).float()
            inputs = inputs.cuda()
            #labels = torch.from_numpy(tf.keras.utils.to_categorical(np.argmax(labels, axis = 1), num_classes=baseClassesNb, dtype='int32')).float()
            labels = labels.cuda()

            val_h = tuple([each.data for each in val_h])

            #print(np.shape(x_support), np.shape(x_query))
            # zero the parameter gradients

            #print(np.shape(labels))
            log_p,val_h = model.forward_offline(inputs,labels,inputs,val_h)
            #print(log_p, y_query)
            #clipwise_output = model(inputs,inputs.shape[0])
            #print("....",np.shape(clipwise_output))
            #clipwise_output = outputs['clipwise_output']
            test_loss = F.binary_cross_entropy(log_p, labels)

            test_output.append(log_p.data.cpu().numpy())
            true_test_output.append(labels.data.cpu().numpy())
            running_test_loss += test_loss
            n_steps += 1
##########################################################################################################################

        

        test_oo = np.argmax(np.vstack(test_output), axis = 1)
        true_test_oo = np.argmax(np.vstack(true_test_output), axis = 1)

        accuracy = metrics.accuracy_score(true_test_oo, test_oo)
        precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_test_oo, test_oo, average='weighted')
        try:
            auc_test = metrics.roc_auc_score(np.vstack(true_test_output), np.vstack(test_output), average="weighted")
        except ValueError:
            auc_test = None
        epoch_test_loss = running_test_loss / n_steps
        print('Test loss: {}'.format(epoch_test_loss))
        print('TEST average_precision: {}'.format(precision))
        print('TEST average f1: {}'.format(fscore))
        print('TEST average recall: {}'.format(recall))
        print('TEST auc: {}'.format(accuracy))

        trainLoss = {'Trainloss': epoch_train_loss}
        #trainLoss = {'Trainloss': loss}

        statistics_container.append(iteration, trainLoss, data_type='Trainloss')
        testLoss = {'Testloss': epoch_test_loss}
        #testLoss = {'Testloss': test_loss}
        statistics_container.append(iteration, testLoss, data_type='Testloss')
        test_f1 = {'test_f1':fscore}
        statistics_container.append(iteration, test_f1, data_type='test_f1')

        statistics_container.dump()

    iteration += 1
    #scheduler.step()

C = confusion_matrix(true_test_oo, test_oo)

#labels = np.argmax(embeddings_list['labels'],axis=1)
labels = true_test_oo.copy()
for i in range(len(true_test_oo)):
    labels[i] = list(mapping.keys())[true_test_oo[i]]

#plt.figure(figsize=(10,10))
#plot_confusion_matrix(C, class_list=np.unique(labels), normalize=True, title='Predicted Results')

#plotCNNStatistics(statistics_path)

## get prototypes from train data and visualize

## extract training embeddings
tr_h = model.extractor.init_hidden(BATCH_SIZE) 

model.eval()
#model.extractor.eval()
embeddings_list = dict()
embeddings_list['embeddings'] = []
embeddings_list['labels'] = []
with torch.no_grad():    
    iteration = 0
    for tr_input, y_tr in train_loader:

        tr_input , y_tr = order_classes(tr_input,np.argmax(y_tr, axis = 1),iteration)
        y_tr = tf.keras.utils.to_categorical(y_tr,num_classes=baseClassesNb,dtype='int32')
        y_tr = torch.from_numpy(y_tr).float()
        tr_input = tr_input.cuda()
        #labels = torch.from_numpy(tf.keras.utils.to_categorical(np.argmax(labels, axis = 1), num_classes=baseClassesNb, dtype='int32')).float()
        y_tr = y_tr.cuda()
        tr_h = tuple(each.data for each in tr_h)

        _,tr_h, embeddings = model.extractor(tr_input, tr_h, BATCH_SIZE)
        embeddings_list['embeddings'].extend(embeddings.data.cpu().numpy())
        embeddings_list['labels'].extend(y_tr.data.cpu().numpy())

cm = plt.get_cmap('gist_rainbow')
NUM_COLORS = len(classes)

colors = [cm((1.*i)/NUM_COLORS) for i in np.arange(NUM_COLORS)]
markers=['.',  'x', 'h','1']

LABELS = ['OpenDoor1', 'OpenDoor2','CloseDoor1','CloseDoor2','OpenFridge','CloseFridge','OpenDishwasher','CloseDishwasher','OpenDrawer1','CloseDrawer1','OpenDrawer2','CloseDrawer2','OpenDrawer3','CloseDrawer3','CleanTable','DrinkFromCup','ToogleSwitch']

 
# plt.figure(figsize=(10,10))
# prot_pca_all = pca.transform(list(model.memory.prototypes.values()))
# emb_pca = pca.transform(embeddings_list['embeddings'])
# emb_labels = np.argmax(np.array(embeddings_list['labels']),axis=1)
# for i in range(len(emb_labels)):
#     emb_labels[i] = list(mapping.keys())[emb_labels[i]]
# #emb_pca, emb_labels = order_classes(emb_pca, emb_labels,0)
# ll = np.array(list(model.memory.prototypes.keys()), dtype=np.int32)
# for k, col in zip(np.unique(emb_labels), colors):
#     #print(k, np.shape(test_embd_pca[y_pred==k,0]))
#     plt.plot(prot_pca_all[ll==mapping[k],0], prot_pca_all[ll==mapping[k],1], 'o',
#             markerfacecolor=col, markeredgecolor=col,
#              marker=markers[mapping[k]%len(markers)],markersize=7)

#     # plt.plot(emb_pca[emb_labels==k,0], emb_pca[emb_labels==k,1], 'o',
#     #         markerfacecolor=col, markeredgecolor=col,
#     #          marker=markers[mapping[k]%len(markers)],markersize=7)

#     # plt.plot(prototypes_pca[mapping[k],0],prototypes_pca[mapping[k],1], 'o',
#     #         markerfacecolor=col, markeredgecolor='k', 
#     #         marker=markers[mapping[k]%len(markers)],markersize=7) 

#     #add label
#     plt.annotate(LABELS[k], (prot_pca_all[mapping[k],0], prot_pca_all[mapping[k],1]),
#                  horizontalalignment='center',
#                  verticalalignment='center',
#                  size=10, weight='bold',rotation=45,
#                  color='k')
# plt.title("Prototypes before updating using all training data")

labels = torch.from_numpy(np.array(embeddings_list['labels'])).float()
#labels = tf.keras.utils.to_categorical(labels, baseClassesNb, dtype='int32')
# for i in range(len(labels)):
#     labels[i] = list(mapping.keys())[labels[i]]

### update prototypes
#prot_mem = PrototypeMemory()

z_proto = torch.from_numpy(np.array(embeddings_list['embeddings'])).float().cuda()
labels = labels.cuda()
model.update_protoMemory(z_proto,labels)

#sys.exit()
#prototypes,prot_std = extract_prototypes(embeddings_list['embeddings'],labels)

## visualization after updating prototypes

### plot prototypes before updating
pca = IncPCA(n_components=2)
pca.partial_fit(list(model.memory.prototypes.values()))
#prototypes_pca = pca.transform(list(prot_mem.prototypes.values()))
prototypes_pca = pca.transform(list(model.memory.prototypes.values()))

fig, ax = plt.subplots(figsize=(10,10))
# ax.set_xlim(-6,6)
# ax.set_ylim(-6,6)
xdata, ydata = [], []
ln, = plt.plot([],[],'ro')

xdata.extend(prototypes_pca[:,0])
ydata.extend(prototypes_pca[:,1])
annotations= set()

def plt_dynamic(x,y,labels,ax,fig,colors,markers=['.',  'x', 'h','1']):
    #print(x,y,labels,x[labels==6],y[labels==6])
    for k, col in zip(np.unique(labels),colors):
        #print(k,x,y,labels)
        xx,yy = x[labels == k], y[labels == k]
        ax.plot(xx,yy, 'o',
        markerfacecolor=col, markeredgecolor=col, 
        marker=markers[k%len(markers)],markersize=20) 

        #add label
        if annotate and LABELS[list(mapping.keys())[k]] not in annotations:
            annotations.add(LABELS[list(mapping.keys())[k]])
            ax.annotate(LABELS[list(mapping.keys())[k]], (xx, yy),
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=10, weight='bold',rotation=45,
                     color='k') 

    fig.canvas.draw()

#ytrue = np.array(list(prot_mem.prototypes.keys()), dtype=np.int32)
ytrue = np.array(list(model.memory.prototypes.keys()), dtype=np.int32)
annotate = True
print(xdata, ydata, ytrue)
plt_dynamic(np.array(xdata), np.array(ydata), ytrue, ax,fig, colors)
plt.title("Prototypes after updating using all training data")
annotate = True
plt.show(block=False)


"""
model adaptation with streaming data 
update model after N time steps
at every time step, prototype updated using online averaging
Assume ground truth is given
"""

N = 20
#N = 100
#sys.exit()
#model.set_memory(prot_mem)

## evaluate on test data to check performance 
prototypes_check =copy.deepcopy(list(model.memory.prototypes.values()))
eval_output = []
true_output = []
test_output = []
true_test_output = []
#h = model.extractor.init_hidden(n_support*baseClassesNb)
h = model.extractor.init_hidden(BATCH_SIZE)
#print(np.shape(val_h[0]))        
embeddings_list = dict()
embeddings_list['embeddings'] = []
embeddings_list['labels'] = []
model.eval()
iteration = 0

with torch.no_grad():
    print('Checking Performance on Test DATA...')
    running_loss1 = 0.0
    n_steps = 0
    for d in train_loader:
        inputs, labels = d
        inputs, labels = order_classes(inputs, np.argmax(labels,axis = 1),iteration)
        labels = tf.keras.utils.to_categorical(labels,num_classes=baseClassesNb,dtype='int32')
        labels = torch.from_numpy(labels).float()      
        inputs = inputs.cuda()
        labels = labels.cuda()

        h = tuple([each.data for each in h])

        #print(np.shape(x_support), np.shape(x_query))
        # zero the parameter gradients

        #print(np.shape(labels))
        _,h,embeddings = model.extractor(inputs, h, BATCH_SIZE)
        embeddings_list['embeddings'].extend(embeddings.data.cpu().numpy())
        embeddings_list['labels'].extend(labels.data.cpu().numpy())
        log_p,h = model.forward_offline(inputs,labels,inputs,h)
        #assert prototypes_check == list(model.memory.prototypes.values())
        #print(log_p, y_query)
        #clipwise_output = model(inputs,inputs.shape[0])
        #print("....",np.shape(clipwise_output))
        #clipwise_output = outputs['clipwise_output']
        test_loss = F.binary_cross_entropy(log_p, labels)
        running_loss1 += test_loss
        n_steps += 1
        test_output.append(log_p.data.cpu().numpy())
        true_test_output.append(labels.data.cpu().numpy())
##########################################################################3

        # x_support, y_support, x_query, y_query = extract_sample(baseClassesNb, n_support, n_support, inputs, np.argmax(labels, axis = 1), seed = iteration,shuffle=False)
        # x_support = x_support.cuda()
        # y_support = tf.keras.utils.to_categorical(y_support, num_classes=baseClassesNb, dtype='int32')
        # y_support = torch.from_numpy(y_support).float().cuda()
        # x_query = x_query.cuda()
        # y_query = tf.keras.utils.to_categorical(y_query, num_classes=baseClassesNb, dtype='int32')
        # y_query = torch.from_numpy(y_query).float().cuda()
        # h = tuple([each.data for each in h])
        # _,_,embeddings = model.extractor(x_query, h, n_support*baseClassesNb)
        # embeddings_list['embeddings'].extend(embeddings.data.cpu().numpy())
        # embeddings_list['labels'].extend(y_query.data.cpu().numpy())

        # #print(np.shape(x_support), np.shape(x_query))
        # # zero the parameter gradients

        # #print(np.shape(labels))
        # log_p,h = model.forward_offline(x_support,y_support,x_query,h)
        # #print(log_p, y_query)
        # #clipwise_output = model(inputs,inputs.shape[0])
        # #print("....",np.shape(clipwise_output))
        # #clipwise_output = outputs['clipwise_output']
        # test_loss = F.binary_cross_entropy(log_p, y_query)

        # test_output.append(log_p.data.cpu().numpy())
        # true_test_output.append(y_query.data.cpu().numpy())

    test_oo = np.argmax(np.vstack(test_output), axis = 1)
    true_test_oo = np.argmax(np.vstack(true_test_output), axis = 1)

    accuracy = metrics.accuracy_score(true_test_oo, test_oo)
    precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_test_oo, test_oo, average='weighted')
    try:
        auc_test = metrics.roc_auc_score(np.vstack(true_test_output), np.vstack(test_output), average="weighted")
    except ValueError:
        auc_test = None
    epoch_loss = running_loss1 / n_steps
    print('Test loss: {}'.format(epoch_loss))
    print('TEST average_precision: {}'.format(precision))
    print('TEST average f1: {}'.format(fscore))
    print('TEST average recall: {}'.format(recall))
    print('TEST auc: {}'.format(accuracy))

C = confusion_matrix(true_test_oo, test_oo)

#labels = np.argmax(embeddings_list['labels'],axis=1)
labels = test_oo.copy()
for i in range(len(test_oo)):
    labels[i] = list(mapping.keys())[test_oo[i]]

#plt.figure(figsize=(10,10))
#plot_confusion_matrix(C, class_list=np.unique(labels), normalize=True, title='Predicted Results')

#test_embd_pca = pca.fit_transform(embeddings_list['embeddings'])
#prototypes_pca = pca.transform(list(model.memory.prototypes.values()))
# fig = plt.figure(figsize=(8, 3))
# markers = ['.',  'x', 'h','1']




# for k, col in zip(np.unique(labels), colors):
#     #print(k, np.shape(test_embd_pca[y_pred==k,0]))
#     plt.plot(test_embd_pca[labels==k,0], test_embd_pca[labels==k,1], 'o',
#             markerfacecolor=col, markeredgecolor=col,
#              marker=markers[mapping[k]%len(markers)],markersize=7)

#     # plt.plot(prototypes_pca[mapping[k],0],prototypes_pca[mapping[k],1], 'o',
#     #         markerfacecolor=col, markeredgecolor='k', 
#     #         marker=markers[mapping[k]%len(markers)],markersize=7) 

#     #add label
#     plt.annotate(LABELS[k], (prototypes_pca[mapping[k],0], prototypes_pca[mapping[k],1]),
#                  horizontalalignment='center',
#                  verticalalignment='center',
#                  size=10, weight='bold',rotation=45,
#                  color='k') 
# plt.title("Training Embeddings labeled using predicted labels after updating prototypes")


# labels = true_test_oo.copy()
# for i in range(len(true_test_oo)):
#     labels[i] = list(mapping.keys())[true_test_oo[i]]
# fig = plt.figure(figsize=(8, 3))

# for k, col in zip(np.unique(labels), colors):
#     #print(k, np.shape(test_embd_pca[y_pred==k,0]))
#     plt.plot(test_embd_pca[labels==k,0], test_embd_pca[labels==k,1], 'o',
#             markerfacecolor=col, markeredgecolor=col,
#              marker=markers[mapping[k]%len(markers)],markersize=7)

#     # plt.plot(prototypes_pca[mapping[k],0],prototypes_pca[mapping[k],1], 'o',
#     #         markerfacecolor=col, markeredgecolor='k', 
#     #         marker=markers[mapping[k]%len(markers)],markersize=7) 

#     #add label
#     plt.annotate(LABELS[k], (np.mean(test_embd_pca[labels==k,0],axis=0), np.mean(test_embd_pca[labels==k,1],axis=0)),
#                  horizontalalignment='center',
#                  verticalalignment='center',
#                  size=10, weight='bold',rotation=45,
#                  color='k') 
# #plt.show()
# plt.title("Training Embeddings labeled using ground truth")



#sys.exit()

val_h = model.extractor.init_hidden(N)
counter = 0
xdata, ydata, ytrue = [], [], []
ll=[]
#optimizer = optim.Adam(model.parameters(), lr=1e-6,betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
print("Started Streaming Data ...")
support_set = []
labels_set = []
map_labels = []
counter = 1
#running_loss = 0.0
n_steps = 0
# embeddings_pca = pca.transform(embeddings_list['embeddings'])
# fig1, ax1 = plt.subplots(figsize=(10,10))
# # ax.set_xlim(-6,6)
# # ax.set_ylim(-6,6)
# ln1, = plt.plot([],[],'ro')
# embx_data, emby_data, embyTrue = embeddings_pca[:,0], embeddings_pca[:,1], np.array(np.argmax(embeddings_list['labels'],axis=1))
# annotate = True
# plt_dynamic(np.array(embx_data), np.array(emby_data), embyTrue, ax1,fig1, colors)
# plt.title("BaseData Embeddings during Model Adaptation")
# annotate = True
# plt.show(block=False)
embx_data, emby_data, embyTrue = [], [], []
while not dataHandler.endOfStream():
    #print(counter)
    d, l = dataHandler.getNextData()
    #d = torch.from_numpy(np.array(d)).float()
    #d = d.cuda()
    #l = l.cuda()
    support_set.append(d)
    labels_set.append(l)
    map_labels.append(mapping[l])
    model.train()

    if counter % N == 0:
        support_set = torch.from_numpy(np.array(support_set)).float()
        map_labels = tf.keras.utils.to_categorical(map_labels, num_classes=baseClassesNb, dtype='int32')
        map_labels = torch.from_numpy(np.array(map_labels)).float()
        #labels_set = torch.from_numpy(np.array(labels_set)).float().cuda()
        support_set , map_labels = order_classes(support_set,np.argmax(map_labels, axis = 1),iteration)
        map_labels = tf.keras.utils.to_categorical(map_labels, num_classes=baseClassesNb, dtype='int32')
        map_labels = torch.from_numpy(np.array(map_labels)).float()

        support_set = support_set.cuda()
        map_labels = map_labels.cuda()

        val_h = tuple([each.data for each in val_h])
        #print(np.shape(val_h[0]), np.shape(val_input))        
        # zero the parameter gradients
        optimizer.zero_grad()

        log_p, val_h,embds = model.forward_online(support_set,map_labels, support_set, val_h)
        #embeddings_list['embeddings'].extend(embds.data.cpu().numpy())
        #embeddings_list['labels'].extend(map_labels.data.cpu().numpy())
        loss = F.binary_cross_entropy(log_p, map_labels)    
        #running_loss += loss
        loss.backward()
        optimizer.step()
        support_set = []
        labels_set = []
        map_labels = []


        eval_output = []
        true_output = []
        test_output = []
        true_test_output = []
        #h = model.extractor.init_hidden(n_support*baseClassesNb)
        h = model.extractor.init_hidden(BATCH_SIZE)

        #print(np.shape(val_h[0]))        

        model.eval()
        with torch.no_grad():
            iteration = 0
            running_test_loss = 0.0
            n_test_steps = 0
            for d in test_loader:
                inputs, labels = d
                #print(labels)
                inputs , labels = order_classes(inputs,np.argmax(labels, axis = 1),iteration)
                labels = tf.keras.utils.to_categorical(labels,num_classes=baseClassesNb,dtype='int32')
                labels = torch.from_numpy(labels).float()

                inputs = inputs.cuda()
                labels = labels.cuda()

                h = tuple([each.data for each in h])

                #print(np.shape(x_support), np.shape(x_query))
                # zero the parameter gradients

                #print(np.shape(labels))
                log_p, h = model.forward_online(inputs,labels,inputs,h)
                #print(log_p, y_query)
                #clipwise_output = model(inputs,inputs.shape[0])
                #print("....",np.shape(clipwise_output))
                #clipwise_output = outputs['clipwise_output']
                test_loss = F.binary_cross_entropy(log_p, labels)
                running_test_loss += test_loss

                test_output.append(log_p.data.cpu().numpy())
                true_test_output.append(labels.data.cpu().numpy())
                n_test_steps += 1


            ######################################################################################################################################3
                # x_support, y_support, x_query, y_query = extract_sample(baseClassesNb, n_support, n_support, inputs, np.argmax(labels, axis = 1), seed = iteration,shuffle=False)
                # x_support = x_support.cuda()
                # y_support = tf.keras.utils.to_categorical(y_support, num_classes=baseClassesNb, dtype='int32')
                # y_support = torch.from_numpy(y_support).float().cuda()
                # x_query = x_query.cuda()
                # y_query = tf.keras.utils.to_categorical(y_query, num_classes=baseClassesNb, dtype='int32')
                # y_query = torch.from_numpy(y_query).float().cuda()
                # h = tuple([each.data for each in h])

                # #print(np.shape(x_support), np.shape(x_query))
                # # zero the parameter gradients

                # #print(np.shape(labels))
                # log_p,h = model.forward_inference(x_support,y_support,x_query,h)
                # #print(log_p, y_query)
                # #clipwise_output = model(inputs,inputs.shape[0])
                # #print("....",np.shape(clipwise_output))
                # #clipwise_output = outputs['clipwise_output']
                # test_loss = F.binary_cross_entropy(log_p, y_query)
                # running_test_loss += test_loss

                # test_output.append(log_p.data.cpu().numpy())
                # true_test_output.append(y_query.data.cpu().numpy())
                # n_test_steps += 1  

            running_loss1 = 0.0
            n_steps = 0
            train_output = []
            true_train_output = []
            h = model.extractor.init_hidden(BATCH_SIZE)
            sub_embeddings = {'embeddings':[],'labels': []}
            for d in train_loader:
                inputs, labels = d
                inputs, labels = order_classes(inputs, np.argmax(labels,axis = 1),iteration)
                labels = tf.keras.utils.to_categorical(labels,num_classes=baseClassesNb,dtype='int32')
                labels = torch.from_numpy(labels).float()      
                inputs = inputs.cuda()
                labels = labels.cuda()

                h = tuple([each.data for each in h])

                #print(np.shape(x_support), np.shape(x_query))
                # zero the parameter gradients

                #print(np.shape(labels))
                _,h,embeddings = model.extractor(inputs, h, BATCH_SIZE)
                embeddings_list['embeddings'].extend(embeddings.data.cpu().numpy())
                embeddings_list['labels'].extend(labels.data.cpu().numpy())
                sub_embeddings['embeddings'].extend(embeddings.data.cpu().numpy())
                sub_embeddings['labels'].extend(labels.data.cpu().numpy())

                log_p,h = model.forward_online(inputs,labels,inputs,h)
                #assert prototypes_check == list(model.memory.prototypes.values())
                #print(log_p, y_query)
                #clipwise_output = model(inputs,inputs.shape[0])
                #print("....",np.shape(clipwise_output))
                #clipwise_output = outputs['clipwise_output']
                train_loss = F.binary_cross_entropy(log_p, labels)
                running_loss1 += train_loss
                n_steps += 1
                train_output.append(log_p.data.cpu().numpy())
                true_train_output.append(labels.data.cpu().numpy())

            train_oo = np.argmax(np.vstack(train_output), axis = 1)
            true_train_oo = np.argmax(np.vstack(true_train_output), axis = 1)

            accuracy = metrics.accuracy_score(true_train_oo, train_oo)
            precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_train_oo, train_oo, average='weighted')
            try:
                auc_test = metrics.roc_auc_score(np.vstack(true_train_output), np.vstack(train_output), average="weighted")
            except ValueError:
                auc_test = None
            epoch_loss = running_loss1 / n_steps
            print('Train loss: {}'.format(epoch_loss))
            print('TRAIN average_precision: {}'.format(precision))
            print('TRAIN average f1: {}'.format(fscore))
            print('TRAIN average recall: {}'.format(recall))
            print('TRAIN auc: {}'.format(accuracy))                

            test_oo = np.argmax(np.vstack(test_output), axis = 1)
            true_test_oo = np.argmax(np.vstack(true_test_output), axis = 1)

            accuracy = metrics.accuracy_score(true_test_oo, test_oo)
            precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_test_oo, test_oo, average='weighted')
            try:
                auc_test = metrics.roc_auc_score(np.vstack(true_test_output), np.vstack(test_output), average="weighted")
            except ValueError:
                auc_test = None
            print('Test loss: {}'.format(running_test_loss / n_test_steps))
            print('TEST average_precision: {}'.format(precision))
            print('TEST average f1: {}'.format(fscore))
            print('TEST average recall: {}'.format(recall))
            print('TEST auc: {}'.format(accuracy))

            trainLoss = {'Trainloss': running_loss1 / n_steps}
            #trainLoss = {'Trainloss': loss}
            statistics_container.append(iteration, trainLoss, data_type='Trainloss')
            testLoss = {'Testloss': running_test_loss / n_test_steps}
            #testLoss = {'Testloss': test_loss}

            statistics_container.append(iteration, testLoss, data_type='Testloss')
            test_f1 = {'test_f1':fscore}
            statistics_container.append(iteration, test_f1, data_type='test_f1')
            statistics_container.dump()
        #print(prot_mem.prototypes)
        pca.partial_fit(list(model.memory.prototypes.values()))
        next_pca = pca.transform(list(model.memory.prototypes.values()))
        pca_geom = IncPCA.geom_trans(prototypes_pca, next_pca)
        prototypes_pca = copy.deepcopy(pca_geom)
        xdata.extend(list(pca_geom[:,0]))
        ydata.extend(list(pca_geom[:,1]))
        ytrue.extend(list(model.memory.prototypes.keys()))
        # embx_data.extend(list(pca.transform(sub_embeddings['embeddings'])[:,0]))
        # emby_data.extend(list(pca.transform(sub_embeddings['embeddings'])[:,1]))
        # embyTrue.extend(np.argmax(sub_embeddings['labels'],axis=1))
        #print(k, xdata[-1],ydata[-1])

        plt_dynamic(np.array(xdata).T, np.array(ydata).T, ytrue, ax,fig, colors) 
        #plt_dynamic(np.array(embx_data).T, np.array(emby_data).T, embyTrue, ax1,fig1, colors)
        # if counter > 40:
        #     break       
    counter += 1
    n_steps += 1


plotCNNStatistics(statistics_path)
shutil.rmtree('./statistics')




#### visualize final prototypes after streaming data 

ipca = IncPCA(2)
ipca.partial_fit(prototypes_check)
before_prot = ipca.transform(prototypes_check)
after_prot = ipca.transform(list(model.memory.prototypes.values()))
after_prot = IncPCA.geom_trans(before_prot, after_prot)
plt.figure(figsize=(10,10))

ll = np.array(list(model.memory.prototypes.keys()), dtype=np.int32)
for k, col in zip(np.unique(ll), colors):
    #print(k, np.shape(test_embd_pca[y_pred==k,0]))
    plt.plot(before_prot[ll==k,0], before_prot[ll==k,1], 'o',
            markerfacecolor=col, markeredgecolor=col,
             marker=markers[k%len(markers)],markersize=7)

    plt.plot(after_prot[ll==k,0], after_prot[ll==k,1], 'o',
            markerfacecolor=col, markeredgecolor=col,
             marker=markers[k%len(markers)],markersize=7)

    #add label
    plt.annotate(LABELS[list(mapping.keys())[k]], (before_prot[k,0], before_prot[k,1]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=10, weight='bold',rotation=45,
                 color='k')
plt.title("Prototypes before and after model adaptation")


plt.show()


