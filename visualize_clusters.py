## streaming data + updating prototypes

import time
import numpy as np
import tensorflow as tf
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
from imblearn.over_sampling import SMOTE
from subprocess import call
from models import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
from data_handler import *
import shutil
from prototype_memory import *
from replay_memory import *
from proto_net import *
from losses import *
from utils import *
import json
import argparse

seed = 1
torch.backends.cudnn.deterministic = True
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


parser = argparse.ArgumentParser(description="Offline ProtoNet")
parser.add_argument('--data', default='Opportunity')
parser.add_argument('--baseClasses', type=int, default = 5)
parser.add_argument('--newClasses', type=int, default = 0)
parser.add_argument('--percentage', type=float, default = 1.)
parser.add_argument('--batch_size', type=int, default = 200)
parser.add_argument('--window_length_PAMAP2', type=float, default = 1.)
parser.add_argument('--window_step_PAMAP2', type=float, default = 0.5)
parser.add_argument('--epochs', type=int, default = 100)
parser.add_argument('--support', type=int, default = 10)
parser.add_argument('--online_epochs', type=int, default=1)
parser.add_argument('--alpha', type=float, default=.5)
parser.add_argument('--replay_size', type=int, default=6)
parser.add_argument('--not_all_buffer_classes', action='store_true', default=False)
parser.add_argument('--random_stream', action='store_true', default=False)
parser.add_argument('--cuda_device', type=str, default='0')
parser.add_argument('--online_batch', type=int, default=20)
parser.add_argument('--contrastive_loss', action='store_true', default=False)
parser.add_argument('--margin', type=int, default=1)
parser.add_argument('--prototypical_contrastive_loss', action='store_true', default=False)
parser.add_argument('--T', type=float, default=1.)
parser.add_argument('--window_length_Skoda', type=int, default=98)
parser.add_argument('--window_step_Skoda', type=int, default=49)
parser.add_argument('--window_length_HAPT', type=float, default=2.56)
parser.add_argument('--contrastive_loss_with_prototypes', action='store_true', default=False)

params = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=params.cuda_device
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

def plotForgettingScore(statistics_path):

    statistics_dict = cPickle.load(open(statistics_path, 'rb'))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    lines = []
 
    bal_alpha = 0.3
    test_alpha = 1.0
    bal_map = np.array([statistics['Forgetting Score'] for statistics in statistics_dict['ForgettingScore']])    # (N, classes_num)

    line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
    lines.append(line)
     
    ax.grid(color='b', linestyle='solid', linewidth=0.3)
    plt.legend(labels=['Forgetting Score'], loc=2)
if params.data == 'Opportunity':
    ##################### Opportunity Dataset ##########################

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
    BATCH_SIZE = params.batch_size
    BATCH_SIZE_VAL = params.batch_size



    print("Loading data...")
    X_train, y_train, X_test, y_test = load_dataset('oppChallenge_gestures.data')
    print(np.shape(y_train))
    assert (NB_SENSOR_CHANNELS_WITH_FILTERING == X_train.shape[1] or NB_SENSOR_CHANNELS == X_train.shape[1])



    X_train, y_train_segments = rearrange(X_train, y_train.reshape((-1,1)), SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    X_test, y_test_segments = rearrange(X_test, y_test.reshape((-1,1)), SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
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
elif params.data == 'PAMAP2':
    ################ PAMAP2 Dataset #############################

    NB_SENSOR_CHANNELS = 52
    NUM_CLASSES = 12


    SAMPLING_FREQ = 100  # 100Hz

    #SLIDING_WINDOW_LENGTH = int(5.12 * SAMPLING_FREQ)
    SLIDING_WINDOW_LENGTH = int(params.window_length_PAMAP2*SAMPLING_FREQ)

    #SLIDING_WINDOW_STEP = int(1*SAMPLING_FREQ)
    SLIDING_WINDOW_STEP = int(params.window_step_PAMAP2*SAMPLING_FREQ)

    print("Extracting...")
    if not os.path.exists("./PAMAP2_Dataset/PAMAP2_Train_Test_{}_{}.data".format(SLIDING_WINDOW_LENGTH,SLIDING_WINDOW_STEP)):
        print('PAMAP2_Train_Test.data not found. Please run python3 PAMAP2_preprocessing.py to extract data')
        raise FileNotFoundError 
    else:
        print("Loading data...")
        X_train, y_train_segments, X_test, y_test_segments = load_dataset("./PAMAP2_Dataset/PAMAP2_Train_Test_{}_{}_normalized.data".format(SLIDING_WINDOW_LENGTH,SLIDING_WINDOW_STEP))

    print(" ..train data: inputs {0}, targets {1}".format(X_train.shape, y_train_segments.shape))
    print(" ..test data : inputs {0}, targets {1}".format(X_test.shape, y_test_segments.shape))

    print(Counter(y_train_segments))


    classes = np.unique(y_train_segments)


    NUM_CLASSES = len(classes)

    # Batch Size
    BATCH_SIZE = params.batch_size
    BATCH_SIZE_VAL = params.batch_size

elif params.data == 'DSADS':
    ################ DSADS Dataset #############################

    NB_SENSOR_CHANNELS =45
    NUM_CLASSES = 19

    SAMPLING_FREQ = 25  # 100Hz
    SLIDING_WINDOW_LENGTH = int(5*SAMPLING_FREQ)


    print("Extracting...")
    if not os.path.exists("./DSADS_Train_Test_normalized.data"):
        print('DSADS_Train_Test_normalized.data not found. Please run python3 DSADS_preprocessing.py to extract data')
        raise FileNotFoundError 
    else:
        print("Loading data...")
        X_train, y_train_segments, X_test, y_test_segments = load_dataset("./DSADS_Train_Test_normalized.data")

    print(" ..train data: inputs {0}, targets {1}".format(X_train.shape, y_train_segments.shape))
    print(" ..test data : inputs {0}, targets {1}".format(X_test.shape, y_test_segments.shape))

    print(Counter(y_train_segments))


    classes = np.unique(y_train_segments)


    NUM_CLASSES = len(classes)

    # Batch Size
    BATCH_SIZE = params.batch_size
    BATCH_SIZE_VAL = params.batch_size
elif params.data == 'Skoda':
    ################ DSADS Dataset #############################

    NB_SENSOR_CHANNELS =30
    NUM_CLASSES = 11

    #SAMPLING_FREQ = 98 # 100Hz
    SLIDING_WINDOW_LENGTH = int(params.window_length_Skoda)

    #SLIDING_WINDOW_STEP = int(1*SAMPLING_FREQ)
    SLIDING_WINDOW_STEP = int(params.window_step_Skoda)


    print("Extracting...")
    if not os.path.exists("./Skoda_data/Skoda_Train_Test_{}_{}.data".format(SLIDING_WINDOW_LENGTH,SLIDING_WINDOW_STEP)):
        print('Skdoa_Train_Test not found. Please run python3 Skdoa_processing.py to extract data')
        raise FileNotFoundError 
    else:
        print("Loading data...")
        X_train, y_train_segments, X_test, y_test_segments = load_dataset("./Skoda_data/Skoda_Train_Test_{}_{}.data".format(SLIDING_WINDOW_LENGTH,SLIDING_WINDOW_STEP))

    print(" ..train data: inputs {0}, targets {1}".format(X_train.shape, y_train_segments.shape))
    print(" ..test data : inputs {0}, targets {1}".format(X_test.shape, y_test_segments.shape))

    print(Counter(y_train_segments))


    classes = np.unique(y_train_segments)


    NUM_CLASSES = len(classes)

    # Batch Size
    BATCH_SIZE = params.batch_size
    BATCH_SIZE_VAL = params.batch_size
elif params.data == 'HAPT':
    ################ DSADS Dataset #############################

    NB_SENSOR_CHANNELS =6
    NUM_CLASSES = 12

    SAMPLING_FREQ = 50 # 100Hz
    SLIDING_WINDOW_LENGTH = int(params.window_length_HAPT*SAMPLING_FREQ)

    #SLIDING_WINDOW_STEP = int(1*SAMPLING_FREQ)
    SLIDING_WINDOW_STEP = int(SLIDING_WINDOW_LENGTH/2)


    print("Extracting...")
    if not os.path.exists("./HAPT_data/HAPT_Train_Test_{}_{}.data".format(SLIDING_WINDOW_LENGTH,SLIDING_WINDOW_STEP)):
        print('HAPT_Train_Test not found. Please run python3 HAPT_processing.py to extract data')
        raise FileNotFoundError 
    else:
        print("Loading data...")
        X_train, y_train_segments, X_test, y_test_segments = load_dataset("./HAPT_data/HAPT_Train_Test_{}_{}.data".format(SLIDING_WINDOW_LENGTH,SLIDING_WINDOW_STEP))

    print(" ..train data: inputs {0}, targets {1}".format(X_train.shape, y_train_segments.shape))
    print(" ..test data : inputs {0}, targets {1}".format(X_test.shape, y_test_segments.shape))

    print(Counter(y_train_segments))


    classes = np.unique(y_train_segments)


    NUM_CLASSES = len(classes)

    # Batch Size
    BATCH_SIZE = params.batch_size
    BATCH_SIZE_VAL = params.batch_size

"""

Get Base Data and Streaming Data 

"""

## streaming data
baseClassesNb = params.baseClasses
percentage = params.percentage #.05 # 20%
dataHandler = DataHandler(nb_baseClasses=baseClassesNb, seed=0, train={'data':X_train,'label':y_train_segments}, ClassPercentage=percentage)
dataHandler.streaming_data(nb_NewClasses=params.newClasses)
baseData = copy.deepcopy(dataHandler.getBaseData())
baseClasses = np.unique(baseData['label'])
NewClasses = dataHandler.NewClasses
newClassesNb = len(NewClasses)


mapping = {}
for i in np.arange(baseClassesNb + newClassesNb):
    if i >= baseClassesNb:
        mapping[NewClasses[i-baseClassesNb]] = i
    else:
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

## select new classes in test data 
X_test_newClasses = []
y_test_newClasses = []
for c in NewClasses:
    d,l = X_test[y_test_segments == c,:], y_test_segments[y_test_segments == c]
    X_test_newClasses.extend(d)
    y_test_newClasses.extend(l)

for x in range(len(y_test_newClasses)):
    y_test_newClasses[x] = mapping[y_test_newClasses[x]]


y_train = tf.keras.utils.to_categorical(baseData['label'], num_classes=baseClassesNb, dtype='int32')
y_test = tf.keras.utils.to_categorical(y_test_select, num_classes=baseClassesNb, dtype='int32')
y_test_newClasses_cat = tf.keras.utils.to_categorical(y_test_newClasses, num_classes=baseClassesNb + newClassesNb, dtype='int32')

#model = InceptionNN(NUM_CLASSES)
extractor = DeepConvLSTM(n_classes=len(np.unique(baseData['label'])), NB_SENSOR_CHANNELS = NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH = SLIDING_WINDOW_LENGTH)
model = ProtoNet(extractor,128,baseClassesNb+newClassesNb)

if torch.cuda.is_available():
    model.cuda()

torch.backends.cudnn.deterministic = True
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
# Statistics
statistics_path = './statistics/OnlineProtoNet_DeepConvLSTM_{}_baseClasses_{}_percentage_{}_online_epochs_{}_WeightedUpdate_{}_not_all_buffer_classes_{}_random_stream_{}.pkl'.format(params.data, 
    params.baseClasses, params.percentage, params.online_epochs, params.alpha, params.not_all_buffer_classes, params.random_stream)
forgetting_path = './forgetting_score/OnlineProtoNet_DeepConvLSTM_{}_baseClasses_{}_percentage_{}_online_epochs_{}_WeightedUpdate_{}_contrastive_loss_{}.pkl'.format(params.data, params.baseClasses, params.percentage, params.online_epochs, params.alpha, params.contrastive_loss)

if not os.path.exists(os.path.dirname(statistics_path)):
    os.makedirs(os.path.dirname(statistics_path))
statistics_container = StatisticsContainer(statistics_path)

if not os.path.exists(os.path.dirname(forgetting_path)):
    os.makedirs(os.path.dirname(forgetting_path))
forgetting_container = ForgettingContainer(forgetting_path)



## pretrain base model
x_train_tensor = torch.from_numpy(np.array(baseData['data'])).float()
y_train_tensor = torch.from_numpy(np.array(y_train)).float()
x_test_tensor = torch.from_numpy(np.array(X_test_select)).float()
x_test_newclasses_tensor = torch.from_numpy(np.array(X_test_newClasses)).float()
y_test_tensor = torch.from_numpy(np.array(y_test)).float()
y_test_newClasses_tensor = torch.from_numpy(np.array(y_test_newClasses_cat)).float()

train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)
test_newClasses_data = TensorDataset(x_test_newclasses_tensor, y_test_newClasses_tensor)

train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                    batch_size=BATCH_SIZE,
                    num_workers=1, pin_memory=True, shuffle = True,drop_last=False)


test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                        batch_size=BATCH_SIZE_VAL,
                        num_workers=1, pin_memory=True, shuffle = True,drop_last=False)

if newClassesNb > 1:
    test_newClasses_loader = torch.utils.data.DataLoader(dataset=test_newClasses_data, 
                            batch_size=BATCH_SIZE_VAL,
                            num_workers=1, pin_memory=True, shuffle = True,drop_last=False)

optimizer = optim.Adam(model.parameters(), lr=1e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

if params.contrastive_loss:
    ContrastiveLoss = OnlineContrastiveLoss(model, PairSelector(balance=False), margin=params.margin)
if params.prototypical_contrastive_loss:
    PrototypicalContrastiveLoss = OnlinePrototypicalContrastiveLoss(model, params.T, baseClassesNb+newClassesNb)
if params.contrastive_loss_with_prototypes:
    ContrastiveLossWithPrototypes = OnlineContrastiveLossWithPrototypes(model, PairSelector(balance=False), margin=params.margin)
n_epochs = params.epochs
n_support = params.support ## HARD CODED
iteration = 0
for epoch in range(n_epochs):
    model.train()   

    running_loss = 0.0
    n_steps = 0
    for d in train_loader:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = d
        x_support, y_support, x_query, y_query = extract_sample(len(np.unique(np.argmax(labels,axis=1))), n_support, n_support, inputs, np.argmax(labels, axis = 1), seed = iteration,shuffle=True)
        h = model.extractor.init_hidden(len(x_support)) 
        query_h = model.extractor.init_hidden(len(x_query))

        #print(y_support)
        y_support = tf.keras.utils.to_categorical(y_support, num_classes=baseClassesNb, dtype='int32')
        #print(y_support)
        #sys.exit()
        y_support = torch.from_numpy(y_support).float().cuda()
        x_support = x_support.cuda()
        x_query = x_query.cuda()
        #y_query = tf.keras.utils.to_categorical(y_query, num_classes=baseClassesNb, dtype='int32')
        y_query = torch.from_numpy(y_query).long().cuda()


        h = tuple([each.data for each in h])
        query_h = tuple([each.data for each in query_h])

        # zero the parameter gradients
        optimizer.zero_grad()

        log_p,h = model.forward_offline(x_support,y_support,x_query,h,query_h)
        key2idx = torch.empty(baseClassesNb,dtype=torch.long).cuda()
        proto_keys = list(model.memory.prototypes.keys())
        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()

        key2idx[proto_keys] = torch.arange(len(proto_keys)).cuda()

        y_query = key2idx[y_query].view(-1,1)
        y_query = tf.keras.utils.to_categorical(y_query.cpu().numpy(), num_classes=len(proto_keys), dtype='int32')
        y_query = torch.from_numpy(y_query).float().cuda() 
        #print(np.argmax(log_p.data.cpu().numpy(),axis=1),np.argmax(y_query.data.cpu().numpy(),axis=1))
        loss = F.binary_cross_entropy(log_p, y_query)
        if params.contrastive_loss:
            _,_,z_query = model.extractor(x_query,query_h, x_query.size(0))
            loss += ContrastiveLoss(z_query, y_query)
        if params.contrastive_loss_with_prototypes:
            _,_,z_query = model.extractor(x_query,query_h, x_query.size(0))
            loss += ContrastiveLossWithPrototypes(z_query, y_query,model)
        # if params.prototypical_contrastive_loss:
        #     _,_,z_query = model.extractor(x_query,query_h, x_query.size(0))
        #     loss += PrototypicalContrastiveLoss(z_query, y_query)
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
       

    model.eval()

    with torch.no_grad():
        print('TESTING !!')
        running_test_loss = 0.0
        n_steps = 0
        for d in test_loader:

            inputs, labels = d
            val_h = model.extractor.init_hidden(len(inputs))
            support_h = model.extractor.init_hidden(len(x_train_tensor))
####### Testing without selecting random support ########################################################################
            #inputs , labels = order_classes(inputs,np.argmax(labels, axis = 1),iteration)
            #labels = tf.keras.utils.to_categorical(labels,num_classes=baseClassesNb,dtype='int32')
            #labels = torch.from_numpy(labels).float()
            inputs = inputs.cuda()
            #labels = torch.from_numpy(tf.keras.utils.to_categorical(np.argmax(labels, axis = 1), num_classes=baseClassesNb, dtype='int32')).float()
            labels = labels.cuda()

            val_h = tuple([each.data for each in val_h])
            support_h = tuple([each.data for each in support_h])
            #print(np.shape(x_support), np.shape(x_query))
            # zero the parameter gradients

            #print(np.shape(labels))
            #log_p,val_h = model.forward_offline(inputs,labels,inputs,support_h,val_h)

            log_p,val_h = model.forward_inference(x_train_tensor.cuda(),y_train_tensor.cuda(),inputs,support_h,val_h)
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
        precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_test_oo, test_oo, labels=np.unique(true_test_oo), average='macro')
        try:
            auc_test = metrics.roc_auc_score(np.vstack(true_test_output), np.vstack(test_output), labels=np.unique(true_test_oo), average='macro')
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


#plotCNNStatistics(statistics_path)
model.eval()
#model.extractor.eval()
embeddings_list = dict()
embeddings_list['embeddings'] = []
embeddings_list['labels'] = []
with torch.no_grad():    
    iteration = 0
    for tr_input, y_tr in train_loader:
        tr_h = model.extractor.init_hidden(len(tr_input)) 
    
        # tr_input , y_tr = order_classes(tr_input,np.argmax(y_tr, axis = 1),iteration)
        # y_tr = tf.keras.utils.to_categorical(y_tr,num_classes=baseClassesNb,dtype='int32')
        # y_tr = torch.from_numpy(y_tr).float()
        tr_input = tr_input.cuda()
        #labels = torch.from_numpy(tf.keras.utils.to_categorical(np.argmax(labels, axis = 1), num_classes=baseClassesNb, dtype='int32')).float()
        y_tr = y_tr.cuda()
        tr_h = tuple(each.data for each in tr_h)

        _,tr_h, embeddings = model.extractor(tr_input, tr_h, len(y_tr))
        embeddings_list['embeddings'].extend(embeddings.data.cpu().numpy())
        embeddings_list['labels'].extend(y_tr.data.cpu().numpy())

labels = torch.from_numpy(np.array(embeddings_list['labels'])).float()

z_proto = torch.from_numpy(np.array(embeddings_list['embeddings'])).float().cuda()
labels = labels.cuda()
model.update_protoMemory(z_proto,labels)

### setup replay memory 
replay_buffer = ReplayMemory(params.replay_size)
yy = [np.argmax(l) for l in y_train]
replay_buffer.update((np.array(baseData['data']), np.array(yy)))


## save prototypes
json_dict = copy.deepcopy(model.memory.prototypes)
for key in model.memory.prototypes.keys():
    print(key, type(key))
    if type(key) is not str:
        json_dict[str(key)] = str(json_dict[key])
        del json_dict[key]

with open("./prototypes_json/Debugging_{}_Data_OfflineShuffle_ModelAdaptation.json".format(percentage), "w") as write_file:
    str_ = json.dumps(json_dict)
    write_file.write(str_)

##ge get train performance on base training data 

model.eval()
embeddings_newClasses_list = dict()
embeddings_newClasses_list['embeddings'] = []
embeddings_newClasses_list['labels'] = []
with torch.no_grad():
    print('Getting Performance on Base Data !!')
    running_train_loss = 0.0
    n_steps = 0
    for d in test_newClasses_loader:

        inputs, labels = d
        val_h = model.extractor.init_hidden(len(inputs))
####### Testing without selecting random support ########################################################################
        #inputs , labels = order_classes(inputs,np.argmax(labels, axis = 1),iteration)
        #labels = tf.keras.utils.to_categorical(labels,num_classes=baseClassesNb,dtype='int32')
        #labels = torch.from_numpy(labels).float()
        inputs = inputs.cuda()
        #labels = torch.from_numpy(tf.keras.utils.to_categorical(np.argmax(labels, axis = 1), num_classes=baseClassesNb, dtype='int32')).float()
        labels = labels.cuda()

        val_h = tuple([each.data for each in val_h])
        _,val_h, embeddings = model.extractor(inputs, val_h, len(labels))
        embeddings_newClasses_list['embeddings'].extend(embeddings.data.cpu().numpy())
        embeddings_newClasses_list['labels'].extend(labels.data.cpu().numpy())
##########################################################################################################################

cm = plt.get_cmap('gist_rainbow')
NUM_COLORS = len(classes)

colors = [cm((1.*i)/NUM_COLORS) for i in np.arange(NUM_COLORS)]
markers=['.',  'x', 'h','1']
## OPPORTUNITY
if params.data == 'Opportunity':
    LABELS = ['OpenDoor1', 'OpenDoor2','CloseDoor1','CloseDoor2','OpenFridge','CloseFridge','OpenDishwasher','CloseDishwasher','OpenDrawer1','CloseDrawer1','OpenDrawer2','CloseDrawer2','OpenDrawer3','CloseDrawer3','CleanTable','DrinkFromCup','ToogleSwitch']
elif params.data == 'PAMAP2':
    ## PAMAP2
    LABELS = {1:'lying',2:'sitting',3:'standing',4: 'walking',5: 'running',6: 'cycling',7: 'Nordic walking',9: 'watching TV',10: 'computer work',11: 'car driving', 12: 'ascending stairs',
    13:'descending stairs',16: 'vacuum cleaning',17: 'ironing',18: 'folding laundry',19: 'house cleaning',20:'playing soccer',24: 'rope jumping'}
elif params.data == 'DSADS':
    LABELS = {1:'sitting',2:'standing',3:'lying on back',4: 'lying on right side',5: 'ascending stairs',6: 'descending stairs',7: 'standing in elevator still',8: 'moving around in elevator',9: 'walking in parking lot',10: 'walking on treadmill w/ speed 4km/h in flat', 11:'walking on treadmill w/ speed 4km/h in 15 deg',12: 'running on treadmill',
    13:'exercising on stepper',14: 'exercising on cross trainer',15: 'cycling on exercise bike in horizontal',16: 'cycling on exercise bike in vertical',17: 'rowing',18: 'jumping',19:'playing basketbal'}    
elif params.data == 'Skoda':
    LABELS = {0: 'null class', 1: 'write on notepad', 2: 'open hood', 3: 'close hood',
              4: 'check gaps on the front door', 5: 'open left front door',
              6: 'close left front door', 7: 'close both left door', 8: 'check trunk gaps',
              9: 'open and close trunk', 10: 'check steering wheel'}    
elif params.data == 'HAPT':
    LABELS = {1:'walking',2:'walking upstairs',3:'walking downstairs',4:'sitting',5:'standing',6:'laying',7:'stand to sit',8:'sit to stand',
    9:'sit to lie',10:'lie to sit',11:'stand to lie',12:'lie to stand'}

 ### starting streaming
pca = PCA(n_components=2)
pca.fit(embeddings_list['embeddings'])
emb_pca = pca.transform(embeddings_list['embeddings'])
emb_labels = np.argmax(np.array(embeddings_list['labels']),axis=1)
#emb_pca, emb_labels = order_classes(emb_pca, emb_labels,0)

#ll = np.array(list(model.memory.prototypes.keys()), dtype=np.int32)[:3]
for k, col in zip(baseClasses[:3], colors):

    plt.plot(emb_pca[emb_labels==mapping[k],0], emb_pca[emb_labels==mapping[k],1], 'o',
            markerfacecolor=col, markeredgecolor=col,
             marker=markers[mapping[k]%len(markers)],markersize=7)

    #add label
    plt.annotate(LABELS[k], (np.mean(emb_pca[emb_labels==mapping[k],0],axis=0), np.mean(emb_pca[emb_labels==mapping[k],1], axis=0)),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=15, weight='bold',rotation=45,
                 color='k')

 ### starting streaming
pca.fit(embeddings_newClasses_list['embeddings'])
emb_pca = pca.transform(embeddings_newClasses_list['embeddings'])
emb_labels = np.argmax(np.array(embeddings_newClasses_list['labels']),axis=1)
#emb_pca, emb_labels = order_classes(emb_pca, emb_labels,0)

#ll = np.array(list(model.memory.prototypes.keys()), dtype=np.int32)
for k, col in zip([NewClasses[0]], colors[len(baseClasses[:3])+1:]):

    plt.plot(emb_pca[emb_labels==mapping[k],0], emb_pca[emb_labels==mapping[k],1], 'o',
            markerfacecolor=col, markeredgecolor=col,
             marker=markers[mapping[k]%len(markers)],markersize=7)

    #add label
    plt.annotate(LABELS[k], (np.mean(emb_pca[emb_labels==mapping[k],0],axis=0), np.mean(emb_pca[emb_labels==mapping[k],1], axis=0)),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=15, weight='bold',rotation=45,
                 color='k')




plt.show()