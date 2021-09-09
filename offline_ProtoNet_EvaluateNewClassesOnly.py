### training ProtoNet offline on all data and get "True Prototypes" for comparision with streaming version 
import os
import time
import numpy as np
import tensorflow as tf
import sys
import pickle


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
sys.path.append('/media/hd4t2/Rebecca/Research-ContinualLearning/streaming-vis-pca-master/')
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
from data_handler import *
import shutil
from prototype_memory import *
from proto_net import *
from utils import *
from losses import *
import json
import argparse


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
parser.add_argument('--window_length_USC_HAD', type=float, default = 1.)
parser.add_argument('--window_step_USC_HAD', type=float, default = 0.5)
parser.add_argument('--cuda_device', type=str, default='1')
parser.add_argument('--window_length_Skoda', type=int, default=98)
parser.add_argument('--window_step_Skoda', type=int, default=49)
parser.add_argument('--window_length_WISDM', type=float, default=5.)
parser.add_argument('--window_step_WISDM', type=float, default=2.5)
parser.add_argument('--WISDM_device', type=str, default='phone')
parser.add_argument('--window_length_HAPT', type=float, default=2.56)
parser.add_argument('--DSoftmax', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1)
params = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=params.cuda_device

seed = params.seed
torch.backends.cudnn.deterministic = True
random.seed(seed)
if params.data =='Skoda':
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
else:
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)   
np.random.seed(seed)

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
    NUM_CLASSES = 17

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
    if not os.path.exists("./PAMAP2_Dataset/PAMAP2_Train_Test_{}_{}_normalized.data".format(SLIDING_WINDOW_LENGTH,SLIDING_WINDOW_STEP)):
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


elif params.data == 'USC-HAD':
    ################ DSADS Dataset #############################

    NB_SENSOR_CHANNELS =6
    NUM_CLASSES = 12

    SAMPLING_FREQ = 100 # 100Hz
    SLIDING_WINDOW_LENGTH = int(params.window_length_USC_HAD*SAMPLING_FREQ)

    #SLIDING_WINDOW_STEP = int(1*SAMPLING_FREQ)
    SLIDING_WINDOW_STEP = int(params.window_step_USC_HAD*SAMPLING_FREQ)


    print("Extracting...")
    if not os.path.exists("./USC-HAD/USC_HAD_Train_Test_{}_{}.data".format(SLIDING_WINDOW_LENGTH,SLIDING_WINDOW_STEP)):
        print('USC_HAD_Train_Test not found. Please run python3 USC_HAD_processing.py to extract data')
        raise FileNotFoundError 
    else:
        print("Loading data...")
        X_train, y_train_segments, X_test, y_test_segments = load_dataset("./USC-HAD/USC_HAD_Train_Test_{}_{}.data".format(SLIDING_WINDOW_LENGTH,SLIDING_WINDOW_STEP))

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
elif params.data == 'WISDM':
    ################ DSADS Dataset #############################

    NB_SENSOR_CHANNELS =8
    NUM_CLASSES = 18

    SAMPLING_FREQ = 20 # 20Hz
    SLIDING_WINDOW_LENGTH = int(params.window_length_WISDM*SAMPLING_FREQ)

    #SLIDING_WINDOW_STEP = int(1*SAMPLING_FREQ)
    SLIDING_WINDOW_STEP = int(params.window_step_WISDM*SAMPLING_FREQ)


    print("Extracting...")
    if not os.path.exists("./wisdm-dataset/WISDM_{}_Train_Test_{}_{}.data".format(params.WISDM_device,SLIDING_WINDOW_LENGTH,SLIDING_WINDOW_STEP)):
        print('WISDM_Train_Test not found. Please run python3 WISDM_processing.py to extract data')
        raise FileNotFoundError 
    else:
        print("Loading data...")
        X_train, y_train_segments, X_test, y_test_segments = load_dataset("./wisdm-dataset/WISDM_{}_Train_Test_{}_{}_avg.data".format(params.WISDM_device, SLIDING_WINDOW_LENGTH,SLIDING_WINDOW_STEP))  

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
# if params.data == 'Opportunity':
#     baseClassesNb = NUM_CLASSES - 1
# elif params.data == 'PAMAP2':
#     baseClassesNb = NUM_CLASSES
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

for x in range(len(y_test_segments)):
    y_test_segments[x] = mapping[y_test_segments[x]]

#import pdb; pdb.set_trace()
y_test = tf.keras.utils.to_categorical(y_test_select, num_classes=baseClassesNb + newClassesNb, dtype='int32')
y_test_newClasses_cat = tf.keras.utils.to_categorical(y_test_newClasses, num_classes=baseClassesNb + newClassesNb, dtype='int32')
y_test_all = tf.keras.utils.to_categorical(y_test_segments, num_classes=baseClassesNb+newClassesNb, dtype='int32')

baseClassesNb = NUM_CLASSES
percentage = 1. #.05 # 20%
dataHandler = DataHandler(nb_baseClasses=baseClassesNb, seed=0, train={'data':X_train,'label':y_train_segments}, ClassPercentage=percentage)
dataHandler.streaming_data(nb_NewClasses=0)
baseData = copy.deepcopy(dataHandler.getBaseData())
baseClasses = np.unique(baseData['label'])
for x in range(len(baseData['label'])):
    baseData['label'][x] = mapping[baseData['label'][x]]

y_train = tf.keras.utils.to_categorical(baseData['label'], num_classes=baseClassesNb, dtype='int32')


#model = InceptionNN(NUM_CLASSES)
extractor = DeepConvLSTM(n_classes=len(np.unique(baseData['label'])), NB_SENSOR_CHANNELS = NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH = SLIDING_WINDOW_LENGTH)
model = ProtoNet(extractor,128,baseClassesNb)

if params.DSoftmax:
    D_softmax = DSoftmaxLoss(torch.Tensor([0.5]).float(), baseClassesNb)

if torch.cuda.is_available():
    model.cuda()

torch.backends.cudnn.deterministic = True
random.seed(seed)
if params.data =='Skoda':
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
else:
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)   
np.random.seed(seed)

# Statistics
statistics_path = './statistics/OfflineProtoNet_DeepConvLSTM_' + params.data + '.pkl'

if not os.path.exists(os.path.dirname(statistics_path)):
    os.makedirs(os.path.dirname(statistics_path))
statistics_container = StatisticsContainer(statistics_path)



## pretrain base model
x_train_tensor = torch.from_numpy(np.array(baseData['data'])).float()
y_train_tensor = torch.from_numpy(np.array(y_train)).float()
x_test_tensor = torch.from_numpy(np.array(X_test_select)).float()
x_test_all_tensor = torch.from_numpy(np.array(X_test)).float()
x_test_newclasses_tensor = torch.from_numpy(np.array(X_test_newClasses)).float()
y_test_tensor = torch.from_numpy(np.array(y_test)).float()
y_test_newClasses_tensor = torch.from_numpy(np.array(y_test_newClasses_cat)).float()
y_test_all_tensor = torch.from_numpy(np.array(y_test_all)).float()

train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)
test_newClasses_data = TensorDataset(x_test_newclasses_tensor, y_test_newClasses_tensor)
test_data_all = TensorDataset(x_test_all_tensor, y_test_all_tensor)

if params.data == 'WISDM':
    train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                    batch_size=BATCH_SIZE,
                    num_workers=1, pin_memory=True, shuffle = True,drop_last=True)
else:
    train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                    batch_size=BATCH_SIZE,
                    num_workers=1, pin_memory=True, shuffle = True,drop_last=False)
test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                        batch_size=BATCH_SIZE_VAL,
                        num_workers=1, pin_memory=True, shuffle = True,drop_last=False)
test_all_loader = torch.utils.data.DataLoader(dataset=test_data_all, batch_size=BATCH_SIZE_VAL, num_workers=1, pin_memory=True, shuffle=True, drop_last=False)
if newClassesNb > 1:
    test_newClasses_loader = torch.utils.data.DataLoader(dataset=test_newClasses_data, 
                            batch_size=BATCH_SIZE_VAL,
                            num_workers=1, pin_memory=True, shuffle = True,drop_last=False)

optimizer = optim.Adam(model.parameters(), lr=1e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)


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
        #print("Nb of classes in support set {}, nb of classes in query set {}".format(len(np.unique(y_support)), len(np.unique(y_query))))
        #print(np.shape(x_support),np.shape(x_query))
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
        #print(np.shape(x_support),np.shape(x_query))


        h = tuple([each.data for each in h])
        query_h = tuple([each.data for each in query_h])

        # zero the parameter gradients
        optimizer.zero_grad()

        log_p,h = model.forward_offline(x_support,y_support,x_query,h,query_h, params.DSoftmax)
        key2idx = torch.empty(baseClassesNb,dtype=torch.long).cuda()
        proto_keys = list(model.memory.prototypes.keys())
        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()

        key2idx[proto_keys] = torch.arange(len(proto_keys)).cuda()

        y_query = key2idx[y_query].view(-1,1)
        y_query = tf.keras.utils.to_categorical(y_query.cpu().numpy(), num_classes=len(proto_keys), dtype='int32')
        y_query = torch.from_numpy(y_query).float().cuda()   
            #print(np.argmax(log_p.data.cpu().numpy(),axis=1),np.argmax(y_query.data.cpu().numpy(),axis=1))
        
        if not params.DSoftmax:
            loss = F.binary_cross_entropy(log_p, y_query)
        else:
            loss = D_softmax(log_p, y_query, model)
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
    model.extractor.eval()
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

        accuracy = metrics.accuracy_score(true_test_oo, test_oo,)
        precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_test_oo, test_oo, labels=np.unique(true_test_oo), average='macro')

        try:
            auc_test = metrics.roc_auc_score(np.vstack(true_test_output), np.vstack(test_output),labels=np.unique(true_test_oo), average='macro' )
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

C = confusion_matrix(true_test_oo, test_oo)

labels = copy.deepcopy(true_test_oo)

for i in range(len(true_test_oo)):
    labels[i] = list(mapping.keys())[true_test_oo[i]]

plt.figure(figsize=(10,10))
plot_confusion_matrix(C, class_list=np.unique(labels), normalize=True, title='Predicted Results')

plotCNNStatistics(statistics_path)

## get prototypes from train data and visualize

## extract training embeddings

model.eval()
model.extractor.eval()
embeddings_list = dict()
embeddings_list['embeddings'] = []
embeddings_list['labels'] = []
with torch.no_grad():    
    iteration = 0
    for tr_input, y_tr in train_loader:

        tr_input , y_tr = order_classes(tr_input,np.argmax(y_tr, axis = 1),iteration)
        tr_h = model.extractor.init_hidden(len(tr_input)) 

        y_tr = tf.keras.utils.to_categorical(y_tr,num_classes=baseClassesNb,dtype='int32')
        y_tr = torch.from_numpy(y_tr).float()
        tr_input = tr_input.cuda()
        #labels = torch.from_numpy(tf.keras.utils.to_categorical(np.argmax(labels, axis = 1), num_classes=baseClassesNb, dtype='int32')).float()
        y_tr = y_tr.cuda()
        tr_h = tuple(each.data for each in tr_h)

        _,tr_h, embeddings = model.extractor(tr_input, tr_h, len(tr_input))
        embeddings_list['embeddings'].extend(embeddings.data.cpu().numpy())
        embeddings_list['labels'].extend(y_tr.data.cpu().numpy())

### plot prototypes before updating
pca = PCA(n_components=2)
before_prot = list(model.memory.prototypes.values())
pca.fit(before_prot)
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
elif params.data == 'USC-HAD':
    LABELS = {1:'walking forward',2:'walking left',3:'walking right',4: 'walking upstairs',5: 'walking downstairs',6: 'running forward',7: 'jumping up',8: 'sitting',9: 'standing',10: 'sleeping', 11:'elevator up',12: 'elevator down'}    
elif params.data == 'Skoda':
    LABELS = {0: 'null class', 1: 'write on notepad', 2: 'open hood', 3: 'close hood',
              4: 'check gaps on the front door', 5: 'open left front door',
              6: 'close left front door', 7: 'close both left door', 8: 'check trunk gaps',
              9: 'open and close trunk', 10: 'check steering wheel'}  
elif params.data =='WISDM':
    LABELS = {0:'walking',1:'jogging',2:'stairs',3:'sitting',4:'standing',5:'typing',6:'teeth',7:'soup',8:'chips',9:'pasta',10:'drinking',11:'sandwich',
    12:'kicking',14:'catch',15:'dribbling', 16:'writing',17:'clapping',18:'folding'}
elif params.data == 'HAPT':
    LABELS = {1:'walking',2:'walking upstairs',3:'walking downstairs',4:'sitting',5:'standing',6:'laying',7:'stand to sit',8:'sit to stand',
    9:'sit to lie',10:'lie to sit',11:'stand to lie',12:'lie to stand'}
  
# plt.figure(figsize=(10,10))
# prot_pca_all = pca.transform(before_prot)

# ll = np.array(list(model.memory.prototypes.keys()), dtype=np.int32)
# for k, col in zip(ll, colors):
#     #print(k, np.shape(test_embd_pca[y_pred==k,0]))
#     plt.plot(prot_pca_all[ll==k,0], prot_pca_all[k,1], 'o',
#             markerfacecolor=col, markeredgecolor=col,
#              marker=markers[k%len(markers)],markersize=7)

#     # plt.plot(emb_pca[emb_labels==k,0], emb_pca[emb_labels==k,1], 'o',
#     #         markerfacecolor=col, markeredgecolor=col,
#     #          marker=markers[mapping[k]%len(markers)],markersize=7)

#     # plt.plot(prototypes_pca[mapping[k],0],prototypes_pca[mapping[k],1], 'o',
#     #         markerfacecolor=col, markeredgecolor='k', 
#     #         marker=markers[mapping[k]%len(markers)],markersize=7) 

#     #add label
#     plt.annotate(LABELS[list(mapping.keys())[k]], (prot_pca_all[k,0], prot_pca_all[k,1]),
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
plt.figure(figsize=(10,10))


#prototypes_pca = pca.transform(list(prot_mem.prototypes.values()))
prototypes_pca = pca.fit_transform(list(model.memory.prototypes.values()))
emb_pca = pca.transform(embeddings_list['embeddings'])
emb_labels = np.argmax(np.array(embeddings_list['labels']),axis=1)
#emb_pca, emb_labels = order_classes(emb_pca, emb_labels,0)

ll = np.array(list(model.memory.prototypes.keys()), dtype=np.int32)
for k, col in zip(ll, colors):
    #print(k, np.shape(test_embd_pca[y_pred==k,0]))
    plt.plot(prototypes_pca[ll==k,0], prototypes_pca[k,1], 'o',
            markerfacecolor=col, markeredgecolor=col,
             marker=markers[k%len(markers)],markersize=7)

    plt.plot(emb_pca[emb_labels==k,0], emb_pca[emb_labels==k,1], 'o',
            markerfacecolor=col, markeredgecolor=col,
             marker=markers[k%len(markers)],markersize=7)

    #add label
    plt.annotate(LABELS[list(mapping.keys())[k]], (prototypes_pca[k,0], prototypes_pca[k,1]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=10, weight='bold',rotation=45,
                 color='k')
plt.title("Prototypes after updating using all training data + training embeddings")


## incremental PCA 

### plot prototypes before updating
ipca = IncPCA(n_components=2)
ipca.partial_fit(before_prot)

cm = plt.get_cmap('gist_rainbow')
NUM_COLORS = len(classes)

colors = [cm((1.*i)/NUM_COLORS) for i in np.arange(NUM_COLORS)]
markers=['.',  'x', 'h','1']

 
plt.figure(figsize=(10,10))
prot_pca_all = ipca.transform(before_prot)

ll = np.array(list(model.memory.prototypes.keys()), dtype=np.int32)
for k, col in zip(ll, colors):
    #print(k, np.shape(test_embd_pca[y_pred==k,0]))
    plt.plot(prot_pca_all[ll==k,0], prot_pca_all[k,1], 'o',
            markerfacecolor=col, markeredgecolor=col,
             marker=markers[k%len(markers)],markersize=7)

    # plt.plot(emb_pca[emb_labels==k,0], emb_pca[emb_labels==k,1], 'o',
    #         markerfacecolor=col, markeredgecolor=col,
    #          marker=markers[mapping[k]%len(markers)],markersize=7)

    # plt.plot(prototypes_pca[mapping[k],0],prototypes_pca[mapping[k],1], 'o',
    #         markerfacecolor=col, markeredgecolor='k', 
    #         marker=markers[mapping[k]%len(markers)],markersize=7) 

    #add label
    plt.annotate(LABELS[list(mapping.keys())[k]], (prot_pca_all[k,0], prot_pca_all[k,1]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=10, weight='bold',rotation=45,
                 color='k')
plt.title("Prototypes before updating using all training data")

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
plt.figure(figsize=(10,10))


#prototypes_pca = pca.transform(list(prot_mem.prototypes.values()))
ipca.partial_fit(list(model.memory.prototypes.values()))
prototypes_pca_next = ipca.transform(list(model.memory.prototypes.values()))
prototypes_pca = IncPCA.geom_trans(prot_pca_all,prototypes_pca_next)
ll = np.array(list(model.memory.prototypes.keys()), dtype=np.int32)
for k, col in zip(ll, colors):
    #print(k, np.shape(test_embd_pca[y_pred==k,0]))
    plt.plot(prototypes_pca[ll==k,0], prototypes_pca[k,1], 'o',
            markerfacecolor=col, markeredgecolor=col,
             marker=markers[k%len(markers)],markersize=7)

    # plt.plot(emb_pca[emb_labels==k,0], emb_pca[emb_labels==k,1], 'o',
    #         markerfacecolor=col, markeredgecolor=col,
    #          marker=markers[mapping[k]%len(markers)],markersize=7)

    # plt.plot(prototypes_pca[mapping[k],0],prototypes_pca[mapping[k],1], 'o',
    #         markerfacecolor=col, markeredgecolor='k', 
    #         marker=markers[mapping[k]%len(markers)],markersize=7) 

    #add label
    plt.annotate(LABELS[list(mapping.keys())[k]], (prototypes_pca[k,0], prototypes_pca[k,1]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=10, weight='bold',rotation=45,
                 color='k')
plt.title("Prototypes after updating using all training data")



#### Evaluate on test data after updating prototypes using all training data 
eval_output = []
true_output = []
test_output = []
true_test_output = []
#h = model.extractor.init_hidden(n_support*baseClassesNb)
#val_h = model.extractor.init_hidden(n_support*baseClassesNb)

#print(np.shape(val_h[0]))        

model.eval()
model.extractor.eval()
with torch.no_grad():
    print('TESTING !!')
    running_test_loss = 0.0
    n_steps = 0
    for d in test_loader:

        inputs, labels = d
        val_h = model.extractor.init_hidden(len(inputs))

        # inputs , labels = order_classes(inputs,np.argmax(labels, axis = 1),iteration)
        # labels = tf.keras.utils.to_categorical(labels,num_classes=baseClassesNb,dtype='int32')
        # labels = torch.from_numpy(labels).float()
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
    precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_test_oo, test_oo, labels=np.unique(true_test_oo), average='macro')
    try:
        auc_test = metrics.roc_auc_score(np.vstack(true_test_output), np.vstack(test_output), labels=np.unique(true_test_oo), average='macro')
    except ValueError:
        auc_test = None
    epoch_test_loss = running_test_loss / n_steps
    print('----------------------------------------------------')
    print('FINAL Test loss: {}'.format(epoch_test_loss))
    print('FINAL TEST average_precision: {}'.format(precision))
    print('FINAL TEST average f1: {}'.format(fscore))
    print('FINAL TEST average recall: {}'.format(recall))
    print('FINAL TEST auc: {}'.format(accuracy))

C = confusion_matrix(true_test_oo, test_oo, labels=np.unique(true_test_oo))

# #labels = np.argmax(embeddings_list['labels'],axis=1)
# labels = true_test_oo.copy()
# for i in range(len(true_test_oo)):
#     labels[i] = list(mapping.keys())[true_test_oo[i]]

plt.figure(figsize=(10,10))
plot_confusion_matrix(C, class_list=np.unique(true_test_oo), normalize=True, title='FINAL Predicted Results')

eval_output_newClass = []
true_output_newClass = []
test_output_newClass = []
true_test_output_newClass = []
model.eval()
model.extractor.eval()
with torch.no_grad():
    print('TESTING !!')
    running_test_loss = 0.0
    n_steps = 0
    for d in test_newClasses_loader:

        inputs, labels = d
        val_h = model.extractor.init_hidden(len(inputs))

        # inputs , labels = order_classes(inputs,np.argmax(labels, axis = 1),iteration)
        # labels = tf.keras.utils.to_categorical(labels,num_classes=baseClassesNb,dtype='int32')
        # labels = torch.from_numpy(labels).float()
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

        test_output_newClass.append(log_p.data.cpu().numpy())
        true_test_output_newClass.append(labels.data.cpu().numpy())
        running_test_loss += test_loss
        n_steps += 1
##########################################################################################################################

    test_oo = np.argmax(np.vstack(test_output_newClass), axis = 1)
    true_test_oo = np.argmax(np.vstack(true_test_output_newClass), axis = 1)

    accuracy = metrics.accuracy_score(true_test_oo, test_oo)
    precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_test_oo, test_oo, labels=np.unique(true_test_oo), average='macro')
    try:
        auc_test = metrics.roc_auc_score(np.vstack(true_test_output_newClass), np.vstack(test_output_newClass), labels=np.unique(true_test_oo), average='macro')
    except ValueError:
        auc_test = None
    epoch_test_loss = running_test_loss / n_steps
    print('----------------------------------------------------')
    print('FINAL New Class Test loss: {}'.format(epoch_test_loss))
    print('FINAL New Class TEST average_precision: {}'.format(precision))
    print('FINAL New Class TEST average f1: {}'.format(fscore))
    print('FINAL New Class TEST average recall: {}'.format(recall))
    print('FINAL New Class TEST auc: {}'.format(accuracy))

C = confusion_matrix(true_test_oo, test_oo,labels = np.unique(true_test_oo))

# #labels = np.argmax(embeddings_list['labels'],axis=1)
# labels = true_test_oo.copy()
# for i in range(len(true_test_oo)):
#     labels[i] = list(mapping.keys())[true_test_oo[i]]

plt.figure(figsize=(10,10))
plot_confusion_matrix(C, class_list=np.unique(true_test_oo), normalize=True, title='FINAL Predicted Results')

#### Evaluate on test data after updating prototypes using all training data 
eval_output = []
true_output = []
test_output = []
true_test_output = []
#h = model.extractor.init_hidden(n_support*baseClassesNb)
#val_h = model.extractor.init_hidden(n_support*baseClassesNb)

#print(np.shape(val_h[0]))        

model.eval()
model.extractor.eval()
with torch.no_grad():
    print('TESTING !!')
    running_test_loss = 0.0
    n_steps = 0
    for d in test_all_loader:

        inputs, labels = d
        val_h = model.extractor.init_hidden(len(inputs))

        # inputs , labels = order_classes(inputs,np.argmax(labels, axis = 1),iteration)
        # labels = tf.keras.utils.to_categorical(labels,num_classes=baseClassesNb,dtype='int32')
        # labels = torch.from_numpy(labels).float()
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
    precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_test_oo, test_oo, labels=np.unique(true_test_oo), average='macro')
    try:
        auc_test = metrics.roc_auc_score(np.vstack(true_test_output), np.vstack(test_output), labels=np.unique(true_test_oo), average='macro')
    except ValueError:
        auc_test = None
    epoch_test_loss = running_test_loss / n_steps
    print('----------------------------------------------------')
    print('FINAL ALL Test loss: {}'.format(epoch_test_loss))
    print('FINAL ALL TEST average_precision: {}'.format(precision))
    print('FINAL ALL TEST average f1: {}'.format(fscore))
    print('FINAL ALL TEST average recall: {}'.format(recall))
    print('FINAL ALL TEST auc: {}'.format(accuracy))

C = confusion_matrix(true_test_oo, test_oo, labels=np.unique(true_test_oo))

# #labels = np.argmax(embeddings_list['labels'],axis=1)
# labels = true_test_oo.copy()
# for i in range(len(true_test_oo)):
#     labels[i] = list(mapping.keys())[true_test_oo[i]]

plt.figure(figsize=(10,10))
plot_confusion_matrix(C, class_list=np.unique(true_test_oo), normalize=True, title='FINAL Predicted Results')




plt.show()


