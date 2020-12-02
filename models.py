import os
import sys
import time
import math
import librosa
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.parameter import Parameter
import datetime
import _pickle as cPickle


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


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pickle'.format(
            os.path.splitext(self.statistics_path)[0], 
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_dict = {'Trainloss': [], 'Testloss': [], 'test_f1': []}

    def append(self, iteration, statistics, data_type):
        statistics['iteration'] = iteration
        self.statistics_dict[data_type].append(statistics)
        
    def dump(self):
        cPickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        cPickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        
    def load_state_dict(self, resume_iteration):
        self.statistics_dict = cPickle.load(open(self.statistics_path, 'rb'))

        resume_statistics_dict = {'Trainloss': [], 'Testloss': [], 'test_f1': []}
        
        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics['iteration'] <= resume_iteration:
                    resume_statistics_dict[key].append(statistics)
                
        self.statistics_dict = resume_statistics_dict

def init_layer(layer):

    if type(layer) == nn.LSTM:
        for name, param in layer.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    else:
        """Initialize a Linear or Convolutional layer. """
        nn.init.xavier_uniform_(layer.weight)
 
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class DeepConvLSTM_old(nn.Module):
    def __init__(self, classes_num, features):
        super(DeepConvLSTM_old, self).__init__()

        self.name = 'DeepConvLSTM'
        self.conv1 = nn.Conv2d(in_channels=1, 
                              out_channels=64,
                              kernel_size=(5, 1), stride=(1,1),
                              padding=(0,0))
        #self.bn1 = nn.BatchNorm2d(64)
                              
        self.conv2 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5,1), stride=(1, 1),
                              padding=(0, 0))
                              
        #self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5, 1), stride=(1,1),
                              padding=(0,0))
        #self.bn3 = nn.BatchNorm2d(64)
                              
        self.conv4 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5,1), stride=(1, 1),
                              padding=(0, 0))
                              
        #self.bn4 = nn.BatchNorm2d(64)

        self.lstm1 = nn.LSTM(64*features, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(128, 128, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(.5)
        self.softmax = nn.Linear(62848, classes_num)

        self.init_weight()

    def init_weight(self):
        # init_bn(self.bn0)
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.lstm1)
        init_layer(self.lstm2)
        #init_layer(self.lstmAcc)
        #init_layer(self.lstmGyr)

        # init_bn(self.bn1)
        # init_bn(self.bn2)
        # init_bn(self.bn3)
        # init_bn(self.bn4)

        #init_layer(self.dense)
        init_layer(self.softmax)

    def forward(self, input):

        x1 = F.relu_(self.conv1(input))
        #x1 = self.bn1(x1)
        x1 = F.relu_(self.conv2(x1))
        #x1 = self.bn2(x1)
        x1 = F.relu_(self.conv3(x1))
        #x1 = self.bn3(x1)
        x1 = F.relu_(self.conv4(x1))
        #x1 = self.bn4(x1)
        #print(x1.shape)
        x1 =  x1.reshape((x1.shape[0], x1.shape[2],-1)) 
        #print(x1.shape)
        self.lstm1.flatten_parameters()
        x1, _ = self.lstm1(x1)
        self.lstm2.flatten_parameters()
        x1, (h1,c1) = self.lstm2(x1)
        #x1 = h1.reshape((h1.shape[1],-1))
        #print(h1.shape)
        x1 = torch.flatten(x1, start_dim=1)
        x1 = self.dropout(x1)
        output = torch.sigmoid(self.softmax(x1))

        return {'clipwise_output': output, 'embedding': x1}

class DeepConvLSTM(nn.Module):
    def __init__(self, n_hidden=128, n_layers=1, n_filters=64, 
                 n_classes=NUM_CLASSES, filter_size=5, drop_prob=0.5):
        super(DeepConvLSTM, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size
             
        self.conv1 = nn.Conv1d(NB_SENSOR_CHANNELS, n_filters, filter_size)
        self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size)
        
        self.lstm1  = nn.LSTM(n_filters, n_hidden, n_layers)
        self.lstm2  = nn.LSTM(n_hidden, n_hidden, n_layers)
        
        self.fc = nn.Linear(n_hidden, n_classes)

        self.dropout = nn.Dropout(drop_prob)

        self.init_weight()

    def init_weight(self):
        # init_bn(self.bn0)
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.lstm1)
        init_layer(self.lstm2)
        #init_layer(self.lstmAcc)
        #init_layer(self.lstmGyr)

        # init_bn(self.bn1)
        # init_bn(self.bn2)
        # init_bn(self.bn3)
        # init_bn(self.bn4)

        #init_layer(self.dense)
        init_layer(self.fc)        

    def forward(self, x, hidden, batch_size):
        
        x = x.view(-1, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = x.view(8, -1, self.n_filters)
        x, hidden = self.lstm1(x, hidden)
        x, hidden = self.lstm2(x, hidden)
        
        x = x.contiguous().view(-1, self.n_hidden)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x))
        
        out = x.view(batch_size, -1, self.n_classes)[:,-1,:]
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (torch.cuda.is_available()):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden