import os
import sys
import time
import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
import random 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.parameter import Parameter
import datetime
import _pickle as cPickle

# seed= 0
# torch.backends.cudnn.deterministic = True
# random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)




# """
# ############# Opportunity Dataset #######################

# #--------------------------------------------
# # Dataset-specific constants and functions
# #--------------------------------------------

# # Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
# NB_SENSOR_CHANNELS = 113
# NB_SENSOR_CHANNELS_WITH_FILTERING = 149

# # Hardcoded number of classes in the gesture recognition problem
# NUM_CLASSES = 18

# # Hardcoded length of the sliding window mechanism employed to segment the data
# SLIDING_WINDOW_LENGTH =24


# # Hardcoded step of the sliding window mechanism employed to segment the data
# SLIDING_WINDOW_STEP = int(SLIDING_WINDOW_LENGTH/2)
# """

# #

# ################ PAMAP2 #################3

# NB_SENSOR_CHANNELS = 52
# SAMPLING_FREQ = 100  # 100Hz

# #SLIDING_WINDOW_LENGTH = int(5.12 * SAMPLING_FREQ)
# SLIDING_WINDOW_LENGTH = int(1.*SAMPLING_FREQ)

# #SLIDING_WINDOW_STEP = int(1*SAMPLING_FREQ)
# SLIDING_WINDOW_STEP = int(SLIDING_WINDOW_LENGTH/2)
# NUM_CLASSES = 12


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pickle'.format(
            os.path.splitext(self.statistics_path)[0], 
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_dict = {'Trainloss': [], 'Testloss': [], 'test_f1': [], 'BaseTrainloss': [], 'BaseTrain_f1': [], 'Testloss_NewClasses':[], 'newClasses_test_f1':[],'Testloss_AllClasses':[], 'allClasses_test_f1':[]}

    def append(self, iteration, statistics, data_type):
        statistics['iteration'] = iteration
        self.statistics_dict[data_type].append(statistics)
        
    def dump(self):
        cPickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        cPickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        
    def load_state_dict(self, resume_iteration):
        self.statistics_dict = cPickle.load(open(self.statistics_path, 'rb'))

        resume_statistics_dict = {'Trainloss': [], 'Testloss': [], 'test_f1': [], 'BaseTrainloss': [], 'BaseTrain_f1': [], 'Testloss_NewClasses':[], 'newClasses_test_f1':[], 'Testloss_AllClasses':[], 'allClasses_test_f1':[]}
        
        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics['iteration'] <= resume_iteration:
                    resume_statistics_dict[key].append(statistics)
                
        self.statistics_dict = resume_statistics_dict


class ForgettingContainer(object):
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pickle'.format(
            os.path.splitext(self.statistics_path)[0], 
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_dict = {'ForgettingScore': []}

    def append(self, iteration, statistics, data_type):
        statistics['iteration'] = iteration
        self.statistics_dict[data_type].append(statistics)
        
    def dump(self):
        cPickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        cPickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        
    def load_state_dict(self, resume_iteration):
        self.statistics_dict = cPickle.load(open(self.statistics_path, 'rb'))

        resume_statistics_dict = {'ForgettingScore': []}
        
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


# class DeepConvLSTM_old(nn.Module):
#     def __init__(self, n_hidden=128, n_layers=1, n_filters=64, 
#                  n_classes=NUM_CLASSES, filter_size=5, drop_prob=0.5):
#         super(DeepConvLSTM_old, self).__init__()

#         self.drop_prob = drop_prob
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden
#         self.n_filters = n_filters
#         self.n_classes = n_classes
#         self.filter_size = filter_size

#         self.name = 'DeepConvLSTM'
#         self.conv1 = nn.Conv2d(in_channels=1, 
#                               out_channels=self.n_filters,
#                               kernel_size=(self.filter_size, 1), stride=(1,1),
#                               padding=(0,0))
#         #self.bn1 = nn.BatchNorm2d(64)
                              
#         self.conv2 = nn.Conv2d(in_channels=self.n_filters, 
#                               out_channels=self.n_filters,
#                               kernel_size=(self.filter_size,1), stride=(1, 1),
#                               padding=(0, 0))
                              
#         #self.bn2 = nn.BatchNorm2d(64)

#         self.conv3 = nn.Conv2d(in_channels=self.n_filters, 
#                               out_channels=self.n_filters,
#                               kernel_size=(self.filter_size, 1), stride=(1,1),
#                               padding=(0,0))
#         #self.bn3 = nn.BatchNorm2d(64)
                              
#         self.conv4 = nn.Conv2d(in_channels=self.n_filters, 
#                               out_channels=self.n_filters,
#                               kernel_size=(self.filter_size,1), stride=(1, 1),
#                               padding=(0, 0))
                              
#         #self.bn4 = nn.BatchNorm2d(64)
#         self.dropout = nn.Dropout(self.drop_prob)
#         self.lstm1 = nn.LSTM(self.n_filters*NB_SENSOR_CHANNELS, hidden_size=self.n_hidden, num_layers=self.n_layers)
#         self.lstm2 = nn.LSTM(self.n_hidden, self.n_hidden, num_layers=self.n_layers)
#         self.softmax = nn.Linear(self.n_hidden, self.n_classes)

#         self.init_weight()

#     def init_weight(self):
#         # init_bn(self.bn0)
#         init_layer(self.conv1)
#         init_layer(self.conv2)
#         init_layer(self.conv3)
#         init_layer(self.conv4)
#         init_layer(self.lstm1)
#         init_layer(self.lstm2)
#         #init_layer(self.lstmAcc)
#         #init_layer(self.lstmGyr)

#         # init_bn(self.bn1)
#         # init_bn(self.bn2)
#         # init_bn(self.bn3)
#         # init_bn(self.bn4)

#         #init_layer(self.dense)
#         init_layer(self.softmax)

#     def forward(self, input, hidden, batch_size):

#         x1 = F.relu_(self.conv1(input))
#         #x1 = self.bn1(x1)
#         x1 = F.relu_(self.conv2(x1))
#         #x1 = self.bn2(x1)
#         x1 = F.relu_(self.conv3(x1))
#         #x1 = self.bn3(x1)
#         x1 = F.relu_(self.conv4(x1))
#         #x1 = self.bn4(x1)
#         #print(x1.shape)
#         #x1 =  x1.reshape((x1.shape[0], x1.shape[2],-1)) 
#         x1 = x1.permute(2,0,1,3)
#         x1 = x1.contiguous()
#         x1 = x1.view(x1.shape[0], x1.shape[1],-1)
#         x1 = self.dropout(x1)
#         #print(x1.shape)
#         #self.lstm1.flatten_parameters()
#         x1, hidden = self.lstm1(x1, hidden)
#         #self.lstm2.flatten_parameters()
#         x1, (h1,c1) = self.lstm2(x1, hidden)
#         x1 = x1.contiguous().view(-1, self.n_hidden)
#         #x1 = h1.reshape((h1.shape[1],-1))
#         #print(h1.shape)
#         output = torch.sigmoid(self.softmax(x1))
#         output = output.view(batch_size, -1, self.n_classes)[:,-1,:]

#         return output, hidden, x1

#     def init_hidden(self, batch_size):
#         ''' Initializes hidden state '''
#         # Create two new tensors with sizes n_layers x batch_size x n_hidden,
#         # initialized to zero, for hidden state and cell state of LSTM
#         weight = next(self.parameters()).data
        
#         if (torch.cuda.is_available()):
#             hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
#                   weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
#         else:
#             hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
#                       weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
#        return hidden


class DeepConvLSTM(nn.Module):
    def __init__(self, n_classes, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH, n_hidden=128, n_layers=1, n_filters=64, 
                filter_size=5, drop_prob=0.5, ):
        super(DeepConvLSTM, self).__init__()

        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size
        self.NB_SENSOR_CHANNELS = NB_SENSOR_CHANNELS
        self.SLIDING_WINDOW_LENGTH = SLIDING_WINDOW_LENGTH
             
        self.conv1 = nn.Conv1d(self.NB_SENSOR_CHANNELS, n_filters, filter_size)
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
        #print(x.shape)
        x = x.view(-1, self.NB_SENSOR_CHANNELS, self.SLIDING_WINDOW_LENGTH)
        #print(x.shape)
        #x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        #print(x.shape)
        x = x.view(x.shape[-1], -1, self.n_filters)
        #print(x.shape)

        #print(np.shape(x), np.shape(hidden))
        x = self.dropout(x)
        x, hidden = self.lstm1(x, hidden)
        #print(x.shape)

        x, hidden = self.lstm2(x, hidden)
        #print(x.shape)

        #print(np.shape(x))

        x = x.contiguous().view(-1, self.n_hidden)
        embeddings = x.contiguous().view(batch_size,-1,self.n_hidden)[:,-1,:]
        x = torch.sigmoid(self.fc(x))
        #print(np.shape(x))
        temp = x.view(batch_size, -1, self.n_classes)
        #print(np.shape(temp))
        out = x.view(batch_size, -1, self.n_classes)[:,-1,:]
        
        return out, hidden, embeddings
    
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

### cosine normalization in last layer to level the difference of the embeddings and biases between all classes (from LUCIR)  ----- idea can apply to prototypes of all formed classes? 
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), \
                F.normalize(self.weight, p=2, dim=1)) # experiment with adding bias?
        if self.sigma is not None:
            out = self.sigma * out
        return out

# ## testing a modification for PAMAP2
# """
# 1- putting dropout before LSTM not after 
# 2- input size to LSTM is filter_size * NB_SENSOR_CHANNELS 
# """
# class DeepConvLSTM_PAMAP2(nn.Module):
#     def __init__(self, n_hidden=128, n_layers=1, n_filters=64, 
#                  n_classes=NUM_CLASSES, filter_size=5, drop_prob=0.5):
#         super(DeepConvLSTM_PAMAP2, self).__init__()

#         self.drop_prob = drop_prob
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden
#         self.n_filters = n_filters
#         self.n_classes = n_classes
#         self.filter_size = filter_size
             
#         self.conv1 = nn.Conv1d(NB_SENSOR_CHANNELS, n_filters, filter_size)
#         self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size)
#         self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size)
#         self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size)
        
#         self.lstm1  = nn.LSTM(n_filters*NB_SENSOR_CHANNELS, n_hidden, n_layers)
#         self.lstm2  = nn.LSTM(n_hidden, n_hidden, n_layers)
        
#         self.fc = nn.Linear(n_hidden, n_classes)

#         self.dropout = nn.Dropout(drop_prob)

#         self.init_weight()

#     def init_weight(self):
#         # init_bn(self.bn0)
#         init_layer(self.conv1)
#         init_layer(self.conv2)
#         init_layer(self.conv3)
#         init_layer(self.conv4)
#         init_layer(self.lstm1)
#         init_layer(self.lstm2)
#         #init_layer(self.lstmAcc)
#         #init_layer(self.lstmGyr)

#         # init_bn(self.bn1)
#         # init_bn(self.bn2)
#         # init_bn(self.bn3)
#         # init_bn(self.bn4)

#         #init_layer(self.dense)
#         init_layer(self.fc)        

#     def forward(self, x, hidden, batch_size):
        
#         x = x.view(-1, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         #print(np.shape(x))
#         #x = x.view(x.shape[-1], -1, self.n_filters)
#         print(x.shape)
#         x = x.permute(2,0,1,3)
#         x = x.contiguous()
#         x = x.view(x.shape[0], x.shape[1],-1)
#         x = self.dropout(x)
#         #print(np.shape(x), np.shape(hidden))
#         x, hidden = self.lstm1(x, hidden)
#         x, hidden = self.lstm2(x, hidden)
#         #print(np.shape(x))

#         x = torch.sigmoid(self.fc(x))
#         #print(np.shape(x))
#         temp = x.view(batch_size, -1, self.n_classes)
#         #print(np.shape(temp))
#         out = x.view(batch_size, -1, self.n_classes)[:,-1,:]
        
#         return out, hidden, embeddings
    
#     def init_hidden(self, batch_size):
#         ''' Initializes hidden state '''
#         # Create two new tensors with sizes n_layers x batch_size x n_hidden,
#         # initialized to zero, for hidden state and cell state of LSTM
#         weight = next(self.parameters()).data
        
#         if (torch.cuda.is_available()):
#             hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
#                   weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
#         else:
#             hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
#                       weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
#         return hidden