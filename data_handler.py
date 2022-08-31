import math
import pickle
import random
from random import shuffle
from operator import itemgetter
import numpy as np
import pandas as pd
import yaml
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from sklearn.utils import shuffle
import torch
# seed =0
# random.seed(seed)

# np.random.seed(seed)


import copy
# from train.visualisations import vis_by_person
# from train.visualisations.training_visualizer import plot_heatmap
# from sklearn.preprocessing import LabelEncoder


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.
    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                #import pdb; pdb.set_trace()
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
class DataHandler:

    def __init__(self, nb_baseClasses, seed, train, ClassPercentage):
        self.nb_baseClasses = nb_baseClasses
        self.seed = seed 
        #self.seed_randomness()
        self.train = train
        self.ClassPercentage = ClassPercentage
        self.counter = 0
        self.base_done = False
        
    def seed_randomness(self):
        np.random.seed(self.seed)
        random.seed()


    def streaming_data(self,nb_NewClasses = 0):
        _labels = sorted(set(self.train['label']))
        print(_labels, self.nb_baseClasses)
        # if ordered_labels:
        #     self.baseClasses = list(range(self.nb_baseClasses))
        # #self.baseClasses = np.random.choice(_labels, self.nb_baseClasses, replace=False).tolist()
        # else:
        print(self.nb_baseClasses, len(np.unique(_labels)))
        #np.random.seed(1)
        self.baseClasses = np.random.choice(_labels, self.nb_baseClasses, replace=False).tolist()
        # self.baseClasses = [2,3,4]
        #np.random.seed(self.seed)
        self.nb_NewClasses = nb_NewClasses
        _labels = sorted(set(self.train['label']))
        _labelsNew = np.setdiff1d(_labels,self.baseClasses)

        self.NewClasses = np.random.choice(_labelsNew, nb_NewClasses, replace=False).tolist()
        # self.NewClasses = [12] # [14, 18] for DSADS#[14,17]

        self.baseData = dict()
        self.baseData['data'] = []
        self.baseData['label'] = []

        self.remData = dict()
        self.remData['data'] = []
        self.remData['label'] = []
        self.remSizePerBaseClass = dict()
        self.remSizePerNewClass = dict()
        for c in _labels:
            d,l = copy.deepcopy(self.train['data'][self.train['label'] == c,:]), copy.deepcopy(self.train['label'][self.train['label'] == c])
            if c in self.baseClasses: 
                self.baseIdx = np.random.choice(np.arange(len(d)),int(self.ClassPercentage*len(d)),replace=False)
                self.baseData['data'].extend(d[self.baseIdx,:])
                self.baseData['label'].extend(l[self.baseIdx])
                self.remIdx = np.setdiff1d(np.arange(len(d)),self.baseIdx)
                self.remData['data'].extend(d[self.remIdx,:])
                self.remData['label'].extend(l[self.remIdx])
                self.remSizePerBaseClass[c] = len(self.remIdx)

            elif c in self.NewClasses:
                #d,l = self.train['data'][self.train['label'] == c,:], self.train['label'][self.train['label'] == c]
                self.remData['data'].extend(d[:])
                self.remData['label'].extend(l[:]) 
                self.remSizePerNewClass[c] = len(l[:])
        self.maxSize = len(self.remData['data'])
        self.shuffleRemData()  


    def streaming_data_IgnoreNewClassInStreaming(self,nb_NewClasses = 0):
        _labels = sorted(set(self.train['label']))
        print(_labels, self.nb_baseClasses)
        # if ordered_labels:
        #     self.baseClasses = list(range(self.nb_baseClasses))
        # #self.baseClasses = np.random.choice(_labels, self.nb_baseClasses, replace=False).tolist()
        # else:
        print(self.nb_baseClasses, len(np.unique(_labels)))
        #np.random.seed(1)
        self.baseClasses = np.random.choice(_labels, self.nb_baseClasses, replace=False).tolist()
        # self.baseClasses = [2,4,6,9,1,3]
        #np.random.seed(self.seed)
        self.nb_NewClasses = nb_NewClasses
        _labels = sorted(set(self.train['label']))
        _labelsNew = np.setdiff1d(_labels,self.baseClasses)

        # self.NewClasses = np.random.choice(_labelsNew, nb_NewClasses, replace=False).tolist()
        self.NewClasses = [1,7] #[14, 18] for DSADS

        self.baseData = dict()
        self.baseData['data'] = []
        self.baseData['label'] = []

        self.remData = dict()
        self.remData['data'] = []
        self.remData['label'] = []
        self.remSizePerBaseClass = dict()
        self.remSizePerNewClass = dict()
        for c in _labels:
            d,l = copy.deepcopy(self.train['data'][self.train['label'] == c,:]), copy.deepcopy(self.train['label'][self.train['label'] == c])
            if c in self.baseClasses: 
                self.baseIdx = np.random.choice(np.arange(len(d)),int(self.ClassPercentage*len(d)),replace=False)
                self.baseData['data'].extend(d[self.baseIdx,:])
                self.baseData['label'].extend(l[self.baseIdx])
                self.remIdx = np.setdiff1d(np.arange(len(d)),self.baseIdx)
                self.remData['data'].extend(d[self.remIdx,:])
                self.remData['label'].extend(l[self.remIdx])
                self.remSizePerBaseClass[c] = len(self.remIdx)

        self.maxSize = len(self.remData['data'])
        self.shuffleRemData()  

    def NIC_generation(self, N):
        if self.counter == 0: 
            num_batches = math.floor(self.maxSize/N)
            self.insertion_point = np.random.choice(int(3*num_batches/4),self.nb_NewClasses,replace=False)
        #get_new = np.random.choice([0,1])
        data, labels = [],[]
        print(self.counter/N, self.insertion_point)
        print('Old Classes {}, New Classes {}'.format(self.remSizePerBaseClass, self.remSizePerNewClass))
        if int(self.counter/N) in self.insertion_point and self.remSizePerNewClass: #get_new or self.base_done:
            # if self.base_done:
            # #     nb_NewClasses = np.random.choice(5)
            # # classes = []
            # # if self.remSizePerNewClass: 
            # #     new_classes = np.random.choice(self.remSizePerNewClass.keys(), min(len(self.remSizePerNewClass),nb_NewClasses), replace=False)
            # #     if self.base_done:
            # #         classes = new_classes
            # #     else:
            # #         if self.remSizePerBaseClass.keys():
            # #             old_classes = np.random.choice(self.remSizePerBaseClass.keys(), self.nb_baseClasses-nb_NewClasses, replace=False)
            # #             classes = np.append(old_classes, new_classes)

            # # #samples_per_class = N/len(classes)
            #     if self.remSizePerNewClass: 
            #         classes = np.random.choice(list(self.remSizePerNewClass.keys()), min(len(self.remSizePerNewClass),5), replace=False)
            #     indxs = []
            #     for c in classes:
            #         ii = np.where(self.remData['label'] == c)[0]
            #         indxs.extend(ii)
            #     #print(np.shape(indxs))
            #     if len(indxs) > 0:
            #         selected_indxs = np.random.choice(indxs, min(N, len(indxs)), replace=False)
            #         data = copy.deepcopy((np.array(self.remData['data'])[selected_indxs]).tolist())
            #         labels = copy.deepcopy((np.array(self.remData['label'])[selected_indxs]).tolist())
            #         self.updateDict(self.remSizePerBaseClass, selected_indxs) 
            #         self.updateDict(self.remSizePerNewClass, selected_indxs, move=self.remSizePerBaseClass)              
            #         self.remData['data'] = np.delete(np.array(self.remData['data']),selected_indxs, axis=0).tolist()
            #         self.remData['label'] = np.delete(np.array(self.remData['label']), selected_indxs, axis=0).tolist() 
            #         self.updateCounter(len(selected_indxs))
             
            # else:
            indxs = []
            old_classes = []
            new_classes = []
            nb_newClass = np.random.choice(np.arange(1,min(len(self.remSizePerNewClass.keys())+1,3)))
            new_classes = np.random.choice(list(self.remSizePerNewClass.keys()), nb_newClass, replace=False)
            if self.remSizePerBaseClass:
                old_classes = np.random.choice(list(self.remSizePerBaseClass.keys()), self.nb_baseClasses-len(new_classes), replace=False)
            classes = np.append(old_classes,new_classes)
            print('Classes Selected {}'.format(classes))
            for c in classes:
                ii = np.where(self.remData['label'] == c)[0]
                indxs.extend(ii)
            #print(np.shape(indxs))
            if len(indxs) > 0:
                selected_indxs = np.random.choice(indxs, min(N, len(indxs)), replace=False)
                data = copy.deepcopy((np.array(self.remData['data'])[selected_indxs]).tolist())
                labels = copy.deepcopy((np.array(self.remData['label'])[selected_indxs]).tolist())
                self.updateDict(self.remSizePerBaseClass, selected_indxs) 
                self.updateDict(self.remSizePerNewClass, selected_indxs, move=self.remSizePerBaseClass)         
                self.remData['data'] = np.delete(np.array(self.remData['data']),selected_indxs, axis=0).tolist()
                self.remData['label'] = np.delete(np.array(self.remData['label']), selected_indxs, axis=0).tolist() 
                self.updateCounter(len(selected_indxs))  
   
        else:
            indxs = []
            classes = np.random.choice(list(self.remSizePerBaseClass.keys()), min(len(self.remSizePerBaseClass.keys()),5), replace=False)
            for c in classes:
                indxs.extend(np.where(self.remData['label'] == c)[0])
            #print(np.shape(indxs))
            #print(len(indxs))
            selected_indxs = np.random.choice(indxs, min(N, len(indxs)), replace=False)
            data = copy.deepcopy((np.array(self.remData['data'])[selected_indxs]).tolist())
            labels = copy.deepcopy((np.array(self.remData['label'])[selected_indxs]).tolist())
            self.updateDict(self.remSizePerBaseClass, selected_indxs)              
            self.remData['data'] = np.delete(np.array(self.remData['data']),selected_indxs, axis=0).tolist()
            self.remData['label'] = np.delete(np.array(self.remData['label']), selected_indxs, axis=0).tolist() 
            self.updateCounter(len(selected_indxs))            

        return data,labels



    def shuffleRemData(self):
        self.remData['data'],self.remData['label'] = shuffle(self.remData['data'],self.remData['label'])

    def endOfStream(self):
        # if self.counter + 1 >= 200:  ## for debugging
        #     return True

        return self.counter+1 >= self.maxSize

    def getNextBatch_controlled(self, N):
        #print(np.shape(self.remData['data']), np.shape(self.remData['label']))
        classes = np.random.choice(list(self.remSizePerNewClass.keys()) + list(self.remSizePerBaseClass.keys()), min(len(list(self.remSizePerNewClass.keys()) + list(self.remSizePerBaseClass.keys())), self.nb_baseClasses), replace=False)
        indxs = []
        for c in classes:
            indxs.extend(np.where(self.remData['label'] == c)[0])
        #print(np.shape(indxs))
        selected_indxs = np.random.choice(indxs, min(N, len(indxs)), replace=False)
        data = copy.deepcopy((np.array(self.remData['data'])[selected_indxs]).tolist())
        labels = copy.deepcopy((np.array(self.remData['label'])[selected_indxs]).tolist())
        self.updateDict(self.remSizePerBaseClass, selected_indxs) 
        self.updateDict(self.remSizePerNewClass, selected_indxs, move=self.remSizePerBaseClass) 
        self.remData['data'] = np.delete(np.array(self.remData['data']),selected_indxs, axis=0).tolist()
        self.remData['label'] = np.delete(np.array(self.remData['label']), selected_indxs, axis=0).tolist() 
        self.updateCounter(len(selected_indxs))

        return data,labels

    def task_incremental(self, N):
        raise NotImplementedError()
    def updateDict(self, oldDict, selected_indxs, move=None):
        keys = list(oldDict.keys())
        for c in keys:
            count = len(np.where(np.array(self.remData['label'])[selected_indxs] == c)[0])
            found =  count > 0
            oldDict[c] -= count

            if oldDict[c] == 0:
                del oldDict[c]

            elif found and move:
                move[c] = copy.deepcopy(oldDict[c])
                del oldDict[c]

    def getNextData(self):
        d,l = copy.deepcopy(self.remData['data'][self.counter]), copy.deepcopy(self.remData['label'][self.counter])
        self.updateCounter()
        return d,l

    def updateCounter(self, count=1):
        self.counter = self.counter + count

    def resetCounter(self):
        self.counter = 0

    def getBaseData(self):
        return copy.deepcopy(self.baseData)

        





