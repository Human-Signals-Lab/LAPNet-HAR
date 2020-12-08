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

# from train.visualisations import vis_by_person
# from train.visualisations.training_visualizer import plot_heatmap
# from sklearn.preprocessing import LabelEncoder


class DataHandler:

    def __init__(self, nb_baseClasses, seed, train, ClassPercentage):
        self.nb_baseClasses = nb_baseClasses
        self.seed = seed 
        self.seed_randomness()
        self.train = train
        self.ClassPercentage = ClassPercentage
        self.counter = 0
        
    def seed_randomness(self):
        np.random.seed(self.seed)


    def streaming_data(self):
        _labels = sorted(set(self.train['label']))

        self.baseClasses = np.random.choice(_labels, self.nb_baseClasses, replace=False).tolist()

        self.baseData = dict()
        self.baseData['data'] = []
        self.baseData['label'] = []

        self.remData = dict()
        self.remData['data'] = []
        self.remData['label'] = []
        for c in _labels:
            d,l = self.train['data'][self.train['label'] == c,:], self.train['label'][self.train['label'] == c]
            if c in self.baseClasses: 
                baseIdx = np.random.choice(np.arange(len(d)),int(self.ClassPercentage*len(d)))
                self.baseData['data'].extend(d[baseIdx,:])
                self.baseData['label'].extend(l[baseIdx])
                remIdx = np.setdiff1d(np.arange(len(d)),baseIdx)
                self.remData['data'].extend(d[remIdx,:])
                self.remData['label'].extend(l[remIdx])

            # to include new activities
            # else:
            #     self.remData['data'].extend(d)
            #     self.remData['label'].extend(l)
        self.shuffleRemData()

    def shuffleRemData(self):
        self.remData['data'],self.remData['label'] = shuffle(self.remData['data'],self.remData['label'])

    def endOfStream(self):
        return self.counter+1 >= len(self.remData['data'])

    def getNextData(self):
        d,l = self.remData['data'][self.counter], self.remData['label'][self.counter]
        self.updateCounter()
        return d,l

    def updateCounter(self):
        self.counter = self.counter + 1

    def resetCounter(self):
        self.counter = 0

    def getBaseData(self):
        return self.baseData

        





