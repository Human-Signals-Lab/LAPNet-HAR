"""
@author: Rebecca Adaimi

Replay Memory
"""

import numpy as np
import copy
import random 
import math
from collections import Counter

# seed = 1
# random.seed(seed)

# np.random.seed(seed)


class ReplayMemory():
    ## A module that stores exemplar samples for replay


    def __init__(self, max_size):
        super(ReplayMemory, self).__init__()
        self.exemplars = dict()
        self.max_size = max_size
        self.class_count = 0


    def get_dict_by_class(self, features, labels):
        classwise_dict_of_features = {}
        for label in np.unique(labels):
            if label not in classwise_dict_of_features:
                #print("Dict by class: ", np.shape(features[labels == label]))
                classwise_dict_of_features[label] = list(features[labels == label])
            else:
                classwise_dict_of_features[label].append(features[labels == label])
        return classwise_dict_of_features


    def get_holdout_size_by_labels(self, count_of_labels, store_num):
        sorted_count_dict = sorted(count_of_labels, key=count_of_labels.get)
        dict_of_store_size = {}

        for label in sorted_count_dict:
            true_size = min(store_num, count_of_labels[label])
            dict_of_store_size[label] = true_size

        for old_cl in self.exemplars:
            if old_cl in sorted_count_dict:
                dict_of_store_size[old_cl] = min(store_num, len(self.exemplars[old_cl]) + count_of_labels[old_cl])
            else:
                dict_of_store_size[old_cl] = min(store_num, len(self.exemplars[old_cl]))

        return dict_of_store_size

    def random_update(self, train_dict):

        for old_cl, value in self.exemplars.items():
            value = np.array(value)
            if old_cl in train_dict: 
                value = np.append(value,train_dict[old_cl], axis=0)
                #print("1: ", len(value))
            self.exemplars[old_cl] = value[np.random.choice(len(value), self.train_store_dict[old_cl], replace=False)]

        for new_cl, value in train_dict.items():
            #print(new_cl)
            if new_cl not in self.exemplars:
                value = np.array(value)
                #print("Random Update: ", np.shape(value))
                random_indices = np.random.choice(len(value), self.train_store_dict[new_cl], replace=False)
                self.exemplars[new_cl] = value[random_indices]

    def update(self, data, strategy='random'):
        x, y = data
        new_class = []
        train_y_counts = Counter(y)
        for cl in train_y_counts:
            if cl not in self.exemplars:
                new_class.append(cl)
                self.class_count += 1
        samples_per_class = self.max_size
        #samples_per_class = self.max_size / self.class_count if self.class_count != 0 else self.max_size
        #samples_per_class = math.ceil(samples_per_class)

        self.train_store_dict = self.get_holdout_size_by_labels(train_y_counts, samples_per_class)
        train_dict = self.get_dict_by_class(x,y)

        print("New classes: {}, Old Classes: {}\n Updated memory size for each old class: {} [Train size: {}".format(new_class, self.exemplars.keys(), int(samples_per_class), self.train_store_dict))
        if len(new_class) > 0: 
            self.newClass = True
        else:
            self.newClass = False
        if strategy == 'random':
            self.random_update(train_dict)

        total_size = 0
        for key, value in self.exemplars.items():
            total_size += len(value)
            #print(f"Total exemplar size: {total_size}")
            assert len(self.exemplars[key]) == self.train_store_dict[key]
            #print(f"Class: {key}, No. of exemplars: {len(value)}")



    def exemplar_train(self, excluded_classes):
        exemplar_train_x = []
        exemplar_train_y = []
        for key, value in self.exemplars.items():
            if key not in excluded_classes:
                #print(np.shape(value))
                for train_x in value:
                    #print("...: ", np.shape(train_x))
                    exemplar_train_x.append(train_x)
                    exemplar_train_y.append(key)
        return exemplar_train_x, exemplar_train_y