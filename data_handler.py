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


# from train.visualisations import vis_by_person
# from train.visualisations.training_visualizer import plot_heatmap
# from sklearn.preprocessing import LabelEncoder


class DataHandler:

	def __init__(self, nb_baseClasses, seed, train, test, ClassPercentage):
		self.nb_baseClasses = nb_baseClasses
		self.seed = seed 
		self.seed_randomness(self.seed)
		self.train = train
		self.test = test 
		self.ClassPercentage = ClassPercentage
		self.counter = 0
	def seed_randomness(self):
        np.random.seed(self.seed)


	def streaming_data(self):
		_labels = sorted(set(self.train['labels']))

		self.baseClasses = np.random.choice(_labels, self.nb_baseClasses, replace=False).tolist()

		self.baseData = dict()
		self.baseData['data'] = []
		self.baseData['label'] = []

		self.remData = dict()
		self.remData['data'] = []
		self.remData['label'] = []
		for c in _labels:
			if c in self.baseClasses:
				d,l = self.train['data'][self.train['label'] == c,:], self.train['label'][self.train['label'] == c]
				baseIdx = np.random.choice(np.arange(len(d),self.ClassPercentage(len(d))))
				self.baseData['data'].append(d[baseIdx,:])
				self.baseData['label'].append(l[baseIdx])
				remIdx = np.setdiff1d(np.arange(len(d)),baseIdx)
				self.remData['data'].append(d[remIdx,:])
				self.remData['label'].append(l[remIdx])

			else:
				self.remData['data'].append(d)
				self.remData['label'].append(l)


	def getNextData(self):
		d,l = self.remData['data'][self.counter], self.remData['label'][self.counter]
		self.updateCounter()
		return d,l

	def updateCounter(self):
		self.counter = self.counter + 1

	def resetCounter(self):
		self.counter = 0

		





