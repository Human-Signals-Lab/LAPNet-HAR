
""" Prototype Network """
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
from prototype_memory import *

class ProtoNet(nn.Module):

	def __init__(self, extractor,n_dim,n_base_classes,memory=None):
		super(ProtoNet,self).__init__()
		self.n_dim = n_dim
		self.n_base_classes = n_base_classes
		self.extractor = extractor.cuda()
		if not memory:
			self.memory = PrototypeMemory()
			#self.memory.zero_initialization(self.n_dim, self.n_base_classes)

	def forward_offline(self, support, y_support, query,hidden):

		## training model on base data

		_,h, z_query = self.extractor.forward(query,hidden,query.size(0))

		if self.training:
			_,_, z_support = self.extractor.forward(support,hidden,support.size(0))
			self.memory.initialize_prototypes(z_support,np.argmax(y_support.data.cpu(),axis=1))
		#print(type(np.array(list(self.memory.prototypes.values()))),np.shape(np.array(list(self.memory.prototypes.values()))))
		z_proto = torch.squeeze(torch.from_numpy(np.array(list(self.memory.prototypes.values()))).float()).cuda()
		#print(np.shape(z_proto),np.shape(z_query))

		dists = self.compute_euclidean(z_query,z_proto)
		#print(np.shape(dists))
		#dists = self.compute_euclidean1(z_query,z_proto)
		#assert dists == dists1

		log_p_y = F.softmax(-dists,dim=1)
		return log_p_y,h


	def forward_inference(self, support, y_support, query, hidden):
		## training model on base data

		_,h, z_query = self.extractor.forward(query,hidden,query.size(0))


		_,_, z_support = self.extractor.forward(support,hidden,support.size(0))
		self.inference_prototypes(z_support,np.argmax(y_support.data.cpu(),axis=1))
		#print(type(np.array(list(self.memory.prototypes.values()))),np.shape(np.array(list(self.memory.prototypes.values()))))
		z_proto = torch.squeeze(torch.from_numpy(np.array(list(self.prototypes.values()))).float()).cuda()
		#print(np.shape(z_proto),np.shape(z_query))

		dists = self.compute_euclidean(z_query,z_proto)
		#print(np.shape(dists))
		#dists = self.compute_euclidean1(z_query,z_proto)
		#assert dists == dists1

		log_p_y = F.softmax(-dists,dim=1)
		return log_p_y,h		

	def inference_prototypes(self, X, y):
		classes = np.sort(np.unique(y))
		self.prototypes = dict()
		self.counters = dict()
		#print(classes)
		for c in classes:
			p_mean = X[y==c].mean(0)
			#print(np.shape(X[y==c]))
			self.prototypes[c] = list(p_mean.data.cpu().numpy().flatten())
			self.counters[c] = len(X[y==c])

	def forward_online(self, support, y_support, query, hidden):
		_,h, z_query = self.extractor.forward(query,hidden,query.size(0))

		if self.training:
			#print("Training ... ")
			_,_, z_support = self.extractor.forward(support,hidden,support.size(0))
			self.memory.update_prototypes(z_support.data.cpu().numpy(),np.argmax(y_support.data.cpu(),axis=1))
		z_proto = torch.squeeze(torch.from_numpy(np.array(list(self.memory.prototypes.values()))).float()).cuda()
		dists = self.compute_euclidean(z_query,z_proto)
		#print(np.shape(dists))
		#dists = self.compute_euclidean1(z_query,z_proto)
		#assert dists == dists1

		log_p_y = F.softmax(-dists,dim=1)
		if self.training:
			return log_p_y,h,z_support	
		else:
			return log_p_y, h


	def update_protoMemory(self, z_support, y_support):
		#_,_, z_support = self.extractor.forward(support,hidden,support.size(0))
		self.memory.initialize_prototypes(z_support,np.argmax(y_support.data.cpu(),axis=1))		


	def compute_euclidean(self,query, proto):
		x = query.unsqueeze(1).expand(query.size(0),proto.size(0),query.size(1))
		y = proto.unsqueeze(0).expand(query.size(0),proto.size(0),query.size(1))
		#print(np.shape(x),np.shape(y))
		return torch.pow(x-y,2).sum(2)


	def set_memory(self, memory):
		self.memory = memory

	def compute_euclidean1(self,a,b):
		a2 = tf.cast(tf.reduce_sum(tf.square(a),[-1],keepdims=True),dtype=tf.float32)
		ab = tf.cast(tf.matmul(a,b, transpose_b=True), dtype=tf.float32)
		b2 = tf.cast(tf.repeat(tf.reduce_sum(tf.square(b),[-1],keepdims=True), len(a),axis=0), dtype=tf.float32)
		#print(np.shape(a),np.shape(b),np.shape(a2),np.shape(ab),np.shape(b2))

		return a2 - 2*ab + b2



