
""" Prototype Network """
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
from prototype_memory import *
import copy
# seed = 0
# torch.backends.cudnn.deterministic = True
# random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)

class ProtoNet(nn.Module):

	def __init__(self, extractor,n_dim,n_classes,memory=None, margin = None):
		super(ProtoNet,self).__init__()
		self.n_dim = n_dim
		self.n_classes = n_classes
		self.extractor = extractor.cuda()
		if not memory:
			self.memory = PrototypeMemory()
			#self.memory.zero_initialization(self.n_dim, self.n_base_classes)
		self.margin = margin

	def forward_offline(self, support, y_support, query,hidden, hidden_query=None, DSoftmax=False):

		if self.training:
			_,_, z_support = self.extractor.forward(support,hidden,support.size(0))
			self.memory.initialize_prototypes(z_support,np.argmax(y_support.data.cpu(),axis=1))

		if hidden_query:
			_,h, z_query = self.extractor.forward(query,hidden_query,query.size(0))
		else:
			_,h, z_query = self.extractor.forward(query,hidden,query.size(0))

		#print(type(np.array(list(self.memory.prototypes.values()))),np.shape(np.array(list(self.memory.prototypes.values()))))
		z_proto = torch.from_numpy(np.array(list(self.memory.prototypes.values()))).float().cuda()
		#print(np.shape(z_proto),np.shape(z_query))

		dists = self.compute_euclidean(z_query,z_proto)
		#print(np.shape(dists))
		#dists = self.compute_euclidean1(z_query,z_proto)
		#assert dists == dists1

		if self.training and DSoftmax:
			return dists, h
		else:
			#print('not training')
			p_y = F.softmax(-dists,dim=1)
		return p_y,h



	def forward_inference(self, support, y_support, query, hidden,hidden_query=None):
		## training model on base data

		## training model on base data
		self.extractor.eval()

		if hidden_query:
			_,h, z_query = self.extractor.forward(query,hidden_query,query.size(0))
		else:
			_,h, z_query = self.extractor.forward(query,hidden,query.size(0))

		_,_, z_support = self.extractor.forward(support,hidden,support.size(0))
		self.inference_prototypes(z_support,np.argmax(y_support.data.cpu(),axis=1))
		#print(type(np.array(list(self.memory.prototypes.values()))),np.shape(np.array(list(self.memory.prototypes.values()))))
		z_proto = torch.squeeze(torch.from_numpy(copy.deepcopy(np.array(list(self.prototypes.values())))).float()).cuda()

		dists = self.compute_euclidean(z_query,z_proto)
		#dists = self.compute_euclideanFIX(z_query)
		#print(np.shape(dists))
		#dists = self.compute_euclidean1(z_query,z_proto)
		#assert dists == dists1

		p_y = F.softmax(-dists,dim=1)
		return p_y,h		

	def inference_prototypes(self, X, y):
		classes = np.sort(np.unique(y))
		self.prototypes = dict()
		self.counters = dict()
		#print(classes)
		for c in classes:
			p_mean = X[y==c].mean(0)
			#print(np.shape(X[y==c]))
			self.prototypes[c] = copy.deepcopy(list(p_mean.data.cpu().numpy().flatten()))
			self.counters[c] = len(X[y==c])

	def prototype_update_momentum(self, support, y_support, momentum,hidden):
		self.extractor.eval()
		_,_, z_support = self.extractor.forward(support,hidden,support.size(0))

		self.inference_prototypes(z_support, np.argmax(y_support.data.cpu(),axis=1))
		self.memory.update_prototypes_momentum(self.prototypes,momentum)



	def forward_online(self, support, y_support, query, hidden_support, hidden_query, lwf=False):

		if self.training:
			#print("Training ... ")
			_,_, z_support = self.extractor.forward(support,hidden_support,support.size(0))
			self.memory.update_prototypes(z_support.data.cpu().numpy(),np.argmax(y_support.data.cpu(),axis=1))
		_,h, z_query = self.extractor.forward(query,hidden_query,query.size(0))

		#z_proto = torch.squeeze(torch.from_numpy(np.array(list(self.memory.prototypes.values()))).float()).cuda()
		#dists = self.compute_euclidean(z_query,z_proto)
		#print(dists)
		#print(log_p_y)
		dists = self.compute_euclideanFIX(z_query)
		#print(dists)
		#print(np.shape(dists))
		#dists = self.compute_euclidean1(z_query,z_proto)
		#assert dists == dists1
		p_y = F.softmax(-dists,dim=1)
		#print(log_p_y)
		#sys.exit()
		if self.training:
			if lwf:
				return dists, h, z_support
			return p_y,h,dists	
		else:
			if lwf:
				return dists, h
			return p_y, h

	def forward_online_QUERY(self, query, hidden_query, lwf=False):

		_,h, z_query = self.extractor.forward(query,hidden_query,query.size(0))

		z_proto = torch.squeeze(torch.from_numpy(np.array(list(self.memory.prototypes.values()))).float()).cuda()
		#dists = self.compute_euclidean(z_query,z_proto)
		dists = self.compute_euclideanFIX(z_query)
		#print(np.shape(dists))
		#dists = self.compute_euclidean1(z_query,z_proto)
		#assert dists == dists1
		if lwf:
			return dists, h
		p_y = F.softmax(-dists,dim=1)
		return p_y, h


	def online_update_prototypes(self, support, y_support, hidden_support):
		_,_,z_support = self.extractor.forward(support,hidden_support,support.size(0))
		self.memory.update_prototypes(z_support.data.cpu().numpy(), np.argmax(y_support.data.cpu(),axis=1))
		return z_support
		#z_proto = torch.squeeze(torch.from_numpy(np.array(list(self.memory.prototypes.values()))).float()).cuda()
		#dists = self.compute_euclidean(z_query, z_proto)

		#log_p_y = F.softmax(-dists,dim=1)
		#return log_p_y, h, z_support

	def compute_euclideanFIX(self, z_query):
		dists = torch.ones((len(z_query),self.n_classes))*float('inf')
		dists = dists.float().cuda()
		#print("CURRENT CLASSES IN PROTOTYPE MEMORY: ", list(self.memory.prototypes.keys()))
		for c in self.memory.prototypes.keys():
			z_proto = torch.from_numpy(self.memory.prototypes[c][None,:]).float().cuda()
			dist = self.compute_euclidean(z_query,z_proto)
			#print(np.shape(dist))
			dists[:,c] = torch.squeeze(dist)
		#print(np.shape(dists))
		return dists


	def update_protoMemory(self, z_support, y_support):
		#_,_, z_support = self.extractor.forward(support,hidden,support.size(0))
		self.memory.initialize_prototypes(z_support,np.argmax(y_support.data.cpu(),axis=1))		


	def compute_euclidean(self,query, proto):
		#print(np.shape(query), np.shape(proto))
		# query_n = torch.linalg.norm(query, dim=1, keepdims=True)
		# proto_n = torch.linalg.norm(proto, dim=1, keepdims=True)
		#import pdb; pdb.set_trace()
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




