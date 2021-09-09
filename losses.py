### Contrastive Loss

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
#from prototype_memory import *
import copy
from itertools import combinations

# seed = 0
# torch.backends.cudnn.deterministic = True
# random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)


class PairSelector:
	def __init__(self, balance=True):
		self.balance = balance

	def get_pairs(self, embeddings, labels):
		labels = labels.cpu().data.numpy()
		all_pairs = np.array(list(combinations(range(len(labels)),2)))
		all_pairs = torch.LongTensor(all_pairs)
		#print(np.shape((labels[all_pairs[:,0]] == labels[all_pairs[:,1]])))
		positive_pairs = all_pairs[(labels[all_pairs[:,0]] == labels[all_pairs[:,1]]).nonzero()]
		negative_pairs = all_pairs[(labels[all_pairs[:,0]] != labels[all_pairs[:,1]]).nonzero()]
		#print(np.shape(positive_pairs), np.shape(negative_pairs))
		if self.balance:
			negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

		return positive_pairs, negative_pairs



class OnlineContrastiveLoss(nn.Module):
	def __init__(self, proto_net, pair_selector, margin=1):
		super(OnlineContrastiveLoss, self).__init__()
		self.proto_net = proto_net
		self.margin = margin
		self.pair_selector = pair_selector

	def forward(self, embeddings, labels):
		labels = torch.argmax(labels, dim=1)
		positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, labels)
		if embeddings.is_cuda:
			positive_pairs = positive_pairs.cuda()
			negative_pairs = negative_pairs.cuda()

		positive_loss = (embeddings[positive_pairs[:,0]] - embeddings[positive_pairs[:,1]]).pow(2).sum(1)
		negative_loss = F.relu(self.margin - (embeddings[negative_pairs[:,0]] - embeddings[negative_pairs[:,1]]).pow(2).sum(1).sqrt()).pow(2)

		loss = torch.cat([positive_loss, negative_loss], dim = 0)
		return loss.mean()

class OnlineContrastiveLossWithPrototypes(nn.Module):
	def __init__(self, proto_net, pair_selector, margin=1):
		super(OnlineContrastiveLossWithPrototypes, self).__init__()
		self.proto_net = proto_net
		self.margin = margin
		self.pair_selector = pair_selector

	def forward(self, embeddings, labels,proto_net):
		labels = torch.argmax(labels, dim=1)
		prototypes = torch.from_numpy(np.array(list(proto_net.memory.prototypes.values()))).float().cuda()
		proto_keys = torch.from_numpy(np.array(list(proto_net.memory.prototypes.keys()))).cuda()
		#import pdb; pdb.set_trace()
		embeddings = torch.cat((embeddings,prototypes),0)
		labels = torch.cat((labels,proto_keys),0)
		positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, labels)
		if embeddings.is_cuda:
			positive_pairs = positive_pairs.cuda()
			negative_pairs = negative_pairs.cuda()

		positive_loss = (embeddings[positive_pairs[:,0]] - embeddings[positive_pairs[:,1]]).pow(2).sum(1)
		negative_loss = F.relu(self.margin - (embeddings[negative_pairs[:,0]] - embeddings[negative_pairs[:,1]]).pow(2).sum(1).sqrt()).pow(2)

		loss = torch.cat([positive_loss, negative_loss], dim = 0)
		return loss.mean()


class OnlinePrototypicalContrastiveLoss(nn.Module):

	def __init__(self, proto_net, T, maxClass):
		super(OnlinePrototypicalContrastiveLoss, self).__init__()
		self.proto_net = proto_net
		self.key2idx = torch.empty(maxClass,dtype=torch.long).cuda()
		self.T = T

	def forward(self, embeddings, labels):
		labels = torch.argmax(labels, dim=1, keepdim=True) ## NX1
		prototypes = torch.from_numpy(np.array(list(self.proto_net.memory.prototypes.values()))).float().cuda()
		proto_keys = list(self.proto_net.memory.prototypes.keys())
		#import pdb; pdb.set_trace()

		self.key2idx[proto_keys] = torch.arange(len(proto_keys)).cuda()
		dist = self.compute_similarity(embeddings, prototypes)   # N X C 
		#print(dist)
		#mask = torch.zeros(dist.shape).cuda()
		#mask = mask.scatter_(1,labels,1.)
		#print(mask)
		#masked_softmax = torch.gather(dist, 1, mask)
		#print(dist*mask)
		#import pdb; pdb.set_trace()
		masked_dist = dist.gather(1,self.key2idx[labels[:,0]].view(-1,1))
		loss = - torch.log(masked_dist)
		#print(loss)
		loss = loss.mean()
		#print(loss)
		return loss
		#loss = 

	def compute_similarity(self,embeddings, prototypes):

		contrast = torch.div(torch.matmul(embeddings, prototypes.T), self.T)
		output = F.softmax(contrast, dim=1)
		return output


class DSoftmaxLoss(nn.Module):

	def __init__(self, d, maxClass):
		super(DSoftmaxLoss, self).__init__()
		self.epsilon = torch.exp(d).cuda()
		self.maxClass = maxClass
		self.key2idx = torch.empty(self.maxClass,dtype=torch.long).cuda()

	def forward(self, distances, labels, proto_net):
		labels = torch.argmax(labels, dim=1, keepdim=True)
		proto_keys = list(proto_net.memory.prototypes.keys())
		#import pdb; pdb.set_trace()
		self.key2idx[proto_keys] = torch.arange(len(proto_keys)).cuda()
		intra_dist = distances.gather(1,self.key2idx[labels[:,0]].view(-1,1))

		#import pdb; pdb.set_trace()
		nb_classes = distances.shape[1]
		inter_dist = distances[torch.ones_like(distances).scatter_(1, labels, 0.).bool()].view(-1,nb_classes-1)
		#inter_dist = distances.gather(1, self.key2idx[inter_labels])
		intra_loss = torch.log(1 + torch.div(self.epsilon, torch.exp(-intra_dist)))
		inter_loss = torch.log(1 + torch.sum(torch.exp(-inter_dist), dim=1, keepdim=True))
		loss = intra_loss + inter_loss
		return loss.mean()










