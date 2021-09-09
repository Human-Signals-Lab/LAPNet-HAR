## Class Prototypes
import numpy as np
#import tensorflow as tf
import copy
import random 
# seed = 0
# random.seed(seed)

# np.random.seed(seed)


class PrototypeMemory():
    # A module that stores prototypes

    def __init__(self):
        super(PrototypeMemory, self).__init__()
        self.prototypes = dict()
        self.counters = dict()

    def zero_initialization(self,n_dim,classes):
        for c in classes:
            self.prototypes[c] = np.zeros(n_dim)
            self.counters[c] = 0


    def initialize_prototypes(self, X, y):
        classes = np.sort(np.unique(y))
        #self.prototypes = dict()
        #print(classes)
        for c in classes:
            #import pdb; pdb.set_trace()
            p_mean = np.mean(X.data.cpu().numpy()[y==c],axis=0,dtype=np.float64)
            #print(np.shape(X[y==c]))
            self.prototypes[c] = copy.deepcopy(p_mean.flatten())
            self.counters[c] = len(X[y==c])

    def update_prototypes(self,X,y):
        classes = np.sort(np.unique(y))
        print(classes)
        for c in classes:
            if c in self.prototypes.keys():
                p_mean_old = copy.deepcopy(np.array(self.prototypes[c]).astype(np.float64))
               # print(c, np.shape(p_mean_old), p_mean_old)
                new_count = len(np.array(X)[y==c])
                #print(np.shape(p_mean_old), np.shape(np.array(X)))
                
                p_mean = float((self.counters[c]/(1.*(self.counters[c]+new_count))))*p_mean_old + np.sum(np.array(X)[y==c],axis=0)/(self.counters[c]+new_count)
                #print(p_mean, p_mean_old)
                #sys.exit()
                self.prototypes[c] = copy.deepcopy(p_mean.flatten().astype(np.float64))
                self.counters[c] += new_count
                #print('old: ',c, p_mean_old, self.prototypes[c])

            else:
                #print('new: ', self.prototypes.keys(), c)
                p_mean = np.mean(X[y==c],axis=0,dtype=np.float64)
                #print(np.shape(X[y==c]))
                self.prototypes[c] = copy.deepcopy(p_mean.flatten())
                self.counters[c] = len(np.array(X)[y==c])    

    def update_prototypes_momentum(self, new_proto, momentum):
        for c in new_proto.keys():
            self.prototypes[c] = copy.deepcopy((momentum*np.array(self.prototypes[c]).astype(np.float64) + (1.-momentum)*np.array(new_proto[c]).astype(np.float64)).flatten().astype(np.float64))

    # def query(self, x, y, t, storage, count, add_new=tf.constant(True), is_training=tf.constant(True), **kwargs):
    #   y_ = self.retrieve(x, storage, count, t, add_new=add_new, is_training=is_training)
    #   storage, count = self.store(x, y, storage, count)
    #   return y_, (storage, count)


    # def retrieve(self,x,storage,count,t,beta=None,gamma=None, add_new=tf.constant(True),is_training=tf.constant(True)):
    #   prototypes = storage
    #   logits = self.compute_logits(x, prototypes)

    
    # def compute_logits(self, x, prototypes):
    #   dist = tf.reduce_sum(tf.square(x - prototypes), [-1])  # [B, K+1]
    #   return -dist

