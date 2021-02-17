## Class Prototypes
import numpy as np
import tensorflow as tf


class PrototypeMemory():
    # A module that stores prototypes

    def __init__(self):
        super(PrototypeMemory, self).__init__()
        self.prototypes = dict()
        self.counters = dict()

    # def zero_initialization(self,n_dim,n_classes):
    #     self.prototypes = tf.zeros([n_classes, n_dim])
    #     self.counters = tf.zeros([n_classes,1])


    def initialize_prototypes(self, X, y):
        classes = np.sort(np.unique(y))
        #print(classes)
        for c in classes:
            p_mean = X[y==c].mean(0)
            #print(np.shape(X[y==c]))
            self.prototypes[c] = list(p_mean.data.cpu().numpy().flatten())
            self.counters[c] = len(X[y==c])
                
    def update_prototypes(self,X,y):
        classes = np.sort(np.unique(y))
        for c in classes:
            if c in self.prototypes:
                #print(self.prototypes[c])
                p_mean_old = np.array(self.prototypes[c]).copy()
               # print(c, np.shape(p_mean_old), p_mean_old)
                new_count = len(np.array(X)[y==c])
                #print(np.shape(p_mean_old), np.shape(np.array(X)))
                
                p_mean = float((self.counters[c]/(self.counters[c]+new_count)))*p_mean_old + np.sum(np.array(X),axis=0)/(self.counters[c]+new_count)
                #print(p_mean, p_mean_old)
                #sys.exit()
                self.prototypes[c] = p_mean.flatten()
                self.counters[c] += new_count
            else:
                #print(self.prototypes.keys(), c)
                p_mean = X[y==c].mean(0)
                #print(np.shape(X[y==c]))
                self.prototypes[c] = list(p_mean.data.cpu().numpy().flatten())
                self.counters[c] = len(X[y==c])               

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

