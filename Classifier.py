################Class which build the fully convolutional neural net###########################################################

import inspect
import os
import TensorflowUtils as utils
import numpy as np
import tensorflow as tf

class Classifier:
    def __init__(self, feature=None):
        self.feature = feature
    
########################################Build Net#####################################################################################################################
    def classify(self):  #Build the fully convolutional neural network (FCN) and load weight for decoder based
        self.flatten = tf.reshape(self.feature, [-1, 3*4*512])
        #self.layer1 = tf.layers.dense(inputs = self.flatten ,units = 128, activation = tf.nn.relu)
        self.predict = tf.layers.dense(inputs = self.flatten ,units = 3, activation = tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        return self.predict

