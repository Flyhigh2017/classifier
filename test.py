################Class which build the fully convolutional neural net###########################################################

import inspect
import os
import TensorflowUtils as utils
import numpy as np
import tensorflow as tf
import Inference as Inf
import cv2
import Data_Reader
from os import listdir
data_path = "/Users/anekisei/Documents/Spine_project_horizontal/train_images/"
TestReader = Data_Reader.Data_Reader(data_path, Train=True)
health = []
disease = []
correct_disease = 0
total_disease = 0
correct_risk = 0
total_risk = 0
correct_health = 0
total_health = 0

while TestReader.batchindex != 0:
    Images,  GTLabels = TestReader.getBatch()
    result = Inf.predict(Images)
    total_disease += np.sum(GTLabels[:,2])
    total_risk += np.sum(GTLabels[:,1])
    total_health += np.sum(GTLabels[:,0])

    for i in range(result.shape[0]):
        if result[i] == 2 and GTLabels[i,2] == 1:
            correct_disease += 1
        if result[i] == 1 and GTLabels[i,1] == 1:
            correct_risk += 1
        if result[i] == 0 and GTLabels[i,0] == 1:
            correct_health += 1
disease_accuracy = (correct_disease + 0.0) / (total_disease + 0.0)
health_accuracy = (correct_health + 0.0) / (total_health + 0.0)
risl_accuracy = (correct_risk + 0.0) / (total_risk + 0.0)
print "disease accuracy is,", disease_accuracy
print "health accuracy is,", health_accuracy
print "risk accuracy is,",risl_accuracy

