import numpy as np
import os
import scipy.misc as misc
import random
import cv2
import parse as par
import random
from os import listdir
#------------------------Class for reading training and  validation data---------------------------------------------------------------------
class Data_Reader:
################################Initiate folders were files are and list of train images############################################################################
    def __init__(self, ImageDir, Train=True):
        self.ImageDir = ImageDir
        self.batchindex = 1
        self.image_name_dic, self.store_mp = par.parsing()#read all existing image name, order fixed
        if Train:
            self.start = 0
            self.end = 48
        else:
            self.start = 49
            self.batchindex = self.start
            self.end = max(self.image_name_dic)

    def getBatch(self):
        while self.batchindex not in self.image_name_dic:
            self.batchindex += 1
        image_batch = self.image_name_dic[self.batchindex]
        #firstname = image_batch[0]
        #first_image = cv2.imread(self.ImageDir + firstname)
        #h, w, c = first_image.shape
        img = np.zeros((len(image_batch),70,110,3))
        label = np.zeros((len(image_batch),3))       #[health, highrisk, disease]
        for i in range(len(image_batch)):
            imname = image_batch[i]
            raw_image = cv2.imread(self.ImageDir + imname)
            resized_image = cv2.resize(raw_image, (110, 70))
            img[i,:,:,:] = resized_image
            if self.store_mp[imname] == 1:#health
                label[i,0] = 1
            if self.store_mp[imname] == 2:#highrisk
                label[i,1] = 1
            if self.store_mp[imname] == 0:#disease
                label[i,2] = 1
        self.batchindex += 1
        if self.batchindex > self.end:
            self.batchindex = 0
        return img, label





