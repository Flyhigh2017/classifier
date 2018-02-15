import numpy as np
import os
from os.path import isfile, join
from os import listdir
import scipy.misc as misc
import cv2
from PIL import Image
import pandas
#parse label.csv sick -- 0, healthy -- 1, high risk -- 2
def sortName(name_lst):
    #store order and sort accending
    order_lst = []
    #store sorted image_name
    new_lst = []
    # temp -- {order : image_name}
    temp = {}
    for i in range(len(name_lst)):
        splitname = name_lst[i].split('_')[2]
        order = splitname.split('.')[0]
        order_lst.append(int(order))
        temp[order] = name_lst[i]
    order_lst = sorted(order_lst)
    for order in order_lst:
        new_lst.append(temp[str(order)])
    return new_lst

def parsing():
    my_data = pandas.read_csv('hori_final_label.csv',header=None).as_matrix()
    image_name_dic = {}
    store_mp = {}
    #image name dic -- {1 : [image name]} store_mp -- {image_name : label}
    for i in range(my_data.shape[0]): #train 0.8, validate 0.2
        image_name = my_data[i][0]
        patient_index = int(image_name.split('_')[1])
        annotated = my_data[i][1]
        store_mp[image_name] = annotated
        if patient_index in image_name_dic:
            image_name_dic[patient_index] += [image_name]
        else:
            image_name_dic[patient_index] = [image_name]
    for index in image_name_dic:
        name_lst = image_name_dic[index]
        image_name_dic[index] = sortName(name_lst)
    return image_name_dic, store_mp

parsing()
