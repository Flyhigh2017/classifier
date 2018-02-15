import numpy as np
import os
from os.path import isfile, join
from os import listdir
import scipy.misc as misc
import cv2
from PIL import Image
import parse as par
def split():
    health = []
    disease = []
    store_mp = par.parse(Train=True)
    for image_name in store_mp:
        if store_mp[image_name] == 1:
            health.append(image_name)
        else:
            disease.append(image_name)
    return health, disease






