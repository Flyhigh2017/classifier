# Run prediction and genertae pixelwise annotation for every pixels in the image using fully coonvolutional neural net
# Output saved as label images, and label image overlay on the original image
# 1) Make sure you you have trained model in logs_dir (See Train.py for creating trained model)
# 2) Set the Image_Dir to the folder where the input image for prediction are located
# 3) Set number of classes number in NUM_CLASSES
# 4) Set Pred_Dir the folder where you want the output annotated images to be save
# 5) Run script
#--------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import sys
import BuildNetVgg16
import TensorflowUtils
from PIL import Image
import os
import Data_Reader
import OverrlayLabelOnImage as Overlay
import CheckVGG16Model
import cv2
import Classifier as C
'''
logs_dir= "logs/"# "path to logs directory where trained model and information will be stored"
Image_Dir="/Users/anekisei/Documents/Spine_project/test_images"# Test image folder
w=0.6# weight of overlay on image
Pred_Dir="/Users/anekisei/Documents/Spine_project/FCN_segment/output/" # Library where the output prediction will be written
'''
model_path="Model_Zoo/vgg16.npy"# "Path to pretrained vgg16 model for encoder"
'''
NameEnd="" # Add this string to the ending of the file name optional
NUM_CLASSES = 3 # Number of classes
'''
#-------------------------------------------------------------------------------------------------------------------------
#CheckVGG16Model.CheckVGG16(model_path)# Check if pretrained vgg16 model avialable and if not try to download it

################################################################################################################################################################################
def predict(imagebatch):
    tf.reset_default_graph()
    logs_dir= "/Users/anekisei/Documents/Spine_project_horizontal/classifier/logs/"# "path to logs directory where trained model and information will be stored"
    Image_Dir="/Users/anekisei/Documents/Spine_project_vertical/test_images/"# Test image folder
    model_path="/Users/anekisei/Documents/Spine_project_vertical/FCN_segment/Model_Zoo/vgg16.npy"# "Path to pretrained vgg16 model for encoder"
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")  # Input image batch first dimension image number second dimension width third dimension height 4 dimension RGB

    # -------------------------Build Net----------------------------------------------------------------------------------------------
    Net = BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path)  # Create class instance for the net
    feature = Net.build(image)
    res = tf.placeholder(tf.float32, shape=[None, 3, 4, 512], name="input_image")
    c = C.Classifier(res)
    logits = c.classify()
    sess = tf.Session() #Start Tensorflow session
    sess.run(tf.global_variables_initializer())
    #print("Setting up Saver...")
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path: # if train model exist restore it
        print "Restore model from:", ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)
    #print("Model restored...")
    else:
        print("ERROR NO TRAINED MODEL IN: "+ckpt.model_checkpoint_path+" See Train.py for creating train network ")
        sys.exit()
    feed_dict = {image:imagebatch}
    output = sess.run(feature, feed_dict=feed_dict)
    feed_dict = {res:output}
    logits = sess.run(logits, feed_dict=feed_dict) # Train one cycle
    predicts = np.argmax(logits, axis=1)
    return predicts

#predict()#Run script
print("Finished")
