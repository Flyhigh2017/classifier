import tensorflow as tf
import numpy as np
import Data_Reader
import BuildNetVgg16
import os
import CheckVGG16Model
import scipy.misc as misc
import Data_Reader
import Classifier as C
#...........................................Input and output folders.................................................
Train_Image_Dir="/Users/anekisei/Documents/Spine_project_horizontal/train_images/" # Images and labels for training
logs_dir= "logs/"# "path to logs directory where trained model and information will be stored"
if not os.path.exists(logs_dir): os.makedirs(logs_dir)
model_path="Model_Zoo/vgg16.npy"# "Path to pretrained vgg16 model for encoder"
learning_rate=1e-4 #Learning rate for Adam Optimizer
CheckVGG16Model.CheckVGG16(model_path)# Check if pretrained vgg16 model avialable and if not try to download it
#-----------------------------Other Paramters------------------------------------------------------------------------
TrainLossTxtFile=logs_dir+"TrainLoss.txt" #Where train losses will be writen
ValidLossTxtFile=logs_dir+"ValidationLoss.txt"# Where validation losses will be writen
Batch_Size=5 # Number of files per training iteration
Weight_Loss_Rate=5e-4# Weight for the weight decay loss function
MAX_ITERATION = int(89400) # Max  number of training iteration

######################################Solver for model   training#####################################################################################################################

def main(argv=None):
    tf.reset_default_graph()
#.........................Placeholders for input image and labels...........................................................................................
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image") #Input image batch first dimension image number second dimension width third dimension height 4 dimension RGB
    GTLabel = tf.placeholder(tf.int32, shape=[None, 3], name="GTLabel")#Ground truth labels for training
  #.........................Build FCN Net...............................................................................................
    Net =  BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path) #Create class for the network
    feature = Net.build(image)# Create the net and load intial weights
#......................................Get loss functions for neural net work  one loss function for each set of label....................................................................................................
    res = tf.placeholder(tf.float32, shape=[None, 3, 4, 512], name="input_image")
    c = C.Classifier(res)
    Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=GTLabel,logits=c.classify(),name="Loss"))  # Define loss function for training

   #....................................Create solver for the net............................................................................................
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Loss)
#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
    TrainReader = Data_Reader.Data_Reader(Train_Image_Dir) #Reader for training data
    sess = tf.Session() #Start Tensorflow session
# -------------load trained model if exist-----------------------------------------------------------------
    print("Setting up Saver...")
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer()) #Initialize variables
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path: # if train model exist restore it
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
#--------------------------- Create files for saving loss----------------------------------------------------------------------------------------------------------

    f = open(TrainLossTxtFile, "w")
    f.write("Iteration\tloss\t Learning Rate="+str(learning_rate))
    f.close()
#..............Start Training loop: Main Training....................................................................
    for itr in range(MAX_ITERATION):
        print "itr:", itr
        Images,  GTLabels = TrainReader.getBatch() # Load  augmeted images and ground true labels for training
        feed_dict = {image:Images}
        output = sess.run(feature, feed_dict=feed_dict)
        feed_dict = {res:output,GTLabel:GTLabels}
        _, loss = sess.run([optimizer,Loss], feed_dict=feed_dict) # Train one cycle
        print "loss is,", loss
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
        if itr % 230 == 0 and itr>0:
            print("Saving Model to file in "+logs_dir)
            saver.save(sess, logs_dir + "model.ckpt", itr) #Save model

#......................Write and display train loss..........................................................................
        '''
        if itr % 11175==0:
            # Calculate train loss
            feed_dict = {image: Images, GTLabel: GTLabels}
            TLoss=sess.run(Loss, feed_dict=feed_dict)
            print("Step "+str(itr)+" Train Loss="+str(TLoss))
            #Write train loss to file
            with open(TrainLossTxtFile, "a") as f:
                f.write("\n"+str(itr)+"\t"+str(TLoss))
                f.close()
        '''

main()#Run script
print("Finished")
