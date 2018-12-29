from preprocessing.read_data import tf_record_parser, scale_image_with_crop_padding
from preprocessing import training
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import network
slim = tf.contrib.slim
import os
import argparse
import json
from PIL import Image

class Dotdict(dict):
     """dot.notation access to dictionary attributes"""
     __getattr__ = dict.get
     __setattr__ = dict.__setitem__
     __delattr__ = dict.__delitem__

class deepLab_InferenceEngine:
    def __init__(self,modelDirectory):
        self.modelDirectory=modelDirectory #Directory where necessary model checkpoints are located 
        with open(modelDirectory + '\data.json', 'r') as fp:
            self.modelConfig = json.load(fp)
            self.modelConfig = Dotdict(self.modelConfig )
    
    def segmentImage(self,inputImage):
        self.testImage = inputImage
        self.testImage = self.testImage.resize((self.modelConfig.crop_size,
                                                self.modelConfig.crop_size),
                                                Image.ANTIALIAS)
        self.testImage = np.array(self.testImage)
        self.testImage = np.expand_dims(self.testImage, axis=0) #This is necessary else Resnet starts bitching about dimensions
        self.testImage = tf.cast(self.testImage, tf.float32)

        self.logits_tf =  network.deeplab_v3(self.testImage,
                                             self.modelConfig,
                                             is_training=False,
                                             reuse=False)
        
        self.predictions_tf = tf.argmax(self.logits_tf , axis=3)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(self.modelDirectory, "model.ckpt"))

        print('Model restored, Segmentation In Progress..')
        resultImg = sess.run(self.predictions_tf)
        print('Segmentation Done !')
        return resultImg[0]
        

if __name__=="__main__":
    sess = tf.Session()
    deepLab = deepLab_InferenceEngine(os.getcwd()+'\model')
    print(deepLab.modelConfig.crop_size)
    testImage = Image.open('C:\\DataSets'+'\\t140.jpg')
    segmentedImage = deepLab.segmentImage(testImage)
    plt.imshow(segmentedImage)
    plt.show()
