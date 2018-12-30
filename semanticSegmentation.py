from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
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

class deepLabV3_InferenceEngine:
    def __init__(self,modelDirectory,sess):
        self.modelDirectory=modelDirectory #Directory where necessary model checkpoints are located
        self.sess=sess
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
                                             reuse=tf.AUTO_REUSE)
        
        self.predictions_tf = tf.argmax(self.logits_tf , axis=3)
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(self.modelDirectory, "model.ckpt"))

        print('DeeplabV3 Model restored')
        print('Segmentation In Progress..')
        resultImg = self.sess.run(self.predictions_tf)
        print('Segmentation Done !')
        return resultImg[0]
        

if __name__=="__main__":
   
    deepLab = deepLabV3_InferenceEngine(os.getcwd()+'\model', tf.Session())
    testImage = Image.open('C:\\DataSets'+'\\t140.jpg')
    segmentedImage = deepLab.segmentImage(testImage)
    plt.imshow(segmentedImage)
    plt.show()

    segmentedImage = deepLab.segmentImage(testImage)
    plt.imshow(segmentedImage)
    plt.show()
