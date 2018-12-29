import tensorflow as tf
print("TF version:", tf.__version__)
import numpy as np
from matplotlib import pyplot as plt
import network
slim = tf.contrib.slim
import os
import argparse
import json
from preprocessing.read_data import tf_record_parser, scale_image_with_crop_padding
from preprocessing import training
from PIL import Image

class Dotdict(dict):
     """dot.notation access to dictionary attributes"""
     __getattr__ = dict.get
     __setattr__ = dict.__setitem__
     __delattr__ = dict.__delitem__

class deepLab_InferenceEngine:
    def __init__(self,modelDirectory,sess):
        self.modelDirectory=modelDirectory
        with open(modelDirectory + '\data.json', 'r') as fp:
            self.modelConfig = json.load(fp)
            self.modelConfig = Dotdict(self.modelConfig )
            self.class_labels = [v for v in range((self.modelConfig.number_of_classes+1))]
            self.class_labels[-1] = 255
    
    def restoreDeepLab(self):
        self.testImage = np.array(Image.open(self.modelDirectory+'\\test.jpg'))
        self.testImage = np.expand_dims(self.testImage, axis=0)
        self.testImage = tf.cast(self.testImage, tf.float32)
        
        
        self.logits_tf =  network.deeplab_v3(self.testImage,
                                             self.modelConfig,
                                             is_training=False,
                                             reuse=False)

        self.predictions_tf = tf.argmax(self.logits_tf , axis=3)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(self.modelDirectory, "model.ckpt"))
        print('Model Restored')

if __name__=="__main__":
    sess = tf.Session()
    deepLab = deepLab_InferenceEngine('D:\Acads\IISc ME\Projects\SemanticSegmentation\deepLabV3_Inference_GUI\model',
                                      sess)
    print(deepLab.modelConfig.crop_size)
    deepLab.restoreDeepLab()
