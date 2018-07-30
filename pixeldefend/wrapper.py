#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 12:29:27 2018

@author: yusu
"""

#import torch
#import numpy as np
import tensorflow as tf
from defense import *
from utils import *
import models.pixelcnn_cifar as pixelcnn

class MyModel:
    
    def __init__(self,model,sess,TRUE_CLASS):
        self.model = model
        self.sess = sess
        self.TRUE_CLASS = TRUE_CLASS
    
    def predict(self,image):
        #saver = tf.train.Saver()
        #saver.restore(self.sess, tf.train.latest_checkpoint('data/models/naturally_trained'))
        
        x = tf.placeholder(tf.float32, (1, 32, 32, 3))
        _, out = pixelcnn.model(self.sess, x)
        
        pixeldefend = make_pixeldefend(self.sess, x, out)
        
        #logits = self.model.pre_softmax
        #probs = tf.nn.softmax(logits)
        #classify = make_classify(self.sess, self.model.x_input, probs)
        #classify(orig)
        
        #grad, = tf.gradients(self.model.xent, self.model.x_input)
        adv_def = pixeldefend(image)
        p = sess.run([self.model.predictions],
                           {self.model.x_input: [adv_def], self.model.y_input: [self.TRUE_CLASS]})
        #print(" prediction of adversarial: ",)
        return p
        