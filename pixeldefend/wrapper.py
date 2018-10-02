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
import numpy as np
import time

class MyModel:
    
    def __init__(self,model,sess,TRUE_CLASS,saver,out,x,bounds):
        self.bounds = bounds
        self.model = model
        self.sess = sess
        self.TRUE_CLASS = TRUE_CLASS
        self.saver = saver
        self.out = out
        self.x = x
        
        self.saver.restore(self.sess, tf.train.latest_checkpoint('data/models/naturally_trained'))
        #timestart = time.time()
        self.pixeldefend = make_pixeldefend(self.sess, self.x, self.out)
        #grad, = tf.gradients(self.model.xent, self.model.x_input)
        
    
    def predict(self,image):
        timestart = time.time()
        if self.bounds[1] == 255.0:
            new_img = image * 255.0
            new_img = np.clip(new_img,0.0,255.0)
        else:
            new_img = np.clip(image,0.0,1.0)

#        new_img = [new_img]

        
        
       # x = tf.placeholder(tf.float32, (1, 32, 32, 3))
        
        
        #logits = self.model.pre_softmax
        #probs = tf.nn.softmax(logits)
        #classify = make_classify(self.sess, self.model.x_input, probs)
        #label = classify(new_img)
        
        #print("the label of the image is :", label)
        
        adv_def = self.pixeldefend(new_img)
        p = self.sess.run(self.model.predictions,
                       {self.model.x_input: [adv_def]})
        timeend = time.time()
        print("time consuming for one query :", timeend -timestart)
        print(p[0])

        return p[0]
        