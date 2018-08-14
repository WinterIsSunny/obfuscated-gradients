#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:20:45 2018

@author: yusu
"""

import torch
import numpy as np
import tensorflow as tf
from utils import *
from defense import *


class MyModel:
    
    def __init__(self,model,sess,bounds):
        self.model = model
        self.sess = sess
        self.bounds = bounds
        self.x = tf.placeholder(tf.float32, (299, 299, 3))
        self.cropped_xs = defend_crop(self.x)
        self.logits, self.preds = self.model.model(self.sess, self.cropped_xs)
#        self.x_expanded = tf.expand_dims(self.x, axis=0)
#        self.logits, self.preds = self.model.model(self.sess, self.x_expanded)
    
    def predict(self,image,y0):
        if self.bounds[1] == 255.0:
            new_img = image * 255.0
            new_img = np.clip(new_img,0.0,255.0)
        else:
            new_img = np.clip(image,0.0,1.0)

#        adv_def = defend_crop(new_img)
        labels = self.sess.run([self.preds], {self.x: new_img})
        if y0 in labels:
            print("single label:",labels[0])
            print("all labels:",labels)
            print("label size:", len(labels))
            return y0 
        else:
            print(labels[0])
            return labels[0]

        