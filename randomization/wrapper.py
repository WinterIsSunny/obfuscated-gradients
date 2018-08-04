#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 17:19:36 2018

@author: yusu
"""
import tensorflow as tf
import numpy as np
from utils import *
from defense import *

class MyModel:
    def __init__(self,model,sess,bounds):
        self.sess = sess
        self.model = model
        self.bounds = bounds
        self.x = tf.placeholder(tf.float32, (299, 299, 3))
        self.x_expanded = tf.expand_dims(self.x, axis=0)
        self.cropped_x = defend(self.x_expanded)
        
        
    def predict(self,image):
        if self.bounds[1] == 255.0:
            new_img = image*255.0
            new_img = np.clip(new_img,0.0,255.0)
        else:
            new_img = np.clip(image,0.0,255.0)
            
        new_img = [new_img]
        cropped_logits, cropped_preds = self.model.model(self.sess, self.cropped_x)
        logits,label = self.sess.run([cropped_logits,cropped_preds],{self.cropped_x:new_img})
        
        return label

        