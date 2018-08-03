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
        
    def predict(self,image):
        if self.bounds[1] == 255.0:
            new_img = image*255.0
            new_img = np.clip(new_img,0.0,255.0)
        else:
            new_img = np.clip(image,0.0,255.0)
            
        new_img = [new_img]
        
        x = tf.placeholder(tf.float32, (299, 299, 3))
        x_expanded = tf.expand_dims(x, axis=0)
        cropped_x = defend(x_expanded)
        cropped_logits, cropped_preds = self.model.model(self.sess, cropped_x)
        logits,label = self.sess.run([cropped_logits,cropped_preds],{cropped_x:new_img})
        
        return label[0]

        