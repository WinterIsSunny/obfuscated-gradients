#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:30:49 2018

@author: yusu
"""
import numpy as np
import tensorflow as tf 

class Model:
    def __init__(self,model, model_logit,sess,bounds):
        self.model = model
        self.model_logit = model_logit
        self.sess = sess
        self.bounds = bounds
        self.x_input = tf.placeholder(tf.float32, (None, 32, 32, 3))
        self.logits = self.model_logit(self.x_input)
    def predict(self,image):
        if self.bounds[1] == 255.0:
            new_img = image * 255.0
            new_img = np.clip(new_img,0.0,255.0)
        else:
            new_img = np.clip(image,-0.5,0.5)

        new_img = [new_img]
        
        label = np.argmax(self.sess.run(self.logits, {self.x_input:new_img}))
        return label
    