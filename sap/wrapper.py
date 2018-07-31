#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:21:41 2018

@author: yusu
"""

import tensorflow as tf

class Model:
    def __init__(self,model,sess):
        self.model = model
        self.sess = sess
        
    def predict(image):
        #xs = tf.placeholder(tf.float32, (1, 32, 32, 3))
        label = self.sess.run(self.model.predictions, {self.model.x_input: image})
        return label
    
    def predict_logit(image):
        logits = self.sess.run(self.model.pre_softmax, {self.model.x_input: image})
        
        