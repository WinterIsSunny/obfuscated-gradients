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
        
    def predict(self,image):
        image = [image]
        #xs = tf.placeholder(tf.float32, (1, 32, 32, 3))
        label = self.sess.run(self.model.predictions, {self.model.x_input: image})
        return label[0]
    
    def predict_logit(self,image):
        image = [image]
        logits = self.sess.run(self.model.pre_softmax, {self.model.x_input: image})
        return logits[0]
        
        