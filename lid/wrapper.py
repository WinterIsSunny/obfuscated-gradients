#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:30:49 2018

@author: yusu
"""
import numpy as np
import tensorflow as tf 

class Model:
    def __init__(self,model, model_logit,sess):
        self.model = model
        self.model_logit = model_logit
        self.sess = sess
        
    def predict(self,images):
        x_input = tf.placeholder(tf.float32, (None, 32, 32, 3))
        logits = self.model_logit(x_input)
        self.sess.run(logits, {x_input:images})
        self.logits = logits
        return np.argmax(logits)
    