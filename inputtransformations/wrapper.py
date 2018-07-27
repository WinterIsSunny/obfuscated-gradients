#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:32:50 2018

@author: yusu
"""

#######3 model wrapper

import torch
import numpy as np
import tensorflow as tf


class MyModel:
    
    def __init__(self,model,sess):
        self.model = model
        self.sess = sess
    
    def predict(self,image):
        x = tf.placeholder(tf.float32, (299, 299, 3))
        x_expanded = tf.expand_dims(x, axis=0)
        logits, preds = self.model.model(self.sess, x_expanded)
        preds = self.sess.run([preds], {x: image})
        return preds[0]
        