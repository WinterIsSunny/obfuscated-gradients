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
    
    def __init__(self,model,sess,bounds):
        self.model = model
        self.sess = sess
        self.bounds = bounds
    
    def predict(self,image):
        if self.bounds[1] == 255.0:
            new_img = image * 255.0
            new_img = np.clip(new_img,0.0,255.0)
        else:
            new_img = np.clip(image,0.0,1.0)

        x = tf.placeholder(tf.float32, (299, 299, 3))
        x_expanded = tf.expand_dims(x, axis=0)
        logits, preds = self.model.model(self.sess, x_expanded)
        preds = self.sess.run([preds], {x: new_img})
        print("type of preds[0] is: ",type(preds[0]))
        return preds[0]
        