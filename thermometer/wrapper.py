#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:48:18 2018

@author: yusu
"""
import tensorflow as tf
from discretization_utils import discretize_uniform
import numpy as np

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

        new_img = [new_img]
        levels = 16
        xs = tf.placeholder(tf.float32, (1, 32, 32, 3))
        encode = discretize_uniform(xs/255.0, levels=levels, thermometer=True)
        thermometer_encoded = self.sess.run(encode, {xs: new_img})
        
        return self.sess.run(self.model.predictions, {self.model.x_input: thermometer_encoded})