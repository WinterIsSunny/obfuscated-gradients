#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:48:18 2018

@author: yusu
"""
import tensorflow as tf
from discretization_utils import discretize_uniform


class MyModel:
    def __init__(self,model,sess):
        self.model = model
        self.sess = sess
        
    def predict(self,image):
        image = [image]
        levels = 16
        xs = tf.placeholder(tf.float32, (1, 32, 32, 3))
        encode = discretize_uniform(xs/255.0, levels=levels, thermometer=True)
        thermometer_encoded = self.sess.run(encode, {xs: image})
        label = self.sess.run(self.model.predictions, {self.model.x_input: thermometer_encoded})
        return label[0]