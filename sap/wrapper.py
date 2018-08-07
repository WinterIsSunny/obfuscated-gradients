#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:21:41 2018

@author: yusu
"""

import tensorflow as tf
import numpy as np

class Model:
    def __init__(self,model,sess,bounds):
        self.bounds = bounds
        self.model = model
        self.sess = sess
        
    def sub_predict(self,new_img):
        new_img = [new_img]
        #xs = tf.placeholder(tf.float32, (1, 32, 32, 3))
        label = self.sess.run(self.model.predictions, {self.model.x_input: new_img})
        return label[0]
    
    def predict(self,image,y0):
        if self.bounds[1] == 255.0:
            new_img = image * 255.0
            new_img = np.clip(new_img,0.0,255.0)
        else:
            new_img = np.clip(image,0.0,1.0)
            
        labels = []
        for i in range(10):
            label = self.sub_predict(new_img)
            labels.append(label)
        if y0 in labels:
#            print(labels)
            return y0
        else:
#            print(labels)
            return labels[0]
            
            
    
    def predict_logit(self,image):
        if self.bounds[1] == 255.0:
            new_img = image * 255.0
            new_img = np.clip(new_img,0.0,255.0)
        else:
            new_img = np.clip(image,0.0,1.0)

        new_img = [new_img]

        logits = self.sess.run(self.model.pre_softmax, {self.model.x_input: new_img})
        return logits[0]
        
