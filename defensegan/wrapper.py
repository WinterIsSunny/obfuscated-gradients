#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:08:20 2018

@author: yusu
"""
import numpy as np
import tensorflow as tf

class Model:
    def __init__(self,model,bounds,sess,gan):
        self.model = model
        self.bounds = bounds
        self.gan = gan
        self.sess = sess
    
    def predict_gan(self,mod,x0):
        modifier = np.expand_dims(np.array(mod),0)
        x_new = tf.placeholder(tf.float32,(1,128))
        mod_gan = self.gan(x_new)
        self.sess.run(mod_gan,{x_new,modifier})
        with self.sess.as_default():
            mod = mod_gan.eval()
        mod = np.sum(mod,0)
        print("shape of mod:", mod.shape)
        print("shape of x0:", x0.shape)
        pred = self.predict(mod+x0)
        return pred

        
    def predict(self,image):
        if self.bounds[1] == 255.0:
            new_img = image*255.0
            new_img = np.clip(new_img,0.0,255.0)
        else:
            new_img = np.clip(image,0.0,255.0)
            
        new_img= np.expand_dims(new_img,0)
        
        #print("shape of new_img ",new_img.shape)
        labels = self.model.predict(new_img)
        #print(labels)
        label = np.argmax(labels[0])
        #print("the current label: ", label)
        return label
    
        
        