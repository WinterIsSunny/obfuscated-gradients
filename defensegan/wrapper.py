#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:08:20 2018

@author: yusu
"""
import numpy as np

class Model:
    def __init__(self,model,bounds):
        self.model = model
        self.bounds = bounds
    
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
    
        
        