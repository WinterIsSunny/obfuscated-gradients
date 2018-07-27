#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:08:20 2018

@author: yusu
"""
import numpy as np

class Model:
    def __init__(self,model):
        self.model = model
    
    def predict(self,images):
        labels = self.model.predict(images)
        label = np.argmax(labels[0])
        print("the current label: ", label)
        return label
    
        
        