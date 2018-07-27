#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:08:20 2018

@author: yusu
"""

class Model:
    def __init__(self,model):
        self.model = model
    
    def predict(self,images):
        labels = self.model.predict(images)
        return labels[0]
    
        
        