#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:32:50 2018

@author: yusu
"""

#######3 model wrapper

import torch
import numpy as np


class MyModel:
    
    def __init__(self,model,sess):
        self.model = model
        self.sess = sess
    
    def predict(image):
        logits, label = self.model.model(self.sess,image)
        
        return label
        