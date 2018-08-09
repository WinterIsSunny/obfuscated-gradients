#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 15:59:45 2018

@author: yusu
"""
import os
import numpy as np

file_name = os.path.join("data/lid_cifar_blackbox.npy")
data = np.load(file_name)
print("shape of this file:", data.shape)
print("shape of one item:",data[0].shape)
print("what is saved in this file:", data)










