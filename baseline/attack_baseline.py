#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:23:24 2018

@author: yusu
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import CW
import utils
from models import PytorchModel
from allmodels import load_mnist_data
from torch.autograd import Variable

import cifar10_input
from cifar_model import Model

cifar = cifar10_input.CIFAR10Data("../cifar10_data")

sess = tf.Session()
model = Model("../models/standard/", tiny=False, mode='eval', sess=sess)
model = PytorchModel(model,[0,255],10)
attack = CW(model)

train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()
real_labels = []
adv_labels = []
count = 0
for i, (xi,yi) in enumerate(test_loader):
    xi_v=Variable(xi)
    adv = attack(xi,yi,False)
    adv_logits = model.predict(adv)
    new_label = np.argmax(adv_logits)
    real_labels.append(yi)
    if yi != new_label:
        count+=1
print("attack %f %:" % (count/len(test_loader)))
    
