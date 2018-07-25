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

import cifar10_input
from cifar_model import Model

cifar = cifar10_input.CIFAR10Data("../cifar10_data")

sess = tf.Session()
model = Model("../models/standard/", tiny=False, mode='eval', sess=sess)
model = PytorchModel(model,[0,255],10)
attack = CW(model)


image = cifar.eval_data.xs[:1]
label = cifar.eval_data.ys[:1]
print("original label is:",label)
adversarial = attack(image,label,False)

new_logits = model(adversarial)
sess = tf.InteractiveSession()
new_logits = new_logits.eval()
new_label = np.argmax(new_logits)
print("new label is :", new_label)

