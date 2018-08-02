#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:19:23 2018

@author: yusu
"""

from __future__ import print_function

import os
#import glob
import csv
import tensorflow as tf
import numpy as np
#from PIL import Image
#from .utils import dense_to_one_hot


def read_raw_images(path):
    
    
    """Reads directory of images in tensorflow
    Args:
    path:
    is_directory:
    Returns:
    """
    
    reader = tf.WholeFileReader()
    images = []
    reader = tf.WholeFileReader()
    for dir in path:
        for image in dir:
            images.append(image)
        

      # Decode if there is a PNG file:
    
    if len(images) > 0:
        jpeg_file_queue = tf.train.string_input_producer(images)
        jkey, jvalue = reader.read(jpeg_file_queue)
        j_img = tf.image.decode_jpeg(jvalue)
    
    return j_img


def read_labels(path):
    
    dirname = path+"train.txt"
    with open(dirname) as d:
        label = d.readlines()
        labels = [data for data in label]
    
    return labels
        
    
images = read_raw_images("/data3/ILSVRC2012/train/")
labels = read_labels("/data3/ILSVRC2012/")

print("the type of images:",type(images))
print("the length of labels:",len(labels))
print("length of image:", len(images))
print("shape of image", np.array(images[0]).shape)