#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:19:23 2018

@author: yusu
"""

from __future__ import print_function

import os
import glob
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
    images = []
    for dirs in os.walk(path):
        for filename in dirs:
            images.append(filename)
#    files_path = glob.glob(os.path.join(path, '*.[jJ][pP][eE][gG]'))
    reader = tf.WholeFileReader()
#    images = []
#    for dir in path:
#        for image in dir:
#            images.append(image)
        
    
    if len(images) > 0:
        jpeg_file_queue = tf.train.string_input_producer(images)
        jkey, jvalue = reader.read(jpeg_file_queue)
        j_img = tf.image.decode_jpeg(jvalue)
    
    return j_img

def get_images(path,img_shape):
    image_list = read_raw_images(path)
    images= []
    labels = []
    for item in image_list:
        image,label = read_and_decode(item,img_shape)
        images.append(image)
        labels.append(label)
        
    return images,labels
        

def read_and_decode(filename, imshape, normalize=False, flatten=True):
    
    """Reads
    Args:
    filename_queue:
    imshape:
    normalize:
    flatten:
    Returns:
    """
    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64)
                    })
    
      # Convert from a scalar string tensor (whose single string has
      # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
      # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    
    if flatten:
        num_elements = 1
        for i in imshape: num_elements = num_elements * i
        print(num_elements)
        image = tf.reshape(image, [num_elements])
        image.set_shape(num_elements)
    else:
        image = tf.reshape(image, imshape)
        image.set_shape(imshape)
    
    if normalize:
        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(image, tf.float32)
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    
      # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)
    
    return image, label

  

        
#image_queue, length= read_raw_images("/data3/ILSVRC2012/train/")
#print("length of images:",length)
images,labels= get_images("/data3/ILSVRC2012/train/",(299, 299, 3))
#labels = read_labels("/data3/ILSVRC2012/")

print("type of images:",type(images))
print("type of labels: ",type(labels))
print("shape of images", tf.shape(images))
print("shape of image:", tf.shape(images[0]))

print("length of images:",len(images))
print("length of labels: ", len(labels))





