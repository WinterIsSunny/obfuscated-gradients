#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 11:19:13 2018

@author: yusu
"""

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
#%matplotlib inline

import cifar10_input

from discretization_utils import one_hot_to_thermometer
from discretization_utils import discretize_uniform
from discretization_attacks import adv_lspga

from cifar_model import Model
import cifar10_input

import torch
from wrapper import MyModel
import time

class blackbox:
    def __init__(self,model):
        self.model = model
        
    def attack_untargeted(self,model, x0, y0, alpha = 0.2, beta = 0.001, iterations = 1000):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        #print(x0.type())
        #print(y0.type())
        #print(model.predict(x0).type())
        y0 = y0.cuda()
        print(y0.type())
        
        if (self.model.predict(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return x0
    
        num_directions = 1000
        best_theta, g_theta = None, float('inf')
        query_count = 0
        
        timestart = time.time()
        for i in range(num_directions):
            theta = torch.randn(x0.shape).type(torch.FloatTensor)
            #print(theta.size())
            initial_lbd = torch.norm(theta)
            theta = theta/torch.norm(theta)
            lbd, count = self.fine_grained_binary_search(x0, y0, theta, initial_lbd, g_theta)
            query_count += count
            if lbd < g_theta:
                best_theta, g_theta = theta,lbd
                print("--------> Found distortion %.4f" % g_theta)
    
        timeend = time.time()
        print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))
    
        
        
        timestart = time.time()
        g1 = 1.0
        theta, g2 = best_theta.clone(), g_theta
        torch.manual_seed(0)
        opt_count = 0
        stopping = 0.01
        prev_obj = 100000
        for i in range(iterations):
            gradient = torch.zeros(theta.shape)
            q = 10
            min_g1 = float('inf')
            for _ in range(q):
                u = torch.randn(theta.shape).type(torch.FloatTensor)
                u = u/torch.norm(u)
                ttt = theta+beta * u
                ttt = ttt/torch.norm(ttt)
                g1, count = self.fine_grained_binary_search_local(self.model, x0, y0, ttt, initial_lbd = g2, tol=beta/500)
                opt_count += count
                gradient += (g1-g2)/beta * u
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt
            gradient = 1.0/q * gradient
    
            if (i+1)%50 == 0:
                print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, torch.norm(g2*theta), opt_count))
                if g2 > prev_obj-stopping:
                    break
                prev_obj = g2
    
            min_theta = theta
            min_g2 = g2
        
            for _ in range(15):
                new_theta = theta - alpha * gradient
                new_theta = new_theta/torch.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local(x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                opt_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                else:
                    break
    
            if min_g2 >= g2:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = theta - alpha * gradient
                    new_theta = new_theta/torch.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local(x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                    opt_count += count
                    if new_g2 < g2:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        break
    
            if min_g2 <= min_g1:
                theta, g2 = min_theta, min_g2
            else:
                theta, g2 = min_ttt, min_g1
    
            if g2 < g_theta:
                best_theta, g_theta = theta.clone(), g2
            
            #print(alpha)
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
                beta = beta * 0.1
                if (beta < 0.0005):
                    break
    
        target = self.model.predict(x0 + g_theta*best_theta)
        timeend = time.time()
        print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
        return g_theta*best_theta, query_count+opt_count
    def fine_grained_binary_search_local(self,x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
         
        if self.model.predict(x0+lbd*theta) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while self.model.predict(x0+lbd_hi*theta) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while self.model.predict(x0+lbd_lo*theta) != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1
    
        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.model.predict(x0 + lbd_mid*theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery
    
    def fine_grained_binary_search(self, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if self.model.predict(x0+current_best*theta) == y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd
        
        ## original version
        #lbd = initial_lbd
        #while model.predict(x0 + lbd*theta) == y0:
        #    lbd *= 2
        #    nquery += 1
        #    if lbd > 100:
        #        return float('inf'), nquery
        
        #num_intervals = 100
    
        # lambdas = np.linspace(0.0, lbd, num_intervals)[1:]
        # lbd_hi = lbd
        # lbd_hi_index = 0
        # for i, lbd in enumerate(lambdas):
        #     nquery += 1
        #     if model.predict(x0 + lbd*theta) != y0:
        #         lbd_hi = lbd
        #         lbd_hi_index = i
        #         break
    
        # lbd_lo = lambdas[lbd_hi_index - 1]
        lbd_hi = lbd
        lbd_lo = 0.0
    
        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.model.predict(x0 + lbd_mid*theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

levels = 16

sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
cifar = cifar10_input.CIFAR10Data()
model = Model('../models/thermometer_advtrain/',
              sess, tiny=False, mode='eval',
              thermometer=True, levels=levels)
model = MyModel(model,sess,levels,[0.0,255.0])

image = np.array(cifar.eval_data.xs[:100],dtype=np.float32)
labels = cifar.eval_data.ys[:100]

new_img = image / 255.0

count = []
for i in range(20):
    label = model.predict(new_img[i])
    if label == labels[i]:
        count.append(1)
    else:
        count.append(0)
    
print("accuracy of this model is:", sum(count)/len(count))


attack = blackbox(model)
dist = []
count = []
for i in range(10):
    print("=============================== this is image ",i+1,"========================================")
    mod,queries = attack.attack_untargeted(new_img[i+4],labels[i+4],alpha = 4, beta = 0.05, iterations = 1000)
    dist.append(np.linalg.norm(mod))
    count.append(queries)
    
    
index = np.nonzero(count)
index = list(index)[0].tolist()

avg_distortion = np.mean(np.array(dist)[index])
avg_count = np.mean(np.array(count)[index])
print("the average distortion for %2d images :"%(len(index)),avg_distortion)
print("the distortions for 15 images :")
for i in dist:
    print(i)
    
print("the number of queries for %2d images :"%(len(index)), avg_count)    
print("the number of queries for 15 images :")
for j in count:
    print(j)


