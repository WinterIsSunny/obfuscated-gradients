#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 17:13:04 2018

@author: yusu
"""

import tensorflow as tf
import numpy as np
import inceptionv3
from utils import *
#from defense import *
import os
from wrapper import MyModel
import torch
import time
import pandas as pd
import random
import foolbox
torch.set_printoptions(precision=10)

class blackbox:
    def __init__(self,model):
        self.model = model
        
    def attack_untargeted(self, x0, y0, alpha = 4, beta = 1e-4, iterations = 1000):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            alpha: learning rate 
            beta: learning rate
            train_dataset: set of training data
            (x0, y0): original image
        """
        pred = self.model.predict(x0)
        print("predicted label:", pred)
        print("true label:", y0)

        if (pred != y0):
            print("Fail to classify the image. No need to attack.")
            return x0,0
    
        num_directions = 5000
        best_theta, g_theta = None, float('inf')
        query_count = 0
        
        timestart = time.time()
        
        for i in range(num_directions):
            #print("iteration",i)
            theta = torch.randn(x0.shape).type(torch.FloatTensor)
            #print(theta.size())
            initial_lbd = torch.norm(theta)
            theta = theta/torch.norm(theta)
            if self.model.predict(x0+np.array(initial_lbd*theta)) != y0:
                query_count += 1 
                lbd, count = self.fine_grained_binary_search( x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta,lbd
#                    print("label for random direction:",self.model.predict(x0+np.array(g_theta*best_theta)))
                    print("--------> Found distortion %.4f in iteration %d" % (g_theta,i))
        timeend = time.time()
        print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))
        
        if g_theta > 100:
            print("this image is hard to attack")
            return x0,0
        
        #timestart = time.time()
        print("the best initialization: ",g_theta)
        g1 = 1.0
        theta, g2 = best_theta.clone(), g_theta
        torch.manual_seed(0)
        opt_count = 0
        stopping = 0.01
        prev_obj = 100000
        for i in range(iterations):
            print("iteration %d , g_theta %.4f" % (i,g_theta))

            if g_theta < 2:
                print("=========================> queries after distortion <2 :",opt_count+query_count)
                break
            
            gradient = torch.zeros(theta.size())
            q = 10
            min_g1 = float('inf')
            for j in range(q):
                u = torch.randn(theta.size()).type(torch.FloatTensor)
                u = u/torch.norm(u)
                ttt = theta+beta * u
                ttt = ttt/torch.norm(ttt)
                #print("inner loop iteration: ", j)
                g1, count = self.fine_grained_binary_search_local( x0, y0, ttt, initial_lbd = g2, tol=max(beta/10,1e-5))
                #print("g1 :",g1)
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
            

            for _ in range(5):
                new_theta = theta - alpha * gradient
                new_theta = new_theta/torch.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local( x0, y0, new_theta, initial_lbd = min_g2, tol=max(beta/10,1e-5))
                opt_count += count
                alpha = alpha * 2
#                print("alpha in the first for loop is: ",alpha)
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                else:
                    break
#            print("=============================================")
    
            if min_g2 >= g2:
                for _ in range(10):
                    alpha = alpha * 0.25
                    new_theta = theta - alpha * gradient
                    new_theta = new_theta/torch.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local( x0, y0, new_theta, initial_lbd = min_g2, tol=max(beta/10,1e-5))
                    opt_count += count
#                    print("alpha in the second for loop is: ",alpha)
                    if new_g2 < g2:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        break
#            print("=============================================")
            if min_g2 <= min_g1:
                theta, g2 = min_theta, min_g2
            else:
                theta, g2 = min_ttt, min_g1
    
            if g2 < g_theta:
                best_theta, g_theta = theta.clone(), g2
            

            if alpha < 1e-6:
                alpha = 1
                print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
                beta = beta * 0.1
                if (beta < 1e-6):
                    print("beta is too samll")
                    break

        print("inceptionv3")
        print("best distortion :", g_theta)
        print("number of queries :", opt_count+query_count)
        return x0 + np.array(g_theta*best_theta), opt_count+query_count
    def fine_grained_binary_search_local(self, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
        
        if self.model.predict(x0+np.array(lbd*theta)) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
#            timestart1 = time.time()
            while self.model.predict(x0+np.array(lbd_hi*theta)) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 100:
                    return float('inf'), nquery
#            timeend1 = time.time()
#            print("1st while time:", timeend1 - timestart1)
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
#            timestart2 = time.time()
            while self.model.predict(x0+ np.array(lbd_lo*theta)) != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1
#            timeend2 = time.time()
#            print("2nd while time:", timeend2 - timestart2)
            
#        timestart3 = time.time()
        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.model.predict(x0 + np.array(lbd_mid*theta)) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
#        timeend3 = time.time()
#        print("3rd while time:",timeend3 - timestart3)
#        print("lbd_low:",lbd_lo)
#        print("lbd_high:", lbd_hi)
#        print("-----------------------------")
        return lbd_hi, nquery
    
    
    def fine_grained_binary_search(self, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if self.model.predict(x0+ np.array(current_best*theta)) == y0:
                nquery += 1
                #print("initial_lbd > current_best & predict == y0, so return inf")
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.
    
        while (lbd_hi - lbd_lo) > 3*1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.model.predict(x0 + np.array(lbd_mid*theta)) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        #print("find a better initialization")
        return lbd_hi, nquery
    

###########test #############################################
sess = tf.Session()
torch.set_printoptions(precision=10)
# load images and lables
index = [1,972,429, 593,281,461,504,676,656,717,938, 812,365,84, 332,627,118,498,601,645]
images,labels = read_images("/data3/ILSVRC2012/train/","/data3/ILSVRC2012/train.txt",index)
model = MyModel(inceptionv3,sess,[0.0,1.0])
attack = blackbox(model)

compare = []
for i in range(len(images)):
    pred = model.predict(images[i])
    if pred == labels[i]:
        compare.append(1)
    else:
        compare.append(0)
print("accuracy of this model:", sum(compare)/len(compare))

dist = []
count = []
for i in range(15):
    print("================attacking image ",i+1,"=======================")
    adv,queries = attack.attack_untargeted(images[i+1],labels[i+1],alpha = 1, beta = 0.01, iterations = 1000)
    dist.append(np.linalg.norm(adv-images[i+1]))
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
    


