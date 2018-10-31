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

#from CW import CW
import utils
from models import PyModel
from allmodels import load_mnist_data
from torch.autograd import Variable
import torch

import cifar10_input
from cifar_model import Model
import foolbox

import time

class blackbox:
    def __init__(self,model):
        self.model = model
        
    def attack_untargeted(self, x0, y0,init, alpha = 2, beta = 0.005, iterations = 1000):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            alpha: learning rate 
            beta: learning rate
            train_dataset: set of training data
            (x0, y0): original image
        """
        
        if (self.model.predict_label(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return np.nan
    
#        num_directions = 500
#        best_theta, g_theta = None, float('inf')
#        query_count = 0
#        
#        #timestart = time.time()
#        for i in range(num_directions):
#            theta = torch.randn(x0.shape).type(torch.FloatTensor)
#            #print(theta.size())
#            initial_lbd = torch.norm(theta)
#            theta = theta/torch.norm(theta)
#            #theta *= 255
#            if self.model.predict_label(x0+np.array(initial_lbd*theta)) != y0:
#                lbd, count = self.fine_grained_binary_search( x0, y0, theta, initial_lbd, g_theta)
#                query_count += count
#                if lbd < g_theta:
#                    best_theta, g_theta = theta,lbd
#                    print("--------> Found distortion %.4f" % g_theta)
            #timeend = time.time()
            #print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))
        
            
        
        
        theta =torch.tensor(init)
        g_theta = torch.norm(theta)
        best_theta = theta/g_theta
        #timestart = time.time()
        #print("the best initialization: ",best_theta)
        g1 = 1.0
        theta, g2 = best_theta.clone(), g_theta
        torch.manual_seed(0)
        opt_count = 0
        stopping = 0.01
        prev_obj = 100000
        
        for i in range(iterations):
            if g_theta < 1:
                print("====================query number after distortion < 1 =======================: ",opt_count)
                break
            #print("n_query:",opt_count)
            #print("best distortion:", g_theta)
            #print("iteration:", i )
            gradient = torch.zeros(theta.size())
            q = 10
            min_g1 = float('inf')
            for _ in range(q):
                u = torch.randn(theta.size()).type(torch.FloatTensor)
                u = u/torch.norm(u)
                ttt = theta+beta * u
                ttt = ttt/torch.norm(ttt)
                g1, count = self.fine_grained_binary_search_local( x0, y0, ttt, initial_lbd = g2, tol=beta/500)
                opt_count += count
                gradient +=  (g1-g2)/beta * u
                #print("norm of gradient:", np.linalg.norm(gradient))
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
                #print("enter first for loop")
                new_theta = theta - alpha * gradient
                new_theta = new_theta/torch.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local( x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                opt_count += count
                alpha = alpha * 2
                #print("alpha in the first for loop is: ",alpha)
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                else:
                    break
    
            if min_g2 >= g2:
                for _ in range(15):
                    #print("enter second for loop")
                    alpha = alpha * 0.95
                    new_theta = theta - alpha * gradient
                    new_theta = new_theta/torch.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local( x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                    opt_count += count
                    #print("alpha in the second for loop is: ",alpha)
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
#            print("%3d th iteration" % i)
            #print("current alpha:",alpha)
            if alpha < 1e-4:
                alpha = 1.0
                #print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
                beta = beta * 0.1
                if (beta < 0.0005):
                    break
    
        #target = model.predict(x0 + g_theta*best_theta)
        
        #timeend = time.time()
        #print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
        print("baseline")
        print("best distortion :", g_theta)
        print("number of queries :", opt_count)
        return np.array(g_theta*best_theta), opt_count
    def fine_grained_binary_search_local(self, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
         
        if self.model.predict_label(x0+np.array(lbd*theta)) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while self.model.predict_label(x0+np.array(lbd_hi*theta)) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while self.model.predict_label(x0+np.array(lbd_lo*theta)) != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1
    
        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.model.predict_label(x0 + np.array(lbd_mid*theta)) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery
    
    def fine_grained_binary_search(self, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if self.model.predict_label(x0+ np.array(current_best*theta)) == y0:
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
            if self.model.predict_label(x0 + np.array(lbd_mid*theta)) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery
    
    

cifar = cifar10_input.CIFAR10Data("../cifar10_data")

sess = tf.Session()
model = Model("../models/standard/", tiny=False, mode='eval', sess=sess)
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
logits = model(x)
fool_model = foolbox.models.TensorFlowModel(x,logits,(0,255))
fool_attack = foolbox.attacks.BoundaryAttack(fool_model)

model = PyModel(model,sess,[0.0,255.0])


images = cifar.eval_data.xs[:1000]
labels = cifar.eval_data.ys[:1000]

#count = 0
#pre_labs = []
#for i in range(100):
##    pre_lab = model.predict_label(images[i])
#    new_img = images[i] / 255.0
#    image = np.clip(new_img*255.0,0.0,255.0)
#    pre_lab = sess.run(model.predictions, {model.x_input: [image]})[0]
#    pre_labs.append(pre_lab)
#    if labels[i] == pre_lab: 
#        count+=1
#print("original label:", labels[:100])
#print("predicted labels", pre_labs)
#print("accuracy of 100 images :", count/100)
#    
#count = 0
#for i in range(1000):
#    pre_lab = model.predict_label(images[i])
#    pre_labs.append(pre_lab)
#    if labels[i] == pre_lab: 
#        count+=1
#
#print("accuracy of 1000 images :", count/1000)
#    





# ==============================================


image = cifar.eval_data.xs[:100]# np.array
test_img = image / 255.0

label = cifar.eval_data.ys[:100]

attack = blackbox(model)

print("original label is:",label)
timestart = time.time()
print("predicted label on clean data is: ", model.predict_label(test_img[0]))
timeend = time.time()
print("time consuming: ", timeend-timestart)

dist = []
count = []
for i in range(20):
    print("================attacking image ",i+1,"=======================")
    init_op = tf.global_initialize_all_variables()
    sess.run(init_op)
    new_img = fool_attack(test_img[i],label[i])
    init_dir = new_img - test_img[i]
    mod,queries = attack.attack_untargeted(test_img[i],label[i],init_dir,alpha = 2, beta = 0.05, iterations = 1000)
    dist.append(np.linalg.norm(mod))
    count.append(queries)

#index1 = np.nonzero(dist)
#index1= list(index1)[0].tolist()
#
#index2 = np.nonzero(count)
#index2 = list(index2)[0].tolist()


#avg_distortion = np.mean(np.array(dist)[index1])
#avg_distortion = np.mean(np.array(count)[index2])
print("the distortions for 15 images :")
for i in dist:
    print(i)
print("the number of queries for 15 images :")
for j in count:
    print(j)
#print("the average distortion of 10 images is:", avg_distortion)
#print("the average queries of 10 images:", avg_distortion)

#new_logits = model.predict(adv)
#new_label = model.predict_label(adv)
#sess = tf.InteractiveSession()
#new_logits = new_logits.eval()
#new_label = np.argmax(new_logits)
#print("new label is :", new_label)


