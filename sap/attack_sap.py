#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 16:48:18 2018

@author: yusu
"""
### SAP

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import cifar10_input
from sap_model import SAPModel


import torch

from wrapper import PyModel


class blackbox:
    def __init__(self,model):
        self.model = model
        
    def attack_untargeted(self, x0, y0, alpha = 2, beta = 0.005, iterations = 1000):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            alpha: learning rate 
            beta: learning rate
            train_dataset: set of training data
            (x0, y0): original image
        """

        if (self.model.single_predict(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return x0, 0
        
        num_directions = 1000
        num_query = 10
        best_theta, g_theta = None, float('inf')
        query_count = 0
        comp_theta = 0
        current_best = float('inf')
        
        timestart = time.time()
        for i in range(num_directions):
            theta = torch.randn(x0.shape).type(torch.FloatTensor)
            initial_lbd = torch.norm(theta)
            theta = theta/torch.norm(theta)
            if self.model.predict(x0+np.array(initial_lbd*theta)) != y0:
                lbd,comp_dec,count = self.fine_grained_binary_search_fix(x0,y0,theta,initial_lbd,g_theta,current_best,num_query)
                query_count += count
                if comp_dec > comp_theta:
                    comp_theta = comp_dec
                    best_theta,g_theta = theta,lbd
                    print("--------> Found abs-distortion %.4f with 10 queries" % g_theta)
                    print("--------> Found comp-distortion %.4f with 10 queries" % comp_dec)
            timeend = time.time()
        print("==========> Found best distortion %.4f in %.4f seconds" % (g_theta, timeend-timestart))
        #print("type of best_theta", type(best_theta))
        #print("type of best_theta", type(g_theta))
        lbd,count = self.fine_grained_binary_search( x0, y0, best_theta, g_theta, current_best)
        g_theta = lbd
        query_count += count


        # num_directions = 1000
        # best_theta, g_theta = None, float('inf')
        # query_count = 0
        
        # #timestart = time.time()
        
        # for i in range(num_directions):
        #     theta = torch.randn(x0.shape).type(torch.FloatTensor)
        #     #print(theta.size())
        #     initial_lbd = torch.norm(theta)
        #     theta = theta/torch.norm(theta)

        #     if  self.model.predict(x0+np.array(initial_lbd*theta),y0) != y0:
        #         lbd, count = self.fine_grained_binary_search( x0, y0, theta, initial_lbd, g_theta)
        #         query_count += count
        #         if lbd < g_theta:
        #             best_theta, g_theta = theta,lbd
        #             print("new g_theta :", g_theta,"***")
        #             print("label for random direction:",self.model.predict(x0+np.array(g_theta*best_theta),y0))
        #             print("norm of theta*lbd 4:", np.linalg.norm(x0+np.array(g_theta*best_theta)))
        #             print("******")
        #             print("--------> Found distortion %.4f" % g_theta)
        
        #     #timeend = time.time()
        #     #print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))
        
        
        
        #timestart = time.time()
        print("the best initialization: ",g_theta)
        g1 = 1.0
        theta, g2 = best_theta.clone(), g_theta
        torch.manual_seed(0)
        opt_count = 0
        stopping = 0.01
        prev_obj = 100000
        for i in range(iterations):
            
            print("iteration:",i)
            if g_theta < 1.2:
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
                g1, count = self.fine_grained_binary_search_local( x0, y0, ttt, initial_lbd = g2, tol=beta/50)
                #print("g1 :",g1)
                opt_count += count
                gradient += (g1-g2)/beta * u
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt
            gradient = 1.0/q * gradient
            #print("=============================================")
    
            if (i+1)%50 == 0:
                
                print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, torch.norm(g2*theta), opt_count))
                if g2 > prev_obj-stopping:
                    break
                prev_obj = g2
    
            min_theta = theta
            min_g2 = g2
            
            #print("gradient:", gradient)
           # print("theta:",theta)
            for _ in range(15):
                new_theta = theta - alpha * gradient
                new_theta = new_theta/torch.norm(new_theta)
                
                new_g2, count = self.fine_grained_binary_search_local( x0, y0, new_theta, initial_lbd = min_g2, tol=beta/50)
                opt_count += count
                alpha = alpha * 2
                print("alpha in the first for loop is: ",alpha)
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                else:
                    break
            #print("=============================================")
    
            if min_g2 >= g2:
                for _ in range(15):
                    alpha = alpha * 0.9
                    new_theta = theta - alpha * gradient
                    new_theta = new_theta/torch.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local( x0, y0, new_theta, initial_lbd = min_g2, tol=beta/50)
                    opt_count += count
                    print("alpha in the second for loop is: ",alpha)
                    if new_g2 < g2:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        break
            #print("=============================================")
            if min_g2 <= min_g1:
                theta, g2 = min_theta, min_g2
            else:
                theta, g2 = min_ttt, min_g1
    
            if g2 < g_theta:
                best_theta, g_theta = theta.clone(), g2
            
            #print(alpha)
            print("%3d th iteration" % i)
            print("current alpha:",alpha)
            print("g_theta")
            print("number of queries:", opt_count+query_count)
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
                beta = beta * 0.1
                if (beta < 0.0005):
                    break
            #print("new label in this iteration:", self.model.single_predict(x0 + np.array(g_theta*best_theta)))
            #print("=-=-=-=-=-=-=-=-=-=-=-=-will enter next iteration=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    
        #target = model.predict(x0 + g_theta*best_theta)
        
        #print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
        print("lid")
        print("best distortion :", g_theta)
        print("number of queries :", opt_count+query_count)
        return np.array(g_theta*best_theta),opt_count+query_count
    def fine_grained_binary_search_local(self, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
        
        if self.model.predict(x0+np.array(lbd*theta),y0) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            #timestart1 = time.time()
            while self.model.predict(x0+np.array(lbd_hi*theta),y0) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
            #timeend1 = time.time()
            #print("1st while time:", timeend1 - timestart1)
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            #timestart2 = time.time()
            while self.model.predict(x0+ np.array(lbd_lo*theta),y0) != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1
            #timeend2 = time.time()
            #print("2nd while time:", timeend2 - timestart2)
            
        #timestart3 = time.time()
        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.model.predict(x0 + np.array(lbd_mid*theta),y0) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        #timeend3 = time.time()
        #print("3rd while time:",timeend3 - timestart3)
        print("lbd_low:",lbd_lo)
        print("lbd_high:", lbd_hi)
        print("-----------------------------")
        return lbd_hi, nquery
        
    def fine_grained_binary_search(self, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
#            tmp_lb = self.model.predict(x0+ np.array(current_best*theta),y0)
            if self.model.predict(x0+ np.array(current_best*theta),y0)  == y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
            print("assign lbd = current_best, lbd = ",lbd,"***")
            print("after assigning lbd = current_best, label :",self.model.predict(x0+ np.array(lbd*theta),y0))
            print("norm of adv 1:", np.linalg.norm(x0+ np.array(lbd*theta)))
            print("******")
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
        print("assign lbd_hi = lbd,  lbd_hi = ",lbd_hi,"***")
        print("label before fine binary search:", self.model.predict(x0+ np.array(lbd_hi*theta),y0))
        print("norm of lbd_hi*theta 2:", np.linalg.norm(x0+ np.array(lbd*theta)))
        print("******")
    
        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.model.predict(x0 + np.array(lbd_mid*theta),y0) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        print("after binary search: lbd_ih:", lbd_hi,"***")
#        tmp_lb = self.model.predict(x0+ np.array(lbd_hi*theta),y0)
        print("label after fine binary search:",self.model.predict(x0+ np.array(lbd_hi*theta),y0))
        print("again, label after fine binary search:", self.model.predict(x0+ np.array(lbd_hi*theta),y0))
        print("norm of lbd_hi*theta 3:", np.linalg.norm(x0+ np.array(lbd_hi*theta)))
        print("******")
        return lbd_hi, nquery
    

# ============ test ====================
sess = tf.Session()

cifar = cifar10_input.CIFAR10Data("/content/drive/My Drive/STAT141C/cifar10_data")
orig_model = Model("/content/drive/My Drive/STAT141C/models/standard/", tiny=False, mode='eval', sess=sess)

model = Model(model,sess,[0.0,255.0])

attack = blackbox(model)

#xs = tf.placeholder(tf.float32, (1, 32, 32, 3))
image = cifar.eval_data.xs[:15]
label = cifar.eval_data.ys[:15]
new_img = image/255.0

#print("original label is :", label[1])
#print(len(image))
#print("label of clean image:", model.predict(new_img[1],label[1]))

dist = []
count = []
for i in range(10):
    print("================attacking image ",i,"=======================")
    mod,queries = attack.attack_untargeted(image[i],label[i],alpha = 4, beta = 0.005, iterations = 1000)
    dist.append(np.linalg.norm(mod))
    count.append(queries)

#========================================================

index = np.nonzero(count)
index= list(index)[0].tolist()


avg_distortion = np.mean(np.array(dist)[index])
avg_count = np.mean(np.array(count)[index])
print("the average distortion for %2d images :"%(len(index)),avg_distortion)
for i in dist:
    print(i)
print("the number of queries for %2d images :"%(len(index)), avg_count)
for j in count:
    print(j)










