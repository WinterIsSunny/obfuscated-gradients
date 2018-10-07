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

class blackbox:
    def __init__(self,model):
        self.model = model
        
    def attack_untargeted(self, x0, y0, alpha = 8, beta = 0.005, iterations = 1000):
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

#        if (self.model.predict(x0) != y0):
#            print("Fail to classify the image. No need to attack.")
#            return x0
    
        num_directions = 100
        best_theta, g_theta = None, float('inf')
        query_count = 0
        
        #timestart = time.time()
        
        
        for i in range(num_directions):
            theta = torch.randn(x0.shape).type(torch.FloatTensor)
            #print(theta.size())
            initial_lbd = torch.norm(theta)
            theta = theta/torch.norm(theta)
            if self.model.predict(x0+np.array(initial_lbd*theta)) != y0:
                query_count += 1 
                #print(type(theta),type(initial_lbd),type(g_theta))
                #print("find a new adv direction, new label:", self.model.predict(x0+np.array(initial_lbd*theta)))
                lbd, count = self.fine_grained_binary_search( x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta,lbd
#                    print("label for random direction:",self.model.predict(x0+np.array(g_theta*best_theta)))
                    print("--------> Found distortion %.4f" % g_theta)
        
            #timeend = time.time()
            #print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))
        
        
        
        
        
        #timestart = time.time()
        print("the best initialization: ",g_theta)
        g1 = 1.0
        theta, g2 = best_theta.clone(), g_theta
        torch.manual_seed(0)
        opt_count = 0
        stopping = 0.01
        prev_obj = 100000
        for i in range(iterations):
            
           # print("iteration:",i)
            if g_theta < 0.005:
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
                g1, count = self.fine_grained_binary_search_local( x0, y0, ttt, initial_lbd = g2, tol=beta/500)
                #print("g1 :",g1)
                opt_count += count
                gradient += (g1-g2)/beta * u
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt
            gradient = 1.0/q * gradient
#            print("=============================================")
    
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
#            print("=============================================")
    
            if min_g2 >= g2:
                for _ in range(15):
                    alpha = alpha * 0.5
                    new_theta = theta - alpha * gradient
                    new_theta = new_theta/torch.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local( x0, y0, new_theta, initial_lbd = min_g2, tol=beta/50)
                    opt_count += count
                    print("alpha in the second for loop is: ",alpha)
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
                    print("beta is too samll")
                    break
            print("distortion in this iteration:", g_theta)
            print("=-=-=-=-=-=-=-=-=-=-=-=-will enter next iteration=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    
        #target = model.predict(x0 + g_theta*best_theta)
        
        #print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
        print("inceptionv3")
        print("best distortion :", g_theta)
        print("number of queries :", opt_count+query_count)
        return x0 + np.array(g_theta*best_theta)
    def fine_grained_binary_search_local(self, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
        
        if self.model.predict(x0+np.array(lbd*theta)) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            timestart1 = time.time()
            while self.model.predict(x0+np.array(lbd_hi*theta)) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
            timeend1 = time.time()
#            print("1st while time:", timeend1 - timestart1)
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            timestart2 = time.time()
            while self.model.predict(x0+ np.array(lbd_lo*theta)) != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1
            timeend2 = time.time()
#            print("2nd while time:", timeend2 - timestart2)
            
        timestart3 = time.time()
        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.model.predict(x0 + np.array(lbd_mid*theta)) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        timeend3 = time.time()
#        print("3rd while time:",timeend3 - timestart3)
#        print("lbd_low:",lbd_lo)
#        print("lbd_high:", lbd_hi)
        print("-----------------------------")
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
        lbd_lo = 0.
        print("label before fine binary search:", self.model.predict(x0+ np.array(lbd_hi*theta)))
    
        while (lbd_hi - lbd_lo) > 1e-3:
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

mypath = os.path.join("data/n01440764")
print("path of images:", mypath)
#files = [load_image(img) for img in os.listdir(mypath)]
files = []
for file in os.listdir(mypath):
    print("this path is correct")
    files.append(load_image(file))
    
images = files[:100]
images = images/255.0



#orig = load_image('cat.jpg')
#image = orig.copy()/255.0
#x = tf.placeholder(tf.float32, (299, 299, 3))
#x_expanded = tf.expand_dims(x, axis=0)
#x_pred = defend(x_expanded)
#logits, preds = inceptionv3.model(sess, x_pred)
#print("prediction :",preds)

#print("Before loading model")
model = MyModel(inceptionv3,sess,[0.0,255.0])


#print("after loading model")
attack = blackbox(model)
#label = 287
y0 = model.predict(images[0])
print("predict pure image:",y0)
#print("predict pure image:", y1)
adv = attack.attack_untargeted(images[0],y0,alpha = 8, beta = 0.005)


print("new label:", model.predict(adv))


