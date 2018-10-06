#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creaated on Thu Jul 26 15:20:02 2018

@author: yusu
"""

from wrapper_jpeg import MyModel
import tensorflow as tf
import numpy as np

from torch.autograd import Variable
import torch

import inceptionv3
from utils import *
from defense import *
from get_image import *
import time

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

        if (self.model.predict(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return x0
    
        num_directions = 1000
        best_theta, g_theta = None, float('inf')
        query_count = 0
        
        #timestart = time.time()
        
        
        for i in range(num_directions):
            theta = torch.randn(x0.shape).type(torch.FloatTensor)
            #print(theta.size())
            if self.model.predict(x0+np.array(theta)) != y0:
                initial_lbd = torch.norm(theta)
                theta = theta/torch.norm(theta)
                lbd, count = self.fine_grained_binary_search( x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta,lbd
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
            if g_theta < 3e-06:
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
#                print("alpha in the first for loop is: ",alpha)
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                else:
                    break
#            print("=============================================")
    
            if min_g2 >= g2:
                for _ in range(15):
                    alpha = alpha * 0.9
                    new_theta = theta - alpha * gradient
                    new_theta = new_theta/torch.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local( x0, y0, new_theta, initial_lbd = min_g2, tol=beta/50)
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
            
#            
#            print("%3d th iteration" % i)
#            print("current alpha:",alpha)
#            print("g_theta")
#            print("number of queries:", opt_count+query_count)
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
                beta = beta * 0.1
                if (beta < 0.0000005):
                    break
            print("new label in this iteration:", self.model.predict(x0+np.array(best_theta*g_theta)))
            print("=-=-=-=-=-=-=-=-=-=-=-=-will enter next iteration=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    
        #target = model.predict(x0 + g_theta*best_theta)
        
        #print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
        print("inputtransformation_jpeg")
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
#            timestart1 = time.time()
            while self.model.predict(x0+np.array(lbd_hi*theta)) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
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
            #print("size of image:",x0.shape)
            #print("size of modifier,",np.array(lbd_mid*theta).shape )
            if self.model.predict(x0 + np.array(lbd_mid*theta)) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

        
#    def attack_targeted(self, x0, y0, x_t,target, alpha = 0.1, beta = 0.001, iterations = 1000):
#        """ Attack the original image and return adversarial example of target t
#            model: (pytorch model)
#            train_dataset: set of training data
#            (x0, y0): original image
#            t: target
#        """
#        
#        # initialize the direction 
#        if (self.model.predict(x0)[0] != y0):
#            print("Fail to classify the image. No need to attack.")
#            return x0
#    
#        # STEP I: find initial direction (theta, g_theta)
#        
#        best_theta, g_theta = None, float('inf')
#        query_count = 0
#        
#        
#        if model.predict(x_t) == target:
#            theta = x_t - x0
#            initial_lbd = torch.norm(theta)
#            theta = theta/torch.norm(theta)
#            theta *= 255
#            lbd, count = self.fine_grained_binary_search_targeted(model, x0, y0, target, theta, initial_lbd)
#            query_count += count
#            if lbd < g_theta:
#                best_theta, g_theta = theta, lbd
#                print("--------> Found distortion %.4f" % g_theta)
#    
#    
#        #print("Searching for the initial direction on %d samples: " % (num_samples))
#        #timestart = time.time()
#        
##        samples = set(random.sample(range(len(train_dataset)), num_samples))
#        
#        ## use targeted picture to attain the initial direction !! 
#    
#       
#        #timeend = time.time()
#        #print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))
#    
#    
#        # STEP II: seach for optimal
#        #timestart = time.time()
#    
#        g1 = 1.0
#        theta, g2 = best_theta.clone(), g_theta
#        opt_count = 0
#    
#        for i in range(iterations):
#            gradient = torch.zeros(theta.size())
#            q = 10
#            min_g1 = float('inf')
#            for _ in range(q):
#                u = torch.randn(theta.size()).type(torch.FloatTensor)
#                u = u/torch.norm(u)
#                ttt = theta+beta * u
#                ttt = ttt/torch.norm(ttt)
#                g1, count = self.fine_grained_binary_search_local_targeted( x0, y0, target, ttt, initial_lbd = g2, tol=beta/500)
#                opt_count += count
#                gradient += (g1-g2)/beta * u
#                if g1 < min_g1:
#                    min_g1 = g1
#                    min_ttt = ttt
#            gradient = 1.0/q * gradient
#    
#            if (i+1)%50 == 0:
#                print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, torch.norm(g2*theta), opt_count))
#    
#            min_theta = theta
#            min_g2 = g2
#        
#            for _ in range(15):
#                new_theta = theta - alpha * gradient
#                new_theta = new_theta/torch.norm(new_theta)
#                new_g2, count = self.fine_grained_binary_search_local_targeted(x0, y0, target, new_theta, initial_lbd = min_g2, tol=beta/500)
#                opt_count += count
#                alpha = alpha * 2
#                if new_g2 < min_g2:
#                    min_theta = new_theta 
#                    min_g2 = new_g2
#                else:
#                    break
#    
#            if min_g2 >= g2:
#                for _ in range(15):
#                    alpha = alpha * 0.25
#                    new_theta = theta - alpha * gradient
#                    new_theta = new_theta/torch.norm(new_theta)
#                    new_g2, count = self.fine_grained_binary_search_local_targeted( x0, y0, target, new_theta, initial_lbd = min_g2, tol=beta/500)
#                    opt_count += count
#                    if new_g2 < g2:
#                        min_theta = new_theta 
#                        min_g2 = new_g2
#                        break
#    
#            if min_g2 <= min_g1:
#                theta, g2 = min_theta, min_g2
#            else:
#                theta, g2 = min_ttt, min_g1
#    
#            if g2 < g_theta:
#                best_theta, g_theta = theta.clone(), g2
#            
#            #print(alpha)
#            if alpha < 1e-4:
#                alpha = 1.0
#                print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
#                beta = beta * 0.1
#                if (beta < 0.0005):
#                    break
#    
#        #target = self.model.predict(x0 + g_theta*best_theta)
#        #timeend = time.time()
#        #print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
#        return x0 + np.array(g_theta*best_theta)
#    
#    def fine_grained_binary_search_local_targeted(self, x0, y0, target, theta, initial_lbd = 1.0, tol=1e-5):
#        nquery = 0
#        lbd = initial_lbd
#        
#        
#        if self.model.predict(x0+np.array(lbd*theta))[0] != target:
#            lbd_lo = lbd
#            lbd_hi = lbd*1.01
#            nquery += 1
#            while self.model.predict(x0+np.array(lbd_hi*theta))[0] != target:
#                lbd_hi = lbd_hi*1.01
#                nquery += 1
#                if lbd_hi > 100: 
#                    return float('inf'), nquery
#        else:
#            lbd_hi = lbd
#            lbd_lo = lbd*0.99
#            nquery += 1
#            while self.model.predict(x0+tf.np.array(lbd_lo*theta))[0] == target:
#                lbd_lo = lbd_lo*0.99
#                nquery += 1
#    
#        while (lbd_hi - lbd_lo) > tol:
#            lbd_mid = (lbd_lo + lbd_hi)/2.0
#            nquery += 1
#            if self.model.predict(x0 + np.array(lbd_mid*theta))[0] == target:
#                lbd_hi = lbd_mid
#            else:
#                lbd_lo = lbd_mid
#        return lbd_hi, nquery
#    
#    def fine_grained_binary_search_targeted(self, x0, y0, target, theta, initial_lbd = 1.0):
#        nquery = 0
#        lbd = initial_lbd
#        
#        print("type of target is:",type(target))
#        print(target)
#        print("new lable is:",self.model.predict(x0 + np.array(lbd*theta))[0])
#        while self.model.predict(x0 + np.array(lbd*theta))[0] != target:
#            lbd *= 1.05
#            nquery += 1
#            if lbd > 100: 
#                return float('inf'), nquery
#    
#        num_intervals = 100
#    
#        lambdas = np.linspace(0.0, lbd, num_intervals)[1:]
#        lbd_hi = lbd
#        lbd_hi_index = 0
#        for i, lbd in enumerate(lambdas):
#            nquery += 1
#            if self.model.predict(x0 + np.array(lbd*theta))[0] == target:
#                lbd_hi = lbd
#                lbd_hi_index = i
#                break
#    
#        lbd_lo = lambdas[lbd_hi_index - 1]
#    
#        while (lbd_hi - lbd_lo) > 1e-7:
#            lbd_mid = (lbd_lo + lbd_hi)/2.0
#            nquery += 1
#            if self.model.predict(x0 + np.array(lbd_mid*theta))[0] == target:
#                lbd_hi = lbd_mid
#            else:
#                lbd_lo = lbd_mid
#    
#        return lbd_hi, nquery
#
#    
#    
sess = tf.Session()
orig = load_image('cat.jpg')
#print("type of orig:. ", type(orig))
#print("size of orig: ", orig.shape)
#print("length of orig: ",len(orig))
#TARGET = 924 # guacamole  

model = MyModel(inceptionv3,sess,[0.0,255.0])
#print(orig.shape)
#image = tf.convert_to_tensor(orig)
#image_extend = tf.expand_dims(image, axis=0)
#print("shape of image_extend: ", image_extend.shape)
image = np.copy(orig)/255.0

#print(len(image),type(image))
true_label = model.predict(image)
print("true label of the original image is: ", true_label)
attack = blackbox(model)
adv = attack.attack_untargeted(image,true_label, alpha = 4, beta = 0.05, iterations = 1000)

#adv = attack.attack_targeted(image,true_label,924)

adv_label = model.predict(adv)#print("target lable is: ", TARGET)
print("label after attack is: ", adv_label)



#classify = make_classify(sess, x, probs)
#classify(orig, target_class=TARGET)


















