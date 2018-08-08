#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:30:27 2018

@author: yusu
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from keras.models import load_model
import keras.backend as K
from util import get_model
from extract_artifacts import get_lid
import collections
from detect_adv_samples import detect
from wrapper import Model
import torch
import time
import cifar10_input


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
            return np.zeros(x0.shape)
    
        num_directions = 1000
        best_theta, g_theta = None, float('inf')
        query_count = 0
        
        #timestart = time.time()
        
        
        for i in range(num_directions):
            theta = torch.randn(x0.shape).type(torch.FloatTensor)
            #print(theta.size())
            initial_lbd = torch.norm(theta)
            theta = theta/torch.norm(theta)
            if self.model.predict(x0+np.array(initial_lbd*theta)) != y0:
                lbd, count = self.fine_grained_binary_search( x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta,lbd
#                    print("new g_theta :", g_theta,"***")
#                    print("label for random direction:",self.model.predict(x0+np.array(g_theta*best_theta)))
#                    print("norm of theta*lbd 4:", np.linalg.norm(x0+np.array(g_theta*best_theta)))
#                    print("******")
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
            if g_theta < 1:
                break
            gradient = torch.zeros(theta.size())
            q = 30
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
                #print("alpha in the first for loop is: ",alpha)
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
                    #print("alpha in the second for loop is: ",alpha)
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
#            print("%3d th iteration" % i)
#            print("current alpha:",alpha)
#            print("g_theta")
#            print("number of queries:", opt_count+query_count)
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
                beta = beta * 0.1
                if (beta < 0.0005):
                    break
#            print("=-=-=-=-=-=-=will enter next iteration=-=-=-=-=-=-=-=")
    
        #target = model.predict(x0 + g_theta*best_theta)
        
        #print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
        
        print("lid")
        print("best distortion :", g_theta)
        print("number of queries :", opt_count+query_count)
#        print("=-=-=-=-=-=-=-=-=-=-=-=-will enter next image=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        return np.array(g_theta*best_theta)
    
    def fine_grained_binary_search_local(self, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
        
        if self.model.predict(x0+np.array(lbd*theta)) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            #timestart1 = time.time()
            while self.model.predict(x0+np.array(lbd_hi*theta)) == y0:
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
            while self.model.predict(x0+ np.array(lbd_lo*theta)) != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1
            #timeend2 = time.time()
            #print("2nd while time:", timeend2 - timestart2)
            
        #timestart3 = time.time()
        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.model.predict(x0 + np.array(lbd_mid*theta)) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        #timeend3 = time.time()
        #print("3rd while time:",timeend3 - timestart3)
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
#            print("assign lbd = current_best, lbd = ",lbd,"***")
#            print("after assigning lbd = current_best,       label :",self.model.predict(x0+ np.array(lbd*theta)))
#            print("norm of adv 1:", np.linalg.norm(x0+ np.array(lbd*theta)))
#            print("******")
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
#        print("assign lbd_hi = lbd,  lbd_hi = ",lbd_hi,"***")
#        print("label before fine binary search:", self.model.predict(x0+ np.array(lbd_hi*theta)))
#        print("norm of lbd_hi*theta 2:", np.linalg.norm(x0+ np.array(lbd*theta)))
#        print("******")
    
        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.model.predict(x0 + np.array(lbd_mid*theta)) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
#        print("after binary search: lbd_ih:", lbd_hi,"***")
#        print("label after fine binary search:", self.model.predict(x0+ np.array(lbd_hi*theta)))
#        print("norm of lbd_hi*theta 3:", np.linalg.norm(x0+ np.array(lbd_hi*theta)))
#        print("******")
        return lbd_hi, nquery
    


model = get_model("cifar", softmax=True)
model.load_weights("data/lid_model_cifar.h5")
model_logits = get_model("cifar", softmax=False)
model_logits.load_weights("data/lid_model_cifar.h5")

sess = K.get_session()
model = Model(model,model_logits,sess,[0.0,1.0])

cifar = cifar10_input.CIFAR10Data("../cifar10_data")
image = cifar.eval_data.xs[:2]/255.0-.5
label = cifar.eval_data.ys[:2]

#timestart = time.time()
#print('Clean Model Prediction', model.predict(image[0]))
#timeend = time.time()
#print("time consuming:", timeend - timestart)
attack = blackbox(model)
#mod = attack.attack_untargeted(image[0],label[0],alpha = 2, beta = 0.05, iterations = 1000)
#adv = image[0] + mod
#print("new label for adversarial sample: ", model.predict(adv))


dist = []
mods = []
for i in range(2):
    print("===========attacking image ",i+1,"=====================")
    mod = attack.attack_untargeted(image[i],label[i])
    mods.append(mod)
    dist.append(np.linalg.norm(mod))
np.save("dist.npy",np.array(dist))
np.save("mods.npy",np.array(mods))

index = np.nonzero(dist)
index = list(index)[0].tolist()
mods_valid = np.array(mods)[index]
dist_valid = np.array(dist)[index]  
avg_dist = np.mean(dist)
image_valid = np.array(image)[index]
n_samples = len(index)
print("shape of image :", image_valid.shape)
print("shape of modi:", mods_valid.shape)
print("length of valid samples:", n_samples)
advs = image_valid + mods_valid
print("length of advs:",len(advs))
print("type of advs:",type(advs))
print("type of elements in advs:",type(advs[0]))
print("average distortion of 100 images is :", avg_dist)

#artifacts, labels = get_lid(model, image, image, adversarial, 20, 100, 'cifar')

artifacts, labels = get_lid(model.model, image_valid, image_valid, advs, 1, n_samples, 'cifar')


T = collections.namedtuple('args', ['dataset', 'attack', 'artifacts', 'test_attack'])
lr, _, scaler = detect(T('cifar', 'cw-l2', 'lid', 'cw-l2'))

t_artifacts = scaler.transform(artifacts)

print('Detection rate clean', np.mean(lr.predict(t_artifacts[:n_samples])))
print('Detection rate adversarial', np.mean(lr.predict(t_artifacts[-n_samples:])))

