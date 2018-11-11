#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 11:52:34 2018

@author: yusu
"""

from keras.datasets import mnist
#import matplotlib
#%matplotlib inline
#import matplotlib.pyplot as plt
import keras
from defense import *
import tensorflow as tf
from torch.autograd import Variable
import torch
import numpy as np
from wrapper import Model
import time
import os


class blackbox:
    def __init__(self,model):
        self.model = model
        
    def attack_untargeted(self, x0, y0, shape ,alpha = 2, beta = 0.005, iterations = 1000):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            alpha: learning rate 
            beta: learning rate
            train_dataset: set of training data
            (x0, y0): original image
        """

        
        if (self.model.predict(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return np.nan
        
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
            if self.model.predict_gan(x0+np.array(initial_lbd*theta)) != y0:
                query_count += 1
                lbd,comp_dec,query = self.fine_grained_binary_search_fix(x0,y0,theta,initial_lbd,g_theta,current_best,num_query)
                query_count += query
                if comp_dec > comp_theta:
                    comp_theta = comp_dec
                    best_theta,g_theta = theta,lbd
                    print("--------> Found abs-distortion %.4f with 10 queries" % g_theta)
                    print("--------> Found comp-distortion %.4f with 10 queries" % comp_dec)
        timeend = time.time()
        print("==========> Found best distortion %.4f in %.4f seconds" % (g_theta, timeend-timestart))
        query_count = (num_directions+1)*num_query
        lbd,count = self.fine_grained_binary_search( x0, y0, best_theta, g_theta, current_best)
        g_theta = lbd
        query_count += count
        
        
        
        #timestart = time.time()
        g1 = 1.0
        theta, g2 = best_theta.clone(), g_theta
        torch.manual_seed(0)
        opt_count = 0
        stopping = 0.01
        prev_obj = 100000
        for i in range(iterations):
            print("type of best_theta*g_thetha", type(best_theta*g_theta))
            _,orig_mod = self.model.predict_gan(best_theta*g_theta,x0)
            mod_norm = np.linalg.norm(orig_mod)
            if mod_norm < 1:
                print("====================query number after distortion < 0.05 =======================: ",opt_count)
                break
            
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
                new_g2, count = self.fine_grained_binary_search_local( x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                opt_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                else:
                    #print("break here 2 ?")
                    break
    
            if min_g2 >= g2:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = theta - alpha * gradient
                    new_theta = new_theta/torch.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local( x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                    opt_count += count
                    if new_g2 < g2:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        #print("break here 3?")
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
                    #print("break here 4?")
                    break
    
        #target = model.predict(x0 + g_theta*best_theta)
        
        #timeend = time.time()
        #print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
        print("defensegan")
        print("best distortion :", g_theta)
        print("number of queries :", opt_count+query_count)
        mod_gan = np.array(g_theta*best_theta)
        print("return g_theta*best_theta, shape of it:", mod_gan.shape)
        return mod_gan, opt_count+query_count
    
    def fine_grained_binary_search_fix(self,x0,y0,theta, initial_lbd = 1.0, tol=1e-5,current_best = float('inf'),num_query = 10):
        nquery = 0
        if initial_lbd > current_best: 
            if self.model.predict_gan(x0+ np.array(current_best*theta)) == y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd
    
        lbd_hi = lbd
        lbd_lo = 0.0
    
        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.model.predict_gan(x0 + np.array(lbd_mid*theta)) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
            if nquery > num_query:
                break
        comp_dec = (initial_lbd - lbd_hi)/initial_lbd
       # print("number of query before return for this direction:",nquery)
        return lbd_hi,comp_dec,nquery
    
    
    def fine_grained_binary_search_local(self, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
        pred,_ = self.model.predict_gan(lbd*theta,x0)
        if pred == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            pred,_ = self.model.predict_gan(lbd_hi*theta,x0)
            while pred == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            pred,_ = self.model.predict_gan(lbd*theta,x0)
            while pred != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1
    
        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            pred,_ = self.model.predict_gan(lbd*theta,x0)
            if pred != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery
    
    def fine_grained_binary_search(self, x0, y0,theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best:
            pred,_ = self.model.predict_gan(current_best*theta,x0)
            if pred == y0:
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
#            modi = self.get_modifier(lbd_mid*theta,x0,gan)
#            print("type of modi:", type(modi))
            pred,_ = self.model.predict_gan(lbd_mid*theta,x0)
            if pred != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery
    

    
# ================================== test ===========================================#
session = keras.backend.get_session()

keras.backend.set_learning_phase(False)
model = keras.models.load_model("data/mnist")

model = Model(model,[0.0,1.0],session,lambda x : Generator(1,x))


#touse = [x for x in tf.trainable_variables() if 'Generator' in x.name]
#saver = tf.train.Saver(touse)
#saver.restore(session, 'data/mnist-gan')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = np.array(x_test, dtype=np.float32)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test /= 255.0
image = x_test[:100]
shape = 128
xin = tf.placeholder(tf.float32, [1, 128])
mygan = Generator(1, xin)
print("True label", y_test[0])
print("Preds",model.predict(image[0]))

attack = blackbox(model)


dist = []
count = []
for i in range(15):
    print("=====================attacking image %2d =========================="%(i))
    print("label of pure image:", model.predict(image[i]))
    adv_mod,query = attack.attack_untargeted(image[i],y_test[i],shape,alpha = 4, beta = 0.005, iterations = 1000)
    adv_mod = np.expand_dims(np.array(adv_mod),0)
    mod = session.run(mygan,{xin:adv_mod})
    dist.append(np.linalg.norm(mod))
    count.append(query)

index = np.nonzero(count)
index = list(index)[0].tolist()

avg_distortion = np.mean(np.array(dist)[index])
avg_count = np.mean(np.array(count)[index])
print("the average distortion for %2d images :"%(len(index)),avg_distortion)
for i in dist:
    print(i)
print("the number of queries for %2d images :"%(len(index)), avg_count)
for j in count:
    print(j)
