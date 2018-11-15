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
            return torch.zeros(shape),0
        
        num_directions = 1000
        best_theta, g_theta = None, float('inf')
        query_count = 0
        print("original label is ", y0)
        
        timestart = time.time()
        for i in range(num_directions):
            #print("generating a new distortion")
            theta = torch.randn(shape)*10
            #print(theta.size())
            initial_lbd = torch.norm(theta)
            theta = theta/initial_lbd
            pred,_ = self.model.predict_gan(theta*initial_lbd,x0)
            #print("predicted label is", pred)
            
            if pred != y0:
#                print("new feasible direction and iteration", pred,i)
                lbd, count = self.fine_grained_binary_search( x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    print("new feasible direction and iteration", pred,i)
                    best_theta, g_theta = theta,lbd
                    print("--------> Found distortion %.4f" % g_theta)
        timeend = time.time()
        print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))
    

        g1 = 1.0
        theta, g2 = best_theta.clone(), g_theta
        torch.manual_seed(0)
        opt_count = 0
        stopping = 0.01
        prev_obj = 100000
        time1 = time.time()
        new_lb,orig_mod = self.model.predict_gan(best_theta*g_theta,x0)
        time2 = time.time()
        print("new label is :", new_lb)
        print("time consuming for one query -- gan:", time2-time1)
        for i in range(iterations):
            print("this is interation:",i)
            _,orig_mod = self.model.predict_gan(best_theta*g_theta,x0)
            #print("loc1")
            mod_norm = np.linalg.norm(orig_mod)
            if mod_norm < 8:
                print("====================query number after distortion < 1 =======================: ",opt_count)
                break
            
            gradient = torch.zeros(theta.size())
            q = 5
            min_g1 = float('inf')
            for _ in range(q):
                u = torch.randn(theta.size()).type(torch.FloatTensor)
                u = u/torch.norm(u)
                ttt = theta+beta * u
                ttt = ttt/torch.norm(ttt)
                g1, count = self.fine_grained_binary_search_local(x0, y0, ttt, initial_lbd = g2, tol=beta/500)
                opt_count += count
                gradient += (g1-g2)/beta * u
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt
            gradient = 1.0/q * gradient
            #print("loc2")
            
    
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
                    break
            #print("loc3")
    
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
                        break
            #print("loc4")
    
            if min_g2 <= min_g1:
                theta, g2 = min_theta, min_g2
            else:
                theta, g2 = min_ttt, min_g1
    
            if g2 < g_theta:
                best_theta, g_theta = theta.clone(), g2
            
            if alpha < 1e-6:
                alpha = 1.0
                print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
                beta = beta * 0.1
                if (beta < 1e-6):
                    print("break because beta is too small")
                    break
    
        print("defensegan")
        print("best distortion :", g_theta)
        print("number of queries :", opt_count+query_count)
        mod_gan = np.array(g_theta*best_theta)
        print("return g_theta*best_theta, shape of it:", mod_gan.shape)
        return mod_gan, opt_count+query_count
    
    def fine_grained_binary_search_local(self, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
        lbd = np.array(lbd)

        if self.model.predict_gan(lbd*theta) == y0:
            lbd_lo = lbd*1
            lbd_hi = lbd*1.01
            nquery += 1
            while self.model.predict_gan(lbd_hi*theta) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 100:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd*1
            lbd_lo = lbd*0.99
            nquery += 1
            while self.model.predict_gan(lbd_lo*theta)!= y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.model.predict_gan(lbd_mid*theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid

        lbd_hi = np.array(lbd_hi)
        lbd_hi = torch.FloatTensor(lbd_hi)

        return lbd_hi, nquery
    
        
    def fine_grained_binary_search(self, x0, y0,theta, initial_lbd, current_best):
        nquery = 0
        theta = np.array(theta)
        initial_lbd = np.array(initial_lbd)
        if initial_lbd > current_best:
            pred,_ = self.model.predict_gan(current_best*theta,x0)
            if pred == y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd
        
        lbd_hi = lbd
        lbd_lo = 0
    
        while not np.isclose(lbd_hi,lbd_lo,1e-5):
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
#            modi = self.get_modifier(lbd_mid*theta,x0,gan)
#            print("type of modi:", type(modi))
            #pred,_ = self.model.predict_gan(lbd_mid*theta,x0)
            if self.model.predict_gan(lbd_mid*theta,x0)[0] != y0:
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
    print("=====================attacking image %2d =========================="%(i+1))
    print("label of pure image:", model.predict(image[i]))
    adv_mod,query = attack.attack_untargeted(image[i],y_test[i],shape,alpha = 4, beta = 0.05, iterations = 1000)
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
