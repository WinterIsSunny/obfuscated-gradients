#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 11:52:34 2018

@author: yusu
"""

import os
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
        best_theta, g_theta = None, float('inf')
        query_count = 0
            
        timestart = time.time()
        for i in range(num_directions):
            theta = np.random.randn(shape)
            #print(theta.size())
            initial_lbd = np.linalg.norm(theta)
            theta = theta/initial_lbd
            lbd, count = self.fine_grained_binary_search( x0, y0, theta, initial_lbd, g_theta)
            query_count += count
            if lbd < g_theta:
                best_theta, g_theta = theta,lbd
                print("--------> Found distortion %.4f" % g_theta)
        timeend = time.time()
        print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))
    
        
        
        
        #timestart = time.time()
        g1 = 1.0
        theta, g2 = best_theta.copy(), g_theta
        torch.manual_seed(0)
        opt_count = 0
        stopping = 0.01
        prev_obj = 100000
        for i in range(iterations):
            _,orig_mod = self.model.predict_gan(best_theta*g_theta,x0)
            mod_norm = np.linalg.norm(orig_mod)
            if mod_norm < 0.05:
                print("====================query number after distortion < 0.05 =======================: ",opt_count)
                break
            
            gradient = np.zeros(theta.shape)
            q = 10
            min_g1 = float('inf')
            for _ in range(q):
                u = np.random.randn(theta.shape).type(np.float32)
                u = u/np.linalg.norm(u)
                ttt = theta+beta * u
                ttt = ttt/np.linalg.norm(ttt)
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
                new_theta = new_theta/np.linalg.norm(new_theta)
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
                    new_theta = new_theta/np.linalg.norm(new_theta)
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
                best_theta, g_theta = theta.copy(), g2
            
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
        return mod_gan
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
#            modi = self.get_modifier(current_best*theta,x0)
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
    
#    def get_modifier(self,modifier,x0):
##        img = np.expand_dims(x0,0)
#        modifier = np.expand_dims(np.array(modifier),0)
#        x_new = tf.placeholder(tf.float32,(1,128))
##        noise = tf.reshape(x_new, [1,128])
##        mod_tf = tf.convert_to_tensor(modifier)
##        print("shape of modifier before GAN:", mod_tf.shape )
##        new_mod = gan(mod_tf)
##        print(type(new_img))
##        new_mod = new_mod[0]
#        new_mod = gan(x_new)
#        with tf.Session() as sess:
#            sess.run(new_mod,{x_new:modifier})
#            print("dimension of img_gan is :",new_mod.get_shape())
#            img_npy = new_mod.eval()
#        
#        new_mod = np.sum(np.expand_dims(x0,0)-img_npy,0)
##        new_mod = np.sum(x0 - img_gan, 0)
#        # return np array modifier 
#        return new_mod
        
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
image = x_test[:1]
shape = 128
xin = tf.placeholder(tf.float32, [1, 128])
mygan = Generator(1, xin)
print("True label", y_test[0])
print("Preds",model.predict(image[0]))

#res = []
##dists =[]

#for i in range(3):
#    modifier = attack1.attack_untargeted(image[0],y_test[0],shape, best_theta = None,
#                                         alpha = 2, beta = 0.05, iterations = 10)
##    dist = pre_adv - image[0]
#    #dist_norm = np.linalg.norm(dist)
#    res.append(modifier)
#    #dists.append(dist_norm)
#res = np.array(res)
#xin = tf.placeholder(tf.float32, [3, 128])
#mygan = Generator(3, xin)
#it = session.run(mygan, {xin: res})
#
#distortion = np.sum((it)**2,(1,2,3))**.5
##print("Distortions", distortion)
#start = np.array([res[np.argmin(distortion)]])

attack = blackbox(model)
print("label of pure image:", model.predict(image[0]))
adv_mod = attack.attack_untargeted(image[0],y_test[0],shape,alpha = 4, beta = 0.005, iterations = 1000)

mod = session.run(mygan,{xin:adv_mod})
adv = image[0]+np.sum(mod,0)
print("new label of adv_sample :", model.predict(adv))

