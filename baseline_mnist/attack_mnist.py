
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
#import l2_attack
import keras
import tensorflow as tf
import time
from torch.autograd import Variable
import torch
import numpy as np
from wrapper import Model

torch.set_printoptions(precision=20)
class blackbox:
    def __init__(self,model):
        self.model = model
        
    def attack_untargeted(self, x0, y0, alpha = 4, beta = 0.005, iterations = 1000):
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

        if (self.model.predict(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return x0,0
    
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
                lbd,comp_dec = self.fine_grained_binary_search_fix(x0,y0,theta,initial_lbd,current_best,num_query,1e-5)
                if comp_dec > comp_theta:
                    comp_theta = comp_dec
                    best_theta,g_theta = theta,lbd
                    print("--------> Found abs-distortion %.4f with 10 queries" % g_theta)
                    print("--------> Found comp-distortion %.4f with 10 queries" % comp_dec)
            timeend = time.time()
        print("==========> Found best distortion %.4f in %.4f seconds" % (g_theta, timeend-timestart))
        query_count = (num_directions+1)*num_query
        #print("type of best_theta", type(best_theta))
        #print("type of best_theta", type(g_theta))
        lbd,count = self.fine_grained_binary_search( x0, y0, best_theta, g_theta, current_best)
        g_theta = lbd
        query_count += count
        
#        timestart = time.time()
#        for i in range(num_directions):
#            theta = torch.randn(x0.shape).type(torch.FloatTensor)
#            #print(theta.size())
#            initial_lbd = torch.norm(theta)
#            theta = theta/torch.norm(theta)
#            if self.model.predict(x0+np.array(initial_lbd*theta)) != y0:
#                query_count += 1 
#                #print(type(theta),type(initial_lbd),type(g_theta))
#                #print("find a new adv direction, new label:", self.model.predict(x0+np.array(initial_lbd*theta)))
#                lbd, count = self.fine_grained_binary_search( x0, y0, theta, initial_lbd, g_theta)
#                query_count += count
#                if lbd < g_theta:
#                    best_theta, g_theta = theta,lbd
##                    print("label for random direction:",self.model.predict(x0+np.array(g_theta*best_theta)))
#                    print("--------> Found distortion %.4f" % g_theta)
#        
#            timeend = time.time()
#            print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))
        
        
        
        
    
        g1 = 1.0
        theta, g2 = best_theta.clone(), g_theta
        torch.manual_seed(0)
        opt_count = 0
        stopping = 0.01
        prev_obj = 100000
        for i in range(iterations):
 
            if g_theta < 2:
                print("====================query number after distortion < 2 =======================: ",opt_count)
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
                    alpha = alpha * 0.25
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
            
            #print(alpha)
#            print("%3d th iteration" % i)
#            print("current alpha:",alpha)
#            print("g_theta")
#            print("number of queries:", opt_count+query_count)
            if alpha < 1e-6:
                alpha = 1.0
                print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
                beta = beta * 0.1
                if (beta < 1e-6):
                    print("beta is too samll")
                    break

    
        #target = model.predict(x0 + g_theta*best_theta)
        
        #print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
        print("mnist_baseline")
        print("best distortion :", g_theta)
        print("number of queries :", opt_count+query_count)
        return np.array(g_theta*best_theta), opt_count+query_count
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

        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            #timestart2 = time.time()
            while self.model.predict(x0+ np.array(lbd_lo*theta)) != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.model.predict(x0 + np.array(lbd_mid*theta)) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery
    
    def fine_grained_binary_search_fix(self,x0,y0,theta, initial_lbd = 1.0, current_best = float('inf'),num_query = 10, tol=1e-5):
        nquery = 0
        if initial_lbd > current_best: 
            if self.model.predict(x0+ np.array(current_best*theta)) == y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd
    
        lbd_hi = lbd
        lbd_lo = 0.0
    
        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.model.predict(x0 + np.array(lbd_mid*theta)) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
            if nquery > num_query:
                break
        comp_dec = (initial_lbd - lbd_hi)/initial_lbd
       # print("number of query before return for this direction:",nquery)
        return lbd_hi,comp_dec,
    
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
        lbd_lo = 0
        print("label before fine binary search:", self.model.predict(x0+ np.array(lbd_hi*theta)))
    
        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.model.predict(x0 + np.array(lbd_mid*theta)) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        #print("find a better initialization")
        return lbd_hi, nquery
    

    


session = keras.backend.get_session()
keras.backend.set_learning_phase(False)
model = keras.models.load_model("data/mnist")
model = Model(model,[0.0,1.0])

attack = blackbox(model)

#touse = [x for x in tf.trainable_variables() if 'Generator' in x.name]
#saver = tf.train.Saver(touse)
#saver.restore(session, 'data/mnist-gan')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = np.array(x_test, dtype=np.float32)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test /= 255.0
#image = x_test[:1]
#adv = attack.attack_untargeted(image[0],y_test[0], alpha = 4, beta = 0.005, iterations = 1000)
count = []
for i in range(20):
    label = model.predict(x_test[i])
    if label == y_test[i]:
        count.append(1)
    else:
        count.append(0)
    
print("accuracy of this model is:", sum(count)/len(count))


dist = []
count = []
for i in range(15):
    print("=========================image ",i+1,"==========================================")
    print("true label:",y_test[i])
    print("predicted label:",model.predict(x_test[i]))
    adv_mod, queries= attack.attack_untargeted(x_test[i],y_test[i], alpha = 4, beta = 0.005, iterations = 1000)
    dist.append(np.linalg.norm(adv_mod))
    count.append(queries)

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





