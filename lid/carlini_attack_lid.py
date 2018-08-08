#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:43:21 2018

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


import cifar10_input
model = get_model("cifar", softmax=True)
model.load_weights("data/lid_model_cifar.h5")
model_logits = get_model("cifar", softmax=False)
model_logits.load_weights("data/lid_model_cifar.h5")
class Attack:
    def __init__(self, model, tol, num_steps, step_size, random_start):
        self.model = model
        self.tol = tol
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start

        self.xs = tf.Variable(np.zeros((1000, 32, 32, 3), dtype=np.float32),
                                    name='modifier')
        self.orig_xs = tf.placeholder(tf.float32, [None, 32, 32, 3])

        self.ys = tf.placeholder(tf.int32, [None])

        self.epsilon = 8.0/255

        delta = tf.clip_by_value(self.xs, 0, 255) - self.orig_xs
        delta = tf.clip_by_value(delta, -self.epsilon, self.epsilon)

        self.do_clip_xs = tf.assign(self.xs, self.orig_xs+delta)

        self.logits = logits = model(self.xs)

        label_mask = tf.one_hot(self.ys, 10)
        correct_logit = tf.reduce_sum(label_mask * logits, axis=1)
        wrong_logit = tf.reduce_max((1-label_mask) * logits - 1e4*label_mask, axis=1)

        self.loss = (correct_logit - wrong_logit)

        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(step_size*1)

        grad,var = optimizer.compute_gradients(self.loss, [self.xs])[0]
        self.train = optimizer.apply_gradients([(tf.sign(grad),var)])

        end_vars = tf.global_variables()
        self.new_vars = [x for x in end_vars if x.name not in start_vars]

    def perturb(self, x, y, sess):
        sess.run(tf.variables_initializer(self.new_vars))
        sess.run(self.xs.initializer)
        sess.run(self.do_clip_xs,
                 {self.orig_xs: x})

        for i in range(self.num_steps):

            sess.run(self.train, feed_dict={self.ys: y})
            sess.run(self.do_clip_xs,
                     {self.orig_xs: x})

        return sess.run(self.xs)

cifar = cifar10_input.CIFAR10Data("../cifar10_data")

sess = K.get_session()
attack = Attack(model_logits,
                      1,
                      100,
                      1/255.0,
                      False)


xs = tf.placeholder(tf.float32, (1, 32, 32, 3))

image = cifar.eval_data.xs[:10]/255.0-.5
label = cifar.eval_data.ys[:10]

#plt.imshow(image[1]+.5)
#plt.show() 
#print("Image Label", label[1])

x_input = tf.placeholder(tf.float32, [None, 32, 32, 3])
logits = model_logits(x_input)

print('Clean Model Prediction', np.argmax(sess.run(logits, {x_input: image[1:2]})))
print('Clean Model Logits', sess.run(logits, {x_input: image[1:2]}))


adversarial = attack.perturb(image, label, sess)

#plt.imshow(adversarial[1]+.5)
#plt.show()

print("Max distortion", np.max(np.abs(adversarial-image)))

print('Adversarial Model Prediction', np.argmax(sess.run(logits, {x_input: adversarial[1:2]})))
print('Adversarial Model Logits', sess.run(logits, {x_input: adversarial[1:2]}))

artifacts, labels = get_lid(model, image, image, adversarial, 2, 10, 'cifar')


T = collections.namedtuple('args', ['dataset', 'attack', 'artifacts', 'test_attack'])
lr, _, scaler = detect(T('cifar', 'cw-l2', 'lid', 'cw-l2'))

t_artifacts = scaler.transform(artifacts)

print('Detection rate clean', np.mean(lr.predict(t_artifacts[:10])))
print('Detection rate adversarial', np.mean(lr.predict(t_artifacts[-10:])))

