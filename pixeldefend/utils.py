import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

def label_to_name(label):
    lut = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    return lut[label]

def make_classify(sess, input_, probs):
    def classify(img, correct_class=None, target_class=None):
#        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
#        fig.sca(ax1)
        p = sess.run(probs, feed_dict={input_: [img]})[0]
#        ax1.imshow(np.clip(img/255.0, 0, 1))
#        fig.sca(ax1)

        topk = list(p.argsort()[-10:][::-1])
        topprobs = p[topk]
#        barlist = ax2.bar(range(10), topprobs)
#        if target_class in topk:
#            barlist[topk.index(target_class)].set_color('r')
#        if correct_class in topk:
#            barlist[topk.index(correct_class)].set_color('g')
#        plt.sca(ax2)
#        plt.ylim([0, 1.1])
#        plt.xticks(range(10),
#                   [label_to_name(i)[:15] for i in topk],
#                   rotation='vertical')
#        fig.subplots_adjust(bottom=0.2)
#        plt.show()
        label = p[topk[0]]
    return label

def int_shape(tensor):
    return list(map(int, tensor.get_shape()))
