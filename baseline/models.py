import torch
from torch.autograd import Variable
import numpy as np
import tensorflow as tf

class PyModel(object):
    def __init__(self,model, sess,bounds):
        self.model = model
        #self.model.eval()
        self.sess = sess
        self.bounds = bounds
        
    
    def predict(self,image):
        image = np.clip(image,0.0,255.0)
        image = [image]

        return self.sess.run(self.model.pre_softmax, {self.model.x_input: image})
    
    def predict_label(self, image):
        if self.bounds[1] == 255.0:
#            print("scale of this model is 255")
            new_img = image * 255.0
            new_img = np.clip(new_img,0.0,255.0)
        else:
#            print("scale of this model is 1")
            new_img = np.clip(image,0.0,1.0)

        new_img = [new_img]
        label = np.argmax(self.sess.run(self.model.predictions, {self.model.x_input: new_img}))
        print("predicted label:", label)

        return label
        

