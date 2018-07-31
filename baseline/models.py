import torch
from torch.autograd import Variable
import numpy as np
import tensorflow as tf

class PytorchModel(object):
    def __init__(self,model, sess,bounds):
        self.model = model
        #self.model.eval()
        self.sess = sess
        self.bounds = bounds
        
    
    def predict(self,image):
        image = np.clip(image,0,255)
        image = [image]

        return self.sess.run(self.model.pre_softmax, {self.model.x_input: image})
    
    def predict_label(self, image):
        if self.bounds[1] == 255:
            image *= 255
            image = np.clip(image,0,255)
        else:
            image = np.clip(image,0,1)

        image = [image]

        return self.sess.run(self.model.predictions, {self.model.x_input: image})
        
    #def get_gradient(self,loss):
     #   loss.backward()
        
