import torch
from torch.autograd import Variable
import numpy as np
import tensorflow as tf

class PytorchModel(object):
    def __init__(self,model, sess):
        self.model = model
        #self.model.eval()
        self.sess = sess
        
    
    def predict(self,image):
       
#        image = torch.clamp(image,self.bounds[0],self.bounds[1]).cuda()
#        # how to use gpu ?
#        #image = tf.clip_by_value(image, self.bounds)
#        if len(image.size())!=4:
#            image = image.unsqueeze(0)
#        #image = Variable(image, volatile=True) # ?? not supported by latest pytorch
#        # convert image from torch to tf !!!
#        image = tf.convert_to_tensor(np.array(image))
#        image = tf.Variable(image)
#        output = self.model(image)
#        sess = tf.InteractiveSession()
#        output = output.eval()
        
        return self.sess.run(self.model.pre_softmax, {self.model.x_input: image})
    
    def predict_label(self, image):
        
#        image = torch.clamp(image,self.bounds[0],self.bounds[1]).cuda()
#        if len(image.size())!=4:
#            image = image.unsqueeze(0)
#        image = Variable(image, volatile=True) # ?? not supported by latest pytorch
#        image = tf.convert_to_tensor(np.array(image))
#        image = tf.Variable(image)
#        output = self.model.predict(image)
#        sess = tf.InteractiveSession()
#        output = output.eval()
#         output is an array???
        
        #image = Variable(image, volatile=True) 
        # ?? not supported by latest pytorch
#        _, predict = torch.max(output.data, 1)
        
        return self.sess.run(self.model.predictions, {self.model.x_input: image})
        
    #def get_gradient(self,loss):
     #   loss.backward()
        
