
import numpy as np
import tensorflow as tf 

class Model:
    def __init__(self,model,bounds,sess,gan):
        self.model = model
        self.bounds = bounds
        self.gan = gan
        self.sess = sess
    
    def predict_gan(self,mod,x0):
        orig_mod = np.expand_dims(np.array(mod),0)
        x_new = tf.placeholder(tf.float32,(1,128))
        mod = self.gan(x_new)
        mod = self.sess.run(mod,{x_new:orig_mod})
        #print("type of mod",type(mod))
        mod = np.sum(mod,0)
        #print("shape of mod",mod.shape)
        pred = self.predict(x0+mod)
        return pred,mod

        
    def predict(self,image):
        if self.bounds[1] == 255:
            new_img = image*255
            new_img = np.clip(new_img,0,255)
        else:
            new_img = np.clip(image,0,1)
            new_img= np.expand_dims(new_img,0)
        #print("shape of image:", image.shape)
        #image = tf.convert_to_tensor(image,np.float32)
        #print("type of image after clip:",type(new_img))

        labels = self.model.predict(new_img)
        #print(labels)
        label = np.argmax(labels[0])
        #print("the current label: ", label)
        return label
    