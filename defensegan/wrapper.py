
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
        mod = np.sum(mod,0)
        pred = self.predict(x0+mod)
        return pred,mod

        
    def predict(self,image):
        #print("shape of image:", image.shape)
        #image = tf.convert_to_tensor(image,np.float32)
        if isinstance(image, np.ndarray ):
            if self.bounds[1] == 255:
                new_img = image*255
                new_img = np.clip(new_img,0,255)
            else:
                new_img = np.clip(image,0,1)
            new_img= np.expand_dims(new_img,0)
        else:     
            if self.bounds[1] == 255:
                new_img = image*255
                new_img = tf.clip_by_value(new_img,0,255)
                new_img = tf.expand_dims(new_img,0)
            else:
                new_img = tf.clip_by_value(image,0,1)
                new_img = tf.expand_dims(new_img,0)
            with self.sess.as_default():
                new_img.eval()
        
        
        #print("type of image after clip:",type(new_img))

        labels = self.model.predict(new_img)
        #print(labels)
        label = np.argmax(labels[0])
        #print("the current label: ", label)
        return label
    