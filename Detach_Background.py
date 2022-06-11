import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import CustomObjectScope

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

#Global parameters
H=512
W=512
smooth=1e-15

def iou(y_true,y_pred):
    def f(y_true,y_pred):
        intersection=(y_true*y_pred).sum()
        union=y_true.sum()+y_pred.sum()-intersection
        x=(intersection+1e-15)/(union + 1e-15)
        x=x.astype(np.float32)
        return x
    return tf.numpy_function(f,[y_true,y_pred],tf.float32)

def dice_coef(y_true, y_pred):
    y_true=tf.keras.layers.Flatten()(y_true)
    y_pred=tf.keras.layers.Flatten()(y_pred)
    intersection=tf.reduce_sum(y_true*y_pred)
    return (2.*intersection+smooth)/(tf.reduce_sum(y_true)+tf.reduce_sum(y_pred)+smooth)

def dice_loss(y_true, y_pred):
    return 1.0-dice_coef(y_true, y_pred)

def detach():
    np.random.seed(42)
    tf.random.set_seed(42)

    with CustomObjectScope({'iou':iou,'dice_coef':dice_coef,'dice_loss':dice_loss}):
        clf=tf.keras.models.load_model("Models/model.h5")

    image=cv2.imread('kodim03.png')
    
    h,w,_=image.shape
    x=cv2.resize(image,(W, H))
    x=x/255.0
    x=x.astype(np.float32)
    x=np.expand_dims(x,axis=0)

    y=clf.predict(x)[0]
    y=cv2.resize(y,(w,h))
    y=np.expand_dims(y,axis=-1)
    y=y>0.5
    photo_mask=y
    background_mask=np.abs(1-y)
    
    cv2.imshow('Object',image*photo_mask)
    cv2.waitKey(0)

if __name__=="__main__":
   detach()