"""
Process an image that we can pass to our networks.
"""
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2
import random
import tensorflow as tf
from keras_preprocessing import image

def process_image(image, target_shape,):
    """Given an image, process it and return the array."""
    # Load the image.
    h, w, _ = target_shape
    image = load_img(image, target_size=(h, w))
#    image = load_img(image, grayscale=True,target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    x = (img_arr / 255.).astype(np.float32)

    return x

def process_image_aug(image, target_shape,h_flip,v_flip,rot,x_shift,y_shift,lighting_k,lighting_b,G_noise):
    """Given an image, process it and return the array."""
    # Load the image.
    h, w, _ = target_shape
    image = load_img(image, target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    if h_flip > 50:
        img_arr = horizon_flip(img_arr)
    if v_flip > 50:
        img_arr = vertical_flip(img_arr)

    x = (img_arr / 255.).astype(np.float32)

    return x

def horizon_flip(img):
    return img[:, ::-1,:] 

def vertical_flip(img):
    return img[::-1]  

def rotate(img, rot):
 

    rows, cols = img.shape[:2]
    center_coordinate = (int(cols / 2), int(rows / 2))
    angle = rot
    M = cv2.getRotationMatrix2D(center_coordinate, angle, 1)


    out_size = (cols, rows)
    rotate_img = cv2.warpAffine(img, M, out_size, borderMode=cv2.BORDER_REPLICATE)

    return rotate_img

def shift(img, x_shift, y_shift):


    rows, cols = img.shape[:2]
    y_shift = y_shift
    x_shift = x_shift

    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])

    img_shift = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

    return img_shift

def lighting_adjust(img, lighting_k, lighting_b):

   
    slope = lighting_k
    bias = lighting_b
    img = img * slope + bias
    img = np.clip(img, 0, 255)

    return img.astype(np.uint8)

def Gaussian_noise(img):
 
    gauss = np.random.normal(loc=0, scale=1, size=img.shape)
    img_gauss = img + gauss
    out = np.clip(img_gauss, 0, 255)

    return out

