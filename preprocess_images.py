# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import sys
import random
import warnings
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import skimage
from scipy import ndimage
#import tensorflow as tf

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))

def read_image_labels(image_id):

    image_file = "../input/stage1_train/{}/images/{}.png".format(image_id,image_id)
    mask_file = "../input/stage1_train/{}/masks/*.png".format(image_id)
    image = skimage.io.imread(image_file)
    masks = skimage.io.imread_collection(mask_file).concatenate()    
    height, width, _ = image.shape
    num_masks = masks.shape[0]
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[masks[index] > 0] = index + 1
    return image, labels
    
    
def data_aug(image, label, angle = 30, resize_rate = 0.9):
    flip = random.randint(0, 1)
    size = image.shape[0]
    rsize = random.randint(np.floor(resize_rate * size), size)
    w_s = random.randint(0, size - rsize)
    h_s = random.randint(0, size - rsize)
    sh = random.random()/2 - 0.25
    rotate_angle = random.random()/180*np.pi*angle
    
    # Create Afine transform
    afine_tf = transform.AffineTransform(shear=sh,rotation=rotate_angle)
    
    # Apply transform to image data
    image = transform.warp(image, inverse_map=afine_tf,mode='edge')
    label = transform.warp(label, inverse_map=afine_tf,mode='edge')
    
    # Randomly cropping image frame
    image = image[w_s:w_s+size, h_s:h_s+size,:]
    label = label[w_s:w_s+size, h_s:h_s+size]
    
    # Randomly flip frame
    if flip:
        image = image[:,::-1,:]
        label = label[:,::-1]
        
    return image, label


# get training image ids
image_ids = check_output(["ls", "../input/stage1_train/"]).decode("utf8").split()

# pick one image at random for now
image_id = image_ids[random.randint(0,len(image_ids))]

image, labels = read_image_labels(image_id)
plt.subplot(221)
plt.imshow(image)
plt.subplot(222)
plt.imshow(labels)

new_image, new_labels = data_aug(image, labels, angle = 30, resize_rate = 0.9)
plt.subplot(223)
plt.imshow(new_image)
plt.subplot(224)
plt.imshow(new_labels)










