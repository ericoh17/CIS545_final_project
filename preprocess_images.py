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

from skimage import io, transform
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from scipy import ndimage
#import tensorflow as tf

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))

def read_image_labels(image_id):

    image_file = "../input/stage1_train/{}/images/{}.png".format(image_id,image_id)
    mask_file = "../input/stage1_train/{}/masks/*.png".format(image_id)
    image = io.imread(image_file)
    masks = io.imread_collection(mask_file).concatenate()    
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


def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten() > 0)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])

# get training image ids
image_ids = check_output(["ls", "../input/stage1_train/"]).decode("utf8").split()

# pick one image at random for now
image_id = image_ids[random.randint(0,len(image_ids))]

image, labels = read_image_labels(image_id)
#plt.subplot(221)
#plt.imshow(image)
#plt.subplot(222)
#plt.imshow(labels)

new_image, new_labels = data_aug(image, labels, angle = 30, resize_rate = 0.9)
#plt.subplot(223)
#plt.imshow(new_image)
#plt.subplot(224)
#plt.imshow(new_labels)

# convert image to greyscale
from skimage.color import rgb2gray
im_gray = rgb2gray(image)

# remove background
from skimage.filters import threshold_otsu, rank, threshold_local

# threshold otsu
thresh_val = threshold_otsu(im_gray)
mask_threshold = np.where(im_gray > thresh_val, 1, 0)

# local thresholding with otsu
#ubyte_im_gray = img_as_ubyte(im_gray)
#radius = 15
#selem = disk(radius)

#local_otsu = rank.otsu(ubyte_im_gray, selem)
#threshold_local_otsu = ubyte_im_gray >= local_otsu
#mask_threshold = np.where(im_gray > threshold_local_otsu, 1, 0)

# Make sure the larger portion of the mask is considered background
if np.sum(mask_threshold == 0) < np.sum(mask_threshold == 1):
    mask_threshold = np.where(mask_threshold, 0, 1)
    
new_labels, nlabels = ndimage.label(mask_threshold)

# Loop through labels and add each to a DataFrame
im_df = pd.DataFrame()
for label_num in range(1, nlabels + 1):
    label_mask = np.where(labels == label_num, 1, 0)
    if label_mask.flatten().sum() > 10:
        rle = rle_encoding(label_mask)
        s = pd.Series({'ImageId': image_id, 'EncodedPixels': rle})
        im_df = im_df.append(s, ignore_index=True)
        

def iou_at_thresholds(target_mask, pred_mask, thresholds=np.arange(0.5,1,0.05)):
    '''Returns True if IoU is greater than the thresholds.'''
    intersection = np.logical_and(target_mask, pred_mask)
    union = np.logical_or(target_mask, pred_mask)
    iou = np.sum(intersection > 0) / np.sum(union > 0)
    return iou > thresholds

def calculate_average_precision(target_masks, pred_masks, thresholds=np.arange(0.5,1,0.05)):
    '''Calculates the average precision over a range of thresholds for one observation (with a single class).'''
    iou_tensor = np.zeros([len(thresholds), len(pred_masks), len(target_masks)])

    for i, p_mask in enumerate(pred_masks):
        for j, t_mask in enumerate(target_masks):
            iou_tensor[:, i, j] = iou_at_thresholds(t_mask, p_mask, thresholds)

    TP = np.sum((np.sum(iou_tensor, axis=2) == 1), axis=1)
    FP = np.sum((np.sum(iou_tensor, axis=1) == 0), axis=1)
    FN = np.sum((np.sum(iou_tensor, axis=2) == 0), axis=1)

    precision = TP / (TP + FP + FN)

    return np.mean(precision)












