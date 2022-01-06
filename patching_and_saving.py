#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 16:01:29 2021

@author: stymex
"""

import glob
import cv2
import numpy as np
import segmentation_models as sm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import random
from patchify import patchify, unpatchify
from math import ceil

#Resize images (height  = X, width = Y)
SIZE_X = 480
SIZE_Y = 1152

def load_dataset():
    training_images = []
    
    for training_img_path in glob.glob("./IMAGES/*_crop_2.png"):
        img = cv2.imread(training_img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        training_images.append(img)
    
    training_images_masks = []
    
    for mask_path in glob.glob("./MASKS/*_crop_2.png"):
        img = cv2.imread(mask_path, 0)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        training_images_masks.append(img)
        
    # Convert lists into numpy arrays
    training_images = np.array(training_images, dtype=object).astype(np.uint8)
    training_images_masks = np.array(training_images_masks, dtype=object).astype(np.uint8)
    
    return training_images, training_images_masks

training_images, training_images_masks = load_dataset()

patch_size = 256
uniqueid=0
for img in training_images:
    # img = training_images[0]
    expanded_image = np.zeros((ceil(training_images[0].shape[0]/patch_size)*patch_size, ceil(training_images[0].shape[1]/patch_size)*patch_size, 3), dtype=np.uint8)
    expanded_image[0:SIZE_X, 0:SIZE_Y] = img[0:SIZE_X, 0:SIZE_Y]
    patches_img = patchify(expanded_image, (256, 256, 3), step=256)
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i,j,:,:]
            path_img = ("./IMAGE_PATCHES/" + "image_id"+ str(uniqueid) + "_" + str(i)+ "_" +str(j)+ ".png")
            uniqueid+=1
            cv2.imwrite(path_img,single_patch_img[0])
    

patch_size = 256
uniqueid=0
for img in training_images_masks:
    # img = training_images[0]
    expanded_mask = np.zeros((ceil(training_images_masks[0].shape[0]/patch_size)*patch_size, ceil(training_images_masks[0].shape[1]/patch_size)*patch_size), dtype=np.uint8)
    expanded_mask[0:SIZE_X, 0:SIZE_Y] = img[0:SIZE_X, 0:SIZE_Y]
    patches_img = patchify(expanded_mask, (256, 256), step=256)
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i,j,:,:]
            path_mask = ("./MASKS_PATCHES/" + "image_id"+ str(uniqueid) + "_" + str(i)+ "_" +str(j)+ ".png")
            uniqueid+=1
            cv2.imwrite(path_mask,single_patch_img)

# reconstructed_image = unpatchify(patches_img, expanded_image.shape)  