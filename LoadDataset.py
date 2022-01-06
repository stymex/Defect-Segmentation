#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 13:08:03 2021

@author: stymex
"""
import glob
import cv2
import numpy as np

#Resize images (height  = X, width = Y)
# SIZE_X = 480 
# SIZE_Y = 1152

SIZE_X = 256 
SIZE_Y = 256

def load_dataset():
    training_images = []
    
    for training_img_path in glob.glob("./IMAGE_PATCHES/*.png"):
        img = cv2.imread(training_img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        training_images.append(img)
    
    training_images_masks = []
    
    for mask_path in glob.glob("./MASKS_PATCHES/*.png"):
        img = cv2.imread(mask_path, 0)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        training_images_masks.append(img)
        
    # Convert lists into numpy arrays
    training_images = np.array(training_images, dtype=object).astype(np.float32)/255.0
    training_images_masks = np.array(training_images_masks, dtype=object).astype(np.float32)/255.0
    
    return training_images, training_images_masks