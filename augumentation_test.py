#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 16:47:52 2021

@author: tymek
"""
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A

SIZE_X = 256 
SIZE_Y = 256

def load_dataset():
    training_images = []
    
    for training_img_path in glob.glob("./IMAGE_PATCHES/*.png"):
        img = cv2.imread(training_img_path, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        training_images.append(img)
    
    training_images_masks = []
    
    for mask_path in glob.glob("./MASKS_PATCHES/*.png"):
        img = cv2.imread(mask_path, 0)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        training_images_masks.append(img)
        
    # Convert lists into numpy arrays
    training_images = np.array(training_images, dtype=np.uint8)
    training_images_masks = np.array(training_images_masks, dtype=np.uint8)
    
    return training_images, training_images_masks

training_images, training_images_masks = load_dataset()
training_images_masks = np.expand_dims(training_images_masks, axis=3)

x_train, x_val, y_train, y_val = train_test_split(training_images, training_images_masks, test_size=0.2, random_state=42)

# augument data

# transform = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     # A.Flip(p=0.5),
#     A.RandomBrightnessContrast(p=1, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.2, 0.2)),
#     A.Blur(p=0.2, blur_limit=(1, 2)),
# ])

transform = A.Compose([
    A.Transpose(always_apply=True, p=0.5),
    A.HorizontalFlip(always_apply=True, p=0.8),
    A.VerticalFlip(always_apply=True, p=0.8),
    A.RandomRotate90(always_apply=True, p=0.8),
    A.RandomBrightnessContrast(always_apply=True, p=0.8, brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3)),
    A.Blur(always_apply=True, p=0.5, blur_limit=(3, 7)),
    A.GaussNoise(always_apply=True, p=0.5, var_limit = (10.0,200.0)),
])

augumented_x = list()
augumented_y = list()
for img, mask in zip(x_train, y_train):
    transformed = transform(image=img, mask=mask)
    augumented_x.append(transformed['image'])
    augumented_y.append(transformed['mask'])

augumented_x = np.array(augumented_x, dtype=np.uint8)
augumented_y = np.array(augumented_y, dtype=np.uint8)


uniqueid=0
for img in augumented_x:
    path_mask = ("./IMAGE_PATCHES_AUGUMENTED/" + "image_id"+ str(uniqueid) + ".png")
    uniqueid+=1
    cv2.imwrite(path_mask,img)

# reconstructed_image = unpatchify(patches_img, expanded_image.shape)  