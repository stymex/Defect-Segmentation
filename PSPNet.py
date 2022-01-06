#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 13:35:38 2021

@author: stymex

"""
import sys
import glob
import cv2
import numpy as np
import segmentation_models as sm
from segmentation_models import losses
from segmentation_models import metrics
from segmentation_models.losses import bce_dice_loss
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback
import wandb
import albumentations as A

resume = sys.argv[-1] == "--resume"

defaults = dict(
    optimizer="rmsprop",
    epoch=100,
    batch_size=64,
    backbone="resnet34",
    architecture="PSPNet"
    )

wandb.init(project="segmentation-pitting", config=defaults, resume=resume)
config = wandb.config

SIZE_X = 240 
SIZE_Y = 240

def load_dataset():
    training_images = []
    
    for training_img_path in glob.glob("./IMAGE_PATCHES/*.png"):
        img = cv2.imread(training_img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        training_images.append(img)
    
    training_images_masks = []
    
    for mask_path in glob.glob("./MASKS_PATCHES/*.png"):
        img = cv2.imread(mask_path, 0)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        training_images_masks.append(img)
        
    # Convert lists into numpy arrays
    training_images = np.array(training_images, dtype=object).astype(np.float32)
    training_images = training_images/255.0
    training_images_masks = np.array(training_images_masks, dtype=object).astype(np.float32)/255.0
    
    return training_images, training_images_masks

training_images, training_images_masks = load_dataset()
training_images_masks = np.expand_dims(training_images_masks, axis=3)

BACKBONE = config.backbone
preprocess_input = sm.get_preprocessing(BACKBONE)

x_train, x_val, y_train, y_val = train_test_split(training_images, training_images_masks, test_size=0.2, random_state=42)

# augument data

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

augumented_x = np.array(augumented_x, dtype=np.float32)
augumented_y = np.array(augumented_y, dtype=np.float32)

x_train = np.concatenate((x_train, augumented_x), axis=0)
y_train = np.concatenate((y_train, augumented_y), axis=0)

# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)
    
# define model
model = sm.PSPNet(BACKBONE, input_shape=(240, 240, 3), classes=1, activation='sigmoid', weights=None, encoder_weights='imagenet', encoder_freeze=True, downsample_factor=4, psp_conv_filters=256, psp_pooling_type='avg', psp_use_batchnorm=True, psp_dropout=0.2)
model.compile(
    optimizer = config.optimizer,
    loss=bce_dice_loss,
    metrics=[metrics.iou_score, metrics.f1_score, metrics.f2_score, metrics.precision, metrics.recall],
)

# WandbCallback auto-saves all metrics from model.fit(), plus predictions on validation_data
logging_callback = WandbCallback(log_evaluation=True)

history = model.fit(x=x_train, y=y_train,
                    epochs=config.epoch,
                    batch_size=config.batch_size,
                    validation_data=(x_val, y_val),
                    callbacks=[logging_callback]
                    )

# Mark the run as finished
wandb.finish()