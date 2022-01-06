#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 12:51:02 2021

@author: tymek
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
from LoadDataset import load_dataset
import albumentations as A

resume = sys.argv[-1] == "--resume"

defaults = dict(
    optimizer="Adam",
    epoch=100,
    batch_size=64,
    backbone="resnet34",
    architecture="Linknet"
    )

wandb.init(project="segmentation-pitting", config=defaults, resume=resume)
config = wandb.config

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
model = sm.Linknet(BACKBONE,
                   input_shape=(None, None, 3), 
                   classes=1, activation='sigmoid', 
                   weights=None, 
                   encoder_weights='imagenet', 
                   encoder_freeze=True, 
                   encoder_features='default', 
                   decoder_block_type='transpose', 
                   decoder_filters=(None, None, None, None, 16), 
                   decoder_use_batchnorm=True)

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