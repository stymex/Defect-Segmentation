#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:11:29 2021

@author: stymex
"""
import numpy as np
import json
import cv2
import glob

def json2mask(json: object):
    h=json['imageHeight']
    w=json['imageWidth']
    
    shapes = json['shapes']
    mask = np.zeros((h, w))
    
    for shape in shapes:
        shape_points = shape['points']
        defect = []
        for point_pair in shape_points:
            defect.append([round(point_pair[0]), round(point_pair[1])])
        
        cv2.fillPoly(mask, pts = [np.array(defect)], color=[255,255,255], lineType=8, shift=0)

    return mask


# image_paths = glob.glob('./BSDATA/BSData-main/data/*.jpg')
mask_paths = glob.glob('./BSDATA/BSData-main/label/*.json')
# mask_paths = glob.glob('./BSDATA/BSData-main/label/01_200906233700_635000_270_crop_2.json')

# p = mask_paths[0]

for p in mask_paths:
    f = open(p)
    # print(json.dumps(p))
    j = json.load(f)
    
    #generate and save mask
    mask = json2mask(j)
    path = './MASKS/' + j['imagePath']
    path = path.replace('.jpg', '.png')
    cv2.imwrite(path,mask)
    
    #move images with defects to separate directory
    path_read = './BSDATA/BSData-main/data/'+j['imagePath']
    img = cv2.imread(path_read)
    image_path = './IMAGES/' + j['imagePath']
    image_path = image_path.replace('.jpg', '.png')
    cv2.imwrite(image_path,img)
