#!/usr/bin/env python
# coding: utf-8


import numpy as np
import cv2
import os
import math
import shutil
import os
import random




# To train your own model;
# - Capture the frames using CaptureFrames function and do TrainValidTestSplit.
# - Using "labelimg" library, create annotations. Dont forget to set it YOLO format. 
# - Modify "dataset.yml" file in "yolov5" folder accordingly.
# - Then train with the following. Options can be changed for sure. Jupyter crashes so use the terminal.
# - To train the model
# 
#         python train.py --batch 2 --epochs 300 --data dataset.yml --weights yolov5s.pt --workers 2 --cache 
#   
#  
# - For inspection, load the best in the path model
# 
#         model = torch.hub.load('ultralytics/yolov5', 'custom', 
#                        path='yolov5/runs/train/exp/weights/best.pt', force_reload=True)
# 

#Create a dataset folder and work in it.
#original video data
video_path = 'dataset/myvideo.mp4'

# raw snapshot dir
image_dir = '/dataset/raw/'
#
# #train,valid and test directories each including 'images' and 'labels' folders
train_dir = '/dataset/train/'
valid_dir = '/dataset/valid/'
test_dir = '/dataset/test/'


#get some information about the video
def GetVideoInfo(video_path):
    cap = cv2.VideoCapture(video_path)
    frameRate = math.floor(cap.get(5))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_lenght_seconds = total_frames/frameRate
    video_lenght_minutes = video_lenght_seconds/60
    print(f'frame_rate: {frameRate}\ntotal_frames: {total_frames}')
    print(f'length(s): {video_lenght_seconds:0.1f}\nlength(mins) : {video_lenght_minutes:0.2f}')
    cap.release()

def CaptureFrames(video_path, image_dir, save_interval=1,key=''):
    """
    capture frames in every save_intervals(seconds)
    """
    cap = cv2.VideoCapture(video_path)
    frameRate = cap.get(5)
    count = 0
    while(cap.isOpened()):
        #get the current frame number
        frameId = cap.get(1) 
        
        #read the frame
        success, frame = cap.read()
        if (success != True):
            break
            
        #framID/frameRate=0 means every second   
        if (frameId % math.floor(frameRate*save_interval) == 0):
            count+=1
            filename = os.path.join(image_dir,str(int(frameId))+key+'.jpg')
            cv2.imwrite(filename, frame)
    cap.release()
    print(f'{count} frames captured!')


def TrainValidTestSplit(image_dir,train_dir,valid_dir,test_dir,
                   train_ratio=0.7,valid_ratio=0.2,test_ratio=0.1):
    """
    get images from image_dir and save to train_dir,valid_dir,test_dir based on ratios 
    Run only once, no safeguard to avoid duplicates. This is random but depending on the
    problem, randomness may not be the best idea. Think about it.
    """
    #get all image paths and shuffle them up
    image_paths = []
    for (dirpath, dirnames, filenames) in os.walk(image_dir):
        for names in filenames:
            image_paths.append(os.path.join(dirpath,names))

    random.shuffle(image_paths)

    train_image_list,temp = np.split(np.array(image_paths),[math.ceil(len(image_paths)*(train_ratio))])
    valid_image_list,test_image_list = np.split(temp,[math.ceil(len(temp)*valid_ratio/(valid_ratio+test_ratio))])
    
    
    
    for train_image in train_image_list:
        shutil.copy(train_image,train_dir+'/images')

    for valid_image in valid_image_list:
        shutil.copy(valid_image,valid_dir+'/images')
    
    for test_image in test_image_list:
        shutil.copy(test_image,test_dir+'/images')
        
    print(f'train_image: {len(train_image_list)} \nvalid_image: {len(valid_image_list)} \
          \ntest_image: {len(test_image_list)}')

    print('Done!')


#check the video info
GetVideoInfo(video_path)

#decide how many frames you need to capture. Default is in every 1 second.
CaptureFrames(video_path, image_dir, save_interval=1,key='')

#decide the ratio of train/valid/test split. 70/20/10 is standard
TrainValidTestSplit(image_dir,train_dir,valid_dir,test_dir,
                   train_ratio=0.7,valid_ratio=0.2,test_ratio=0.1):
