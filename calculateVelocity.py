from pyexpat import model
from track import detect
from CalculateDepth import ret_depth,load_ADA
from ClassModel import myModel
import torch
import sys
import os
import cv2
import numpy as np
sys.path.append(os.path.abspath('./yolov5'))
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from yolov5.models.common import DetectMultiBackend

#device
device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    


#load depth_model
depth_model=load_ADA('./AdaBins_kitti.pt',device)

 #Deepsort
config_deepsort="deep_sort/configs/deep_sort.yaml"
deep_sort_model="osnet_x0_25"
cfg = get_config()
cfg.merge_from_file(config_deepsort)

deepsort = DeepSort(deep_sort_model,
                    device,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    )


#load yolov5 tracker
yolo_model = DetectMultiBackend("yolov5m.pt", device=device, dnn=False)

#velocity_model

model1=myModel('./model.pt',device)
model1.load_model()

#video_info
#video 
file_path='./imgs_video.avi'
output_file_name='./output.avi'
video=cv2.VideoCapture(file_path)
frame_width = int(video.get(3))
frame_height = int(video.get(4))
frame_size=(frame_width,frame_height)
FPS=video.get(cv2.CAP_PROP_FPS)
video_writer= cv2.VideoWriter(output_file_name, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         FPS, frame_size)
ROI=(frame_width/2,frame_width,frame_height/3,frame_height) # Region Of Interest (left,right,top,bottom)
#bounding box
RECT_COLOR_BBOX=(0,255,255) #yellow #BGR
RECT_COLOR_ROI=(0,255,0)
TEXT_COLOR=(0,255,255)



#Checking if the video is loaded successfully

if (video.isOpened()):
    print("Successfully Opened Video :) ")
else:
    print("Error Opening Video")
    exit()

#frames=get_40_frames(file_path)
depths=ret_depth(frames) # frames is the list of np.ndarray
