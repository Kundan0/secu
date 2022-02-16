
import torch
from sklearn.linear_model import LinearRegression
import numpy as np
from torch.utils.data import Dataset
import json
import os
import sys
sys.path.append(os.path.abspath('./YoloTracker'))
from track import detect
class myDataset(Dataset):
    def __init__(self,yolo_model,deep_sort,annotation_dir,data_dir,json_dir):
        self.annotation_dir=annotation_dir
        self.data_dir=data_dir
        self.json_dir=json_dir
        self.json_data=json.load(open(self.json_dir))
        #self.yolo_model=torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.limit=500
        self.yolo_model=yolo_model
        self.deep_sort=deep_sort
    
    def __getitem__(self,index):
        json_data=self.json_data[index]
        folder=json_data["folder"]
        an_index=json_data["an_index"]
        annotation_data=json.load(open(os.path.join(self.annotation_dir,"annotation"+folder+".json")))[an_index]
        velocity=torch.tensor(annotation_data['velocity'])
        
        position=torch.tensor(annotation_data["position"])
        label=torch.cat((velocity,position),dim=0)

        bbox=annotation_data["bbox"]
        left=bbox["left"]
        top=bbox["top"]
        right=bbox["right"]
        bottom=bbox["bottom"]
        bbox=(left,top,right,bottom)

        center=(left+right)/2,(top+bottom)/2
        print("center of bbox",center)

        track_result=detect(os.path.join(self.data_dir,folder,"imgs"),self.yolo_model,self.deep_sort)
        track_result.pop(0)
        track_result.pop(0)
        id=None
        
        third_frame=track_result[0]
        third_frame_track_centers=[]
        for values in third_frame:
            left,top,right,bottom=values[2]
            third_frame_track_centers.append(((right+left)/2,(bottom+top)/2))
        
        print("tracked centers on third frame",third_frame_track_centers)

        
        id=third_frame[self.match(center,third_frame_track_centers)][1] # returning the id of best match vehicle index 1 stores id and match returns the index of vehicle
        print('original id ',id)
        myTracks=[]
        
        for idx,frame in enumerate(track_result):# 
            found=False #frame=[(16,1,(),2),(16,2,(),2)]
            updated_tracks=[]
            for values in frame:
                left_,top_,right_,bottom_=values[2]
                updated_tracks.append(((left_+right_)/2,(top_+bottom_)/2))
                
                if values[1]==id:
                    print("tracks ",values[2])
                    myTracks.append(values[2])
                    last_track_center=updated_tracks[-1]
                    found=True
                    
            if not found:
                print("not found for folder ",folder)
                print("frame ",frame)
                id_=id
                returned_match=self.match(last_track_center,updated_tracks)
                if(returned_match is not None):
                    id=frame[returned_match][1]
                    print("id changed to ",id)
                    for values in frame:
                        
                        if values[1]==id:
                            print("new tracks ",values[2])
                            myTracks.append(values[2])
                            last_track_center=((left_+right_)/2,(top_+bottom_)/2)
                            
                        
                            
                else:
                    X_values=np.arange(1,len(myTracks)+1).reshape(-1,1)
                    
                    left_values=np.array([val[0] for val in myTracks])
                    
                    top_values=np.array([val[1] for val in myTracks])
                    right_values=np.array([val[2] for val in myTracks])
                    bottom_values=np.array([val[3] for val in myTracks])
                    print("x",X_values)
                    print("l",left_values)
                    print('t',top_values)
                    print('b',bottom_values)
                    print('r',right_values)
                    lr=LinearRegression()
                    model_left=lr.fit(X_values,left_values)
                    model_right=lr.fit(X_values,right_values)
                    model_top=lr.fit(X_values,top_values)
                    model_bottom=lr.fit(X_values,bottom_values)
                    l=model_left.predict(np.array([len(myTracks)+1]).reshape(-1,1)).item()
                    t=model_top.predict(np.array([len(myTracks)+1]).reshape(-1,1)).item()
                    r=model_right.predict(np.array([len(myTracks)+1]).reshape(-1,1)).item()
                    b=model_bottom.predict(np.array([len(myTracks)+1]).reshape(-1,1)).item()
                    print("left",l)
                    print("right",r)
                    print('bottom',b)
                    print("top",t)
                    track_=(l,t,r,b)
                    myTracks.append(track_)
                    print("linearly regretted track ",track_)
                    last_track_center=((l+r)/2,(t+b)/2)
                    id=id_
                
               
        myTracks=torch.tensor(myTracks)
        
        
        return (myTracks,label)

    def __len__(self):
        return len(self.json_data)
    
    def match(self,bbox_center,tracks):
        
        loss=[self.Calcloss(bbox_center,track) for track in tracks]
        
        mini=min(loss)
        print("distances ",loss)
        print("the minimum distace got is ",mini)
        if (mini>self.limit):
            print("Limit crossed")
            return 
        return loss.index(mini)


    def Calcloss(self,center1,center2):
        return (center1[0]-center2[0])**2+(center1[1]-center2[1])**2
    