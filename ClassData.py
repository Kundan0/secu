
import torch
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

        track_result=detect(os.path.join(self.data_dir,folder,"imgs"),self.yolo_model,self.deep_sort)
        track_result.pop(0)
        track_result.pop(0)
        id=None
        
        third_frame=track_result[0]
        third_frame_track_centers=[]
        for values in third_frame:
            left,top,right,bottom=values[2]
            third_frame_track_centers.append(((right+left)/2,(bottom+top)/2))
        

        
        id=third_frame[self.match(center,third_frame_track_centers)][1] # returning the id of best match vehicle index 1 stores id and match returns the index of vehicle
        myTracks=[]
        
        for frame in track_result:# 
            found=False #result=[(16,1,(),2),(16,2,(),2)]
            updated_tracks=[]
            for values in frame:
                left_,top_,right_,bottom_=values[2]
                updated_tracks.append(((left_+right_)/2,(top_+bottom_)/2))
                  
                if values[1]==id:
                    
                    myTracks.append(values[2])
                    last_track_center=updated_tracks[-1]
                    found=True
                    
            if not found:
                print("not found")
                try:
                    id=frame[self.match(last_track_center,updated_tracks)][1]
                    
                    for values in frame:
                        
                        if values[1]==id:
                            
                            myTracks.append(values[2])
                            last_track_center=((left_+right_)/2,(top_+bottom_)/2)
                            
                        
                            
                except:
                    print("Vehicle Not found")
                    sys.exit()
            
                
               
        myTracks=torch.tensor(myTracks)
        print(myTracks)
        print(myTracks.size())
        return (myTracks,label)

    def __len__(self):
        return len(self.json_data)
    
    def match(self,bbox_center,tracks):
        
        loss=[self.Calcloss(bbox_center,track) for track in tracks]
        
        mini=min(loss)
        return loss.index(mini)


    def Calcloss(self,center1,center2):
        return (center1[0]-center2[0])**2+(center1[1]-center2[1])**2
    