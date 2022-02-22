import torch
from sklearn.linear_model import LinearRegression
import numpy as np
from torch.utils.data import Dataset
import json
import os
import sys
import pickle

sys.path.append(os.path.abspath('./yolov5'))
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from yolov5.models.common import DetectMultiBackend
from track import detect
class GenTracks():
    def __init__(self,yolo_model,deep_sort,annotation_dir,data_dir,json_dir):
        self.annotation_dir=annotation_dir
        self.data_dir=data_dir
        self.json_dir=json_dir
        self.json_data=json.load(open(self.json_dir))
        #self.yolo_model=torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.limit=900 # 30pixels limit
        self.yolo_model=yolo_model
        self.deep_sort=deep_sort
    def __getitem__(self,index):
        
        json_data=self.json_data[index]
        folder=json_data["folder"]
        an_index=json_data["an_index"]
        print("index {} folder ".format(index,folder))
        if folder=="301":
            print('folder 301 to 333')
            folder="333"
            an_index=0
        
        annotation_data=json.load(open(os.path.join(self.annotation_dir,"annotation"+folder+".json")))[an_index]
        velocity=annotation_data['velocity']
        
        position=annotation_data["position"]
        

        bbox=annotation_data["bbox"]
        left=bbox["left"]
        top=bbox["top"]
        right=bbox["right"]
        bottom=bbox["bottom"]
        bbox=(left,top,right,bottom)

        center=(left+right)/2,(top+bottom)/2
        #print("center of bbox",center)

        track_result=detect(os.path.join(self.data_dir,folder,"imgs"),self.yolo_model,self.deep_sort)
        track_result.pop(0)
        track_result.pop(0)
        id=None
        
        last_frame=track_result[-1]
        if (len(last_frame)==0):
            print("not tracked in last frame  ",folder)
        last_frame_track_centers=[]
        for values in last_frame:
            left,top,right,bottom=values[2]
            last_frame_track_centers.append(((right+left)/2,(bottom+top)/2))
        
        

        returned_match=self.match(center,last_frame_track_centers)
        
        id=last_frame[returned_match][1] # returning the id of best match vehicle index 1 stores id and match returns the index of vehicle
        #print('original id ',id)
        myTracks=[]
        
        for idx,frame in enumerate(track_result[::-1]):# 
            found=False #frame=[(16,1,(),2),(16,2,(),2)]
            updated_tracks=[]
            for values in frame:
                left_,top_,right_,bottom_=values[2]
                updated_tracks.append(((left_+right_)/2,(top_+bottom_)/2))
                
                if values[1]==id:
                    #print("tracks ",values[2])
                    myTracks.append(values[2])
                    last_track_center=updated_tracks[-1]
                    found=True
                    
            if not found:
                print("id not found for folder ",folder,"index",idx,"trying to match from last tracks ")
                
                id_=id
                returned_match=self.match(last_track_center,updated_tracks)
                if(returned_match is not None):
                    id=frame[returned_match][1]
                    #print("id changed to ",id)
                    for values in frame:
                        
                        if values[1]==id:
                            print("found id ,added ",values[2])
                            left_,top_,right_,bottom_=values[2]
                            #print("new tracks ",values[2])
                            myTracks.append(values[2])
                            last_track_center=((left_+right_)/2,(top_+bottom_)/2)
                            
                        
                            
                elif len(myTracks)>3:
                    
                    
                    X_values=np.arange(1,len(myTracks)+1).reshape(-1,1)
                    
                    left_values=np.array([val[0] for val in myTracks])
                    
                    top_values=np.array([val[1] for val in myTracks])
                    right_values=np.array([val[2] for val in myTracks])
                    bottom_values=np.array([val[3] for val in myTracks])
                    # print("x",X_values)
                    # print("l",left_values)
                    # print('t',top_values)
                    # print('b',bottom_values)
                    # print('r',right_values)
                    lr=LinearRegression()
                    x0=np.array([len(myTracks)+1])
                    #print(x0)
                    model=lr.fit(X_values,left_values)
                    l=model.predict((x0).reshape(-1,1)).item()
                    #print("prdicted left is ",l)
                    model=lr.fit(X_values,right_values)
                    r=model.predict((x0).reshape(-1,1)).item()
                    #print("prdicted right is ",r)
                    model=lr.fit(X_values,top_values)
                    t=model.predict((x0).reshape(-1,1)).item()
                    #print("prdicted top is ",t)
                    model=lr.fit(X_values,bottom_values)
                    b=model.predict((x0).reshape(-1,1)).item()
                    #print("prdicted bottom is ",b)

                    track_=(int(l),int(t),int(r),int(b))
                    myTracks.append(track_)
                    print("linearly regretted track ",track_)
                    last_track_center=((l+r)/2,(t+b)/2)
                    id=id_
                else:
                    print("copying same")
                    myTracks.append(myTracks[-1])
                    id=id_
               
        myTracks=myTracks[::-1]
        
        
        return {"track":myTracks,"velocity":velocity,"position":position}

    def __len__(self):
        return len(self.json_data)
    
    def match(self,bbox_center,tracks):
        
        loss=[self.Calcloss(bbox_center,track) for track in tracks]
        
        mini=min(loss)
        # print("distances ",loss)
        print("the minimum distace got is ",mini)
        if (mini>self.limit):
            print("Limit crossed")

            return
        return loss.index(mini)


    def Calcloss(self,center1,center2):
        return (center1[0]-center2[0])**2+(center1[1]-center2[1])**2

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)    

if __name__=="__main__":

    PATH=os.path.join("/content")
    PATHJ=os.path.join("/content","secu")
    PATHS=os.path.join(PATH,"drive","MyDrive","State2")

    data_dir=os.path.join(PATH,"clips")
    an_dir=os.path.join(PATHJ,"Annotations")
    json_dir=os.path.join(PATHJ,"JSON.json")
    #device 
    device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    print(device)    




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

    # Load model

    yolo_model="yolov5m.pt"
    dnn=False
    yolo_model = DetectMultiBackend(yolo_model, device=device, dnn=dnn)


    myObject=GenTracks(yolo_model,deepsort,an_dir,data_dir,json_dir)
    dataset=[]
    for index in range(len(myObject)):
        dataset.append(myObject[index])
        
    with open('dataset.json','w') as f:
        json.dump(dataset,f,cls=NpEncoder)
    # with open('dataset.json') as f:
    #     data=json.load(f)
    #     track=data[0]["track"]
    #     vel=data[0]["velocity"]
    #     pos=data[0]["position"]
    #     print(torch.tensor(track))
    #     print(torch.tensor(vel))
    #     print(torch.tensor(pos))
        