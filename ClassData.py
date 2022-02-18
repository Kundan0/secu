
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import json
import os

class myDataset(Dataset):
    def __init__(self,dataset_dir,json_dir,depth_dir,height_ratio=3,width_ratio=4):
        self.data=json.load(open(dataset_dir))
        self.json_data=json.load(open(json_dir))
        self.depth_dir=depth_dir
        self.height_ratio=height_ratio
        self.width_ratio=width_ratio
    def __getitem__(self,index):
        track=torch.tensor(self.data[index]['track']).to(torch.float32)
        folder=self.json_data[index]["folder"]
        folder=os.path.join(self.depth_dir,folder,"depth.pt")
        track_index=[0,18,36]
        #print("folder",folder)
        #three_tracks=[self.data[index]["track"][x] for x in track_index]
        depths=torch.load(folder)
        cropped_depths=[]
        for i in range(3):
            track_=self.data[index]["track"][i*18]
            left,top,right,bottom=track_
            #print("not resized  bbox",left,top,right,bottom)
            left=int(left/self.width_ratio)
            top=int(top/self.height_ratio)
            right=int(right/self.width_ratio)
            bottom=int(bottom/self.height_ratio)
            #print("resized bbox",left,top,right,bottom)
            depth=depths[i]
            
            try:
                # print("top -5",top-5)
                # print("bottom +5",bottom+5)
                # print("left -5",left-5)
                # print("right+5",right+5)
                crop=depths[i][max(top-5,0):min(bottom+5,238),max(left-5,0):min(right+5,318)]
            except:
                print("inside exception ",left,top,right,bottom)
                crop=depths[i][top:bottom,left:right]
            cropped_depths.append(crop)
        #three_tracks_centers=[]
        # for i in range(3):
        #     left,top,right,bottom=three_tracks[i][0],three_tracks[i][1],three_tracks[i][2],three_tracks[i][3]
        #     center=(int((top+bottom)/2),int((left+right)/2))
        #     three_tracks_centers.append(center)
        # print("three tracks",three_tracks)
        
        #depths=[depths[i][int(three_tracks[i][1]/self.height_ratio)-5:int(three_tracks[i][3]/self.height_ratio)+5,int(three_tracks[i][0]/self.width_ratio)-5:int(three_tracks[i][2]/self.width_ratio)+5] for i in range(3)]
        
        filtered_depth=[]
        for depth in cropped_depths:
            #print("depth.shape ",depth.shape)
            depth=torch.flatten(depth.detach().cpu()).numpy()
            if len(depth)==0:
                print("empty depth")
            depth = depth[~np.isnan(depth)]
            filtered_depth.append(depth)
        depths=[np.nanmean(x) for x in filtered_depth]
        if depths.count(np.nan)==3:
            print('all nan')
            depths=np.array([60]).repeat(3)
            print("converted nans to ",depths)
        else:
            if np.nan in depths:
                print("gotchyaa hahahaha ")
                depths=np.array([np.nanmean(depths)]).repeat(3)
        #cropping
        # print("after cropping depth size",depths[0].size())
        #flat_depths=[torch.flatten(d).detach().cpu().numpy() for d in depths]
        # avg_depth=[np.nanmean(x.detach().cpu().numpy()).item() for x in depths]
        # std_depth=[np.std(x.detach().cpu().numpy()).item() for x in depths]
        # # print("avg depth before ",avg_depth)
        # # print("std depth before ",std_depth)
        # averages=[]
        # stds=[]
        # #removing outliers
        # for i in range(3):
        #     avg=avg_depth[i]
        #     std=std_depth[i]
        #     filtered=[x for x in flat_depths[i] if x>avg-2*std]
           
        #     avg=[x for x in filtered if x < avg+2*std]
        #     stds.append(np.std(avg))
        #     averages.append(np.nanmean(avg))
            
        # averages=torch.tensor(averages).to(torch.float32)
        # # print("average depth",averages)
        # # print("std after ",stds)
        # depths=[depth.detach().cpu().numpy() for depth in depths]
        # depths=torch.tensor([np.nanmean(depth) for depth in depths]).to(torch.float32)
        # print(depths)

        velocity=torch.tensor(self.data[index]['velocity'])
        position=torch.tensor(self.data[index]['position'])
        label=torch.cat((velocity,position),dim=0).to(torch.float32)
        #print("result forwarded ",(track,averages,label))
        
        return (track,torch.tensor(depths).to(torch.float32),label)


    def __len__(self):
        return (len(self.data))



    #     json_data=self.json_data[index]
    #     folder=json_data["folder"]
    #     an_index=json_data["an_index"]
    #     annotation_data=json.load(open(os.path.join(self.annotation_dir,"annotation"+folder+".json")))[an_index]
    #     velocity=torch.tensor(annotation_data['velocity'])
        
    #     position=torch.tensor(annotation_data["position"])
    #     label=torch.cat((velocity,position),dim=0)

    #     bbox=annotation_data["bbox"]
    #     left=bbox["left"]
    #     top=bbox["top"]
    #     right=bbox["right"]
    #     bottom=bbox["bottom"]
    #     bbox=(left,top,right,bottom)

    #     center=(left+right)/2,(top+bottom)/2
    #     #print("center of bbox",center)

    #     track_result=detect(os.path.join(self.data_dir,folder,"imgs"),self.yolo_model,self.deep_sort)
    #     track_result.pop(0)
    #     track_result.pop(0)
    #     id=None
        
    #     third_frame=track_result[0]
    #     if (len(third_frame)==0):
    #         print("not found for ",folder)
    #     third_frame_track_centers=[]
    #     for values in third_frame:
    #         left,top,right,bottom=values[2]
    #         third_frame_track_centers.append(((right+left)/2,(bottom+top)/2))
        
        

    #     returned_match=self.match(center,third_frame_track_centers,True)
        
    #     id=third_frame[returned_match][1] # returning the id of best match vehicle index 1 stores id and match returns the index of vehicle
    #     #print('original id ',id)
    #     myTracks=[]
        
    #     for idx,frame in enumerate(track_result):# 
    #         found=False #frame=[(16,1,(),2),(16,2,(),2)]
    #         updated_tracks=[]
    #         for values in frame:
    #             left_,top_,right_,bottom_=values[2]
    #             updated_tracks.append(((left_+right_)/2,(top_+bottom_)/2))
                
    #             if values[1]==id:
    #                 #print("tracks ",values[2])
    #                 myTracks.append(values[2])
    #                 last_track_center=updated_tracks[-1]
    #                 found=True
                    
    #         if not found:
    #             print("not found for folder ",folder)
    #             print("frame ",frame)
    #             id_=id
    #             returned_match=self.match(last_track_center,updated_tracks)
    #             if(returned_match is not None):
    #                 id=frame[returned_match][1]
    #                 #print("id changed to ",id)
    #                 for values in frame:
                        
    #                     if values[1]==id:
    #                         #print("new tracks ",values[2])
    #                         myTracks.append(values[2])
    #                         last_track_center=((left_+right_)/2,(top_+bottom_)/2)
                            
                        
                            
    #             elif len(myTracks)>2:
    #                 X_values=np.arange(1,len(myTracks)+1).reshape(-1,1)
                    
    #                 left_values=np.array([val[0] for val in myTracks])
                    
    #                 top_values=np.array([val[1] for val in myTracks])
    #                 right_values=np.array([val[2] for val in myTracks])
    #                 bottom_values=np.array([val[3] for val in myTracks])
    #                 # print("x",X_values)
    #                 # print("l",left_values)
    #                 # print('t',top_values)
    #                 # print('b',bottom_values)
    #                 # print('r',right_values)
    #                 lr=LinearRegression()
    #                 x0=np.array([len(myTracks)+1])
    #                 #print(x0)
    #                 model=lr.fit(X_values,left_values)
    #                 l=model.predict((x0).reshape(-1,1)).item()
    #                 #print("prdicted left is ",l)
    #                 model=lr.fit(X_values,right_values)
    #                 r=model.predict((x0).reshape(-1,1)).item()
    #                 #print("prdicted right is ",r)
    #                 model=lr.fit(X_values,top_values)
    #                 t=model.predict((x0).reshape(-1,1)).item()
    #                 #print("prdicted top is ",t)
    #                 model=lr.fit(X_values,bottom_values)
    #                 b=model.predict((x0).reshape(-1,1)).item()
    #                 #print("prdicted bottom is ",b)

    #                 track_=(l,t,r,b)
    #                 myTracks.append(track_)
    #                 # print("linearly regretted track ",track_)
    #                 last_track_center=((l+r)/2,(t+b)/2)
    #                 id=id_
    #             else:
    #                 myTracks.append(myTracks[-1])
    #                 id=id_
               
    #     myTracks=torch.tensor(myTracks)
        
        
    #     return (myTracks,label)

    # def __len__(self):
    #     return len(self.json_data)
    
    # def match(self,bbox_center,tracks,third=False):
        
    #     loss=[self.Calcloss(bbox_center,track) for track in tracks]
        
    #     mini=min(loss)
    #     # print("distances ",loss)
    #     # print("the minimum distace got is ",mini)
    #     if (mini>self.limit) and not third:
    #         #print("Limit crossed")

    #         return
    #     return loss.index(mini)


    # def Calcloss(self,center1,center2):
    #     return (center1[0]-center2[0])**2+(center1[1]-center2[1])**2
    