
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import json
import os

class myDataset(Dataset):
    def __init__(self,dataset_dir,json_dir,depth_dir,map_dir,height_ratio=3,width_ratio=4):
        self.data=json.load(open(dataset_dir))
        self.json_data=json.load(open(json_dir))
        self.depth_dir=depth_dir
        self.height_ratio=height_ratio
        self.width_ratio=width_ratio
        self.map_data=json.load(open(map_dir))
    def __getitem__(self,index):
        index=self.map_data[2][index]
        track=torch.tensor(self.data[index]['track']).to(torch.float32)
        
        folder=self.json_data[index]["folder"]
        folder=os.path.join(self.depth_dir,folder,"depth.pt")
        # track_index=[0,18,36]
        #print("folder",folder)
        #three_tracks=[self.data[index]["track"][x] for x in track_index]
        depths=torch.load(folder)
        depths=[(depth-torch.mean(depth))/(torch.std(depth)) for depth in depths]
        cropped_depths=[]
        for i in range(len(depths)):
            track_=self.data[index]["track"][i*18]
            left,top,right,bottom=track_
            #print("not resized  bbox",left,top,right,bottom)
            left=round(left/self.width_ratio)
            top=round(top/self.height_ratio)
            right=round(right/self.width_ratio)
            bottom=round(bottom/self.height_ratio)
            #print("resized bbox",left,top,right,bottom)
            depth=depths[i]
            cropped_depths.append(depth[top:bottom,left:right])

        flat_depths=[torch.flatten(d).detach().cpu().numpy() for d in cropped_depths]
        avg_depth=[np.nanmean(x.detach().cpu().numpy()).item() for x in cropped_depths]
        std_depth=[np.std(x.detach().cpu().numpy()).item() for x in cropped_depths]
        # print("avg depth before ",avg_depth)
        # print("std depth before ",std_depth)
        averages=[]
        stds=[]
        #removing outliers
        for i in range(3):
            avg=avg_depth[i]
            std=std_depth[i]
            filtered=[x for x in flat_depths[i] if x>avg-std]
           
            avg=[x for x in filtered if x < avg+std]
            stds.append(np.std(avg))
            averages.append(np.nanmean(avg))
            
        averages=torch.tensor(averages).to(torch.float32)
        # print("average depth after ",averages)
        # print("std after ",stds)
        
        

        velocity=torch.tensor(self.data[index]['velocity'])
        position=torch.tensor(self.data[index]['position'])
        label=torch.cat((velocity,position),dim=0).to(torch.float32)
        #print("result forwarded ",(track,averages,label))
        permutted_track=track.permute(1,0)
        permutted_track[0]=permutted_track[0]/1280
        permutted_track[2]=permutted_track[2]/1280
        permutted_track[1]=permutted_track[1]/720
        permutted_track[3]=permutted_track[3]/720
        track=permutted_track.permute(1,0)
        return (track,averages,label)


    def __len__(self):
        return (len(self.map_data[2]))


# import torch
# import cv2
# import numpy as np
# from torch.utils.data import Dataset
# import json
# import os

# class myDataset(Dataset):
#     def __init__(self,dataset_dir,json_dir,depth_dir,map_dir,height_ratio=3,width_ratio=4):
#         self.data=json.load(open(dataset_dir))
#         self.json_data=json.load(open(json_dir))
#         self.depth_dir=depth_dir
#         self.height_ratio=height_ratio
#         self.width_ratio=width_ratio
#         self.map_data=json.load(open(map_dir))
#     def __getitem__(self,index):
#         index=self.map_data[2][index]
#         track=torch.tensor(self.data[index]['track']).to(torch.float32)
        
#         folder=self.json_data[index]["folder"]
#         folder=os.path.join(self.depth_dir,folder,"depth.pt")
#         track_index=[0,18,36]
#         #print("folder",folder)
#         #three_tracks=[self.data[index]["track"][x] for x in track_index]
#         depths=torch.load(folder)
#         depths=[(depth-torch.mean(depth))/(torch.std(depth)) for depth in depths]
#         cropped_depths=[]
#         for i in range(len(depths)):
#             track_=self.data[index]["track"][i*18]
#             left,top,right,bottom=track_
#             #print("not resized  bbox",left,top,right,bottom)
#             left=round(left/self.width_ratio)
#             top=round(top/self.height_ratio)
#             right=round(right/self.width_ratio)
#             bottom=round(bottom/self.height_ratio)
#             #print("resized bbox",left,top,right,bottom)
#             depth=depths[i]
#             cropped_depths.append(depth[top:bottom,left:right])

#         flat_depths=[torch.flatten(d).detach().cpu().numpy() for d in cropped_depths]
#         avg_depth=[np.nanmean(x.detach().cpu().numpy()).item() for x in cropped_depths]
#         std_depth=[np.std(x.detach().cpu().numpy()).item() for x in cropped_depths]
#         # print("avg depth before ",avg_depth)
#         # print("std depth before ",std_depth)
#         averages=[]
#         stds=[]
#         #removing outliers
#         for i in range(3):
#             avg=avg_depth[i]
#             std=std_depth[i]
#             filtered=[x for x in flat_depths[i] if x>avg-std]
           
#             avg=[x for x in filtered if x < avg+std]
#             stds.append(np.std(avg))
#             averages.append(np.nanmean(avg))
            
#         averages=torch.tensor(averages).to(torch.float32)
#         # print("average depth after ",averages)
#         # print("std after ",stds)
        
        

#         velocity=torch.tensor(self.data[index]['velocity'])
#         position=torch.tensor(self.data[index]['position'])
#         label=torch.cat((velocity,position),dim=0).to(torch.float32)
#         #print("result forwarded ",(track,averages,label))
#         permutted_track=track.permute(1,0)
#         permutted_track[0]=permutted_track[0]/1280
#         permutted_track[2]=permutted_track[2]/1280
#         permutted_track[1]=permutted_track[1]/720
#         permutted_track[3]=permutted_track[3]/720
#         track=permutted_track.permute(1,0)
#         return (track,averages,label)


#     def __len__(self):
#         return (len(self.map_data[2]))

