from pyexpat import model
from track import detect
from CalculateDepth import ret_depth,load_ADA
from ClassModel import myModel
import torch
import sys
import os
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

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
width_ratio=frame_width/320
height_ratio=frame_height/240


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

#track


tracks=detect(file_path,yolo_model,deepsort)  #[[[frameId,objectId,(left,top,bottom,right),classId],[]],[],[]]
tracks.pop(0)
tracks.pop(0)



def match(bbox_center,tracks):
        
        loss=[Calcloss(bbox_center,track) for track in tracks]
        
        mini=min(loss)
        # print("distances ",loss)
        # print("the minimum distace got is ",mini)
        if (mini>625):
            print("Limit crossed with loss ",mini)
            return
        return loss.index(mini)


def Calcloss(center1,center2):
    return (center1[0]-center2[0])**2+(center1[1]-center2[1])**2


# what if frame exists in previous frame but not on current frames ?
# there are  two possibilities :
# 1) the vehicle no more exists in incoming frames (most likely)
# 2) the vehicle has been mis identified , (less likely but possible)
# how can we deal with these two problems
#possible solution
# for each frame , if the previous identified vehicle is not found, we push an empty track for all those ids.
# if last 5 tracks are empty, we stop pushing the track, that marks the end of existence of that vehicle in the video
# but if the previous identified vehicle is found in other succesive frames after being missed in some frames :count <5 .

# we fill the hole . (hole is the empty track that was pushed.)
# how to fill hole ?

# we interpolate from all the tracks 



bucket=[{"tracks":[tracks[0][x][2]],"id":tracks[0][x][1],"startIdx":0,"endIdx":None,"lastFill":0,"depths":[],"velocity":[]} for x in range(len(tracks[0]))] # bucket[{"id":id,"tracks":[tracks[0][0][2]]},{},{}]
    
for frameIdx,frame in enumerate(tracks):
    if frameIdx ==0 :
        continue
    for vehicle in frame:
        id_got=vehicle[1]
        left,top,right,bottom=vehicle[2]
        location=None
        for elem_idx,each_elem in enumerate(bucket):
            if each_elem["id"]==id_got:
                location=elem_idx
                break
        
        if location is not None: # found same id on next frame 
            print("Found")
            
             #check if there are holes , 
            lastFill=bucket[location]["lastFill"]
            diff_index=frameIdx-lastFill-1
            if diff_index!=0: #holes present 
                #fill holes 
                fillTracks=bucket[location]["tracks"][:lastFill+1]
                X_values=np.arange(0,lastFill+1).reshape(-1,1)
                
                left_values=np.array([val[0] for val in fillTracks])
                top_values=np.array([val[1] for val in fillTracks])
                right_values=np.array([val[2] for val in fillTracks])
                bottom_values=np.array([val[3] for val in fillTracks])
                
                # print("x",X_values)
                # print("l",left_values)
                # print('t',top_values)
                # print('b',bottom_values)
                # print('r',right_values)

                lr=LinearRegression()
                x0=np.array([lastFill+i for i in range(1,diff_index+1)]).reshape(-1,1)
                #print(x0)
                model=lr.fit(X_values,left_values)
                l=list(model.predict(x0))
                #print("prdicted left is ",l)
                model=lr.fit(X_values,right_values)
                r=list(model.predict(x0))
                #print("prdicted right is ",r)
                model=lr.fit(X_values,top_values)
                t=list(model.predict(x0))
                #print("prdicted top is ",t)
                model=lr.fit(X_values,bottom_values)
                b=list(model.predict(x0))
                for i in range(len(l)):
                    bucket[location]["tracks"][lastFill+i+1]=[l[i],t[i],r[i],b[i]]
            bucket[location]["tracks"].append([left,top,right,bottom])
            bucket[location]["lastFill"]=frameIdx

           


        else:
            print("not found") # if not found, id may have been changed for the same vehicle , so checking the distance 
            
            center=(left+right)/2,(top+bottom)/2
            last_track_center=[((left_+right_)/2,(top_+bottom_)/2) for left_,top_,right_,bottom_ in [y["tracks"][y["lastFill"]] for y in bucket]]
            mat=match(center,last_track_center)
            if mat is not None: # finds a match having sq-distance less than limit (625)
                id_=bucket[mat]["id"]
                for elem_idx,each_elem in enumerate(bucket):
                    if each_elem["id"]==id_:
                        location=elem_idx
                        break
                bucket[location]["tracks"].append(vehicle[2])
                bucket[location]["lastFill"]=frameIdx
            else: # if couldn't found , that's a new vehicle 
                bucket.append({"tracks":[vehicle[2]],"id":vehicle[1],"startIdx":frameIdx,"endIdx":None,"lastFill":frameIdx,"depths":[],"velocity":[]})
                
                for each_elem in bucket:
                    if each_elem["tracks"][-5]==[[] for _ in range(5)]: # if last five tracks are empty
                        each_elem["tracks"]=each_elem["tracks"][:-5] # delete those 
                        each_elem["endIdx"]=frameIdx-5 # change endIdx to 5 idx before 
                        
                    else:

                        each_elem["tracks"].append([]) # if not empty , add empty




# delete all the empty arrays that couldn't be deleted as only 5 arrays could be deleted at once 

for each_elem in bucket:
    lastfill=each_elem["lastFill"]
    each_elem["tracks"]=each_elem["tracks"][:lastFill+1]
    track_length=lastfill-each_elem["startIdx"]+1
    lastfill=track_length-track_length%38 # remove greater than divisible by 38
    
    each_elem["tracks"]=each_elem["tracks"][:lastFill]




                



#loop
frames=[]
count=0




while (video.isOpened()):
    video.set(2,2) # read from third frames
    ret,frame=video.read()
    if not ret:
        print("Couldn't read video ")
        sys.exit()
    frames.append(frame)
    if count%37 !=0 and count==0:
        continue

    
    
    

        
    
    depth0=ret_depth(frames[0:18],depth_model,device)
    depth1=ret_depth(frames[18:38],depth_model,device)
    depth=torch.cat((depth0,depth1))

    depth=(depth-torch.mean(depth))/(torch.std(depth)).detach().cpu()
    # kaslai chaiyeko xa liyera jaao hai id haru


    for each_elem in bucket:
        if count>=each_elem["startIdx"] and count<=each_elem["endIdx"] :
            
            tracks=each_elem["tracks"][each_elem["startIdx"]:count+1]
                
            for i in range(len(tracks)):
                left_,top_,right_,bottom_=tracks[i]
                left_=round(left_/width_ratio)
                top_=round(top_/height_ratio)
                right_=round(right_/width_ratio)
                bottom_=round(bottom_/height_ratio)
                cropped_depth=depth[each_elem["startIdx"]%38][0][top_:bottom_,left_:right_]
                flat_depth=torch.flatten(cropped_depth).numpy()
                avg=np.nanmean(flat_depth).item()
                std=np.std(flat_depth).item()
                filtered=[x for x in flat_depth if x>avg-std]
                avg=[x for x in filtered if x < avg+std]
                each_elem["depths"].append(np.nanmean(avg))
    
# calculate velocity

for each_item in bucket:
    tracks=each_item["tracks"]
    depths=each_item["depths"]
    lengthOfDataset=int(len(tracks)/38)
    tracks=[tracks[38*i:(i+1)*38] for i in range(lengthOfDataset)]
    depths=[depths[38*i:(i+1)*38] for i in range(lengthOfDataset)]
    #dataset=[(tracks[i],depths[i]) for i in range(lengthOfDataset)]
    for i in range(lengthOfDataset):
        track=torch.tensor(tracks[i],dtype=torch.float32,device=device)
        permutted_track=track.permute(1,0)
        permutted_track[0]=permutted_track[0]/frame_width
        permutted_track[2]=permutted_track[2]/frame_width
        permutted_track[1]=permutted_track[1]/frame_height
        permutted_track[3]=permutted_track[3]/frame_height
        track=permutted_track.permute(1,0)
        depth=torch.tensor(depths[i],dtype=torch.float32,device=device)
        
        with torch.no_grad():
            output=model1(track,depth)
            velx,vely,posx,posy=output
            each_item["velocity"].append((velx,vely))
frame_count=0
while (video.isOpened()):
    video.set(2,2)
    ret,frame=video.read()
    if not ret:
        print("Couldn't read video ")
        sys.exit()
    for each_elem in bucket:
        start=each_elem["startIdx"]
        end=each_elem["endIdx"]
        
        if frame_count>=start and frame_count<=end:
            left,top,right,bottom=each_elem["tracks"][frame_count-start]
            velx=each_elem["velocity"][int(frame_count/38)][0]
            vely=each_elem["velocity"][int(frame_count/38)][1]
            cv2.rectangle(frame,(left,top),(right,bottom),RECT_COLOR_BBOX,thickness=5)
            cv2.putText(frame,"V({},{})".format(round(velx,2),vely))
    video_writer.write(frame)

            


    



    
    





                
            



                

            



        






    

    







                    



    

    frames=[]
        