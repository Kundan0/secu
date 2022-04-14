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

camera_speed=45

#load yolov5 tracker
yolo_model = DetectMultiBackend("yolov5m.pt", device=device, dnn=False)

#velocity_model

model1=myModel('./model.zip',device).to(device)
model1.load_model()



#video_info
#video 
file_path='./WR.mp4'
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
        if (mini>900):
            #print("Limit crossed with loss ",mini)
            return
        return loss.index(mini)


def Calcloss(center1,center2):
    return (center1[0]-center2[0])**2+(center1[1]-center2[1])**2


# # what if frame exists in previous frame but not on current frames ?
# # there are  two possibilities :
# # 1) the vehicle no more exists in incoming frames (most likely)
# # 2) the vehicle has been mis identified , (less likely but possible)
# # how can we deal with these two problems
# #possible solution
# # for each frame , if the previous identified vehicle is not found, we push an empty track for all those ids.
# # if last some tracks are empty, we stop pushing the track, that marks the end of existence of that vehicle in the video
# # but if the previous identified vehicle is found in other succesive frames after being missed in some frames :count < some value .

# # we fill the hole . (hole is the empty track that was pushed.)
# # how to fill hole ?

# # we interpolate from all the tracks 



bucket=[{"tracks":[tracks[0][x][2]],"id":tracks[0][x][1],"startIdx":0,"endIdx":None,"lastFill":0,"depths":[],"velocity":[]} for x in range(len(tracks[0]))] # bucket[{"id":id,"tracks":[tracks[0][0][2]]},{},{}]
# print("length of tracks",len(tracks))
# print("bucket",bucket)   
for frameIdx,frame in enumerate(tracks):
    #print("frameIdx outside",frameIdx)
    if frameIdx ==0 :
        continue
    for vehicle in frame:
        
        id_got=vehicle[1]
       # print("for vehicle id ",id_got)
        #print("vehilces position ",vehicle[2])
        left,top,right,bottom=vehicle[2]
        location=None
        for elem_idx,each_elem in enumerate(bucket):
            if each_elem["id"]==id_got:
                location=elem_idx
                break
        
        if location is not None: # found same id on next frame 
            #print("Found")
            
             #check if there are holes ,
            buc_loc=bucket[location] 
            
            starti=buc_loc["startIdx"]
            endi=buc_loc["endIdx"]
            lastFill=buc_loc["lastFill"]
            len_buct=len(buc_loc["tracks"])
            #diff_index=starti+len_buct-lastFill-1
            diff_index=len_buct-(lastFill-starti)-1
            #print(len_buct,lastFill,starti,diff_index,frameIdx)
            #print("diff index ",diff_index)
            if diff_index!=0: #holes present
                #print("regressing for holes") 
                if lastFill-starti<6: # if no elements to linearly regress , start from new
                  bucket[location]={"tracks":[vehicle[2]],"id":vehicle[1],"startIdx":frameIdx,"endIdx":None,"lastFill":frameIdx,"depths":[],"velocity":[]}
                  #print("not enough size to regress")
                else:
                  #fill holes 
                  fillTracks=bucket[location]["tracks"][:lastFill-starti+1]
                  
                  X_values=np.arange(0,lastFill-starti+1).reshape(-1,1)
                  #print("fillTracks",fillTracks)
                  left_values=np.array([val[0] for val in fillTracks])
                  #print("left",left_values)
                  top_values=np.array([val[1] for val in fillTracks])
                  #print("t",top_values)
                  right_values=np.array([val[2] for val in fillTracks])
                  #print("r",right_values)
                  bottom_values=np.array([val[3] for val in fillTracks])
                  #print("b",bottom_values)
                  
                  #print("x",X_values)
                  #print("len l",len(left_values))
                  # print('t',top_values)
                  # print('b',bottom_values)
                  # print('r',right_values)

                  lr=LinearRegression()
                  x0=np.array([i for i in range(lastFill-starti+1,len_buct)]).reshape(-1,1)
                  #print("xo",x0)
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
                  #print('before adding ',bucket[location])
                  #print("length of l",len(l))
                  for i in range(len(l)):
                        bucket[location]["tracks"][lastFill-starti+i+1]=(round(l[i]),round(t[i]),round(r[i]),round(b[i]))
                  #print('after adding ',bucket[location])

                  bucket[location]["tracks"].append(vehicle[2])
                  bucket[location]["lastFill"]=frameIdx
            else:
                bucket[location]["tracks"].append(vehicle[2])
                bucket[location]["lastFill"]=frameIdx
              


        else:
            #print("not found") # if not found, id may have been changed for the same vehicle , so checking the distance 
            
            # center=(left+right)/2,(top+bottom)/2
            # #print("last tracks ",[y["tracks"][y["lastFill"]-y["startIdx"]-1] for y in bucket])
            # last_track_center=[((left_+right_)/2,(top_+bottom_)/2) for left_,top_,right_,bottom_ in [y["tracks"][y["lastFill"]-y["startIdx"]-1] for y in bucket]]
            # mat=match(center,last_track_center)
            # if mat is not None: # finds a match having sq-distance less than limit
            #     id_=bucket[mat]["id"]
            #     for elem_idx,each_elem in enumerate(bucket):
            #         if each_elem["id"]==id_:
            #             location=elem_idx
            #             break
            #     bucket[location]["tracks"].append(vehicle[2])
            #     bucket[location]["lastFill"]=frameIdx
            #     #bucket[location]["endIdx"]+=1
    
            # else: # if couldn't found , that's a new vehicle 
            #     #print(" New vehicle found ,frameIdx inside,",frameIdx)
            bucket.append({"tracks":[vehicle[2]],"id":vehicle[1],"startIdx":frameIdx,"endIdx":None,"lastFill":frameIdx,"depths":[],"velocity":[]})
                
    id_in_bucket=[x["id"] for x in bucket]
   # print(id_in_bucket)
    id_in_frame=[vehicle[1] for vehicle in frame]
    #print(id_in_frame)

    for loc,each_id_in_bucket in enumerate(id_in_bucket):
        if each_id_in_bucket not in id_in_frame and bucket[loc]["endIdx"] is None:
            bucket[loc]["tracks"].append(())
            
    
    for loc,each_elem in enumerate(bucket):
        cut=30
        if each_elem["tracks"][-cut:]==[() for _ in range(cut)]: # if last cut number of frames tracks are empty
            #print('last fifteen empty')
            #iid=each_elem["id"]
            #print("previous id ",iid)
            #each_elem["id"]=each_elem["id"]
            #print('after changing ',each_elem["id"])
            #print("type of id ",type(each_elem["id"]))
            each_elem["tracks"]=each_elem["tracks"][:-cut] # delete those 
            ending=each_elem["endIdx"]
            if ending is not None:
                each_elem["endIdx"]=ending-cut # change endIdx to -cut  idx before 
            else:
                each_elem["endIdx"]=frameIdx-cut

        if each_elem["endIdx"] is not None and len(each_elem["tracks"])<38:
            bucket.pop(loc)
        
    


# # delete all the empty arrays that couldn't be deleted as only 5 arrays could be deleted at once 
#print("removing empty arrays")
for loc,each_elem in enumerate(bucket):
    lastfill=each_elem["lastFill"]
    # print("the original lastfill is",lastfill)
    # print("trying to keep from start to lastfill ,length before ",len(each_elem["tracks"]))
       
      
    
        
    each_elem["tracks"]=each_elem["tracks"][:lastfill-each_elem["startIdx"]+1]
    # print("after ",len(each_elem["tracks"]))
        
    track_length=len(each_elem["tracks"])
    
    
    lastfill=track_length-track_length%38 # remove greater than divisible by 38
    
    
    # print("removing greater than 38")
    each_elem["tracks"]=each_elem["tracks"][:lastfill]
    #print("now length has reduced to ",len(each_elem["tracks"]))
    each_elem["endIdx"]=each_elem["startIdx"]+lastfill-1
    

#print("before deleting ",bucket)

def remove_empty():
    empty_count=0
    loc=0
    for l,each_elem in enumerate(bucket):
        if len(each_elem["tracks"])==0:
            loc=l
            empty_count+=1
        else:
            continue
        if empty_count:
            bucket.pop(loc)
            remove_empty()
        else:
            return 

remove_empty()
#print("after deleting 0 len ",bucket)




#loop
frames=[]
count=-1

unitsize=20

video.set(cv2.CAP_PROP_POS_FRAMES,2) # read from third frames
    
while (video.isOpened()):
    
    ret,frame=video.read()
    if not ret:
        print("Couldn't read video ")
        break
    count+=1
    frames.append(frame)
    
    print("count ",count)
    if (count+1)%unitsize ==0 :
        
        

    
    
    

        
    
        depths=ret_depth(frames[0:unitsize],depth_model,device)
        for i in range(len(depths)):
            depths[i]=(depths[i]-torch.mean(depths[i]))/torch.std(depths[i])
        
        frames=[]
        
        # kaslai chaiyeko xa liyera jaao hai id haru

        print(depths.size())
        for each_elem in (bucket):
            #print("for id ",each_elem["id"])
            tracks=each_elem["tracks"]
            
            for i in range(count-(unitsize-1),count+1):
                if i in range(each_elem["startIdx"],each_elem["endIdx"]+1):
                    #print("         for index ",i)
                    left_,top_,right_,bottom_=tracks[i-each_elem["startIdx"]]
                    #print("             track got ",(left_,top_,right_,bottom_))
                    left_=round(left_/width_ratio)
                    top_=round(top_/height_ratio)
                    right_=round(right_/width_ratio)
                    bottom_=round(bottom_/height_ratio)
                    #print("              after normallizing ",(left_,top_,right_,bottom_))
                    #print("              depth size ",depths.size())
                    cropped_depth=depths[i%unitsize,:,top_:bottom_,left_:right_]
                    #print("              cropped depth size ",cropped_depth.size())
                    flat_depth=torch.flatten(cropped_depth).detach().cpu().numpy()
                    #print("               after flattening ",len(flat_depth)," type ",type(flat_depth))
                    avg=np.nanmean(flat_depth).item()
                    #print("               average calculate ",avg)
                    std=np.std(flat_depth).item()
                    filtered=[x for x in flat_depth if x>avg-std and x < avg+std]
                    #print("                filtered ",len(filtered))
                    #avg=[x for x in filtered if x < avg+std]
                    #print("                 nan mean of avg ",np.nanmean(avg))
                    if each_elem["id"]==7:
                        print("adding depth for frame index ",i," for ")
                    each_elem["depths"].append(np.nanmean(avg))
# for each_elem in bucket:
#     if len(each_elem["depths"])!=len(each_elem["tracks"]):
#         print("Alert error depth count ")
#         print(each_elem["id"]," has ",len(each_elem["depths"])," start at ",each_elem["startIdx"]," end at ", each_elem["endIdx"])
        


print("with depth",bucket)   
for loc,each_elem in enumerate(bucket):
    length_of_track=len(each_elem["tracks"])
    length_of_depth=len(each_elem["depths"])
    print("loc ",loc)
    print("tracklength",length_of_track)
    print("depths length",length_of_depth)
    if (length_of_depth<length_of_track):
        final_length=int(length_of_depth/38)*38
        each_elem["depths"]=each_elem["depths"][:final_length]
        each_elem["tracks"]=each_elem["tracks"][:final_length]
        each_elem["endIdx"]=each_elem["startIdx"]+lastfill-1
    # elif (length_of_depth>length_of_track):
    #     final_length=int(length_of_track/38)*38
    #     each_elem["depths"]=each_elem["depths"][:final_length]
    #     each_elem["tracks"]=each_elem["tracks"][:final_length]
        
for loc,each_elem in enumerate(bucket):
    length_of_track=len(each_elem["tracks"])
    length_of_depth=len(each_elem["depths"])
    print("loc ",loc)
    print("tracklength",length_of_track)
    print("depths length",length_of_depth)



# calculate velocity

for each_item in bucket:
    tracks=each_item["tracks"]
    depths=each_item["depths"]
    lengthOfDataset=int(len(tracks)/38)
    tracks=[tracks[38*i:(i+1)*38] for i in range(lengthOfDataset)]
    depths=[depths[38*i:(i+1)*38] for i in range(lengthOfDataset)]
    #dataset=[(tracks[i],depths[i]) for i in range(lengthOfDataset)]
    for i in range(lengthOfDataset-1,-1,-1):
        track=torch.tensor(tracks[i],dtype=torch.float32,device=device)
        permutted_track=track.permute(1,0)
        permutted_track[0]=permutted_track[0]/frame_width
        permutted_track[2]=permutted_track[2]/frame_width
        permutted_track[1]=permutted_track[1]/frame_height
        permutted_track[3]=permutted_track[3]/frame_height
        track=permutted_track.permute(1,0)
        depth=torch.tensor(depths[i],dtype=torch.float32,device=device)
        
        with torch.no_grad():
            output=model1(track.unsqueeze(0),depth.unsqueeze(0))
            velx,vely,posx,posy=output.squeeze(0)
            each_item["velocity"].append((velx.item(),vely.item()))
print("vel bucket",bucket)
frame_count=0
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
thickness = 1

video.set(cv2.CAP_PROP_POS_FRAMES,2) # read from third frames

while (video.isOpened()):

    ret,frame=video.read()
    if not ret:
        print("Couldn't read video ")
        sys.exit()
    for each_elem in bucket:
        start=each_elem["startIdx"]
        end=each_elem["endIdx"]
        
        if frame_count>=start and frame_count<=end:
            left,top,right,bottom=each_elem["tracks"][frame_count-start]
            velx=each_elem["velocity"][int((frame_count-start)/38)][0]
            vely=each_elem["velocity"][int((frame_count-start)/38)][1]
            if velx>0:
                RECT_COLOR_BBOX=(0,255,0)
            else:
                RECT_COLOR_BBOX=(0,0,255)
            cv2.rectangle(frame,(left,top),(right,bottom),RECT_COLOR_BBOX,thickness=2)
            cv2.putText(frame,"V({},{})".format(round(velx,2),round(vely,2)),(left,top-15),font,fontScale,TEXT_COLOR,thickness,cv2.LINE_AA)
    frame_count+=1
    video_writer.write(frame)

            


    



    
    





                
            



                

            



        






    

    







                    



    

#     frames=[]
        