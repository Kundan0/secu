import sys
import json
import pickle
from track import detect
limit=500
def Calcloss(center1,center2):
    return (center1[0]-center2[0])**2+(center1[1]-center2[1])**2
def match(bbox_center,tracks):
        
    loss=[Calcloss(bbox_center,track) for track in tracks]
    print("losses ",loss)
    mini=min(loss)
    if mini<limit:
        return loss.index(mini)
    else:
        return




annotation_data=json.load(open("./Annotations/annotation249.json"))[0]
        
bbox=annotation_data["bbox"]
print(type(bbox))
left=bbox["left"]
top=bbox["top"]
right=bbox["right"]
bottom=bbox["bottom"]
bbox=(left,top,right,bottom)

center=(left+right)/2,(top+bottom)/2

write=True
if not write:
    track_result=detect("./imgs")
    track_result.pop(0)
    track_result.pop(0)
    with open("result.json",'wb') as f:
        pickle.dump(track_result,f)
else:
    track_result=pickle.load(open('result.json','rb'))
#print(track_result)
id=None

third_frame=track_result[0]
third_frame_track_centers=[]
for values in third_frame:
    left,top,right,bottom=values[2]
    third_frame_track_centers.append(((right+left)/2,(bottom+top)/2))



id=third_frame[match(center,third_frame_track_centers)][1] # returning the id of best match vehicle index 1 stores id and match returns the index of vehicle
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
            print("found")
    if not found:
        print("not found")
        try:
            id=frame[match(last_track_center,updated_tracks)][1]
            
            for values in frame:
                
                if values[1]==id:
                    
                    myTracks.append(values[2])
                    last_track_center=((left_+right_)/2,(top_+bottom_)/2)
                    
                
                    
        except:
            print("Vehicle Not found")
            sys.exit()
    
        
        
print(myTracks)
print(len(myTracks))