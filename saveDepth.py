import torch
from CalculateDepth import ret_depth,load_ADA
import json
import os
import matplotlib.image as mpimg
device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
SAVE_PATH="../DepthTen"
IMG_PATH="/content/clips"
model=load_ADA("./AdaBins_kitti.pt",device)

try:
    os.mkdir(SAVE_PATH)
except Exception as e:
    print(e)
    
annotation_data=json.load(open('./JSON.json'))

for idx,data in enumerate(annotation_data):
    folder=data["folder"]
    save_folder=os.path.join(SAVE_PATH,folder)
    img_folder=os.path.join(IMG_PATH,folder,"imgs")
    
    try:
        os.mkdir(save_folder)
    except:
        pass
    filenames=[os.path.join(img_folder,x)for x in ["003.jpg","021.jpg","039.jpg"]]
    depth=[ret_depth(x,model,device) for x in filenames]
    print('depth ',idx,depth)
    torch.save(depth,os.path.join(save_folder,"depth.pt"))
    
    if idx==2:
        print("idx 2 039 depth",depth[-1])
        print("saved depth shape ",depth[-1].shape)
        mpimg.imsave('./savedDepth.png',depth[-1].detach(),cmap='gray')
        depth=[]
        break
    depth=[]

data=torch.load(os.path.join(save_folder,"depth.pt"))
print("data ",data)
print("last tensor ",data[2])

