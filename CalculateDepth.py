# from time import time
# import sys
# import os
# sys.path.append(os.path.abspath('./AdaBins'))

# from AdaBins import model_io
# from AdaBins.models import UnetAdaptiveBins

# from PIL import Image
# import torchvision.transforms as transform
# import torch
# import numpy as np
# import matplotlib.image as mpimg
# import cv2
# from torchvision.transforms import ToPILImage,ToTensor
# unloader = ToPILImage()
# loader = ToTensor()  

# def image_loader(img):
#     size=640, 480
#     inter_tensor=None
#     if (isinstance(img,np.ndarray)):
            
        
#         img=cv2.resize(img,(640,480))
    
#         img=torch.from_numpy((img/255)).to(torch.float32)
#         img=img.permute(2,0,1)
#     elif (isinstance(img,str)):
#             #img=loader(Image.open(img).convert('RGB').resize(size, Image.ANTIALIAS))
#             img=cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2RGB)/255.
#             img=torch.from_numpy(cv2.resize(img,(640,480))).to(torch.float32).permute(2,0,1)
            

#     return img.unsqueeze(0)
    
    
# def ret_depth(batch,model,device):
#     imgs=image_loader(batch)            
    
    
#     start=time()
#     _,depth=model(imgs.to(device))
#     #print(f"took {time()-start}") 
#     #print(depth.squeeze(0).squeeze(0).size())
#     return depth.squeeze(0)

# def load_ADA(pretrained,device):
#     MIN_DEPTH = 1e-3
#     #MAX_DEPTH_NYU = 10
#     MAX_DEPTH_KITTI = 80
#     N_BINS = 256 
    
#     model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
#     model, _, _ = model_io.load_checkpoint(pretrained, model)
#     return model.to(device)


# if __name__=="__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     pretrained = "./AdaBins_kitti.pt"
#     model=load_ADA(pretrained,device)
#     imgs=['./001.jpg']

#     depth=ret_depth(imgs[0],model) #  
#     print(depth)
#     # print(depth.detach().size())
#     # mpimg.imsave('./depth2.jpg',depth.detach(),cmap='gray')

    
    
from time import time
import sys
import os
sys.path.append(os.path.abspath('./AdaBins'))

from AdaBins import model_io
from AdaBins.models import UnetAdaptiveBins

from PIL import Image
import torchvision.transforms as transform
import torch
import numpy as np
import matplotlib.image as mpimg
import cv2
from torchvision.transforms import ToPILImage,ToTensor
unloader = ToPILImage()
loader = ToTensor()  

def image_loader(imgs):
    size=(640, 480)
    inter_tensor=None
    # if (isinstance(img,np.ndarray)):
            
        
    #     img=cv2.resize(img,(640,480))
    
    #     img=torch.from_numpy((img/255)).to(torch.float32)
    #     img=img.permute(2,0,1)
    # elif (isinstance(img,str)):
    #         #img=loader(Image.open(img).convert('RGB').resize(size, Image.ANTIALIAS))
    #         img=cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2RGB)/255.
    #         img=torch.from_numpy(cv2.resize(img,(640,480))).to(torch.float32).permute(2,0,1)
            
    res_images=[cv2.resize(image,size) for image in imgs]
    torch_images=[torch.from_numpy((res_image/255)).permute(2,0,1).unsqueeze(0) for res_image in res_images]
    output=torch.tensor(len(imgs),3,480,640)
    torch.cat(torch_images,out=output)
    return output
    
    
def ret_depth(batch,model,device):

    imgs=image_loader(batch)            
    
    
    start=time()
    with torch.no_grad():
        _,depth=model(imgs.to(device))
    print(f"took {time()-start}") 
    #print(depth.squeeze(0).squeeze(0).size())
    return depth

def load_ADA(pretrained,device):
    MIN_DEPTH = 1e-3
    #MAX_DEPTH_NYU = 10
    MAX_DEPTH_KITTI = 80
    N_BINS = 256 
    
    model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
    model, _, _ = model_io.load_checkpoint(pretrained, model)
    return model.to(device)


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained = "./AdaBins_kitti.pt"
    model=load_ADA(pretrained,device)
    folder='./imgs'
    imgs=[]
    for img in os.listdir(folder):
        filename=folder+"/"+img
        imgs.append(cv2.imread(filename))

    depth=ret_depth(imgs,model) #  
    print(depth.size)
    # print(depth.detach().size())
    # mpimg.imsave('./depth2.jpg',depth.detach(),cmap='gray')

    
    
    

