import torch
import os
from torch.utils.data import DataLoader,random_split
from ClassData import myDataset
from ClassModel import myModel
from DeviceData import DeviceDataLoader
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json

#colab
PATH=os.path.join("/content")
PATHJ=os.path.join("/content","secu")
PATHS=os.path.join(PATH,"drive","MyDrive","State2")
#kaggle
# PATH=os.path.join("/kaggle","working")
# PATHJ=os.path.join(PATH,"Major")
# PATHS=os.path.join(PATH,"models")
#PATH=os.curdir


device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
print(device)    
data_dir=os.path.join(PATH,"clips")
an_dir=os.path.join(PATHJ,"Annotations")
json_dir=os.path.join(PATHJ,"JSON.json")


dataset=myDataset(an_dir,data_dir,json_dir)
dataset_size=len(dataset)
print("length of dataset ",dataset_size)
train_size=int(dataset_size*0.8)
train_ds, val_ds = random_split(dataset, [train_size,dataset_size-train_size])
train_dl=DataLoader(train_ds,batch_size=4,shuffle=True)
val_dl=DataLoader(val_ds,batch_size=4,shuffle=True)

train_dl=DeviceDataLoader(train_dl,device)
val_dl=DeviceDataLoader(val_dl,device)
lr_rate=0.001
try:
    print("Creating saving directory")
    os.mkdir(PATHS)
except Exception as e:
    print(e)
chkpt_file_pth=os.path.join(PATHS,"model")
model=myModel(chkpt_file_pth).to(device)
losses=[]
try:
    with open(os.path.join(PATHS,"losses.json"),'rb') as f:
                losses=pickle.load(f)
except Exception as e:
    print(e)
            


def plot_losses():
    
    plt.plot([loss[1] for loss in losses ], '-bx')
    plt.plot([loss[2] for loss in losses], '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')

def evaluate(model, val_dl):
    model.eval()
    outputs=[]
    for idx,batch in enumerate(val_dl):
      print("validating batch number ",idx+1)
      output=model.validation_step(batch)
      outputs.append(output)
      
      
    
    return model.validation_epoch_end(outputs)


def fit(epochs,optim,learning_rate,model,train_dl,val_dl):
    optimizer=optim(model.parameters(),learning_rate)
    
    
    try:
        print("Loading Model ...")
        optimizer,trained_epoch,last_tl,last_vl=model.load_model(optimizer)
        print(f"Training loss {last_tl} and validation loss {last_vl}for last epoch {trained_epoch} ")
        print("Successfully loaded the model")
        print("Starting from epoch ",trained_epoch+1)
        
    except:
        trained_epoch=-1
        print("Cannot Load Model")
    
    for ep in range(trained_epoch+1,epochs):
        print("epoch",ep)
        model.train()
        
        train_losses=[]
        for idx,batch in enumerate(train_dl):
            
            print("idx",idx)
            optimizer.zero_grad()
            loss=model.training_step(batch)
            l=loss.detach()
            print("loss",l)
            loss.backward()
            optimizer.step() 
            train_losses.append(l)
            #print("average_Loss for last 20 batches",np.average([x.item() for x in train_losses[-20:]]))
        mean_tl=torch.stack(train_losses).mean().item()
        print("Performing Model Evaluation   ... wait ")
        mean_vl=evaluate(model,val_dl)
        print("Saving model")
        model.save_model(ep,mean_tl,mean_vl,optimizer)
        
        
        print("Saved ")
        losses.append((ep,mean_tl,mean_vl))
        
        with open(os.path.join(PATHS,"losses.json"),'wb') as f:
            pickle.dump(losses,f)
        print(f"mean validation loss for this epoch {ep}is {mean_vl} /n mean training loss is {mean_tl}")
        
            
            
        
        

fit(500,torch.optim.Adam,lr_rate,model,train_dl,val_dl)

plot_losses()


