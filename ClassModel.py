import torch
import torch.nn as nn
class myModel(nn.Module):
    #hl_dims are hidden layer dimensions
    def __init__(self,checkpoint_pth,device,batch_size=16,fps=20,output_dims=4,loss=nn.MSELoss()):
        super().__init__()
        self.output_dims=output_dims
        self.checkpoint_path=checkpoint_pth
        
        self.loss=loss
        self.device=device
        self.batch_size=batch_size
        self.fps=torch.tensor(fps).to(self.device)
        self.n1=nn.Sequential(
            
            nn.Linear(38*4+3,256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),

            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.2)
        
        )
        self.n2=nn.Sequential(
            
            nn.Linear(128+3,64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.2),
            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.2),
        
        )
        self.n3=nn.Sequential(
            
            nn.Linear(32+3,16),
            nn.BatchNorm1d(16),
            nn.Dropout(p=0.2),
            nn.Linear(16,8),
            nn.BatchNorm1d(8),
            nn.Dropout(p=0.2),
            nn.Linear(8,4)
        )
       
        
        
        

    def forward(self,track,depths):
        print("in forward size of depths and tracks ",depths.shape,track.shape)
        track=torch.flatten(track,start_dim=1)
        print("after flattening track shape ",track.shape)
        input=torch.cat((track,depths),dim=1)
        print("input forwarded to n1 ",input.shape)
        result=self.n1(input.to(torch.float32))
        print("result obtained from n1 ",result.shape)
        result=self.n2(torch.cat((result,depths),dim=1))
        print("result obtained from n2 ",result.shape)
        result=self.n3(torch.cat((result,depths),dim=1))
        print("result obtained from n3 ",result.shape)
        
        #result=self.n2(torch.cat((result.permute(1,0),self.fps.repeat(1,self.batch_size))).permute(1,0))
        #print("final result shape ",result.shape)
        print(result)
        return result
        


    def training_step(self,batch):
        track,depths,label=batch
        result=self(track,depths)
        loss=self.loss(result.to(torch.float32),label.to(torch.float32))
        print("loss obtatained for this batch is ",loss)
        return loss
    
    def validation_step(self,batch):
        
        track,depths,label=batch
        result=self(track,depths)
        loss=self.loss(result.to(torch.float32),label.to(torch.float32))
        
        
        return loss.detach()
    
    def validation_epoch_end(self, outputs):
        
        
        epoch_loss = torch.stack(outputs).mean()   # Combine losses
        
        return epoch_loss.item()

    def save_model(self,ep,train_loss,validation_loss,optimizer):
        checkpoint={
            'train_loss':train_loss,
            'validation_loss':validation_loss,
            'epoch':ep,
            'optimizer':optimizer.state_dict(),
            'model':self.state_dict(),
        }
        torch.save(checkpoint,self.checkpoint_path)
        
    def load_model(self,optimizer=None):
        self.eval()
        checkpoint=torch.load(self.checkpoint_path)
        self.load_state_dict(checkpoint['model'])
        if optimizer==None:
            return
        optimizer.load_state_dict(checkpoint['optimizer'])
        return optimizer,checkpoint['epoch'],checkpoint['train_loss'],checkpoint['validation_loss']