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
            
        nn.Linear(38*4,256),
        nn.Linear(256,256),
        nn.Linear(256,128),
        nn.Linear(128,32),
        nn.Linear(32,4)
        )
        self.n2=nn.Sequential(
            nn.Linear(5,10),
            nn.Linear(10,4)
        )
        
        
        

    def forward(self,track):
        print("forward first track size ",track.shape)
        track=torch.flatten(track,start_dim=1)
        print("after flattening",track.shape)
        result=self.n1(track.to(torch.float32))
        print("result from n1 ",result.shape)
        result=self.n2(torch.cat((result.permute(1,0),self.fps.repeat(1,self.batch_size))).permute(1,0))
        print("final result shape ",result.shape)
        return result
        


    def training_step(self,batch):
        track,label=batch
        result=self(track)
        loss=self.loss(result.to(torch.float32),label.to(torch.float32))
        
        return loss
    
    def validation_step(self,batch):
        
        track,label=batch
        result=self(track)
        loss=self.loss(result.to(torch.float32),label.to(torch.float32))
        
        print("val loss for this batch ",loss.detach().item())
        return loss.detach()
    
    def validation_epoch_end(self, outputs):
        print("calculating mean val loss")
        
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