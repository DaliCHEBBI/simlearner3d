import torch
from pytorch_lightning import LightningModule
from torch import nn

from simlearner3d.models.PSMNet.models import PSMNet

import torch.nn.functional as F
from simlearner3d.utils import utils

log = utils.get_logger(__name__)

MODEL_ZOO = [PSMNet]


def get_neural_net_class(class_name: str) -> nn.Module:
    """A Class Factory to class of neural net based on class name.

    :meta private:

    Args:
        class_name (str): the name of the class to get.

    Returns:
        nn.Module: CLass of requested neural network.
    """
    for neural_net_class in MODEL_ZOO:
        print(class_name, neural_net_class.__name__)
        if class_name in neural_net_class.__name__:
            return neural_net_class
    raise KeyError(f"Unknown class name {class_name}")



class ModelReg(LightningModule):
    """Model training, validation, test.

    Read the Pytorch Lightning docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html

    """

    def __init__(self, **kwargs):
        """Initialization method of the Model lightning module.

        Everything needed to train/evaluate/test/predict with a neural architecture, including
        the architecture class name and its hyperparameter.

        See config files for a list of kwargs.

        """
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        #self.save_hyperparameters(ignore=["criterion"])
        self.criterion = kwargs.get("criterion")
        self.model=kwargs.get("model")
        self.nanvalue=kwargs.get("value_nan")
        self.maxdisp=kwargs.get("max_disparity")
        self.learning_rate=kwargs.get("learning_rate")
        self.load_pretrained=kwargs.get("load_pretrained")
        self.channel=kwargs.get("channel")
        self.nbsteps=0
        neural_net_class = get_neural_net_class(kwargs.get("neural_net_class_name"))
        self.regressor = neural_net_class(**kwargs.get("neural_net_hparams"))

    def load_trained_assets(self, model_tar : str):
        #device="cuda"
        state_dict = torch.load(model_tar,map_location="cpu")['state_dict'] # ,map_location=device
        state_dict_new={k.replace("module.",""):v for k,v in zip(state_dict.keys(),state_dict.values())}
        self.regressor.load_state_dict(state_dict_new)

    def training_step(self,batch, batch_idx: int):
        x0,x1,dispnoc0,_,_=batch
        """if self.channel>1 and x0.shape[1]==1:
            x0=x0.tile((1,self.channel,1,1))
            x1=x1.tile((1,self.channel,1,1)) """ 

        dispnoc0=dispnoc0.squeeze()
        mask = (dispnoc0 < self.maxdisp) * (dispnoc0 >= self.nanvalue) # add non defined values in case where sparse disparity
        mask.detach_()

        if self.model == 'stackhourglass':
            output1, output2, output3 = self.regressor(x0,x1)
            output1 = torch.squeeze(output1,1)
            output2 = torch.squeeze(output2,1)
            output3 = torch.squeeze(output3,1)
            training_loss = 0.5*F.smooth_l1_loss(output1[mask], dispnoc0[mask], size_average=True) \
                + 0.7*F.smooth_l1_loss(output2[mask], dispnoc0[mask], size_average=True) \
                    + F.smooth_l1_loss(output3[mask], dispnoc0[mask], size_average=True) 
        else:
            output = self.regressor(x0,x1)
            output = torch.squeeze(output,1)
            training_loss = F.smooth_l1_loss(output[mask], dispnoc0[mask], size_average=True)

        self.log("training_loss",
                 training_loss, 
                 prog_bar=True,
                 logger=True, 
                 on_step=True, 
                 on_epoch=True,
                 sync_dist=True)
        return training_loss

    """def validation_step(self,batch,batch_idx: int):
        x0,x1,dispnoc0,_,_=batch
        if self.channel>1 and x0.shape[1]==1:
            x0=x0.tile((1,self.channel,1,1))
            x1=x1.tile((1,self.channel,1,1))
                  
        #device='cuda' if x0.is_cuda else 'cpu'
        mask = (dispnoc0 < self.maxdisp) * (dispnoc0!=self.nanvalue) # add non defined values in case where sparse disparity
        mask.detach_()
        if self.model == 'stackhourglass':
            output1, output2, output3 = self.regressor(x0,x1)
            output1 = torch.squeeze(output1,1)
            output2 = torch.squeeze(output2,1)
            output3 = torch.squeeze(output3,1)
            validation_loss = 0.5*F.smooth_l1_loss(output1[mask], dispnoc0[mask], size_average=True) \
                + 0.7*F.smooth_l1_loss(output2[mask], dispnoc0[mask], size_average=True) \
                    + F.smooth_l1_loss(output3[mask], dispnoc0[mask], size_average=True) 
        else:
            output = self.regressor(x0,x1)
            output = torch.squeeze(output,1)
            validation_loss = F.smooth_l1_loss(output[mask], dispnoc0[mask], size_average=True)

        self.log("val_loss",
                 validation_loss.data, 
                 prog_bar=True,
                 logger=True, 
                 on_step=True, 
                 on_epoch=True,
                 sync_dist=True)
        
        return validation_loss.data"""
    

    def validation_step(self,batch,batch_idx: int):
        x0,x1,dispnoc0,_,_=batch
        """if self.channel>1 and x0.shape[1]==1:
            x0=x0.tile((1,self.channel,1,1))
            x1=x1.tile((1,self.channel,1,1))"""
        
        dispnoc0=dispnoc0.squeeze()
        #device='cuda' if x0.is_cuda else 'cpu'
        mask = (dispnoc0 < self.maxdisp) * (dispnoc0 >= self.nanvalue) # add non defined values in case where sparse disparity
        mask.detach_()

        print(mask.shape)

        output = self.regressor(x0,x1)
        output = torch.squeeze(output,1)
        validation_loss = F.smooth_l1_loss(output[mask], dispnoc0[mask], size_average=True)

        self.log("val_loss",
                 validation_loss, 
                 prog_bar=True,
                 logger=True, 
                 on_step=True, 
                 on_epoch=True,
                 sync_dist=True)
        
        return validation_loss
    
    def test_step(self,batch,batch_idx: int):
        x0,x1,dispnoc0,_,_=batch
        """if self.channel>1 and x0.shape[1]==1:
            x0=x0.tile((1,self.channel,1,1))
            x1=x1.tile((1,self.channel,1,1)) """

        dispnoc0=dispnoc0.squeeze()       
        mask = (dispnoc0 < self.maxdisp) * (dispnoc0 >= self.nanvalue) # add non defined values in case where sparse disparity
        mask.detach_()

        if x0.shape[2] % 16 != 0:
            times = x0.shape[2]//16       
            top_pad = (times+1)*16 -x0.shape[2]
        else:
            top_pad = 0

        if x0.shape[3] % 16 != 0:
            times = x0.shape[3]//16                       
            right_pad = (times+1)*16-x0.shape[3]
        else:
            right_pad = 0  

        x0 = F.pad(x0,(0,right_pad, top_pad,0))
        x1 = F.pad(x1,(0,right_pad, top_pad,0))

        with torch.no_grad():
            output3 = self.regressor(x0,x1)
            output3 = torch.squeeze(output3)
        
        if top_pad !=0:
            img = output3[:,top_pad:,:]
        else:
            img = output3

        if len(dispnoc0[mask])==0:
           test_loss = 0
        else:
           test_loss = F.l1_loss(img[mask],dispnoc0[mask])
        self.log("test_loss",
                 test_loss.data, 
                 prog_bar=True,
                 logger=True, 
                 on_step=True, 
                 on_epoch=True,
                 sync_dist=True)
        
        return test_loss
    def forward(self,imgL,imgR):
        #with torch.no_grad():
        disp = self.regressor(imgL,imgR)
        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()
        return pred_disp
    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Returns:
            An optimizer, or a config of a scheduler and an optimizer.

        """
        self.lr = self.hparams.lr  # aliasing for Lightning auto_find_lr
        optimizer = self.hparams.optimizer(
            params=self.parameters(),#filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
        )
        if self.hparams.lr_scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": self.hparams.lr_scheduler(optimizer),
            "monitor": self.hparams.monitor,
        }