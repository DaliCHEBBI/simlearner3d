import torch
from pytorch_lightning import LightningModule
from torch import nn

from simlearner3d.models.modules.resnet_fpn import ResNetFPN_8_1, ResNetFPN_16_4, ResNet34Encoder_FPNDecoder
from simlearner3d.models.modules.msaff import MSNet
from simlearner3d.models.modules.unet import UNet
from simlearner3d.models.modules.unetgatedattention import UNetGatedAttention
from simlearner3d.models.modules.decision_net import DecisionNetwork
import torch.nn.functional as F
from simlearner3d.utils import utils
from simlearner3d.utils.utils import coords_grid
import numpy as np

log = utils.get_logger(__name__)

MODEL_ZOO = [ResNet34Encoder_FPNDecoder, ResNetFPN_8_1,MSNet,UNet,UNetGatedAttention]

NODATA=-9999.0

def get_neural_net_class(class_name: str) -> nn.Module:
    """A Class Factory to class of neural net based on class name.

    :meta private:

    Args:
        class_name (str): the name of the class to get.

    Returns:
        nn.Module: CLass of requested neural network.
    """
    for neural_net_class in MODEL_ZOO:
        if class_name in neural_net_class.__name__:
            return neural_net_class
    raise KeyError(f"Unknown class name {class_name}")


STEPS= [1.0,0.5]
PROBAS=[0.5,0.5]

DEFAULT_MODE="feature"

class Model(LightningModule):
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
        #self.true1=kwargs.get("true1")
        self.false1=kwargs.get("false1")
        self.false2=kwargs.get("false2")
        #self.inplanes = kwargs.get("in_planes")
        self.learning_rate=kwargs.get("learning_rate")
        self.load_pretrained=kwargs.get("load_pretrained")
        self.mode=DEFAULT_MODE
        if kwargs.get("mode"):
            self.mode=kwargs.get("mode")
        #self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.nbsteps=0
        neural_net_class = get_neural_net_class(kwargs.get("neural_net_class_name"))
        self.feature = neural_net_class(**kwargs.get("neural_net_hparams"))
        #self.feature=MSNet(self.inplanes)
        if self.mode=="feature+decision":
            self.decisionNet=DecisionNetwork(2*64)

    def load_trained_assets(self, model_tar : str):
        #device="cuda"
        state_dict = torch.load(model_tar)['state_dict'] # ,map_location=device
        if self.mode=="feature":
            state_dict_feature={k.replace("feature.",""):v for k,v in zip(state_dict.keys(),state_dict.values()) if k.startswith('feature')}
            self.feature.load_state_dict(state_dict_feature)
        else:
            state_dict_feature={k.replace("feature.",""):v for k,v in zip(state_dict.keys(),state_dict.values()) if k.startswith('feature')}
            state_dict_decision={k.replace("decisionNet.",""):v for k,v in zip(state_dict.keys(),state_dict.values()) if k.startswith('decisionNet')}
            self.feature.load_state_dict(state_dict_feature)
            self.decisionNet.load_state_dict(state_dict_decision)

    def training_step(self,batch, batch_idx: int):
        x0,x1,dispnoc0,Mask0,_=batch        
        MaskDef=(dispnoc0!=NODATA)
        device='cuda' if x0.is_cuda else 'cpu'
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)

        B,D,H1,W1  = FeatsL.shape
        _,_,_, W2  = FeatsR.shape

        aStep= np.random.choice(STEPS,p=PROBAS)

        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size(),device=device) + self.false2) * aStep
        RandSens=torch.rand(dispnoc0.size(),device=device)
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0))
        Offset_neg=Offset_neg*RandSens


        coords = coords_grid(B,H1,W1,device) # B,2,H1,W1 
        xgrid0, ygrid = coords.split([1,1], dim=1)
        xgrid = xgrid0 - dispnoc0
        xgrid_pos = torch.clamp(torch.round(xgrid/aStep)*aStep,0,W2-1)
        xgrid_neg = torch.clamp(torch.round((xgrid-Offset_neg)/aStep)*aStep,0,W2-1)
        # noramlize between [-1,1]
        # grid of positive samples
        xgrid_pos = 2*xgrid_pos/(W2-1) - 1
        # grid of negative samples 
        xgrid_neg = 2*xgrid_neg/(W2-1) - 1
        ygrid = 2*ygrid/(H1-1) - 1

        grid_p = torch.cat([xgrid_pos,ygrid], dim=1).permute(0,2,3,1) # B,H1,W1, 2
        grid_n = torch.cat([xgrid_neg,ygrid], dim=1).permute(0,2,3,1) # B,H1,W1, 2
        # compute features at coordinates
        FeatsR_plus = F.grid_sample(FeatsR, grid_p, align_corners=True)
        FeatsR_minus = F.grid_sample(FeatsR, grid_n, align_corners=True)
        if self.mode==DEFAULT_MODE:
            training_loss=self.criterion(FeatsL,
                                           FeatsR_plus,
                                           FeatsR_minus,
                                           OCCLUDED)
        else:

            ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
            ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
            sample = torch.cat((ref_pos, ref_neg), dim=0)
            target = torch.cat((torch.ones(x0.size(),device=device)-OCCLUDED.float(),
                                 torch.zeros(x0.size(),device=device)), 
                                 dim=0)
            training_loss=self.criterion(sample+1e-20, target)*torch.cat((MaskDef,MaskDef),0)
        training_loss=training_loss.mul(MaskDef).sum().div(MaskDef.count_nonzero()+1e-12)
        self.log("training_loss",
                 training_loss, 
                 prog_bar=True,
                 logger=True, 
                 on_step=True, 
                 on_epoch=True,
                 sync_dist=True)
        return training_loss

    def validation_step(self,batch,batch_idx: int):
        x0,x1,dispnoc0,Mask0,_=batch        
        MaskDef=(dispnoc0!=NODATA)
        device='cuda' if x0.is_cuda else 'cpu'
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)

        B,D,H1,W1  = FeatsL.shape
        _,_,_, W2  = FeatsR.shape

        aStep= np.random.choice(STEPS,p=PROBAS)

        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size(),device=device) + self.false2) * aStep
        RandSens=torch.rand(dispnoc0.size(),device=device)
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0))
        Offset_neg=Offset_neg*RandSens

        
        coords = coords_grid(B,H1,W1,device) # B,2,H1,W1 
        xgrid0, ygrid = coords.split([1,1], dim=1)
        xgrid = xgrid0 - dispnoc0
        xgrid_pos = torch.clamp(torch.round(xgrid/aStep)*aStep,0,W2-1)
        xgrid_neg = torch.clamp(torch.round((xgrid-Offset_neg)/aStep)*aStep,0,W2-1)
        # noramlize between [-1,1]
        # grid of positive samples
        xgrid_pos = 2*xgrid_pos/(W2-1) - 1
        # grid of negative samples 
        xgrid_neg = 2*xgrid_neg/(W2-1) - 1
        ygrid = 2*ygrid/(H1-1) - 1

        grid_p = torch.cat([xgrid_pos,ygrid], dim=1).permute(0,2,3,1) # B,H1,W1, 2
        grid_n = torch.cat([xgrid_neg,ygrid], dim=1).permute(0,2,3,1) # B,H1,W1, 2
        # compute features at coordinates
        FeatsR_plus = F.grid_sample(FeatsR, grid_p, align_corners=True)
        FeatsR_minus = F.grid_sample(FeatsR, grid_n, align_corners=True)
        if self.mode==DEFAULT_MODE:
            validation_loss=self.criterion(FeatsL,
                                           FeatsR_plus,
                                           FeatsR_minus,
                                           OCCLUDED)
        else:
            ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
            ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
            sample = torch.cat((ref_pos, ref_neg), dim=0)
            target = torch.cat((torch.ones(x0.size(),device=device)-OCCLUDED.float(),
                                 torch.zeros(x0.size(),device=device)), 
                                 dim=0)
            validation_loss=self.criterion(sample+1e-20, target)*torch.cat((MaskDef,MaskDef),0)
        validation_loss=validation_loss.mul(MaskDef).sum().div(MaskDef.count_nonzero()+1e-12)

        self.log("val_loss",
                 validation_loss, 
                 prog_bar=True,
                 logger=True, 
                 on_step=True, 
                 on_epoch=True,
                 sync_dist=True)
        return validation_loss
    
    def test_step(self,batch,batch_idx: int):
        x0,x1,dispnoc0,Mask0,_=batch        
        MaskDef=(dispnoc0!=NODATA)
        device='cuda' if x0.is_cuda else 'cpu'
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)

        B,D,H1,W1  = FeatsL.shape
        _,_,_, W2  = FeatsR.shape

        aStep= np.random.choice(STEPS,p=PROBAS)

        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size(),device=device) + self.false2) * aStep
        RandSens=torch.rand(dispnoc0.size(),device=device)
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0))
        Offset_neg=Offset_neg*RandSens

        
        coords = coords_grid(B,H1,W1,device) # B,2,H1,W1 
        xgrid0, ygrid = coords.split([1,1], dim=1)
        xgrid = xgrid0 - dispnoc0
        xgrid_pos = torch.clamp(torch.round(xgrid/aStep)*aStep,0,W2-1)
        xgrid_neg = torch.clamp(torch.round((xgrid-Offset_neg)/aStep)*aStep,0,W2-1)
        # noramlize between [-1,1]
        # grid of positive samples
        xgrid_pos = 2*xgrid_pos/(W2-1) - 1
        # grid of negative samples 
        xgrid_neg = 2*xgrid_neg/(W2-1) - 1
        ygrid = 2*ygrid/(H1-1) - 1

        grid_p = torch.cat([xgrid_pos,ygrid], dim=1).permute(0,2,3,1) # B,H1,W1, 2
        grid_n = torch.cat([xgrid_neg,ygrid], dim=1).permute(0,2,3,1) # B,H1,W1, 2
        # compute features at coordinates
        FeatsR_plus = F.grid_sample(FeatsR, grid_p, align_corners=True)
        FeatsR_minus = F.grid_sample(FeatsR, grid_n, align_corners=True)
        if self.mode==DEFAULT_MODE:
            test_loss=self.criterion(FeatsL,
                                           FeatsR_plus,
                                           FeatsR_minus,
                                           OCCLUDED)
        else:
            ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
            ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
            sample = torch.cat((ref_pos, ref_neg), dim=0)
            target = torch.cat((torch.ones(x0.size(),device=device)-OCCLUDED.float(),
                                 torch.zeros(x0.size(),device=device)), 
                                 dim=0)
            test_loss=self.criterion(sample+1e-20, target)*torch.cat((MaskDef,MaskDef),0)
        test_loss=test_loss.mul(MaskDef).sum().div(MaskDef.count_nonzero()+1e-12)

        self.log("test_loss",
                 test_loss, 
                 prog_bar=True,
                 logger=True, 
                 on_step=True, 
                 on_epoch=True,
                 sync_dist=True)
        return test_loss
    
    def forward(self,x):
        f_all=self.feature(x)
        # shape 2,64,w,h
        if self.mode==DEFAULT_MODE:
            return f_all
        ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
        out=self.decisionNet(ref_other)
        return out
    
    """def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150,200],gamma=0.7)
        return [optimizer],[scheduler]"""
    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Returns:
            An optimizer, or a config of a scheduler and an optimizer.

        """
        self.lr = self.hparams.lr  # aliasing for Lightning auto_find_lr
        optimizer = self.hparams.optimizer(
            params=filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
        )
        if self.hparams.lr_scheduler is None:
            return optimizer

        lr_scheduler_partial = self.hparams.lr_scheduler
        if lr_scheduler_partial.func is torch.optim.lr_scheduler.OneCycleLR:
            # OneCycleLR needs the total number of optimizer steps and must be
            # stepped every batch. Let Lightning compute total_steps so that
            # steps_per_epoch/epochs do not have to be set by hand.
            scheduler = lr_scheduler_partial(
                optimizer, total_steps=self.trainer.estimated_stepping_batches
            )
            lr_scheduler_config = {"scheduler": scheduler, "interval": "step"}
        else:
            lr_scheduler_config = {
                "scheduler": lr_scheduler_partial(optimizer),
                "monitor": self.hparams.monitor,
            }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}