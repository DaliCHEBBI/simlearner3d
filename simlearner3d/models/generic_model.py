import torch
from pytorch_lightning import LightningModule
from torch import nn

from simlearner3d.models.modules.resnet_fpn import ResNetFPN_8_1, ResNetFPN_16_4
from simlearner3d.models.modules.msaff import MSNet
from simlearner3d.models.modules.unet import UNet
from simlearner3d.models.modules.unetgatedattention import UNetGatedAttention
from simlearner3d.models.modules.decision_net import DecisionNetwork
import torch.nn.functional as F
from simlearner3d.utils import utils

log = utils.get_logger(__name__)

MODEL_ZOO = [ResNetFPN_8_1,MSNet,UNet,UNetGatedAttention]


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

    def training_step(self,batch, batch_idx: int):
        x0,x1,dispnoc0,Mask0,x_offset=batch        
        MaskDef=dispnoc0!=0.0
        # ADD DIM 1
        #dispnoc0=dispnoc0.unsqueeze(1)
        # DISPNOC0 ++> DENSE DISPARITY Map 
        #MaskDef=MaskDef.unsqueeze(1)  # ==> defined (occluded + non occluded + non available) disparity values 
        # MASK 0 ==> non available + non occluded areas ==> we search for occluded areas  
        #Mask0=Mask0.unsqueeze(1)
        device='cuda' if x0.is_cuda else 'cpu'
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        #print('OCCLUDEDE SHAPE ',torch.max(OCCLUDED))
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_pos=- (0.5) * torch.rand(dispnoc0.size(),device=device) + (0.5)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size(),device=device) + self.false2)
        RandSens=torch.rand(dispnoc0.size(),device=device)
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0))
        Offset_pos=Offset_pos*RandSens
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0+Offset_pos
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1],device=device)
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlobP=MaskDef*MaskOffPositive
        MaskGlobN=MaskDef*MaskOffNegative
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
            training_loss=self.criterion(sample+1e-20, target)*torch.cat((MaskGlobP,MaskGlobN),0)

        training_loss=training_loss.sum().div(MaskGlobP.count_nonzero()+MaskGlobN.count_nonzero()+1e-12)
        self.log("training_loss",
                 training_loss, 
                 prog_bar=True,
                 logger=True, 
                 on_step=True, 
                 on_epoch=True,
                 sync_dist=True)
        return training_loss

    def validation_step(self,batch,batch_idx: int):
        x0,x1,dispnoc0,Mask0,x_offset=batch
        MaskDef=dispnoc0!=0.0
        # ADD DIM 1
        #dispnoc0=dispnoc0.unsqueeze(1)
        # DISPNOC0 ++> DENSE DISPARITY Map 
        #MaskDef=MaskDef.unsqueeze(1)  # ==> defined (occluded + non occluded + non available) disparity values 
        # MASK 0 ==> non available + non occluded areas ==> we search for occluded areas  
        #Mask0=Mask0.unsqueeze(1)
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        #print('OCCLUDEDE SHAPE ',OCCLUDED.shape)
        # Forward
        device='cuda' if x0.is_cuda else 'cpu'
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_pos=- (0.5) * torch.rand(dispnoc0.size(),device=device) + (0.5)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size(),device=device) + self.false2)
        RandSens=torch.rand(dispnoc0.size(),device=device)
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0))
        Offset_pos=Offset_pos*RandSens
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0+Offset_pos
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1],device=device)
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlobP=MaskDef*MaskOffPositive
        MaskGlobN=MaskDef*MaskOffNegative
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
            validation_loss=self.criterion(sample+1e-20, target)*torch.cat((MaskGlobP,MaskGlobN),0)

        validation_loss=validation_loss.sum().div(MaskGlobP.count_nonzero()
                                                  + MaskGlobN.count_nonzero()+1e-12)
        self.log("val_loss",
                 validation_loss, 
                 prog_bar=True, 
                 logger=True, 
                 on_step=True, 
                 on_epoch=True, 
                 sync_dist=True)
        return validation_loss
    
    def test_step(self,batch,batch_idx: int):
        x0,x1,dispnoc0,Mask0,x_offset=batch
        MaskDef=dispnoc0!=0.0
        # ADD DIM 1
        #dispnoc0=dispnoc0.unsqueeze(1)
        # DISPNOC0 ++> DENSE DISPARITY Map 
        #MaskDef=MaskDef.unsqueeze(1)  # ==> defined (occluded + non occluded + non available) disparity values 
        # MASK 0 ==> non available + non occluded areas ==> we search for occluded areas  
        #Mask0=Mask0.unsqueeze(1)
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        #print('OCCLUDEDE SHAPE ',OCCLUDED.shape)
        # Forward
        device='cuda' if x0.is_cuda else 'cpu'
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_pos=- (0.5) * torch.rand(dispnoc0.size(),device=device) + (0.5)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size(),device=device) + self.false2)
        RandSens=torch.rand(dispnoc0.size(),device=device)
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0))
        Offset_pos=Offset_pos*RandSens
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0+Offset_pos
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1],device=device)
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlobP=MaskDef*MaskOffPositive
        MaskGlobN=MaskDef*MaskOffNegative
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
            test_loss=self.criterion(sample+1e-20, target)*torch.cat((MaskGlobP,MaskGlobN),0)

        test_loss=test_loss.sum().div(MaskGlobP.count_nonzero()
                                                  + MaskGlobN.count_nonzero()+1e-12)
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

        return {
            "optimizer": optimizer,
            "lr_scheduler": self.hparams.lr_scheduler(optimizer),
            "monitor": self.hparams.monitor,
        }