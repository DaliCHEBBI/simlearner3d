import torch
from pytorch_lightning import LightningModule
from torch import nn

from simlearner3d.models.modules.resnet_fpn import ResNet34Encoder_FPNDecoder, ResNetFPN_8_1
from simlearner3d.models.modules.msaff import MSNet
from simlearner3d.models.modules.unet import UNet
from simlearner3d.models.modules.unetgatedattention import UNetGatedAttention
from simlearner3d.models.modules.decision_net import DecisionNetwork
import torch.nn.functional as F
from simlearner3d.utils import utils
from simlearner3d.utils.utils import coords_grid

log = utils.get_logger(__name__)

MODEL_ZOO = [ResNet34Encoder_FPNDecoder,ResNetFPN_8_1,MSNet,UNet,UNetGatedAttention]


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

NODATA=-9999.0
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

        # training mode 
        self.n_uplets = True

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

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        #corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr #/ torch.sqrt(torch.tensor(D).float())

    def training_step (self, batch, batch_idx: int):

        left,right, disp, masq_occ,_ = batch
        masq_defined= (disp != NODATA)
        device='cuda' if left.is_cuda else 'cpu'
        #images = (2 * (images / 255.0) - 1.0).contiguous()

        fmap1 = self.feature(left)
        fmap2 = self.feature(right)

        fmap1 = F.normalize(fmap1,p=2.0, dim=1)
        fmap2 = F.normalize(fmap2,p=2.0, dim=1)

        B,D,H1,W1  = fmap1.shape
        _,_,_, W2  = fmap2.shape

        coords = coords_grid(B,H1,W1,device) # B,2,H1,W1 

        xgrid0, ygrid = coords.split([1,1], dim=1)

        xgrid = xgrid0 + disp
        # noramlize between [-1,1]

        xgrido = 2*xgrid/(W2-1) - 1
        ygrido = 2*ygrid/(H1-1) - 1

        grid = torch.cat([xgrido,ygrido], dim=1).permute(0,2,3,1) # B,H1,W1, 2
        # compute features at coordinates
        f_map2_pos = F.grid_sample(fmap2, grid, align_corners=True)

        # compute all pairs cosine similarities 
        # B,H,W1,W2
        all_corr = Model.corr(fmap1,fmap2)

        # get all possible negatives 
        # B,1,H,W1
        corr_matching =  torch.sum( fmap1 * f_map2_pos, dim=1) 

        # select non matching cosine similarity values B, H1, W1
        x_lower_bound =  torch.round(xgrid).long().permute(0,2,3,1)
        mask_non_matching = torch.ones_like(all_corr).bool()


        masq_defined = torch.logical_and( masq_defined.permute(0,2,3,1), 
                                        x_lower_bound.ge(0) & x_lower_bound.le(W2-1))

        x_lower_bound[torch.logical_not(masq_defined)] = 0


        mask_non_matching.scatter_(3,x_lower_bound,0)
        corr_matching = corr_matching.unsqueeze(-1).repeat(1,1,1,W2)

        training_loss = self.criterion(corr_matching,all_corr,mask_non_matching)

        training_loss= training_loss.sum().div( 
            torch.logical_and(mask_non_matching, masq_defined).count_nonzero() 
                                            + 1e-12)
        self.log("training_loss",
                 training_loss, 
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True,
                 sync_dist=True)
        return training_loss

    def validation_step(self,batch,batch_idx: int):
        left,right, disp, masq_occ,_ = batch
        masq_defined= (disp != NODATA)
        device='cuda' if left.is_cuda else 'cpu'
        #images = (2 * (images / 255.0) - 1.0).contiguous()

        fmap1 = self.feature(left)
        fmap2 = self.feature(right)

        fmap1 = F.normalize(fmap1,p=2.0, dim=1)
        fmap2 = F.normalize(fmap2,p=2.0, dim=1)

        B,D,H1,W1  = fmap1.shape
        _,_,_, W2  = fmap2.shape

        coords = coords_grid(B,H1,W1,device) # B,2,H1,W1 

        xgrid0, ygrid = coords.split([1,1], dim=1)

        xgrid = xgrid0 + disp
        # noramlize between [-1,1]

        xgrido = 2*xgrid/(W2-1) - 1
        ygrido = 2*ygrid/(H1-1) - 1

        grid = torch.cat([xgrido,ygrido], dim=1).permute(0,2,3,1) # B,H1,W1, 2
        # compute features at coordinates
        f_map2_pos = F.grid_sample(fmap2, grid, align_corners=True)

        # compute all pairs cosine similarities 
        # B,H,W1,W2
        all_corr = Model.corr(fmap1,fmap2)

        # get all possible negatives 
        # B,1,H,W1
        corr_matching =  torch.sum( fmap1 * f_map2_pos, dim=1) 

        # select non matching cosine similarity values B, H1, W1
        x_lower_bound =  torch.round(xgrid).long().permute(0,2,3,1)
        mask_non_matching = torch.ones_like(all_corr).bool()


        masq_defined = torch.logical_and( masq_defined.permute(0,2,3,1), 
                                        x_lower_bound.ge(0) & x_lower_bound.le(W2-1))

        x_lower_bound[torch.logical_not(masq_defined)] = 0


        mask_non_matching.scatter_(3,x_lower_bound,0)
        corr_matching = corr_matching.unsqueeze(-1).repeat(1,1,1,W2)

        validation_loss = self.criterion(corr_matching,all_corr,mask_non_matching)

        validation_loss= validation_loss.sum().div( 
            torch.logical_and(mask_non_matching, masq_defined).count_nonzero() 
                                            + 1e-12)
        self.log("val_loss",
                 validation_loss, 
                 prog_bar=True, 
                 logger=True, 
                 on_step=True, 
                 on_epoch=True, 
                 sync_dist=True)
        return validation_loss
    
    def test_step(self,batch,batch_idx: int):
        left,right, disp, masq_occ,_ = batch
        masq_defined= (disp != NODATA)
        device='cuda' if left.is_cuda else 'cpu'
        #images = (2 * (images / 255.0) - 1.0).contiguous()

        fmap1 = self.feature(left)
        fmap2 = self.feature(right)

        fmap1 = F.normalize(fmap1,p=2.0, dim=1)
        fmap2 = F.normalize(fmap2,p=2.0, dim=1)

        B,D,H1,W1  = fmap1.shape
        _,_,_, W2  = fmap2.shape

        coords = coords_grid(B,H1,W1,device) # B,2,H1,W1 


        xgrid0, ygrid = coords.split([1,1], dim=1)

        xgrid = xgrid0 + disp
        # noramlize between [-1,1]

        xgrido = 2*xgrid/(W2-1) - 1
        ygrido = 2*ygrid/(H1-1) - 1

        grid = torch.cat([xgrido,ygrido], dim=1).permute(0,2,3,1) # B,H1,W1, 2
        # compute features at coordinates
        f_map2_pos = F.grid_sample(fmap2, grid, align_corners=True)

        # compute all pairs cosine similarities 
        # B,H,W1,W2
        all_corr = Model.corr(fmap1,fmap2)


        # get all possible negatives 
        # B,1,H,W1
        corr_matching =  torch.sum( fmap1 * f_map2_pos, dim=1) 

        # select non matching cosine similarity values B, H1, W1
        x_lower_bound =  torch.round(xgrid).long().permute(0,2,3,1)
        mask_non_matching = torch.ones_like(all_corr).bool()


        masq_defined = torch.logical_and( masq_defined.permute(0,2,3,1), 
                                        x_lower_bound.ge(0) & x_lower_bound.le(W2-1))

        x_lower_bound[torch.logical_not(masq_defined)] = 0


        mask_non_matching.scatter_(3,x_lower_bound,0)
        corr_matching = corr_matching.unsqueeze(-1).repeat(1,1,1,W2)

        test_loss = self.criterion(corr_matching,all_corr,mask_non_matching)

        test_loss= test_loss.sum().div( 
            torch.logical_and(mask_non_matching, masq_defined).count_nonzero() 
                                            + 1e-12)
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
        #self.lr = self.hparams.lr  # aliasing for Lightning auto_find_lr
        optimizer = self.hparams.optimizer(
            params=filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
        if self.hparams.lr_scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": self.hparams.lr_scheduler(optimizer),
            "monitor": self.hparams.monitor,
        }