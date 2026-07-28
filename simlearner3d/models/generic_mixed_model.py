from numpy import disp
import torch
from pytorch_lightning import LightningModule
from torch import nn

from simlearner3d.models.RAFTStereo.core.raft_stereo import RAFTStereo

import torch.nn.functional as F
from simlearner3d.utils import utils

log = utils.get_logger(__name__)

MODEL_ZOO = [RAFTStereo]


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



class ModelMix(LightningModule):
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
        self.model=kwargs.get("model")
        self.nanvalue=kwargs.get("value_nan")
        self.maxdisp=kwargs.get("max_disparity")
        self.learning_rate=kwargs.get("learning_rate")
        self.load_pretrained=kwargs.get("load_pretrained")
        self.channel=kwargs.get("channel")
        self.nbsteps=0
        neural_net_class = get_neural_net_class(kwargs.get("neural_net_class_name"))
        self.regressor = neural_net_class(**kwargs.get("neural_net_hparams"))

        self.train_nb_iterrations=kwargs.get("train_nb_iterations")
        self.validation_nb_iterrations=kwargs.get("validation_nb_iterations")


    def load_trained_assets(self, model_tar : str):
        #device="cuda"
        state_dict = torch.load(model_tar,map_location="cpu")['state_dict'] # ,map_location=device
        state_dict_new={k.replace("module.",""):v for k,v in zip(state_dict.keys(),state_dict.values())}
        self.regressor.load_state_dict(state_dict_new)


    def sequence_loss(self, flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
        """ Loss function defined over sequence of flow predictions """

        n_predictions = len(flow_preds)
        assert n_predictions >= 1
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1).sqrt()

        # exclude extremly large displacements
        valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
        assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
        assert not torch.isinf(flow_gt[valid.bool()]).any()

        for i in range(n_predictions):
            assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
            # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
            adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
            flow_loss += i_weight * i_loss[valid.bool()].mean()

        epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        return flow_loss, metrics

    def training_step(self,batch, batch_idx: int):
        x0,x1,dispnoc0,valid,_=batch

        flow_preds = self.regressor(x0, x1,
                            iters=self.train_nb_iterrations if self.training else self.validation_nb_iterrations, 
                            test_mode=not self.training)

        training_loss, metrics = self.sequence_loss(flow_preds, dispnoc0, valid)

        self.log("training_loss",
                 training_loss, 
                 prog_bar=True,
                 logger=True, 
                 on_step=True, 
                 on_epoch=True,
                 sync_dist=True)


        for key, value in metrics.items():
            self.log(f"train_{key}", value, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        return training_loss

    def validation_step(self,batch, batch_idx: int):
        x0,x1,dispnoc0,valid,_=batch


        flow_preds = self.regressor(x0, x1,
                            iters=self.train_nb_iterrations if self.training else self.validation_nb_iterrations, 
                            test_mode=not self.training)

        validation_loss, metrics = self.sequence_loss(flow_preds, dispnoc0, valid)

        self.log("validation_loss",
                 validation_loss, 
                 prog_bar=True,
                 logger=True, 
                 on_step=True, 
                 on_epoch=True,
                 sync_dist=True)

        
        for key, value in metrics.items():
            self.log(f"validation_{key}", value, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return validation_loss
    
    def test_step(self,batch, batch_idx: int):
        x0,x1,dispnoc0,valid,_=batch

        flow_preds = self.regressor(x0, x1,
                            iters=self.train_nb_iterrations if self.training else self.validation_nb_iterrations, 
                            test_mode=not self.training)

        test_loss, metrics = self.sequence_loss(flow_preds, dispnoc0, valid)

        self.log("test_loss",
                 test_loss, 
                 prog_bar=True,
                 logger=True, 
                 on_step=True, 
                 on_epoch=True,
                 sync_dist=True)

        
        for key, value in metrics.items():
            self.log(f"test_{key}", value, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return test_loss

    
    def forward(self,imgL,imgR):
        #with torch.no_grad():
        disp = self.regressor(imgL,imgR,
                            iters=self.train_nb_iterrations if self.training else self.validation_nb_iterrations, 
                            test_mode=not self.training)
        return disp
    

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