import copy
import os
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningModule,
)
import torch
from torch import nn

from simlearner3d.models.generic_model import Model

from simlearner3d.models.modules.msaff import MSNet,MSNETInferenceGatedAttention
from simlearner3d.models.modules.unet import UNet,UNetInference
from simlearner3d.models.modules.unetgatedattention import UNetGatedAttention, UNetInferenceGatedAttention
from simlearner3d.models.modules.decision_net import DecisionNetworkOnCube
from simlearner3d.utils import utils

log = utils.get_logger(__name__)

NEURAL_NET_ARCHITECTURE_CONFIG_GROUP = "neural_net"

MODEL_ZOO = [MSNet,UNet,UNetGatedAttention]
MODEL_INFERENCE_ZOO=[MSNETInferenceGatedAttention,UNetInference,UNetInferenceGatedAttention]


def get_inference_neural_net_class(class_training: nn.Module) -> nn.Module:

    for neural_net_class,neural_inference_net_class in zip(MODEL_ZOO,MODEL_INFERENCE_ZOO):
        if isinstance(class_training,neural_net_class):
            return neural_inference_net_class
        
    raise KeyError(f"Unknown training class name {class_training.__name__}")



def extract(config: DictConfig):
    """
    extracts scripted feature and mlp models from checkpointed pytorchlightning model

    """
    # Set seed for random number generators in pytorch, numpy and python.random

    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    log.info("Extracting models architectures for inference!")
    # Instantiates the Model but overwrites everything with current config,
    # except module related params (nnet architecture)
    kwargs_to_override = copy.deepcopy(model.hparams)
    kwargs_to_override.pop(
        NEURAL_NET_ARCHITECTURE_CONFIG_GROUP, None
    )  # removes that key if it's there
    model = Model.load_from_checkpoint(config.model.ckpt_path, **kwargs_to_override)

    neural_inference_network=get_inference_neural_net_class(model.feature)
    print(neural_inference_network)

    print(kwargs_to_override.keys(),kwargs_to_override.values())
    feature_inference=neural_inference_network(**kwargs_to_override["neural_net_hparams"])

    # copy parameters 
    feature_inference.load_state_dict(model.feature.state_dict())
    

    for p1,p2 in zip (feature_inference.parameters(),model.feature.parameters()):
        assert(torch.equal(p1.cpu(),p2.cpu()))
        
    feature_inference_scrpt=torch.jit.script(feature_inference)

    out_feature_inference=config.model.ckpt_path.replace('.ckpt','_FEATURES.pt')
    torch.jit.save(feature_inference_scrpt,out_feature_inference)

    print("Model Feature is saved as : ", out_feature_inference)

    MODE=model.mode
    if MODE=="feature+decision":
        decision_network_inference=DecisionNetworkOnCube(128)
        
        decision_network_inference.load_state_dict(model.decisionNet.state_dict())
        
        for p1,p2 in zip (decision_network_inference.parameters(),model.decisionNet.parameters()):
            assert(torch.equal(p1.cpu(),p2.cpu())) 
        
        decision_network_inference_scrpt=torch.jit.script(decision_network_inference)
        
        out_decision_inference=config.model.ckpt_path.replace('.ckpt','_DECISION_NET.pt')
        
        torch.jit.save(decision_network_inference_scrpt,out_decision_inference)

        print("Model Decision is saved as : ", out_decision_inference)











