# -*- coding: utf-8 -*-
from pathlib import Path
from  models.UNetDecisionEBM import UnetMlpCubeMatcher,UNetInference
from  models.model import DecisionNetworkOnCube
import torch
import torch.nn as nn
#import tifffile as tff
import numpy as np
import os 
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch.nn.functional as F
import gc
  
        
def MemStatus(loc):
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t -m').readlines()[-1].split()[1:])
    # Memory usage
    print("RAM memory % used before @ "+ loc, round((used_memory/total_memory) * 100, 2))

def collate_func(batch):
    return torch.cat(batch,0)

if __name__ == '__main__':
    TheUnetMlpCubeMatcher=UnetMlpCubeMatcher.load_from_checkpoint(
        "/tmp/PyTorchLightning/PythonProject/trained_models/Shapelearning_v1_with_features/version_0/logs/epoch=5-step=678.ckpt")
    ModelUNETInf=UNetInference()
    ModelDecisionInf=DecisionNetworkOnCube(128)
    if TheUnetMlpCubeMatcher.FEATS_ON:
        ModelCubeInf=Matcher3D(in_channels=2*64+1)
    else:
        ModelCubeInf=Matcher3D()

    ModelUNETInf.load_state_dict(TheUnetMlpCubeMatcher.feature.state_dict())
    ModelDecisionInf.load_state_dict(TheUnetMlpCubeMatcher.decisionNet.state_dict())
    ModelCubeInf.load_state_dict(TheUnetMlpCubeMatcher.Matcher3D.state_dict())
    
    for p1,p2 in zip (ModelUNETInf.parameters(),TheUnetMlpCubeMatcher.feature.parameters()):
        assert(torch.equal(p1,p2))
    for p1,p2 in zip (ModelDecisionInf.parameters(),TheUnetMlpCubeMatcher.decisionNet.parameters()):
        assert(torch.equal(p1,p2))
        
    for p1,p2 in zip (ModelCubeInf.parameters(),TheUnetMlpCubeMatcher.Matcher3D.parameters()):
        assert(torch.equal(p1,p2))
    
    ModelUNETInf=ModelUNETInf.to(torch.device('cpu'))
    ModelDecisionInf=ModelDecisionInf.to(torch.device('cpu'))
    ModelCubeInf=ModelCubeInf.to(torch.device('cpu'))
    
    scriptModuleFeatures=torch.jit.script(ModelUNETInf)

    torch.jit.save(scriptModuleFeatures,'./UNET_FEATURES_DUB_201022.pt')
    
    scriptModuleDECISION=torch.jit.script(ModelDecisionInf)

    torch.jit.save(scriptModuleDECISION,'./UNET_DECISION_NET_DUB_201022.pt')
    
    scriptModuleMATCHER=torch.jit.script(ModelCubeInf)

    if TheUnetMlpCubeMatcher.FEATS_ON:
        torch.jit.save(scriptModuleMATCHER,'./UNET_MATCHER_NET_DUB_FEATS_ON_201022.pt')
    else:
        torch.jit.save(scriptModuleMATCHER,'./UNET_MATCHER_NET_DUB_FEATS_OFF_201022.pt')
    
    
    